#!/usr/bin/env python3
"""
Merged canonical + alternatives scoring for the 3-action inverse studies
(1a food_inv_desire, 1b food_inv_joint_de, 2a food_inv_intimacy, 2b
food_inv_joint_ie). Pick the study with --study.

For each scenario, builds a unified action list combining (i) the 3 canonical
actions from `scenarios.csv` and (ii) the unique LM-generated alternatives
from the study's `outputs/lm/<slug>/lm_alternatives.csv` (deduped
case-insensitively). The LM then rates this single unified list on risk, effort,
and g in separate prompts — so slot 0 (canonical observed action) and slots
1..k (alts) end up on the same comparative scale by construction.

Three design choices baked in:

1. CANONICAL + ALTS SCORED TOGETHER. The same prompt rates the unified action
   list, giving the LM a single comparative reference frame for all actions
   that will ultimately populate the padded table. This addresses the slot-0
   vs slot-1..k calibration mismatch that arose from the prior split scoring.

2. RISK IS EFFORT-MARGINAL. The model treats risk as an action property
   modulated by intimacy via (1-I)^gamma in the utility — risk(a|s) is
   formally intimacy- and effort-independent. The risk scoring prompt
   therefore omits the effort paragraph; risk is elicited once per scenario
   and broadcast across effort_condition in the output CSVs.

3. EFFORT IS EFFORT-CONDITIONAL, V IS DESIRE-CONDITIONAL. Both are scored
   under the context they genuinely depend on (effort paragraph for effort;
   state paragraph for V). Neither shows intimacy.

Desire representation: desire enters the utility as w_v · desire · g(a|s). g is
the desire-free goal-satisfaction of the action (replaces the old signed-valence
V); desire is the inferred latent in 1a/1b, and an LM-rated per-condition scalar
in the given-desire studies 2a/2b.

Outputs — all written into the study's own folder, outputs/lm/<slug>/:
  - lm_scenario_params.csv (canonical risk + effort; risk broadcast
    across effort_condition)
  - lm_scenario_params_marginal.csv (canonical risk only, no effort_
    condition column — same values as above, kept for the effort-marginal loader)
  - lm_scenario_g.csv (canonical g per (scenario, action); desire-free)
  - lm_scenario_desire.csv (per (scenario, desire_condition) desire scalar;
    given-desire studies 2a/2b only)
  - lm_alternatives_features.csv (alts risk + effort; risk broadcast
    across effort_condition; the row's cell columns are the study's generation
    cell, plus an effort_condition feature column. For studies whose observer
    INFERS effort, effort is not a generation axis, so each alt gets a feature
    row for BOTH effort conditions.)
  - lm_alternatives_g.csv (alts g per (scenario, observed, <cell cols>,
    alt_idx); desire-free, no desire axis)

Cost (per study, NUM_RUNS=10 per prompt): risk 16 × 10 = 160 calls; effort
16 × 2 × 10 = 320; g 16 × 10 = 160; desire (2a/2b only) 16 × 2 × 10 = 320;
~640-960 calls per study.

Canonical-table scope: the canonical actions are re-scored per study, in the
comparative frame of that study's own alternative set, and written into the
study's folder (NOT a single shared file). This is deliberate — within a study,
canonical and alts share one comparative scale (what the actor softmax needs);
across studies the canonical values may differ by frame, so they are kept
separate rather than overwriting one shared table.

Usage:
    uv run python model/lm/score_merged.py --study food_inv_desire

Requires:
    - TOGETHER_API_KEY in env or .env
    - outputs/lm/<slug>/lm_alternatives.csv produced by
      generate_alternatives.py --study <slug>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from together import Together

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _features_dispatcher import (
    _max_tokens_for,
    format_risk_prompt_variable,
    format_effort_prompt_variable,
    format_g_prompt_variable,
    normalize_risk,
    normalize_effort,
    normalize_g,
    numeric_desire_schema,
    parse_action_response_variable,
    parse_desire_response,
)
from client import (
    MODEL_ID,
    aggregate_action_ratings,
    get_ratings_concurrent,
    load_api_key,
    numeric_action_schema,
)
from prompts import DESIRE_SYSTEM_PROMPT, desire_user_prompt
from prompts import system_prompt as build_system_prompt


N_ACTIONS = 3
# Canonical action names (the scenarios.csv columns), in slot order. The output
# CSVs label the `action` column with these instead of the integer index 0/1/2,
# so the saved tables are traceable to the experiment's variables. The LM-facing
# protocol stays neutral (positional action_0/1/2) to avoid leaking the risk
# level into the ratings.
CANONICAL_ACTIONS = ["no_share", "low_risk_share", "high_risk_share"]
EFFORT_CONDITIONS = ["low", "high"]
DESIRES = ["low", "high"]

# Each study writes its own canonical CSVs into outputs/lm/<slug>/ (re-scored in
# that study's alt set as comparative context). `cell_cols` are the
# generation-cell key columns in the study's alternatives CSV (besides
# scenario_label + observed_action);
# the alt feature/V rows carry these. `effort_inferred` flags studies whose
# observer infers effort: their generation cell does NOT include effort, so the
# alt's effort feature is emitted for BOTH effort conditions (the effort feature
# is always scored per condition; it is a feature axis, not a generation axis).
# `desire_given` flags the studies where desire is observer-visible context
# (2a, 2b): for those the LM additionally rates a per-(scenario, desire
# condition) desire scalar -> lm_scenario_desire.csv. For the inferred-
# desire studies (1a, 1b) desire is the latent and is never elicited. Either
# way, g (goal-satisfaction, desire-free) replaces the old signed-valence V.
_STUDY_CONFIG = {
    "food_inv_desire": {
        "scenarios": "scenarios.csv",
        "alternatives_input": "lm_alternatives.csv",
        "canonical_params_output": "lm_scenario_params.csv",
        "canonical_params_marginal_output": "lm_scenario_params_marginal.csv",
        "canonical_g_output": "lm_scenario_g.csv",
        "alternatives_features_output": "lm_alternatives_features.csv",
        "alternatives_g_output": "lm_alternatives_g.csv",
        "cell_cols": ("effort_condition", "intimacy_condition"),
        "effort_inferred": False,
        "desire_given": False,
    },
    "food_inv_joint_de": {
        "scenarios": "scenarios.csv",
        "alternatives_input": "lm_alternatives.csv",
        "canonical_params_output": "lm_scenario_params.csv",
        "canonical_params_marginal_output": "lm_scenario_params_marginal.csv",
        "canonical_g_output": "lm_scenario_g.csv",
        "alternatives_features_output": "lm_alternatives_features.csv",
        "alternatives_g_output": "lm_alternatives_g.csv",
        "cell_cols": ("intimacy_condition",),
        "effort_inferred": True,
        "desire_given": False,
    },
    "food_inv_intimacy": {
        "scenarios": "scenarios.csv",
        "alternatives_input": "lm_alternatives.csv",
        "canonical_params_output": "lm_scenario_params.csv",
        "canonical_params_marginal_output": "lm_scenario_params_marginal.csv",
        "canonical_g_output": "lm_scenario_g.csv",
        "canonical_desire_output": "lm_scenario_desire.csv",
        "alternatives_features_output": "lm_alternatives_features.csv",
        "alternatives_g_output": "lm_alternatives_g.csv",
        "cell_cols": ("desire_condition", "effort_condition"),
        "effort_inferred": False,
        "desire_given": True,
    },
    "food_inv_joint_ie": {
        "scenarios": "scenarios.csv",
        "alternatives_input": "lm_alternatives.csv",
        "canonical_params_output": "lm_scenario_params.csv",
        "canonical_params_marginal_output": "lm_scenario_params_marginal.csv",
        "canonical_g_output": "lm_scenario_g.csv",
        "canonical_desire_output": "lm_scenario_desire.csv",
        "alternatives_features_output": "lm_alternatives_features.csv",
        "alternatives_g_output": "lm_alternatives_g.csv",
        "cell_cols": ("desire_condition",),
        "effort_inferred": True,
        "desire_given": True,
    },
}


def _norm(text):
    return text.lower().strip()


def _build_merged_actions(scenario_row, alt_rows_for_scenario):
    """Build the unified action list for a scenario.

    Positions 0..2 are the 3 canonical actions (no_share / low_risk_share /
    high_risk_share from scenarios.csv). Positions 3..N are the unique LM-generated alternative
    texts, deduped case-insensitively *and* excluding any alt whose normalized
    text matches one of the canonical actions.

    Returns (merged_action_texts, canonical_norms, alt_norms_in_order):
      - merged_action_texts: list[str] of length 3 + n_unique_alts
      - canonical_norms: list[str] of length 3 (normalized canonical action texts)
      - alt_norms_in_order: list[str] of length n_unique_alts (normalized alt
        texts in the order they appear in merged_action_texts)
    """
    canonical_actions = [scenario_row[c] for c in CANONICAL_ACTIONS]
    canonical_norms = [_norm(a) for a in canonical_actions]
    canonical_norm_set = set(canonical_norms)

    alt_norms_in_order = []
    alt_texts_unique = []
    seen = set()
    for _, r in alt_rows_for_scenario.iterrows():
        norm = _norm(r["action_text"])
        if norm in canonical_norm_set or norm in seen:
            continue
        seen.add(norm)
        alt_norms_in_order.append(norm)
        alt_texts_unique.append(r["action_text"])

    merged = canonical_actions + alt_texts_unique
    return merged, canonical_norms, alt_norms_in_order


def _score_one_call(client, system_prompt, user_prompt, n_actions, label):
    """Single concurrent-runs LM call returning per-action mean/std ratings."""
    ratings, n_failures = get_ratings_concurrent(
        client,
        system_prompt,
        user_prompt,
        lambda t: parse_action_response_variable(t, n_actions),
        max_tokens=_max_tokens_for(n_actions),
        response_format=numeric_action_schema(n_actions),
        label=label,
    )
    agg = aggregate_action_ratings(ratings, n_actions)
    return agg, len(ratings), n_failures


def _score_scenario(client, scenario_row, alt_rows_for_scenario, system_prompts, cfg):
    """Run the rating types on the merged action list for one scenario.

    Returns a dict:
        {
          "merged_actions":   list of action texts,
          "canonical_norms":  list of 3 normalized canonical action texts,
          "alt_norms_in_order": list of normalized alt texts (positions 3..N),
          "risk":     dict[norm] -> {raw, raw_std, n_runs, n_failures}
                         (single value per action; risk is effort-marginal),
          "effort":     dict[(effort_cond, norm)] -> {raw, raw_std, n_runs, n_failures},
          "g":          dict[norm] -> {raw, raw_std, n_runs, n_failures}
                         (single value per action; goal-satisfaction, desire-free),
          "desire":     dict[desire] -> {raw, raw_std, n_runs, n_failures}
                         (per (scenario, desire condition); only for the
                         given-desire studies, else empty),
        }
    """
    scenario = scenario_row["scenario_label"]
    merged, canonical_norms, alt_norms = _build_merged_actions(
        scenario_row, alt_rows_for_scenario
    )
    n_actions = len(merged)
    all_norms = canonical_norms + alt_norms

    # --- risk: 1 prompt per scenario, no effort paragraph ---
    print(f"  risk (1 prompt, n_actions={n_actions}, vignette only)...", flush=True)
    risk_agg, risk_n_runs, risk_n_fail = _score_one_call(
        client,
        system_prompts["risk"],
        format_risk_prompt_variable(scenario_row["vignette"], merged),
        n_actions,
        label=f"{scenario}/merged/risk",
    )
    risk = {}
    for i, norm in enumerate(all_norms):
        a_mean, a_std = risk_agg[f"action_{i}"]
        risk[norm] = {
            "raw": a_mean,
            "raw_std": a_std,
            "n_runs": risk_n_runs,
            "n_failures": risk_n_fail,
        }

    # --- effort: 1 prompt per (scenario, effort_condition) ---
    effort = {}
    for effort_cond in EFFORT_CONDITIONS:
        vignette_with_effort = (
            f"{scenario_row['vignette']} {scenario_row[f'effort_{effort_cond}']}"
        )
        print(f"  effort (effort={effort_cond}, n_actions={n_actions})...", flush=True)
        effort_agg, eff_n_runs, eff_n_fail = _score_one_call(
            client,
            system_prompts["effort"],
            format_effort_prompt_variable(vignette_with_effort, merged),
            n_actions,
            label=f"{scenario}/merged/effort[{effort_cond}]",
        )
        for i, norm in enumerate(all_norms):
            e_mean, e_std = effort_agg[f"action_{i}"]
            effort[(effort_cond, norm)] = {
                "raw": e_mean,
                "raw_std": e_std,
                "n_runs": eff_n_runs,
                "n_failures": eff_n_fail,
            }

    # --- g: 1 prompt per scenario, desire-free (goal-satisfaction) ---
    print(f"  g (1 prompt, n_actions={n_actions}, vignette only)...", flush=True)
    g_agg, g_n_runs, g_n_fail = _score_one_call(
        client,
        system_prompts["g"],
        format_g_prompt_variable(
            scenario_row["vignette"], merged, scenario_row["desire_object"]
        ),
        n_actions,
        label=f"{scenario}/merged/g",
    )
    g = {}
    for i, norm in enumerate(all_norms):
        g_mean, g_std = g_agg[f"action_{i}"]
        g[norm] = {
            "raw": g_mean,
            "raw_std": g_std,
            "n_runs": g_n_runs,
            "n_failures": g_n_fail,
        }

    # --- desire: 1 scalar per (scenario, desire_condition), given-desire only ---
    # In the given-desire studies (2a, 2b) desire is observer-visible context, so
    # the LM rates how much the two people would like the thing under each state. In
    # the inferred-desire studies (1a, 1b) desire is the latent and is skipped.
    desire = {}
    if cfg.get("desire_given", False):
        for desire_cond in DESIRES:
            state = scenario_row[f"desire_{desire_cond}"]
            print(f"  desire (state={desire_cond})...", flush=True)
            ratings, d_n_fail = get_ratings_concurrent(
                client,
                DESIRE_SYSTEM_PROMPT,
                desire_user_prompt(
                    scenario_row["vignette"], state, scenario_row["desire_object"]
                ),
                parse_desire_response,
                max_tokens=64,
                response_format=numeric_desire_schema(),
                label=f"{scenario}/merged/desire[{desire_cond}]",
            )
            vals = [r for r in ratings if r is not None]
            d_mean = float(np.mean(vals)) if vals else np.nan
            d_std = float(np.std(vals)) if vals else np.nan
            desire[desire_cond] = {
                "raw": d_mean,
                "raw_std": d_std,
                "n_runs": len(ratings),
                "n_failures": d_n_fail,
            }

    return {
        "merged_actions": merged,
        "canonical_norms": canonical_norms,
        "alt_norms_in_order": alt_norms,
        "risk": risk,
        "effort": effort,
        "g": g,
        "desire": desire,
    }


def _build_canonical_params_row(
    scenario, effort_cond, action_idx, canonical_norm, risk, effort
):
    a = risk[canonical_norm]
    e = effort[(effort_cond, canonical_norm)]
    return {
        "scenario_label": scenario,
        "effort_condition": effort_cond,
        "action": CANONICAL_ACTIONS[action_idx],
        "risk_raw": a["raw"],
        "risk_raw_std": a["raw_std"],
        "risk": normalize_risk(a["raw"]) if not np.isnan(a["raw"]) else np.nan,
        "effort_raw": e["raw"],
        "effort_raw_std": e["raw_std"],
        "effort": normalize_effort(e["raw"]) if not np.isnan(e["raw"]) else np.nan,
        "n_runs_risk": a["n_runs"],
        "n_runs_effort": e["n_runs"],
        "n_failures_risk": a["n_failures"],
        "n_failures_effort": e["n_failures"],
    }


def _build_canonical_marginal_row(scenario, action_idx, canonical_norm, risk):
    a = risk[canonical_norm]
    return {
        "scenario_label": scenario,
        "action": CANONICAL_ACTIONS[action_idx],
        "risk_raw": a["raw"],
        "risk_raw_std": a["raw_std"],
        "risk": normalize_risk(a["raw"]) if not np.isnan(a["raw"]) else np.nan,
        "n_runs_risk": a["n_runs"],
        "n_failures_risk": a["n_failures"],
    }


def _build_canonical_g_row(scenario, action_idx, canonical_norm, g):
    val = g[canonical_norm]
    return {
        "scenario_label": scenario,
        "action": CANONICAL_ACTIONS[action_idx],
        "g_raw": val["raw"],
        "g_raw_std": val["raw_std"],
        "g": normalize_g(val["raw"]) if not np.isnan(val["raw"]) else np.nan,
        "n_runs": val["n_runs"],
        "n_failures": val["n_failures"],
    }


def _build_canonical_desire_row(scenario, desire_cond, desire):
    val = desire[desire_cond]
    return {
        "scenario_label": scenario,
        "desire_condition": desire_cond,
        "desire_raw": val["raw"],
        "desire_raw_std": val["raw_std"],
        # desire is rated directly on 0-100; the model uses the [0, 1] scale.
        "desire": val["raw"] / 100.0 if not np.isnan(val["raw"]) else np.nan,
        "n_runs": val["n_runs"],
        "n_failures": val["n_failures"],
    }


def _cell_col_values(alt_row, cell_cols):
    """Copy the study's generation-cell columns off an alt row, normalizing
    intimacy_condition to int."""
    out = {}
    for col in cell_cols:
        v = alt_row[col]
        out[col] = int(v) if col == "intimacy_condition" else v
    return out


def _build_alt_features_row(alt_row, risk, effort, cell_cols, effort_cond):
    """Build an alts-features CSV row for a given effort_condition (the effort
    feature axis). risk is effort-marginal (looked up per action_norm);
    effort is looked up at (effort_cond, action_norm). Copies the study's
    generation-cell columns plus an explicit `effort_condition` feature column.

    If the alt's action_norm matches a canonical action (rare but possible — the
    LM occasionally proposes an alternative coinciding with a canonical text
    under case-insensitive match), the lookup hits the canonical's rating, which
    is correct: same physical action, same rating.
    """
    norm = _norm(alt_row["action_text"])
    a = risk.get(norm)
    e = effort.get((effort_cond, norm))
    if a is None or e is None:
        return None  # action not in the merged list this scenario — shouldn't happen
    row = {
        "scenario_label": alt_row["scenario_label"],
        "observed_action": alt_row["observed_action"],
    }
    row.update(_cell_col_values(alt_row, cell_cols))
    row["effort_condition"] = effort_cond  # feature axis (may equal a cell col)
    row["alt_idx"] = int(alt_row["alt_idx"])
    row.update(
        {
            "risk_raw": a["raw"],
            "risk_raw_std": a["raw_std"],
            "risk": normalize_risk(a["raw"]) if not np.isnan(a["raw"]) else np.nan,
            "effort_raw": e["raw"],
            "effort_raw_std": e["raw_std"],
            "effort": normalize_effort(e["raw"]) if not np.isnan(e["raw"]) else np.nan,
            "n_runs_risk": a["n_runs"],
            "n_runs_effort": e["n_runs"],
            "n_failures_risk": a["n_failures"],
            "n_failures_effort": e["n_failures"],
        }
    )
    return row


def _build_alt_g_row(alt_row, g, cell_cols):
    norm = _norm(alt_row["action_text"])
    val = g.get(norm)
    if val is None:
        return None
    row = {
        "scenario_label": alt_row["scenario_label"],
        "observed_action": alt_row["observed_action"],
    }
    row.update(_cell_col_values(alt_row, cell_cols))
    row["alt_idx"] = int(alt_row["alt_idx"])
    row.update(
        {
            "g_raw": val["raw"],
            "g_raw_std": val["raw_std"],
            "g": normalize_g(val["raw"]) if not np.isnan(val["raw"]) else np.nan,
            "n_runs": val["n_runs"],
            "n_failures": val["n_failures"],
        }
    )
    return row


def _load_existing(path, key_cols):
    """Load existing CSV at path; return (list-of-records, set-of-done-keys).

    If file doesn't exist, returns ([], set())."""
    if not Path(path).exists():
        return [], set()
    df = pd.read_csv(path)
    records = df.to_dict("records")
    done = set(tuple(r[k] for k in key_cols) for r in records)
    return records, done


def main(study):
    if study not in _STUDY_CONFIG:
        raise SystemExit(
            f"Unknown study: {study!r}. Supported: {sorted(_STUDY_CONFIG.keys())}"
        )
    cfg = _STUDY_CONFIG[study]
    api_key = load_api_key()

    # Load scenarios + alternatives. All LM outputs for a study live in that
    # study's folder, outputs/lm/<slug>/.
    scenarios_path = get_project_root() / "experiments" / cfg["scenarios"]
    study_dir = get_project_root() / "model" / "outputs" / "lm" / study
    alts_path = study_dir / cfg["alternatives_input"]
    if not alts_path.exists():
        raise SystemExit(
            f"Alternatives CSV not found at {alts_path}. "
            f"Run model/lm/generate_alternatives.py --study {study} first."
        )
    scenarios_df = pd.read_csv(scenarios_path).set_index("scenario_label", drop=False)
    alts_df = pd.read_csv(alts_path)

    print(
        f"Loaded {len(scenarios_df)} scenarios; {len(alts_df)} alternative rows",
        flush=True,
    )

    output_dir = study_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    desire_given = cfg.get("desire_given", False)
    paths = {
        "canonical_params": output_dir / cfg["canonical_params_output"],
        "canonical_marginal": output_dir / cfg["canonical_params_marginal_output"],
        "canonical_g": output_dir / cfg["canonical_g_output"],
        "alt_features": output_dir / cfg["alternatives_features_output"],
        "alt_g": output_dir / cfg["alternatives_g_output"],
    }
    if desire_given:
        paths["canonical_desire"] = output_dir / cfg["canonical_desire_output"]

    # Per-scenario resumability: skip scenarios present in ALL output CSVs.
    canonical_params_records, done_canonical_params = _load_existing(
        paths["canonical_params"], ("scenario_label",)
    )
    canonical_marginal_records, _ = _load_existing(
        paths["canonical_marginal"], ("scenario_label",)
    )
    canonical_g_records, done_canonical_g = _load_existing(
        paths["canonical_g"], ("scenario_label",)
    )
    alt_features_records, done_alt_features = _load_existing(
        paths["alt_features"], ("scenario_label",)
    )
    alt_g_records, done_alt_g = _load_existing(paths["alt_g"], ("scenario_label",))
    if desire_given:
        canonical_desire_records, done_canonical_desire = _load_existing(
            paths["canonical_desire"], ("scenario_label",)
        )
    else:
        canonical_desire_records, done_canonical_desire = [], None

    fully_done = (
        done_canonical_params & done_canonical_g & done_alt_features & done_alt_g
    )
    if desire_given:
        fully_done = fully_done & done_canonical_desire
    if fully_done:
        # Convert from tuple to bare scalar (since key_cols is single)
        fully_done_scalars = {t[0] for t in fully_done}
        print(
            f"{len(fully_done_scalars)} scenarios already complete in all output CSVs — resuming.",
            flush=True,
        )
    else:
        fully_done_scalars = set()

    # Filter records lists to only keep rows for scenarios that are already
    # fully done across all output CSVs. This drops stale rows from a prior run
    # (e.g. canonical CSVs from an OLD-prompt elicitation) before we append
    # new merged-scoring rows. Without this filter, records from old runs
    # accumulate in-memory and produce CSVs with duplicate (scenario, ...)
    # keys after the next checkpoint write.
    def _keep_done(records):
        return [r for r in records if r["scenario_label"] in fully_done_scalars]

    canonical_params_records = _keep_done(canonical_params_records)
    canonical_marginal_records = _keep_done(canonical_marginal_records)
    canonical_g_records = _keep_done(canonical_g_records)
    alt_features_records = _keep_done(alt_features_records)
    alt_g_records = _keep_done(alt_g_records)
    canonical_desire_records = _keep_done(canonical_desire_records)

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    system_prompts = {
        "risk": build_system_prompt("risk", n_actions=None),
        "effort": build_system_prompt("effort", n_actions=None),
        "g": build_system_prompt("g", n_actions=None),
    }

    scenarios_to_run = [s for s in scenarios_df.index if s not in fully_done_scalars]
    print(
        f"\n{len(scenarios_to_run)} scenarios to score "
        f"(total: {len(scenarios_df)}; already done: {len(fully_done_scalars)}).",
        flush=True,
    )

    for sc_idx, scenario in enumerate(scenarios_to_run, start=1):
        scenario_row = scenarios_df.loc[scenario]
        alt_rows = alts_df[alts_df["scenario_label"] == scenario]
        n_unique_alts = (
            alt_rows["action_text"].str.lower().str.strip().drop_duplicates().shape[0]
        )
        print(
            f"\n[{sc_idx}/{len(scenarios_to_run)}] {scenario} "
            f"({len(alt_rows)} alt rows, ~{n_unique_alts} unique)",
            flush=True,
        )

        result = _score_scenario(client, scenario_row, alt_rows, system_prompts, cfg)
        risk = result["risk"]
        effort = result["effort"]
        g = result["g"]
        desire = result["desire"]
        canonical_norms = result["canonical_norms"]

        # Build canonical params rows (16 × 2 × 3 = 96 rows for full table).
        for effort_cond in EFFORT_CONDITIONS:
            for action_idx in range(N_ACTIONS):
                canonical_params_records.append(
                    _build_canonical_params_row(
                        scenario,
                        effort_cond,
                        action_idx,
                        canonical_norms[action_idx],
                        risk,
                        effort,
                    )
                )
        # Marginal risk rows (16 × 3 = 48 rows for full table).
        for action_idx in range(N_ACTIONS):
            canonical_marginal_records.append(
                _build_canonical_marginal_row(
                    scenario, action_idx, canonical_norms[action_idx], risk
                )
            )
        # Canonical g rows (16 × 3 = 48 for full table; desire-free).
        for action_idx in range(N_ACTIONS):
            canonical_g_records.append(
                _build_canonical_g_row(
                    scenario, action_idx, canonical_norms[action_idx], g
                )
            )
        # Canonical desire scalars (16 × 2 = 32; given-desire studies only).
        if desire_given:
            for desire_cond in DESIRES:
                canonical_desire_records.append(
                    _build_canonical_desire_row(scenario, desire_cond, desire)
                )

        # Alts features rows. When effort is inferred (effort not a generation
        # cell col), emit a row per effort_condition (effort is a feature axis);
        # otherwise emit one row at the cell's observed effort_condition.
        for _, alt_row in alt_rows.iterrows():
            if cfg["effort_inferred"]:
                effort_conds = list(EFFORT_CONDITIONS)
            else:
                effort_conds = [alt_row["effort_condition"]]
            for ec in effort_conds:
                r = _build_alt_features_row(alt_row, risk, effort, cfg["cell_cols"], ec)
                if r is not None:
                    alt_features_records.append(r)

        # Alts g rows (one per generation cell; desire-free).
        for _, alt_row in alt_rows.iterrows():
            r = _build_alt_g_row(alt_row, g, cfg["cell_cols"])
            if r is not None:
                alt_g_records.append(r)

        # Checkpoint after each scenario.
        pd.DataFrame(canonical_params_records).to_csv(
            paths["canonical_params"], index=False
        )
        pd.DataFrame(canonical_marginal_records).to_csv(
            paths["canonical_marginal"], index=False
        )
        pd.DataFrame(canonical_g_records).to_csv(paths["canonical_g"], index=False)
        pd.DataFrame(alt_features_records).to_csv(paths["alt_features"], index=False)
        pd.DataFrame(alt_g_records).to_csv(paths["alt_g"], index=False)
        if desire_given:
            pd.DataFrame(canonical_desire_records).to_csv(
                paths["canonical_desire"], index=False
            )
        print("  checkpointed", flush=True)

    print("\n=== Done ===")
    print(f"Wrote:")
    for p in paths.values():
        if Path(p).exists():
            n = len(pd.read_csv(p))
            print(f"  {p.name}  ({n} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=tuple(_STUDY_CONFIG.keys()),
        default="food_inv_desire",
    )
    args = parser.parse_args()
    main(args.study)
