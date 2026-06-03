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

Outputs — two tables, written into the study's own folder, outputs/lm/<slug>/:
  - lm_scenario.csv (the 3 canonical actions: risk + effort + g in one row per
    (scenario, effort_condition, action). risk is scored effort-marginally —
    vignette only, no effort paragraph — then broadcast across effort_condition;
    g is desire-free, repeated across effort_condition.)
  - lm_alternatives.csv (the SAME file generate_alternatives.py wrote, now with
    the alternatives' risk/effort/g columns filled in alongside action_text /
    is_share. effort is a feature axis: for studies whose observer INFERS effort,
    each alternative gets a row for BOTH effort conditions, with risk/g repeated.)
  - lm_scenario_desire.csv (per (scenario, desire_condition) desire scalar;
    given-desire studies 2a/2b only)

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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    NUM_RUNS,
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

# How many scenarios to score concurrently. Each scenario still fans its NUM_RUNS
# calls out internally, so the in-flight request count is ~SCENARIO_WORKERS ×
# NUM_RUNS. Tune to your Together tier with --scenario-workers; if you also run
# several studies as parallel processes, lower this so the product stays under
# the tier's concurrency / RPM limit.
SCENARIO_WORKERS = 4

# Each study writes its own canonical table (lm_scenario.csv) into
# outputs/lm/<slug>/ (re-scored in that study's alt set as comparative context),
# plus the scored alternatives back into lm_alternatives.csv. `cell_cols` are the
# generation-cell key columns in the study's alternatives CSV (besides
# scenario_label + observed_action); the scored alt rows carry these.
# `effort_inferred` flags studies whose observer infers effort: their generation
# cell does NOT include effort, so each alt gets a scored row for BOTH effort
# conditions (effort is a feature axis, not a generation axis; risk/g repeat).
# `desire_given` flags the studies where desire is observer-visible context
# (2a, 2b): for those the LM additionally rates a per-(scenario, desire
# condition) desire scalar -> lm_scenario_desire.csv. For the inferred-
# desire studies (1a, 1b) desire is the latent and is never elicited. Either
# way, g (goal-satisfaction, desire-free) replaces the old signed-valence V.
_STUDY_CONFIG = {
    "food_inv_desire": {
        "scenarios": "scenarios.csv",
        "alternatives": "lm_alternatives.csv",
        "canonical_output": "lm_scenario.csv",
        "cell_cols": ("effort_condition", "intimacy_condition"),
        "effort_inferred": False,
        "desire_given": False,
    },
    "food_inv_joint_de": {
        "scenarios": "scenarios.csv",
        "alternatives": "lm_alternatives.csv",
        "canonical_output": "lm_scenario.csv",
        "cell_cols": ("intimacy_condition",),
        "effort_inferred": True,
        "desire_given": False,
    },
    "food_inv_intimacy": {
        "scenarios": "scenarios.csv",
        "alternatives": "lm_alternatives.csv",
        "canonical_output": "lm_scenario.csv",
        "canonical_desire_output": "lm_scenario_desire.csv",
        "cell_cols": ("desire_condition", "effort_condition"),
        "effort_inferred": False,
        "desire_given": True,
    },
    "food_inv_joint_ie": {
        "scenarios": "scenarios.csv",
        "alternatives": "lm_alternatives.csv",
        "canonical_output": "lm_scenario.csv",
        "canonical_desire_output": "lm_scenario_desire.csv",
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
    # The effort manipulation lives on the low_risk_share action's paragraph
    # (`low_risk_share_effort_{low,high}` in scenarios.csv), the same column
    # generate_alternatives.py appends as the effort context.
    effort = {}
    for effort_cond in EFFORT_CONDITIONS:
        vignette_with_effort = (
            f"{scenario_row['vignette']} "
            f"{scenario_row[f'low_risk_share_effort_{effort_cond}']}"
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


def _build_canonical_row(
    scenario, effort_cond, action_idx, canonical_norm, risk, effort, g
):
    """One canonical (slot-0 reference) row for lm_scenario.csv: risk + effort
    (per effort_condition) + goal-satisfaction g. g is desire-free, so the same
    value is written into both effort_condition rows of an action."""
    a = risk[canonical_norm]
    e = effort[(effort_cond, canonical_norm)]
    gv = g[canonical_norm]
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
        "g_raw": gv["raw"],
        "g_raw_std": gv["raw_std"],
        "g": normalize_g(gv["raw"]) if not np.isnan(gv["raw"]) else np.nan,
        "n_runs_risk": a["n_runs"],
        "n_runs_effort": e["n_runs"],
        "n_runs_g": gv["n_runs"],
        "n_failures_risk": a["n_failures"],
        "n_failures_effort": e["n_failures"],
        "n_failures_g": gv["n_failures"],
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


def _build_alt_row(alt_row, risk, effort, g, cell_cols, effort_cond):
    """One scored alternatives row for lm_alternatives.csv at a given
    effort_condition (the effort feature axis): the alternative's text + is_share
    plus risk/effort/g, all on one row. risk and g are effort-marginal /
    desire-free (the same value across effort rows); effort is looked up at
    (effort_cond, action_norm). Copies the study's generation-cell columns plus
    an explicit `effort_condition` feature column.

    If the alt's action_norm matches a canonical action (rare but possible — the
    LM occasionally proposes an alternative coinciding with a canonical text
    under case-insensitive match), the lookup hits the canonical's rating, which
    is correct: same physical action, same rating. Returns None if the action
    wasn't in the scored merged list (shouldn't happen).
    """
    norm = _norm(alt_row["action_text"])
    a = risk.get(norm)
    e = effort.get((effort_cond, norm))
    gv = g.get(norm)
    if a is None or e is None or gv is None:
        return None
    row = {
        "scenario_label": alt_row["scenario_label"],
        "observed_action": alt_row["observed_action"],
    }
    row.update(_cell_col_values(alt_row, cell_cols))
    row["effort_condition"] = effort_cond  # feature axis (may equal a cell col)
    row["alt_idx"] = int(alt_row["alt_idx"])
    row["action_text"] = alt_row["action_text"]
    row["is_share"] = (
        int(alt_row["is_share"])
        if not pd.isna(alt_row["is_share"])
        else alt_row["is_share"]
    )
    row.update(
        {
            "risk_raw": a["raw"],
            "risk_raw_std": a["raw_std"],
            "risk": normalize_risk(a["raw"]) if not np.isnan(a["raw"]) else np.nan,
            "effort_raw": e["raw"],
            "effort_raw_std": e["raw_std"],
            "effort": normalize_effort(e["raw"]) if not np.isnan(e["raw"]) else np.nan,
            "g_raw": gv["raw"],
            "g_raw_std": gv["raw_std"],
            "g": normalize_g(gv["raw"]) if not np.isnan(gv["raw"]) else np.nan,
            "n_runs_risk": a["n_runs"],
            "n_runs_effort": e["n_runs"],
            "n_runs_g": gv["n_runs"],
            "n_failures_risk": a["n_failures"],
            "n_failures_effort": e["n_failures"],
            "n_failures_g": gv["n_failures"],
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


def _scored_scenarios(path):
    """Scenarios in a canonical/alternatives CSV that are actually SCORED — i.e.
    have a non-null `g`. Mere presence isn't enough: a migrated risk/effort-only
    table has `g`=NaN and must be re-scored, and a row only ever gets `g` once the
    whole feature set was written for it."""
    if not Path(path).exists():
        return set()
    df = pd.read_csv(path)
    if "g" not in df.columns or df.empty:
        return set()
    return set(df[df["g"].notna()]["scenario_label"])


def main(study, scenario_workers=SCENARIO_WORKERS):
    if study not in _STUDY_CONFIG:
        raise SystemExit(
            f"Unknown study: {study!r}. Supported: {sorted(_STUDY_CONFIG.keys())}"
        )
    cfg = _STUDY_CONFIG[study]
    api_key = load_api_key()

    # All LM tables for a study live in its folder, outputs/lm/<slug>/. The
    # alternatives table is both the stage-1 input (action texts from
    # generate_alternatives.py) and the stage-2 output (this script fills in the
    # risk/effort/g columns), so we read it, recover the action list, score, and
    # write it back with the feature columns.
    scenarios_path = get_project_root() / "experiments" / cfg["scenarios"]
    study_dir = get_project_root() / "model" / "outputs" / "lm" / study
    study_dir.mkdir(parents=True, exist_ok=True)
    alts_path = study_dir / cfg["alternatives"]
    if not alts_path.exists():
        raise SystemExit(
            f"Alternatives CSV not found at {alts_path}. "
            f"Run model/lm/generate_alternatives.py --study {study} first."
        )
    scenarios_df = pd.read_csv(scenarios_path).set_index("scenario_label", drop=False)
    alts_df = pd.read_csv(alts_path)

    desire_given = cfg.get("desire_given", False)
    canonical_path = study_dir / cfg["canonical_output"]
    desire_path = study_dir / cfg["canonical_desire_output"] if desire_given else None

    # Recover the stage-1 action list (texts), robust to a prior scoring run that
    # added feature columns / doubled effort rows: select the generation-cell key
    # columns + action_text/is_share and drop duplicates.
    cell_cols = list(cfg["cell_cols"])
    stage1_cols = [
        "scenario_label",
        "observed_action",
        *cell_cols,
        "alt_idx",
        "action_text",
        "is_share",
    ]
    stage1_df = alts_df[stage1_cols].drop_duplicates()
    print(
        f"Loaded {len(scenarios_df)} scenarios; {len(stage1_df)} alternative actions",
        flush=True,
    )

    # Resume: a scenario is done only when it's actually SCORED in every output
    # table — keyed on a non-null `g`, NOT mere presence. (The migrated
    # risk/effort-only tables have rows with `g`=NaN; those must re-score.)
    canonical_records, _ = _load_existing(canonical_path, ("scenario_label",))
    done_canonical = _scored_scenarios(canonical_path)
    if desire_given:
        desire_records, done_desire = _load_existing(desire_path, ("scenario_label",))
        done_desire = {t[0] for t in done_desire}
    else:
        desire_records, done_desire = [], None

    scored_alts_by_scenario = {}
    if "g" in alts_df.columns:
        for s, grp in alts_df[alts_df["g"].notna()].groupby("scenario_label"):
            scored_alts_by_scenario[s] = grp.to_dict("records")
    done_alts = set(scored_alts_by_scenario)

    fully_done = done_canonical & done_alts
    if desire_given:
        fully_done = fully_done & done_desire

    # Drop partials from an interrupted prior run so we re-score them cleanly.
    canonical_records = [
        r for r in canonical_records if r["scenario_label"] in fully_done
    ]
    if desire_given:
        desire_records = [
            r for r in desire_records if r["scenario_label"] in fully_done
        ]
    scored_alts_by_scenario = {
        s: rows for s, rows in scored_alts_by_scenario.items() if s in fully_done
    }

    def _write_checkpoint():
        pd.DataFrame(canonical_records).to_csv(canonical_path, index=False)
        if desire_given:
            pd.DataFrame(desire_records).to_csv(desire_path, index=False)
        # lm_alternatives.csv = scored rows (done scenarios, with features) +
        # stage-1 rows (texts only) for scenarios not yet scored.
        scored_rows = [r for rows in scored_alts_by_scenario.values() for r in rows]
        pending = stage1_df[~stage1_df["scenario_label"].isin(scored_alts_by_scenario)]
        frames = []
        if scored_rows:
            frames.append(pd.DataFrame(scored_rows))
        if len(pending):
            frames.append(pending)
        out = pd.concat(frames, ignore_index=True) if frames else stage1_df.iloc[0:0]
        out.to_csv(alts_path, index=False)

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)
    system_prompts = {
        "risk": build_system_prompt("risk", n_actions=None),
        "effort": build_system_prompt("effort", n_actions=None),
        "g": build_system_prompt("g", n_actions=None),
    }

    scenarios_to_run = [s for s in scenarios_df.index if s not in fully_done]
    print(
        f"\n{len(scenarios_to_run)} scenarios to score "
        f"(total: {len(scenarios_df)}; already done: {len(fully_done)}; "
        f"{scenario_workers} scenarios concurrent × {NUM_RUNS} runs each).",
        flush=True,
    )

    def _process_scenario(scenario):
        """Score one scenario and RETURN its rows. Does NOT touch shared state,
        so it's safe to run on a worker thread; aggregation + checkpointing happen
        on the main thread as results arrive."""
        scenario_row = scenarios_df.loc[scenario]
        alt_rows = stage1_df[stage1_df["scenario_label"] == scenario]
        result = _score_scenario(client, scenario_row, alt_rows, system_prompts, cfg)
        risk, effort, g, desire = (
            result["risk"],
            result["effort"],
            result["g"],
            result["desire"],
        )
        canonical_norms = result["canonical_norms"]
        # Canonical (slot-0) rows -> lm_scenario.csv (risk + effort + g, 96 rows).
        canon_rows = [
            _build_canonical_row(scenario, ec, ai, canonical_norms[ai], risk, effort, g)
            for ec in EFFORT_CONDITIONS
            for ai in range(N_ACTIONS)
        ]
        # Per-condition desire scalars (given-desire studies only).
        desire_rows = (
            [_build_canonical_desire_row(scenario, dc, desire) for dc in DESIRES]
            if desire_given
            else []
        )
        # Scored alternatives (Long): when effort is inferred, emit a row per
        # effort_condition (effort is a feature axis); otherwise one row at the
        # cell's observed effort_condition. risk/g repeat across effort rows.
        alt_out = []
        for _, alt_row in alt_rows.iterrows():
            effort_conds = (
                list(EFFORT_CONDITIONS)
                if cfg["effort_inferred"]
                else [alt_row["effort_condition"]]
            )
            for ec in effort_conds:
                r = _build_alt_row(alt_row, risk, effort, g, cfg["cell_cols"], ec)
                if r is not None:
                    alt_out.append(r)
        return scenario, canon_rows, desire_rows, alt_out

    # Score up to `scenario_workers` scenarios concurrently (each still fans its
    # NUM_RUNS calls out internally). Aggregate + checkpoint on the main thread as
    # each scenario completes, so no lock is needed on the shared records.
    done_count = 0
    with ThreadPoolExecutor(max_workers=max(1, scenario_workers)) as ex:
        futures = {ex.submit(_process_scenario, s): s for s in scenarios_to_run}
        for fut in as_completed(futures):
            scenario, canon_rows, desire_rows, alt_out = fut.result()
            canonical_records.extend(canon_rows)
            if desire_given:
                desire_records.extend(desire_rows)
            scored_alts_by_scenario[scenario] = alt_out
            done_count += 1
            _write_checkpoint()
            print(
                f"  [{done_count}/{len(scenarios_to_run)}] {scenario} done — checkpointed",
                flush=True,
            )

    print("\n=== Done ===")
    print("Wrote:")
    out_paths = [canonical_path, alts_path] + ([desire_path] if desire_given else [])
    for p in out_paths:
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
    parser.add_argument(
        "--scenario-workers",
        type=int,
        default=SCENARIO_WORKERS,
        help=(
            "How many scenarios to score concurrently (in-flight requests ≈ this "
            f"× NUM_RUNS={NUM_RUNS}). Lower it if running several studies in "
            "parallel so the total stays under your Together tier's limit."
        ),
    )
    args = parser.parse_args()
    main(args.study, scenario_workers=args.scenario_workers)
