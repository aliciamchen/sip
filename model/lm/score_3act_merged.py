#!/usr/bin/env python3
"""
Merged canonical + alternatives scoring for Study 3b (food_inv_desire_3act).

For each scenario, builds a unified action list combining (i) the 3 canonical
actions from `scenarios_3act.csv` and (ii) the unique LM-generated alternatives
from `lm_alternatives_food_inv_desire_3act.csv` (deduped case-insensitively).
The LM then rates this single unified list on access, effort, and V in
separate prompts — so slot 0 (canonical observed action) and slots 1..k (alts)
end up on the same comparative scale by construction.

Three design choices baked in:

1. CANONICAL + ALTS SCORED TOGETHER. The same prompt rates the unified action
   list, giving the LM a single comparative reference frame for all actions
   that will ultimately populate the padded table. This addresses the slot-0
   vs slot-1..k calibration mismatch that arose from the prior split scoring.

2. ACCESS IS EFFORT-MARGINAL. The model treats access as an action property
   modulated by intimacy via (1-I)^gamma in the utility — access(a|s) is
   formally intimacy- and effort-independent. The access scoring prompt
   therefore omits the effort paragraph; access is elicited once per scenario
   and broadcast across effort_condition in the output CSVs.

3. EFFORT IS EFFORT-CONDITIONAL, V IS MOTIVATION-CONDITIONAL. Both are scored
   under the context they genuinely depend on (effort paragraph for effort;
   state paragraph for V). Neither shows intimacy.

Outputs (existing schemas preserved — no downstream loader changes needed):
  - lm_scenario_params_3act.csv (canonical access + effort; access broadcast
    across effort_condition)
  - lm_scenario_params_3act_marginal.csv (canonical access only, no effort_
    condition column — same values as above, kept for Study 3a's loader)
  - lm_scenario_v_3act.csv (canonical V per (scenario, action, motivation))
  - lm_alternatives_features_food_inv_desire_3act.csv (alts access + effort;
    access broadcast across effort_condition; one row per generation cell)
  - lm_alternatives_v_food_inv_desire_3act.csv (alts V per (scenario, observed,
    effort, intimacy, alt_idx, motivation_query))

Cost (Study 3b only, NUM_RUNS=10 per prompt):
  - access: 16 scenarios × 1 × 10 runs = 160 calls
  - effort: 16 × 2 effort × 10 runs   = 320 calls
  - V:      16 × 2 motivation × 10 runs = 320 calls
  - Total: ~800 calls (vs ~1,920 for split scoring; ~60% reduction)

Note on the canonical CSV scope: this script writes the SHARED canonical
tables (lm_scenario_params_3act.csv, lm_scenario_v_3act.csv) that the other
4 inverse experiments (Studies 2, 3a, 4a, 4b) also read. When this script
runs for 3b, the canonical ratings get the comparative context of 3b's
specific alternative set. That's a feature (richer reference points produce
better-calibrated canonical ratings) but worth being aware of: if you later
migrate the other 4 studies to padded-alts, you'd want to re-elicit canonical
with the union of all studies' alts.

Usage:
    uv run python model/lm/score_3act_merged.py --study food_inv_desire_3act

Requires:
    - TOGETHER_API_KEY in env or .env
    - lm_alternatives_food_inv_desire_3act.csv produced by
      generate_alternatives_3act.py
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
    format_access_prompt_variable,
    format_effort_prompt_variable,
    format_v_prompt_variable,
    normalize_access,
    normalize_effort,
    normalize_v,
    parse_action_response_variable,
)
from client import (
    MODEL_ID,
    aggregate_action_ratings,
    get_ratings_concurrent,
    load_api_key,
    numeric_action_schema,
)
from prompts import system_prompt as build_system_prompt


N_ACTIONS_3ACT = 3
EFFORT_CONDITIONS = ["low", "high"]
MOTIVATIONS = ["low", "high"]

_STUDY_CONFIG = {
    "food_inv_desire_3act": {
        "scenarios": "scenarios_3act.csv",
        "alternatives_input": "lm_alternatives_food_inv_desire_3act.csv",
        "canonical_params_output": "lm_scenario_params_3act.csv",
        "canonical_params_marginal_output": "lm_scenario_params_3act_marginal.csv",
        "canonical_v_output": "lm_scenario_v_3act.csv",
        "alternatives_features_output": "lm_alternatives_features_food_inv_desire_3act.csv",
        "alternatives_v_output": "lm_alternatives_v_food_inv_desire_3act.csv",
    },
}


def _norm(text):
    return text.lower().strip()


def _build_merged_actions(scenario_row, alt_rows_for_scenario):
    """Build the unified action list for a scenario.

    Positions 0..2 are the 3 canonical actions (action_0/1/2 from
    scenarios_3act.csv). Positions 3..N are the unique LM-generated alternative
    texts, deduped case-insensitively *and* excluding any alt whose normalized
    text matches one of the canonical actions.

    Returns (merged_action_texts, canonical_norms, alt_norms_in_order):
      - merged_action_texts: list[str] of length 3 + n_unique_alts
      - canonical_norms: list[str] of length 3 (normalized canonical action texts)
      - alt_norms_in_order: list[str] of length n_unique_alts (normalized alt
        texts in the order they appear in merged_action_texts)
    """
    canonical_actions = [scenario_row[f"action_{i}"] for i in range(N_ACTIONS_3ACT)]
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


def _score_scenario(client, scenario_row, alt_rows_for_scenario, system_prompts):
    """Run all three rating types on the merged action list for one scenario.

    Returns a dict:
        {
          "merged_actions":   list of action texts,
          "canonical_norms":  list of 3 normalized canonical action texts,
          "alt_norms_in_order": list of normalized alt texts (positions 3..N),
          "access":     dict[norm] -> {raw, raw_std, n_runs, n_failures}
                         (single value per action; access is effort-marginal),
          "effort":     dict[(effort_cond, norm)] -> {raw, raw_std, n_runs, n_failures},
          "v":          dict[(motivation_query, norm)] -> {raw, raw_std, n_runs, n_failures},
        }
    """
    scenario = scenario_row["scenario_label"]
    merged, canonical_norms, alt_norms = _build_merged_actions(
        scenario_row, alt_rows_for_scenario
    )
    n_actions = len(merged)
    all_norms = canonical_norms + alt_norms

    # --- access: 1 prompt per scenario, no effort paragraph ---
    print(f"  access (1 prompt, n_actions={n_actions}, vignette only)...", flush=True)
    access_agg, access_n_runs, access_n_fail = _score_one_call(
        client,
        system_prompts["access"],
        format_access_prompt_variable(scenario_row["vignette"], merged),
        n_actions,
        label=f"{scenario}/merged/access",
    )
    access = {}
    for i, norm in enumerate(all_norms):
        a_mean, a_std = access_agg[f"action_{i}"]
        access[norm] = {
            "raw": a_mean,
            "raw_std": a_std,
            "n_runs": access_n_runs,
            "n_failures": access_n_fail,
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

    # --- V: 1 prompt per (scenario, motivation_query), with state paragraph ---
    v = {}
    for motivation in MOTIVATIONS:
        state = scenario_row[f"reward_{motivation}"]
        print(
            f"  V (motivation_query={motivation}, n_actions={n_actions})...",
            flush=True,
        )
        v_agg, v_n_runs, v_n_fail = _score_one_call(
            client,
            system_prompts["v"],
            format_v_prompt_variable(scenario_row["vignette"], state, merged),
            n_actions,
            label=f"{scenario}/merged/V[{motivation}]",
        )
        for i, norm in enumerate(all_norms):
            v_mean, v_std = v_agg[f"action_{i}"]
            v[(motivation, norm)] = {
                "raw": v_mean,
                "raw_std": v_std,
                "n_runs": v_n_runs,
                "n_failures": v_n_fail,
            }

    return {
        "merged_actions": merged,
        "canonical_norms": canonical_norms,
        "alt_norms_in_order": alt_norms,
        "access": access,
        "effort": effort,
        "v": v,
    }


def _build_canonical_params_row(
    scenario, effort_cond, action_idx, canonical_norm, access, effort
):
    a = access[canonical_norm]
    e = effort[(effort_cond, canonical_norm)]
    return {
        "scenario_label": scenario,
        "effort_condition": effort_cond,
        "action": action_idx,
        "access_raw": a["raw"],
        "access_raw_std": a["raw_std"],
        "access": normalize_access(a["raw"]) if not np.isnan(a["raw"]) else np.nan,
        "effort_raw": e["raw"],
        "effort_raw_std": e["raw_std"],
        "effort": normalize_effort(e["raw"]) if not np.isnan(e["raw"]) else np.nan,
        "n_runs_access": a["n_runs"],
        "n_runs_effort": e["n_runs"],
        "n_failures_access": a["n_failures"],
        "n_failures_effort": e["n_failures"],
    }


def _build_canonical_marginal_row(scenario, action_idx, canonical_norm, access):
    a = access[canonical_norm]
    return {
        "scenario_label": scenario,
        "action": action_idx,
        "access_raw": a["raw"],
        "access_raw_std": a["raw_std"],
        "access": normalize_access(a["raw"]) if not np.isnan(a["raw"]) else np.nan,
        "n_runs_access": a["n_runs"],
        "n_failures_access": a["n_failures"],
    }


def _build_canonical_v_row(scenario, action_idx, motivation, canonical_norm, v):
    val = v[(motivation, canonical_norm)]
    return {
        "scenario_label": scenario,
        "action": action_idx,
        "motivation": motivation,
        "v_raw": val["raw"],
        "v_raw_std": val["raw_std"],
        "v": normalize_v(val["raw"]) if not np.isnan(val["raw"]) else np.nan,
        "n_runs": val["n_runs"],
        "n_failures": val["n_failures"],
    }


def _build_alt_features_row(alt_row, access, effort):
    """Build an alts-features CSV row, looking up the rating by action_norm.

    If the alt's action_norm matches a canonical action (rare but possible —
    the LM occasionally proposes an alternative that coincides with one of
    the 3 canonical action texts under case-insensitive match), the lookup
    will hit the canonical's rating, which is correct: same physical action,
    same rating.
    """
    norm = _norm(alt_row["action_text"])
    a = access.get(norm)
    e = effort.get((alt_row["effort_condition"], norm))
    if a is None or e is None:
        return None  # action not in the merged list this scenario — shouldn't happen
    return {
        "scenario_label": alt_row["scenario_label"],
        "observed_action": alt_row["observed_action"],
        "effort_condition": alt_row["effort_condition"],
        "intimacy_condition": int(alt_row["intimacy_condition"]),
        "alt_idx": int(alt_row["alt_idx"]),
        "access_raw": a["raw"],
        "access_raw_std": a["raw_std"],
        "access": normalize_access(a["raw"]) if not np.isnan(a["raw"]) else np.nan,
        "effort_raw": e["raw"],
        "effort_raw_std": e["raw_std"],
        "effort": normalize_effort(e["raw"]) if not np.isnan(e["raw"]) else np.nan,
        "n_runs_access": a["n_runs"],
        "n_runs_effort": e["n_runs"],
        "n_failures_access": a["n_failures"],
        "n_failures_effort": e["n_failures"],
    }


def _build_alt_v_row(alt_row, motivation_query, v):
    norm = _norm(alt_row["action_text"])
    val = v.get((motivation_query, norm))
    if val is None:
        return None
    return {
        "scenario_label": alt_row["scenario_label"],
        "observed_action": alt_row["observed_action"],
        "effort_condition": alt_row["effort_condition"],
        "intimacy_condition": int(alt_row["intimacy_condition"]),
        "alt_idx": int(alt_row["alt_idx"]),
        "motivation_query": motivation_query,
        "v_raw": val["raw"],
        "v_raw_std": val["raw_std"],
        "v": normalize_v(val["raw"]) if not np.isnan(val["raw"]) else np.nan,
        "n_runs": val["n_runs"],
        "n_failures": val["n_failures"],
    }


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

    # Load scenarios + alternatives.
    scenarios_path = get_project_root() / "experiments" / cfg["scenarios"]
    alts_path = (
        get_project_root() / "model" / "outputs" / "lm" / cfg["alternatives_input"]
    )
    if not alts_path.exists():
        raise SystemExit(
            f"Alternatives CSV not found at {alts_path}. "
            f"Run model/lm/generate_alternatives_3act.py --study {study} first."
        )
    scenarios_df = pd.read_csv(scenarios_path).set_index("scenario_label", drop=False)
    alts_df = pd.read_csv(alts_path)

    print(
        f"Loaded {len(scenarios_df)} scenarios; {len(alts_df)} alternative rows",
        flush=True,
    )

    output_dir = get_project_root() / "model" / "outputs" / "lm"
    output_dir.mkdir(exist_ok=True)
    paths = {
        "canonical_params": output_dir / cfg["canonical_params_output"],
        "canonical_marginal": output_dir / cfg["canonical_params_marginal_output"],
        "canonical_v": output_dir / cfg["canonical_v_output"],
        "alt_features": output_dir / cfg["alternatives_features_output"],
        "alt_v": output_dir / cfg["alternatives_v_output"],
    }

    # Per-scenario resumability: skip scenarios that appear in ALL 5 output CSVs.
    canonical_params_records, done_canonical_params = _load_existing(
        paths["canonical_params"], ("scenario_label",)
    )
    canonical_marginal_records, _ = _load_existing(
        paths["canonical_marginal"], ("scenario_label",)
    )
    canonical_v_records, done_canonical_v = _load_existing(
        paths["canonical_v"], ("scenario_label",)
    )
    alt_features_records, done_alt_features = _load_existing(
        paths["alt_features"], ("scenario_label",)
    )
    alt_v_records, done_alt_v = _load_existing(paths["alt_v"], ("scenario_label",))

    fully_done = (
        done_canonical_params & done_canonical_v & done_alt_features & done_alt_v
    )
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
    # fully done across all 5 CSVs. This drops stale rows from a prior run
    # (e.g. canonical CSVs from an OLD-prompt elicitation) before we append
    # new merged-scoring rows. Without this filter, records from old runs
    # accumulate in-memory and produce CSVs with duplicate (scenario, ...)
    # keys after the next checkpoint write.
    canonical_params_records = [
        r for r in canonical_params_records if r["scenario_label"] in fully_done_scalars
    ]
    canonical_marginal_records = [
        r
        for r in canonical_marginal_records
        if r["scenario_label"] in fully_done_scalars
    ]
    canonical_v_records = [
        r for r in canonical_v_records if r["scenario_label"] in fully_done_scalars
    ]
    alt_features_records = [
        r for r in alt_features_records if r["scenario_label"] in fully_done_scalars
    ]
    alt_v_records = [
        r for r in alt_v_records if r["scenario_label"] in fully_done_scalars
    ]

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    system_prompts = {
        "access": build_system_prompt("access", n_actions=None),
        "effort": build_system_prompt("effort", n_actions=None),
        "v": build_system_prompt("v", n_actions=None),
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

        result = _score_scenario(client, scenario_row, alt_rows, system_prompts)
        access = result["access"]
        effort = result["effort"]
        v = result["v"]
        canonical_norms = result["canonical_norms"]

        # Build canonical params rows (16 × 2 × 3 = 96 rows for full table).
        for effort_cond in EFFORT_CONDITIONS:
            for action_idx in range(N_ACTIONS_3ACT):
                canonical_params_records.append(
                    _build_canonical_params_row(
                        scenario,
                        effort_cond,
                        action_idx,
                        canonical_norms[action_idx],
                        access,
                        effort,
                    )
                )
        # Marginal access rows (16 × 3 = 48 rows for full table).
        for action_idx in range(N_ACTIONS_3ACT):
            canonical_marginal_records.append(
                _build_canonical_marginal_row(
                    scenario, action_idx, canonical_norms[action_idx], access
                )
            )
        # Canonical V rows (16 × 3 × 2 = 96 for full table).
        for action_idx in range(N_ACTIONS_3ACT):
            for motivation in MOTIVATIONS:
                canonical_v_records.append(
                    _build_canonical_v_row(
                        scenario,
                        action_idx,
                        motivation,
                        canonical_norms[action_idx],
                        v,
                    )
                )

        # Alts features rows (one per generation cell in this scenario).
        for _, alt_row in alt_rows.iterrows():
            r = _build_alt_features_row(alt_row, access, effort)
            if r is not None:
                alt_features_records.append(r)

        # Alts V rows (× 2 motivations per generation cell).
        for _, alt_row in alt_rows.iterrows():
            for motivation in MOTIVATIONS:
                r = _build_alt_v_row(alt_row, motivation, v)
                if r is not None:
                    alt_v_records.append(r)

        # Checkpoint after each scenario.
        pd.DataFrame(canonical_params_records).to_csv(
            paths["canonical_params"], index=False
        )
        pd.DataFrame(canonical_marginal_records).to_csv(
            paths["canonical_marginal"], index=False
        )
        pd.DataFrame(canonical_v_records).to_csv(paths["canonical_v"], index=False)
        pd.DataFrame(alt_features_records).to_csv(paths["alt_features"], index=False)
        pd.DataFrame(alt_v_records).to_csv(paths["alt_v"], index=False)
        print(f"  checkpointed (5 CSVs updated)", flush=True)

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
        default="food_inv_desire_3act",
    )
    args = parser.parse_args()
    main(args.study)
