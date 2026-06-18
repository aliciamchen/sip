#!/usr/bin/env python3
"""
Per-run scoring of the K-run simulated-observer elicitation for the 3-action
inverse studies (1a food_inv_desire, 1b food_inv_joint_de, 2a food_inv_intimacy,
2b food_inv_joint_ie). Pick the study with --study.

This is the second step of the two-step elicitation. The first step
(generate_alternatives.py) elicited, for each (scenario × condition) cell, K
independent alternative sets — one per elicitation run, tagged with `run_id`.
Here, for each (scenario, run) we build a unified action list — the 3 canonical
actions from `scenarios.csv` plus that run's unique alternatives — and have the
LM rate it ONCE on risk, effort, and goal-satisfaction g (slot 0 = the observed
canonical action, slots 1..k = the run's alternatives, all on one comparative
scale). There is NO inner rating-averaging: each run is a single scoring pass, so
the run-to-run spread of both alternatives AND feature scores becomes part of the
model's predicted distribution (the simulated-observer mixture).

Design choices carried over from the single-run pipeline:
  1. Canonical + alts scored together (one comparative reference frame).
  2. Risk is effort-marginal (vignette only, no effort paragraph), broadcast
     across effort_condition.
  3. Effort is effort-conditional; g is desire-free. Neither shows intimacy.

Given-magnitude scalars (run-independent, scored once):
  - given-desire studies (2a, 2b): per-(scenario, desire_condition) desire scalar.
  - given-relationship studies (1a, 1b): per-level intimacy scalar, rated from the
    DE-ANCHORED relationship descriptors (rating the anchored descriptor would be
    circular). Scenario-independent → 4 values.

Output (one folder per study, outputs/lm/<slug>/):
  - lm_runs.jsonl — one record per (run_id, cell), each carrying the run's scored
    actions (slot 0 canonical + slots 1..k alternatives). Consumed by the run-axis
    table loaders in model/tables.py.
  - lm_given.json — the study's given-magnitude scalars (`desire` and/or
    `relationship`).

Usage:
    uv run python model/lm/score_merged.py --study food_inv_desire
    # K runs / temperature / concurrency via env: K_RUNS, ALT_T, --scenario-workers

Requires:
    - TOGETHER_API_KEY in env or .env
    - outputs/lm/<slug>/lm_alternatives.csv produced by
      generate_alternatives.py --study <slug> (now carrying a run_id column)
"""

import argparse
import itertools
import json
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
    format_effort_prompt_variable,
    format_g_prompt_variable,
    format_risk_prompt_variable,
    normalize_effort,
    normalize_g,
    normalize_risk,
    numeric_desire_schema,
    numeric_intimacy_schema,
    parse_action_response_variable,
    parse_desire_response,
    parse_intimacy_response,
)
from client import (
    MODEL_ID,
    aggregate_action_ratings,
    get_ratings_concurrent,
    load_api_key,
    numeric_action_schema,
)
from prompts import (
    DESIRE_SYSTEM_PROMPT,
    INTIMACY_SYSTEM_PROMPT,
    RELATIONSHIP_DESCRIPTORS_NOANCHOR,
    desire_user_prompt,
    relationship_user_prompt,
)
from prompts import system_prompt as build_system_prompt

N_ACTIONS = 3
CANONICAL_ACTIONS = ["no_share", "low_risk_share", "high_risk_share"]
# is_share for the canonical actions (no_share is the one non-sharing action).
CANONICAL_IS_SHARE = {"no_share": 0, "low_risk_share": 1, "high_risk_share": 1}
EFFORT_CONDITIONS = ["low", "high"]
DESIRES = ["low", "high"]
INTIMACY_LEVELS = ["max_formal", "neither", "somewhat_intimate", "max_intimate"]
# Levels for each generation-cell condition column, used to enumerate cells.
_LEVELS = {
    "desire_condition": DESIRES,
    "effort_condition": EFFORT_CONDITIONS,
    "intimacy_condition": INTIMACY_LEVELS,
}

SCENARIO_WORKERS = 4

# Per-study config. `cell_cols` are the generation-cell key columns in the
# alternatives CSV (besides scenario_label + observed_action + run_id).
# `effort_inferred` flags studies whose observer infers effort: their generation
# cell does NOT include effort, so each cell emits a record for BOTH effort
# conditions (effort is a feature axis; risk/g repeat). `desire_given` /
# `relationship_given` select which given-magnitude scalar block lm_given.json
# carries.
_STUDY_CONFIG = {
    "food_inv_desire": {
        "cell_cols": ("effort_condition", "intimacy_condition"),
        "effort_inferred": False,
        "desire_given": False,
        "relationship_given": True,
    },
    "food_inv_joint_de": {
        "cell_cols": ("intimacy_condition",),
        "effort_inferred": True,
        "desire_given": False,
        "relationship_given": True,
    },
    "food_inv_intimacy": {
        "cell_cols": ("desire_condition", "effort_condition"),
        "effort_inferred": False,
        "desire_given": True,
        "relationship_given": False,
    },
    "food_inv_joint_ie": {
        "cell_cols": ("desire_condition",),
        "effort_inferred": True,
        "desire_given": True,
        "relationship_given": False,
    },
}


def _norm(text):
    return text.lower().strip()


def _build_merged_actions(scenario_row, alt_rows_for_run):
    """Unified action list for one (scenario, run): the 3 canonical actions
    followed by the run's unique alternative texts (deduped case-insensitively
    and excluding any alt matching a canonical text). Returns
    (merged_action_texts, canonical_norms, alt_norms_in_order)."""
    canonical_actions = [scenario_row[c] for c in CANONICAL_ACTIONS]
    canonical_norms = [_norm(a) for a in canonical_actions]
    canonical_norm_set = set(canonical_norms)

    alt_norms_in_order, alt_texts_unique, seen = [], [], set()
    for _, r in alt_rows_for_run.iterrows():
        norm = _norm(r["action_text"])
        if norm in canonical_norm_set or norm in seen:
            continue
        seen.add(norm)
        alt_norms_in_order.append(norm)
        alt_texts_unique.append(r["action_text"])

    merged = canonical_actions + alt_texts_unique
    return merged, canonical_norms, alt_norms_in_order


def _score_one_call(client, system_prompt, user_prompt, n_actions, label):
    """Single LM scoring pass (num_runs=1, no inner averaging) returning per-action
    ratings. The K elicitation runs are the variation axis, not repeated calls."""
    ratings, n_failures = get_ratings_concurrent(
        client,
        system_prompt,
        user_prompt,
        lambda t: parse_action_response_variable(t, n_actions),
        num_runs=1,
        max_tokens=_max_tokens_for(n_actions),
        response_format=numeric_action_schema(n_actions),
        label=label,
    )
    agg = aggregate_action_ratings(ratings, n_actions)
    return agg, n_failures


def _score_actions(client, scenario_row, alt_rows_for_run, system_prompts):
    """Score risk / effort / g on one (scenario, run)'s merged action list.

    Returns {merged_actions, canonical_norms, alt_norms_in_order, risk, effort, g}
    where risk/g are dict[norm] -> normalized [0,1] value (single value per
    action) and effort is dict[(effort_cond, norm)] -> normalized [0,1]."""
    scenario = scenario_row["scenario_label"]
    merged, canonical_norms, alt_norms = _build_merged_actions(
        scenario_row, alt_rows_for_run
    )
    n_actions = len(merged)
    all_norms = canonical_norms + alt_norms

    # risk: one prompt, vignette only (effort-marginal).
    risk_agg, _ = _score_one_call(
        client,
        system_prompts["risk"],
        format_risk_prompt_variable(scenario_row["vignette"], merged),
        n_actions,
        label=f"{scenario}/risk",
    )
    risk = {
        norm: normalize_risk(risk_agg[f"action_{i}"][0])
        if not np.isnan(risk_agg[f"action_{i}"][0])
        else np.nan
        for i, norm in enumerate(all_norms)
    }

    # effort: one prompt per effort_condition (effort paragraph appended).
    effort = {}
    for ec in EFFORT_CONDITIONS:
        vignette_eff = (
            f"{scenario_row['vignette']} {scenario_row[f'low_risk_share_effort_{ec}']}"
        )
        eff_agg, _ = _score_one_call(
            client,
            system_prompts["effort"],
            format_effort_prompt_variable(vignette_eff, merged),
            n_actions,
            label=f"{scenario}/effort[{ec}]",
        )
        for i, norm in enumerate(all_norms):
            v = eff_agg[f"action_{i}"][0]
            effort[(ec, norm)] = normalize_effort(v) if not np.isnan(v) else np.nan

    # g: one prompt, desire-free goal-satisfaction.
    g_agg, _ = _score_one_call(
        client,
        system_prompts["g"],
        format_g_prompt_variable(
            scenario_row["vignette"], merged, scenario_row["desire_object"]
        ),
        n_actions,
        label=f"{scenario}/g",
    )
    g = {
        norm: normalize_g(g_agg[f"action_{i}"][0])
        if not np.isnan(g_agg[f"action_{i}"][0])
        else np.nan
        for i, norm in enumerate(all_norms)
    }

    return {
        "merged_actions": merged,
        "canonical_norms": canonical_norms,
        "alt_norms_in_order": alt_norms,
        "risk": risk,
        "effort": effort,
        "g": g,
    }


def _build_run_records(study, scenario_row, run_id, run_alt_rows, scored, cfg):
    """Assemble the per-(run, cell) JSONL records for one (scenario, run).

    Enumerates the full generation cell grid (observed_action × generation
    condition levels) so every cell gets a record with its canonical slot 0,
    even cells whose run produced zero alternatives. For effort-inferred studies
    each cell emits one record per effort_condition (the effort feature axis).
    """
    scenario = scenario_row["scenario_label"]
    risk, effort, g = scored["risk"], scored["effort"], scored["g"]
    cell_cols = list(cfg["cell_cols"])
    level_lists = [_LEVELS[c] for c in cell_cols]

    records = []
    for observed_action in CANONICAL_ACTIONS:
        obs_norm = _norm(scenario_row[observed_action])
        for cond_values in itertools.product(*level_lists) if level_lists else [()]:
            cond = dict(zip(cell_cols, cond_values))
            # The run's alternatives for this exact cell, in alt_idx order.
            mask = run_alt_rows["observed_action"] == observed_action
            for c, v in cond.items():
                mask = mask & (run_alt_rows[c] == v)
            cell_alts = run_alt_rows[mask].sort_values("alt_idx")

            record_effort_conds = (
                EFFORT_CONDITIONS
                if cfg["effort_inferred"]
                else [cond["effort_condition"]]
            )
            for ec in record_effort_conds:
                actions = [
                    {
                        "slot": 0,
                        "is_canonical": True,
                        "action_text": scenario_row[observed_action],
                        "is_share": CANONICAL_IS_SHARE[observed_action],
                        "risk": _f(risk.get(obs_norm)),
                        "effort": _f(effort.get((ec, obs_norm))),
                        "g": _f(g.get(obs_norm)),
                    }
                ]
                for _, alt in cell_alts.iterrows():
                    a_norm = _norm(alt["action_text"])
                    actions.append(
                        {
                            "slot": int(alt["alt_idx"]) + 1,
                            "alt_idx": int(alt["alt_idx"]),
                            "is_canonical": False,
                            "action_text": alt["action_text"],
                            "is_share": int(alt["is_share"])
                            if not pd.isna(alt["is_share"])
                            else None,
                            "risk": _f(risk.get(a_norm)),
                            "effort": _f(effort.get((ec, a_norm))),
                            "g": _f(g.get(a_norm)),
                        }
                    )
                record = {
                    "run_id": int(run_id),
                    "scenario_label": scenario,
                    "observed_action": observed_action,
                }
                record.update(cond)
                record["effort_condition"] = ec  # loader keys canon on this
                record["actions"] = actions
                records.append(record)
    return records


def _f(x):
    """JSON-safe float (NaN -> None)."""
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else x


def _rate_desire_scalars(client, scenarios_df):
    """Per-(scenario, desire_condition) desire scalar in [0, 1] (run-independent).
    Single scoring pass per (scenario, condition)."""
    out = {}
    for scenario, row in scenarios_df.iterrows():
        out[scenario] = {}
        for dc in DESIRES:
            ratings, _ = get_ratings_concurrent(
                client,
                DESIRE_SYSTEM_PROMPT,
                desire_user_prompt(
                    row["vignette"], row[f"desire_{dc}"], row["desire_object"]
                ),
                parse_desire_response,
                num_runs=1,
                max_tokens=64,
                response_format=numeric_desire_schema(),
                label=f"{scenario}/desire[{dc}]",
            )
            out[scenario][dc] = (float(ratings[0]) / 100.0) if ratings else None
    return out


def _rate_relationship_values(client):
    """Per-level intimacy scalar in [0, 1] from the de-anchored relationship
    descriptors (scenario-independent → 4 values). Single scoring pass per level."""
    out = {}
    for level in INTIMACY_LEVELS:
        ratings, _ = get_ratings_concurrent(
            client,
            INTIMACY_SYSTEM_PROMPT,
            relationship_user_prompt(RELATIONSHIP_DESCRIPTORS_NOANCHOR[level]),
            parse_intimacy_response,
            num_runs=1,
            max_tokens=64,
            response_format=numeric_intimacy_schema(),
            label=f"relationship[{level}]",
        )
        out[level] = (float(ratings[0]) / 100.0) if ratings else None
    return out


def main(study, scenario_workers=SCENARIO_WORKERS):
    if study not in _STUDY_CONFIG:
        raise SystemExit(
            f"Unknown study: {study!r}. Supported: {sorted(_STUDY_CONFIG.keys())}"
        )
    cfg = _STUDY_CONFIG[study]
    api_key = load_api_key()

    scenarios_path = get_project_root() / "experiments" / "scenarios.csv"
    study_dir = get_project_root() / "model" / "outputs" / "lm" / study
    study_dir.mkdir(parents=True, exist_ok=True)
    alts_path = study_dir / "lm_alternatives.csv"
    runs_path = study_dir / "lm_runs.jsonl"
    given_path = study_dir / "lm_given.json"
    if not alts_path.exists():
        raise SystemExit(
            f"Alternatives CSV not found at {alts_path}. Run "
            f"model/lm/generate_alternatives.py --study {study} first."
        )

    scenarios_df = pd.read_csv(scenarios_path).set_index("scenario_label", drop=False)
    alts_df = pd.read_csv(alts_path)
    if "run_id" not in alts_df.columns:
        raise SystemExit(
            f"{alts_path} has no run_id column — re-run generate_alternatives.py "
            "with the K-run pipeline first."
        )

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    # Given-magnitude scalars (run-independent, scored once) -> lm_given.json.
    given = {}
    if cfg["desire_given"]:
        print("Rating per-(scenario, condition) desire scalars...", flush=True)
        given["desire"] = _rate_desire_scalars(client, scenarios_df)
    if cfg["relationship_given"]:
        print("Rating per-level relationship intimacy (de-anchored)...", flush=True)
        given["relationship"] = _rate_relationship_values(client)
    with open(given_path, "w") as f:
        json.dump(given, f, indent=2)
    print(f"Wrote {given_path}", flush=True)

    system_prompts = {
        "risk": build_system_prompt("risk", n_actions=None),
        "effort": build_system_prompt("effort", n_actions=None),
        "g": build_system_prompt("g", n_actions=None),
    }

    # Resume: skip (scenario, run) units already written to lm_runs.jsonl.
    done_units = set()
    existing_records = []
    if runs_path.exists():
        with open(runs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing_records.append(rec)
                done_units.add((rec["scenario_label"], int(rec["run_id"])))
        print(
            f"Found {len(done_units)} (scenario, run) units already scored — resuming.",
            flush=True,
        )

    run_ids = sorted(alts_df["run_id"].dropna().unique().astype(int))
    units = [
        (s, r)
        for s in scenarios_df.index
        for r in run_ids
        if (s, int(r)) not in done_units
    ]
    print(
        f"\n{len(units)} (scenario, run) units to score "
        f"(scenarios={len(scenarios_df)}, runs={len(run_ids)}; "
        f"{scenario_workers} concurrent).",
        flush=True,
    )

    def _process_unit(scenario, run_id):
        scenario_row = scenarios_df.loc[scenario]
        run_alt_rows = alts_df[
            (alts_df["scenario_label"] == scenario) & (alts_df["run_id"] == run_id)
        ]
        scored = _score_actions(client, scenario_row, run_alt_rows, system_prompts)
        return _build_run_records(
            study, scenario_row, run_id, run_alt_rows, scored, cfg
        )

    all_records = list(existing_records)
    done_count = 0
    with ThreadPoolExecutor(max_workers=max(1, scenario_workers)) as ex:
        futures = {ex.submit(_process_unit, s, r): (s, r) for s, r in units}
        for fut in as_completed(futures):
            s, r = futures[fut]
            all_records.extend(fut.result())
            done_count += 1
            # Checkpoint: rewrite the full JSONL (append-only set of records).
            with open(runs_path, "w") as f:
                for rec in all_records:
                    f.write(json.dumps(rec) + "\n")
            print(
                f"  [{done_count}/{len(units)}] {s} / run {r} done — checkpointed",
                flush=True,
            )

    print("\n=== Done ===")
    print(f"  {runs_path.name}  ({len(all_records)} records)")
    print(f"  {given_path.name}  (keys: {sorted(given.keys())})")


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
        help="How many (scenario, run) units to score concurrently.",
    )
    args = parser.parse_args()
    main(args.study, scenario_workers=args.scenario_workers)
