#!/usr/bin/env python3
"""
Score access + effort for LM-generated alternatives in the 3-action inverse
experiments.

For each scenario, deduplicates alternative-action strings (case-insensitive),
then runs the access and effort LM passes per effort_condition (since the
3-act access and effort tables depend on the effort paragraph appearing in
the vignette — keep parity with score_3act_features.py). Features are
broadcast back to every (observed_action, effort_condition, intimacy_condition,
alt_idx) row that used that action text.

Output (Study 3b):
    model/outputs/lm/lm_alternatives_features_food_inv_desire.csv

Usage:
    uv run python model/lm/score_alternatives_3act.py --study food_inv_desire

Requires:
    - TOGETHER_API_KEY environment variable or in .env file
    - lm_alternatives_food_inv_desire.csv produced by generate_alternatives_3act.py
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
    normalize_access,
    normalize_effort,
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


_STUDY_CONFIG = {
    "food_inv_desire": {
        "scenarios": "scenarios_3act.csv",
        "alternatives_input": "lm_alternatives_food_inv_desire.csv",
        "alternatives_output": "lm_alternatives_features_food_inv_desire.csv",
        # The 3-act access/effort tables depend on effort_condition, so each
        # alt is scored under both effort levels (the effort paragraph is part
        # of the vignette text passed to the LM, mirroring score_3act_features.py).
        "score_per_effort": True,
    },
}


def _build_vignette(row, effort_condition):
    """Vignette + effort paragraph — same as score_3act_features.format_full_vignette."""
    return f"{row['vignette']} {row[f'effort_{effort_condition}']}"


def main(study):
    if study not in _STUDY_CONFIG:
        raise SystemExit(
            f"Unknown study: {study!r}. Supported: {sorted(_STUDY_CONFIG.keys())}"
        )
    cfg = _STUDY_CONFIG[study]
    api_key = load_api_key()

    access_system_prompt = build_system_prompt("access", n_actions=None)
    effort_system_prompt = build_system_prompt("effort", n_actions=None)

    print(f"Loading scenarios and LM alternatives (study={study})...", flush=True)
    scenarios_path = get_project_root() / "experiments" / cfg["scenarios"]
    scenarios_df = pd.read_csv(scenarios_path)
    alt_path = (
        get_project_root() / "model" / "outputs" / "lm" / cfg["alternatives_input"]
    )
    if not alt_path.exists():
        raise SystemExit(
            f"Error: {alt_path} not found. Run "
            "model/lm/generate_alternatives_3act.py first."
        )
    alt_df = pd.read_csv(alt_path)
    alt_df["action_norm"] = alt_df["action_text"].str.lower().str.strip()
    n_cells = alt_df.groupby(
        ["scenario_label", "observed_action", "effort_condition", "intimacy_condition"]
    ).ngroups
    print(
        f"Loaded {len(alt_df)} alternatives across {n_cells} cells "
        f"({alt_df['action_norm'].nunique()} unique action strings)",
        flush=True,
    )

    output_dir = get_project_root() / "model" / "outputs" / "lm"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / cfg["alternatives_output"]

    results = []
    already_done_scenarios = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_done_scenarios = set(existing["scenario_label"].unique())
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with "
            f"{len(already_done_scenarios)} scenarios already scored — resuming.",
            flush=True,
        )

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    scenario_lookup = scenarios_df.set_index("scenario_label").to_dict("index")
    effort_levels = ("low", "high") if cfg["score_per_effort"] else (None,)

    scenarios_in_data = sorted(alt_df["scenario_label"].unique())
    for sc_idx, scenario in enumerate(scenarios_in_data, start=1):
        if scenario in already_done_scenarios:
            print(
                f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} — already scored, skipping.",
                flush=True,
            )
            continue

        sc_group = alt_df[alt_df["scenario_label"] == scenario]
        # Dedupe per scenario by normalized action text — access/effort depend
        # on (vignette+effort_paragraph, action_text), so per-effort_condition
        # scoring is run on the same unique action set.
        unique_df = sc_group.drop_duplicates("action_norm").reset_index(drop=True)
        unique_actions = unique_df["action_text"].tolist()
        unique_norms = unique_df["action_norm"].tolist()
        n_unique = len(unique_actions)
        sc_meta = scenario_lookup[scenario]

        print(
            f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} | "
            f"{len(sc_group)} rows | {n_unique} unique actions",
            flush=True,
        )
        if n_unique == 0:
            continue

        feats_by_effort_by_norm = {}
        for effort_condition in effort_levels:
            vignette = (
                _build_vignette(sc_meta, effort_condition)
                if effort_condition is not None
                else sc_meta["vignette"]
            )
            print(
                f"  scoring access (effort={effort_condition}, concurrent, structured)...",
                flush=True,
            )
            access_ratings, access_failures = get_ratings_concurrent(
                client,
                access_system_prompt,
                format_access_prompt_variable(vignette, unique_actions),
                lambda t: parse_action_response_variable(t, n_unique),
                max_tokens=_max_tokens_for(n_unique),
                response_format=numeric_action_schema(n_unique),
                label=f"{scenario}/alts_3act/access[{effort_condition}]",
            )
            access_agg = aggregate_action_ratings(access_ratings, n_unique)

            print(
                f"  scoring effort (effort={effort_condition}, concurrent, structured)...",
                flush=True,
            )
            effort_ratings, effort_failures = get_ratings_concurrent(
                client,
                effort_system_prompt,
                format_effort_prompt_variable(vignette, unique_actions),
                lambda t: parse_action_response_variable(t, n_unique),
                max_tokens=_max_tokens_for(n_unique),
                response_format=numeric_action_schema(n_unique),
                label=f"{scenario}/alts_3act/effort[{effort_condition}]",
            )
            effort_agg = aggregate_action_ratings(effort_ratings, n_unique)

            for i, norm in enumerate(unique_norms):
                key = f"action_{i}"
                a_mean, a_std = access_agg[key]
                e_mean, e_std = effort_agg[key]
                feats_by_effort_by_norm.setdefault(effort_condition, {})[norm] = {
                    "access_raw": a_mean,
                    "access_raw_std": a_std,
                    "effort_raw": e_mean,
                    "effort_raw_std": e_std,
                    "access": normalize_access(a_mean)
                    if not np.isnan(a_mean)
                    else np.nan,
                    "effort": normalize_effort(e_mean)
                    if not np.isnan(e_mean)
                    else np.nan,
                    "n_runs_access": len(access_ratings),
                    "n_runs_effort": len(effort_ratings),
                    "n_failures_access": access_failures,
                    "n_failures_effort": effort_failures,
                }

        new_rows = 0
        for _, row in sc_group.iterrows():
            effort_condition = row["effort_condition"]
            f = feats_by_effort_by_norm.get(effort_condition, {}).get(
                row["action_norm"]
            )
            if f is None:
                continue
            results.append(
                {
                    "scenario_label": scenario,
                    "observed_action": row["observed_action"],
                    "effort_condition": effort_condition,
                    "intimacy_condition": int(row["intimacy_condition"]),
                    "alt_idx": int(row["alt_idx"]),
                    **f,
                }
            )
            new_rows += 1

        pd.DataFrame(results).to_csv(output_path, index=False)
        print(
            f"  +{new_rows} rows | checkpoint written ({len(results)} total)",
            flush=True,
        )

    print(
        f"\nFinal: saved {len(results)} alternative-feature rows to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=tuple(_STUDY_CONFIG.keys()),
        default="food_inv_desire",
    )
    args = parser.parse_args()
    main(args.study)
