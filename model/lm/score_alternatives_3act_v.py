#!/usr/bin/env python3
"""
Score signed-valence V for LM-generated alternatives in the 3-action inverse
experiments.

V is independent of the effort paragraph and the intimacy descriptor (it
depends only on (action, motivation, scenario) — same convention as
score_3act_v.py). Each unique alternative-action string is scored under both
motivation_query ∈ {low, high} so the observer's `thinks[reward in
RewardConditions, …]` block has V at both reward values.

Output (Study 3b):
    model/outputs/lm/lm_alternatives_v_food_inv_desire.csv

Schema:
    scenario_label, observed_action, effort_condition, intimacy_condition,
    alt_idx, motivation_query, v_raw, v_raw_std, v, n_runs, n_failures

Usage:
    uv run python model/lm/score_alternatives_3act_v.py --study food_inv_desire

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
    format_v_prompt_variable,
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


_STUDY_CONFIG = {
    "food_inv_desire": {
        "scenarios": "scenarios.csv",
        "alternatives_input": "lm_alternatives_food_inv_desire.csv",
        "alternatives_v_output": "lm_alternatives_v_food_inv_desire.csv",
    },
}


def main(study):
    if study not in _STUDY_CONFIG:
        raise SystemExit(
            f"Unknown study: {study!r}. Supported: {sorted(_STUDY_CONFIG.keys())}"
        )
    cfg = _STUDY_CONFIG[study]
    api_key = load_api_key()

    v_system_prompt = build_system_prompt("v", n_actions=None)

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

    output_dir = get_project_root() / "model" / "outputs" / "lm"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / cfg["alternatives_v_output"]

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

    scenarios_in_data = sorted(alt_df["scenario_label"].unique())
    for sc_idx, scenario in enumerate(scenarios_in_data, start=1):
        if scenario in already_done_scenarios:
            print(
                f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} — already scored, skipping.",
                flush=True,
            )
            continue

        sc_group = alt_df[alt_df["scenario_label"] == scenario]
        unique_df = sc_group.drop_duplicates("action_norm").reset_index(drop=True)
        unique_actions = unique_df["action_text"].tolist()
        unique_norms = unique_df["action_norm"].tolist()
        n_unique = len(unique_actions)
        sc_meta = scenario_lookup[scenario]
        vignette = sc_meta["vignette"]

        print(
            f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} | "
            f"{len(sc_group)} rows | {n_unique} unique actions",
            flush=True,
        )
        if n_unique == 0:
            continue

        v_by_norm_by_query = {}
        for motivation_query in ("low", "high"):
            print(
                f"  scoring V (motivation_query={motivation_query}, concurrent, structured)...",
                flush=True,
            )
            state = sc_meta[f"desire_{motivation_query}"]
            ratings, n_failures = get_ratings_concurrent(
                client,
                v_system_prompt,
                format_v_prompt_variable(vignette, state, unique_actions),
                lambda t: parse_action_response_variable(t, n_unique),
                max_tokens=_max_tokens_for(n_unique),
                response_format=numeric_action_schema(n_unique),
                label=f"{scenario}/alts_3act/V[{motivation_query}]",
            )
            agg = aggregate_action_ratings(ratings, n_unique)
            for i, norm in enumerate(unique_norms):
                key = f"action_{i}"
                v_mean, v_std = agg[key]
                v_by_norm_by_query.setdefault(norm, {})[motivation_query] = {
                    "v_raw": v_mean,
                    "v_raw_std": v_std,
                    "v": normalize_v(v_mean) if not np.isnan(v_mean) else np.nan,
                    "n_runs": len(ratings),
                    "n_failures": n_failures,
                }

        new_rows = 0
        for _, row in sc_group.iterrows():
            for motivation_query in ("low", "high"):
                f = v_by_norm_by_query.get(row["action_norm"], {}).get(motivation_query)
                if f is None:
                    continue
                results.append(
                    {
                        "scenario_label": scenario,
                        "observed_action": row["observed_action"],
                        "effort_condition": row["effort_condition"],
                        "intimacy_condition": int(row["intimacy_condition"]),
                        "alt_idx": int(row["alt_idx"]),
                        "motivation_query": motivation_query,
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
        f"\nFinal: saved {len(results)} alternative-V rows to {output_path}",
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
