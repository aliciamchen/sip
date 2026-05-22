#!/usr/bin/env python3
"""Side-by-side comparison of OLD vs NEW _ACCESS_BODY prompt on a small
scenario set. Exploratory tool used to gauge whether the revised, lit-grounded
prompt shifts access ratings before committing to a full re-elicitation.

Test scope:
  - 3 scenarios chosen to span the access range: basketball (mid-range),
    drinks (high substance-transmission), wedding (formal/sensitive).
  - Both effort conditions per scenario.
  - 5 runs per (scenario, effort_condition) cell.
  - Total: 3 × 2 × 5 = 30 LM calls.

The script:
  1. Loads OLD access ratings from lm_scenario_params_3act.csv (under the
     prompt body that existed before the lit-grounded revision).
  2. Calls the LM with the *current* prompts.py:_ACCESS_BODY (the revised
     version) on the same scenarios.
  3. Writes side-by-side CSV to
     model/outputs/lm/_prompt_comparison_access_3b.csv and prints a delta
     table to stdout.

Does not touch the main lm_scenario_params_3act.csv. Underscore-prefixed
output filename signals this is a temp artifact — not loaded by any of the
table loaders.

Usage:
    uv run python model/lm/_prompt_compare_access_3b.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from together import Together

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _features_dispatcher import normalize_access
from client import (
    MODEL_ID,
    aggregate_action_ratings,
    find_json,
    get_ratings_concurrent,
    load_api_key,
    numeric_action_schema,
    strip_leading_plus,
)
from prompts import system_prompt as build_system_prompt
from prompts import user_prompt as build_user_prompt


N_ACTIONS = 3
NUM_RUNS = 5
TEST_SCENARIOS = ["basketball", "drinks", "wedding"]
EFFORT_CONDITIONS = ["low", "high"]


def _full_vignette(row, effort_condition):
    return f"{row['vignette']} {row[f'effort_{effort_condition}']}"


def _action_texts(row):
    return [row[f"action_{i}"] for i in range(N_ACTIONS)]


def _parse(response_text):
    if response_text is None:
        return None
    js = find_json(response_text)
    if js is None:
        return None
    js = strip_leading_plus(js)
    try:
        ratings = json.loads(js)
        expected = {f"action_{i}" for i in range(N_ACTIONS)}
        if expected.issubset(ratings.keys()):
            return {k: float(ratings[k]) for k in expected}
    except (json.JSONDecodeError, ValueError):
        return None
    return None


def main():
    api_key = load_api_key()

    scenarios_path = get_project_root() / "experiments" / "scenarios_3act.csv"
    scenarios_df = pd.read_csv(scenarios_path)
    test_rows = scenarios_df[
        scenarios_df["scenario_label"].isin(TEST_SCENARIOS)
    ].set_index("scenario_label")

    old_df = pd.read_csv(
        get_project_root() / "model" / "outputs" / "lm" / "lm_scenario_params_3act.csv"
    )
    old_sub = old_df[old_df["scenario_label"].isin(TEST_SCENARIOS)][
        [
            "scenario_label",
            "effort_condition",
            "action",
            "access_raw",
            "access_raw_std",
            "access",
        ]
    ].rename(
        columns={
            "access_raw": "old_access_raw",
            "access_raw_std": "old_access_raw_std",
            "access": "old_access",
        }
    )

    access_system_prompt = build_system_prompt("access", n_actions=N_ACTIONS)
    print(f"Initializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    new_rows = []
    for scenario in TEST_SCENARIOS:
        row = test_rows.loc[scenario]
        for effort in EFFORT_CONDITIONS:
            user_prompt = build_user_prompt(
                "access",
                _full_vignette(row, effort),
                _action_texts(row),
            )
            print(f"\n[{scenario} / effort={effort}] {NUM_RUNS} runs...", flush=True)
            ratings, n_failures = get_ratings_concurrent(
                client,
                access_system_prompt,
                user_prompt,
                _parse,
                response_format=numeric_action_schema(N_ACTIONS),
                num_runs=NUM_RUNS,
                label=f"{scenario}/effort={effort}/access_NEW",
            )
            agg = aggregate_action_ratings(ratings, n_actions=N_ACTIONS)
            for action in range(N_ACTIONS):
                a_mean, a_std = agg[f"action_{action}"]
                new_rows.append(
                    {
                        "scenario_label": scenario,
                        "effort_condition": effort,
                        "action": action,
                        "new_access_raw": a_mean,
                        "new_access_raw_std": a_std,
                        "new_access": normalize_access(a_mean)
                        if not np.isnan(a_mean)
                        else np.nan,
                        "new_n_runs": len(ratings),
                        "new_n_failures": n_failures,
                    }
                )

    new_df = pd.DataFrame(new_rows)
    merged = new_df.merge(
        old_sub, on=["scenario_label", "effort_condition", "action"], how="left"
    )
    merged["delta_access_raw"] = merged["new_access_raw"] - merged["old_access_raw"]

    output_dir = get_project_root() / "model" / "outputs" / "lm"
    output_path = output_dir / "_prompt_comparison_access_3b.csv"
    merged.to_csv(output_path, index=False)

    display_cols = [
        "scenario_label",
        "effort_condition",
        "action",
        "old_access_raw",
        "new_access_raw",
        "delta_access_raw",
        "new_access_raw_std",
    ]
    print("\n=== Side-by-side comparison (raw 0-6 scale) ===\n")
    print(merged[display_cols].to_string(index=False, float_format="%.2f"))

    print(f"\nFull comparison saved to {output_path}")
    print(
        "\nNOTE: NEW std reflects NUM_RUNS=5 (vs OLD's 10), so NEW std will tend "
        "to look noisier even if the underlying distribution is the same. Look "
        "at means for the prompt-effect signal."
    )


if __name__ == "__main__":
    main()
