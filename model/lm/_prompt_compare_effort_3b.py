#!/usr/bin/env python3
"""Side-by-side comparison of OLD vs NEW _EFFORT_BODY prompt on a small
scenario set. Exploratory tool used to gauge whether the revised, NUC-grounded
prompt shifts effort ratings before committing to a full re-elicitation.

Test scope:
  - 4 scenarios: basketball, drinks, wedding (parity with the access test),
    plus oysters (chosen because its effort_high paragraph is an explicit
    time + coordination cost case — "server is slammed — flagging him down
    for extra small plates and forks would take a long time").
  - Both effort conditions per scenario.
  - 5 runs per (scenario, effort_condition) cell.
  - Total: 4 × 2 × 5 = 40 LM calls.

Reports both absolute deltas (per cell) and manipulation deltas
(effort_high − effort_low for action_1 within each scenario, since action_1
is what the effort manipulation targets). A sharper manipulation delta under
the new prompt means the effort construct has more leverage in the model.

Output: model/outputs/lm/_prompt_comparison_effort_3b.csv (underscored;
not picked up by any loader).

Usage:
    uv run python model/lm/_prompt_compare_effort_3b.py
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
from _features_dispatcher import normalize_effort
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
TEST_SCENARIOS = ["basketball", "drinks", "wedding", "oysters"]
EFFORT_CONDITIONS = ["low", "high"]


def _full_vignette(row, effort_condition):
    return f"{row['vignette']} {row[f'effort_{effort_condition}']}"


def _action_texts(row):
    return [row[c] for c in ("no_share", "low_risk_share", "high_risk_share")]


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

    scenarios_path = get_project_root() / "experiments" / "scenarios.csv"
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
            "effort_raw",
            "effort_raw_std",
            "effort",
        ]
    ].rename(
        columns={
            "effort_raw": "old_effort_raw",
            "effort_raw_std": "old_effort_raw_std",
            "effort": "old_effort",
        }
    )

    effort_system_prompt = build_system_prompt("effort", n_actions=N_ACTIONS)
    print(f"Initializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    new_rows = []
    for scenario in TEST_SCENARIOS:
        row = test_rows.loc[scenario]
        for effort in EFFORT_CONDITIONS:
            user_prompt = build_user_prompt(
                "effort",
                _full_vignette(row, effort),
                _action_texts(row),
            )
            print(f"\n[{scenario} / effort={effort}] {NUM_RUNS} runs...", flush=True)
            ratings, n_failures = get_ratings_concurrent(
                client,
                effort_system_prompt,
                user_prompt,
                _parse,
                response_format=numeric_action_schema(N_ACTIONS),
                num_runs=NUM_RUNS,
                label=f"{scenario}/effort={effort}/effort_NEW",
            )
            agg = aggregate_action_ratings(ratings, n_actions=N_ACTIONS)
            for action in range(N_ACTIONS):
                e_mean, e_std = agg[f"action_{action}"]
                new_rows.append(
                    {
                        "scenario_label": scenario,
                        "effort_condition": effort,
                        "action": action,
                        "new_effort_raw": e_mean,
                        "new_effort_raw_std": e_std,
                        "new_effort": normalize_effort(e_mean)
                        if not np.isnan(e_mean)
                        else np.nan,
                        "new_n_runs": len(ratings),
                        "new_n_failures": n_failures,
                    }
                )

    new_df = pd.DataFrame(new_rows)
    merged = new_df.merge(
        old_sub, on=["scenario_label", "effort_condition", "action"], how="left"
    )
    merged["delta_effort_raw"] = merged["new_effort_raw"] - merged["old_effort_raw"]

    output_dir = get_project_root() / "model" / "outputs" / "lm"
    output_path = output_dir / "_prompt_comparison_effort_3b.csv"
    merged.to_csv(output_path, index=False)

    display_cols = [
        "scenario_label",
        "effort_condition",
        "action",
        "old_effort_raw",
        "new_effort_raw",
        "delta_effort_raw",
        "new_effort_raw_std",
    ]
    print("\n=== Side-by-side comparison (raw 0-6 scale) ===\n")
    print(merged[display_cols].to_string(index=False, float_format="%.2f"))

    # Manipulation delta: effort_high - effort_low for action_1 (the action
    # the effort_low/effort_high paragraphs target). Sharper delta under the
    # new prompt means the effort manipulation has more leverage in the model.
    print(
        "\n=== Manipulation delta (action_1: effort_high - effort_low, raw scale) ===\n"
    )
    pivot = merged[merged["action"] == 1].pivot(
        index="scenario_label",
        columns="effort_condition",
        values=["old_effort_raw", "new_effort_raw"],
    )
    delta_rows = []
    for scenario in TEST_SCENARIOS:
        old_low = pivot.loc[scenario, ("old_effort_raw", "low")]
        old_high = pivot.loc[scenario, ("old_effort_raw", "high")]
        new_low = pivot.loc[scenario, ("new_effort_raw", "low")]
        new_high = pivot.loc[scenario, ("new_effort_raw", "high")]
        delta_rows.append(
            {
                "scenario": scenario,
                "old_low": old_low,
                "old_high": old_high,
                "old_delta": old_high - old_low,
                "new_low": new_low,
                "new_high": new_high,
                "new_delta": new_high - new_low,
                "delta_of_deltas": (new_high - new_low) - (old_high - old_low),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    print(delta_df.to_string(index=False, float_format="%.2f"))

    print(f"\nFull comparison saved to {output_path}")
    print(
        "\nNOTE: NEW std reflects NUM_RUNS=5 (vs OLD's 10), so look at means "
        "for the prompt-effect signal. The manipulation delta is the most "
        "load-bearing number for effort — it's what the effort_condition "
        "design lever moves."
    )


if __name__ == "__main__":
    main()
