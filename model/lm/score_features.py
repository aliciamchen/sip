#!/usr/bin/env python3
"""
Score risk + effort features for the 3-action set (Studies 1a, 1b, 2a, 2b).

For each of the 16 scenarios in experiments/scenarios.csv, estimates per
(effort_condition, action):

- risk(a): physical / informational / spatial exposure  (0-6 -> [0, 1])
- effort(a): physical / logistical cost                   (0-6 -> [0, 1])

Same elicitation pattern as score_effort_features.py — the LM is prompted with
the full (vignette + effort_paragraph) text so the effort manipulation shows
up in the ratings — but with 3 actions instead of 2.

Also produces an effort-marginal risk table (vignette without effort
paragraph) for studies where the observer infers effort and so does not
see the effort context.

Outputs:
- model/outputs/lm/lm_scenario_params.csv         (96 rows: 16 x 2 efforts x 3 actions)
- model/outputs/lm/lm_scenario_params_marginal.csv (48 rows: 16 x 3 actions)

10 runs per parameter-type per (scenario, effort_condition), aggregated to
mean/std. Resumes per scenario if the output CSV already exists.

Requires TOGETHER_API_KEY in env or .env.
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
from client import (
    MODEL_ID,
    aggregate_action_ratings,
    find_json,
    get_ratings_concurrent,
    load_api_key,
    numeric_action_schema,
    strip_leading_plus,
)
from _features_dispatcher import normalize_risk, normalize_effort
from prompts import system_prompt as build_system_prompt
from prompts import user_prompt as build_user_prompt


N_ACTIONS = 3
# Canonical action names (the scenarios.csv columns), in slot order. The output
# CSV labels the `action` column with these instead of the integer index 0/1/2.
# The LM-facing protocol stays neutral (positional action_0/1/2).
CANONICAL_ACTIONS = ["no_share", "low_risk_share", "high_risk_share"]
EFFORT_CONDITIONS = ["low", "high"]

RISK_SYSTEM_PROMPT = build_system_prompt("risk", n_actions=N_ACTIONS)
EFFORT_SYSTEM_PROMPT = build_system_prompt("effort", n_actions=N_ACTIONS)


def load_scenarios():
    scenarios_path = get_project_root() / "experiments" / "scenarios.csv"
    return pd.read_csv(scenarios_path)


def _action_texts_3(row):
    return [row[c] for c in CANONICAL_ACTIONS]


def format_full_vignette(row, effort_condition):
    return f"{row['vignette']} {row[f'effort_{effort_condition}']}"


def format_risk_prompt(row, effort_condition):
    return build_user_prompt(
        "risk",
        format_full_vignette(row, effort_condition),
        _action_texts_3(row),
    )


def format_risk_prompt_marginal(row):
    return build_user_prompt("risk", row["vignette"], _action_texts_3(row))


def format_effort_prompt(row, effort_condition):
    return build_user_prompt(
        "effort",
        format_full_vignette(row, effort_condition),
        _action_texts_3(row),
    )


def parse_action_response(response_text):
    """Parse JSON ratings with action_0 / action_1 / action_2 keys."""
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
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Failed to parse JSON: {e}")
    return None


def run_effort_conditional(client, scenarios_df, output_path):
    """One row per (scenario, effort_condition, action) — 96 rows total."""
    results = []
    already_done = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_done = set(existing["scenario_label"].unique())
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with "
            f"{len(already_done)} scenarios already scored — resuming.",
            flush=True,
        )

    for idx, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        if scenario in already_done:
            print(
                f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} — already scored, skipping.",
                flush=True,
            )
            continue

        for effort_condition in EFFORT_CONDITIONS:
            print(
                f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} (effort={effort_condition})",
                flush=True,
            )

            print(
                "  Getting risk ratings (concurrent, structured, effort-conditional)...",
                flush=True,
            )
            risk_ratings, risk_failures = get_ratings_concurrent(
                client,
                RISK_SYSTEM_PROMPT,
                format_risk_prompt(row, effort_condition),
                parse_action_response,
                response_format=numeric_action_schema(N_ACTIONS),
                label=f"{scenario}/effort={effort_condition}/risk",
            )
            risk_agg = aggregate_action_ratings(risk_ratings, n_actions=N_ACTIONS)

            print("  Getting effort ratings (concurrent, structured)...", flush=True)
            effort_ratings, effort_failures = get_ratings_concurrent(
                client,
                EFFORT_SYSTEM_PROMPT,
                format_effort_prompt(row, effort_condition),
                parse_action_response,
                response_format=numeric_action_schema(N_ACTIONS),
                label=f"{scenario}/effort={effort_condition}/effort",
            )
            effort_agg = aggregate_action_ratings(effort_ratings, n_actions=N_ACTIONS)

            for action in range(N_ACTIONS):
                key = f"action_{action}"
                a_mean, a_std = risk_agg[key]
                e_mean, e_std = effort_agg[key]
                results.append(
                    {
                        "scenario_label": scenario,
                        "effort_condition": effort_condition,
                        "action": CANONICAL_ACTIONS[action],
                        "risk_raw": a_mean,
                        "risk_raw_std": a_std,
                        "effort_raw": e_mean,
                        "effort_raw_std": e_std,
                        "risk": normalize_risk(a_mean)
                        if not np.isnan(a_mean)
                        else np.nan,
                        "effort": normalize_effort(e_mean)
                        if not np.isnan(e_mean)
                        else np.nan,
                        "n_runs_risk": len(risk_ratings),
                        "n_runs_effort": len(effort_ratings),
                        "n_failures_risk": risk_failures,
                        "n_failures_effort": effort_failures,
                    }
                )

            acc_str = [f"{risk_agg[f'action_{i}'][0]:.1f}" for i in range(N_ACTIONS)]
            eff_str = [f"{effort_agg[f'action_{i}'][0]:.1f}" for i in range(N_ACTIONS)]
            print(f"  Risk (raw): {acc_str}", flush=True)
            print(f"  Effort (raw): {eff_str}", flush=True)

        pd.DataFrame(results).to_csv(output_path, index=False)

    results_df = pd.DataFrame(results)
    print(f"\nSaved effort-conditional results to {output_path}")
    print(
        f"Total rows: {len(results_df)} (expected 96 = 16 scenarios × 2 effort × 3 actions)"
    )
    for col, target in [("risk", "[0, 1]"), ("effort", "[0, 1]")]:
        print(
            f"\n{col.capitalize()} (normalized, target {target}):"
            f"\n  Mean: {results_df[col].mean():.2f}, Std: {results_df[col].std():.2f}"
            f"\n  Range: [{results_df[col].min():.2f}, {results_df[col].max():.2f}]"
        )

    print("\n=== Effort manipulation sanity (low_risk_share effort: low vs high) ===")
    act1 = results_df[results_df["action"] == "low_risk_share"]
    wide = act1.pivot(
        index="scenario_label", columns="effort_condition", values="effort"
    )
    wide["delta"] = wide["high"] - wide["low"]
    print(wide.round(3).to_string())
    print(
        f"\nMean Δ effort(action_1): {wide['delta'].mean():+.3f} (positive = manipulation worked)"
    )


def run_marginal_risk(client, scenarios_df, output_path):
    """One row per (scenario, action) — 48 rows total."""
    results = []
    already_done = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_done = set(existing["scenario_label"].unique())
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with "
            f"{len(already_done)} scenarios already scored — resuming.",
            flush=True,
        )

    for idx, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        if scenario in already_done:
            print(
                f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} — already scored, skipping.",
                flush=True,
            )
            continue

        print(
            f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} (effort-marginal risk)",
            flush=True,
        )

        risk_ratings, risk_failures = get_ratings_concurrent(
            client,
            RISK_SYSTEM_PROMPT,
            format_risk_prompt_marginal(row),
            parse_action_response,
            response_format=numeric_action_schema(N_ACTIONS),
            label=f"{scenario}/marginal/risk",
        )
        risk_agg = aggregate_action_ratings(risk_ratings, n_actions=N_ACTIONS)

        for action in range(N_ACTIONS):
            key = f"action_{action}"
            a_mean, a_std = risk_agg[key]
            results.append(
                {
                    "scenario_label": scenario,
                    "action": action,
                    "risk_raw": a_mean,
                    "risk_raw_std": a_std,
                    "risk": normalize_risk(a_mean) if not np.isnan(a_mean) else np.nan,
                    "n_runs_risk": len(risk_ratings),
                    "n_failures_risk": risk_failures,
                }
            )

        acc_str = [f"{risk_agg[f'action_{i}'][0]:.1f}" for i in range(N_ACTIONS)]
        print(f"  Risk (raw): {acc_str}", flush=True)

        pd.DataFrame(results).to_csv(output_path, index=False)

    results_df = pd.DataFrame(results)
    print(f"\nSaved effort-marginal risk to {output_path}")
    print(f"Total rows: {len(results_df)} (expected 48 = 16 scenarios × 3 actions)")
    print(
        f"\nRisk (normalized, target [0, 1]):"
        f"\n  Mean: {results_df['risk'].mean():.2f}, Std: {results_df['risk'].std():.2f}"
        f"\n  Range: [{results_df['risk'].min():.2f}, {results_df['risk'].max():.2f}]"
    )


def main():
    api_key = load_api_key()

    print("Loading 3-action scenarios...")
    scenarios_df = load_scenarios()
    print(f"Loaded {len(scenarios_df)} scenarios")

    print(f"\nInitializing Together AI client for {MODEL_ID}...")
    client = Together(api_key=api_key)

    output_dir = get_project_root() / "model" / "outputs" / "lm"
    output_dir.mkdir(exist_ok=True)
    cond_path = output_dir / "lm_scenario_params.csv"
    marg_path = output_dir / "lm_scenario_params_marginal.csv"

    print("\n=== Effort-conditional pass (3 actions) ===")
    run_effort_conditional(client, scenarios_df, cond_path)

    print("\n=== Effort-marginal risk pass (3 actions) ===")
    run_marginal_risk(client, scenarios_df, marg_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
