#!/usr/bin/env python3
"""
Generate LLM-derived scenario-specific parameters for the effort-manipulation
experiment (scenarios_effort).

For each of the 16 scenarios in experiments/scenarios_effort.csv, estimates
per (effort_condition, action):

- access(a): physical / informational / spatial exposure per action  (0-6 -> [0, 2])
- effort(a): physical / logistical cost per action                   (0-6 -> [0, 1])

Unlike the 4-action canonical pipeline, effort scenarios have only 2 actions
(action_1 = non-saliva-share, action_2 = saliva-share) and a 2-level effort
manipulation carried by the effort_low / effort_high vignette paragraphs.
The LM is prompted with the full (vignette + effort_paragraph) text so the
manipulation shows up in the ratings (the non-share action's effort should be
higher under effort_high than under effort_low).

Reward is NOT elicited from the LLM — reward is fixed at HIGH for all effort
scenarios and V(a|s) = 1 is stipulated uniformly for both actions.

10 runs per parameter-type per (scenario, effort_condition), aggregated to
mean/std. Outputs (model/outputs/):
- lm_scenario_params_effort.csv (64 rows: 16 scenarios × 2 efforts × 2 actions)
  Effort-conditional access + effort. Used by food_forw_intimacy_effort and
  food_inv-intimacy_effort_alt, where the actor / observer sees the effort paragraph.
- lm_scenario_params_effort_marginal.csv (32 rows: 16 scenarios × 2 actions)
  Effort-marginal access only (vignette without effort paragraph). Used by
  food_inv-effort_intimacy_alt, where the observer does not see the effort
  paragraph and so must reason about access from the base vignette alone.

Usage:
    uv run python model/lm/scenario_params_effort.py

If the conditional CSV already exists the conditional pass is skipped (useful
when only refreshing the marginal table).

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

# Shared LM-call infrastructure + canonical-script helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import (
    MODEL_ID,
    aggregate_action_ratings,
    find_json,
    get_ratings_concurrent,
    load_api_key,
    numeric_action_schema,
)
from scenario_params import normalize_access, normalize_effort
# Imported with aliases so they don't collide with parameters below.
from prompts import system_prompt as build_system_prompt
from prompts import user_prompt as build_user_prompt


EFFORT_CONDITIONS = ["low", "high"]


# Module-level system-prompt constants (kept for backwards compatibility with
# any external callers; the script itself uses the build_system_prompt helper
# directly at the call site).
ACCESS_SYSTEM_PROMPT = build_system_prompt("access", n_actions=2)
EFFORT_SYSTEM_PROMPT = build_system_prompt("effort", n_actions=2)


def load_scenarios():
    scenarios_path = get_project_root() / "experiments" / "scenarios_effort.csv"
    return pd.read_csv(scenarios_path)


def format_full_vignette(row, effort_condition):
    """Concatenate the shared vignette with the effort_low or effort_high paragraph."""
    effort_paragraph = row[f"effort_{effort_condition}"]
    return f"{row['vignette']} {effort_paragraph}"


def _action_texts_2(row):
    return [row["action_1"], row["action_2"]]


def format_access_prompt(row, effort_condition):
    return build_user_prompt(
        "access", format_full_vignette(row, effort_condition), _action_texts_2(row)
    )


def format_access_prompt_marginal(row):
    """Access prompt that omits the effort paragraph. Used to estimate access
    as it would be perceived by an observer who does not see the effort context
    (specifically, the food_inv-effort_intimacy_alt experiment)."""
    return build_user_prompt("access", row["vignette"], _action_texts_2(row))


def format_effort_prompt(row, effort_condition):
    return build_user_prompt(
        "effort", format_full_vignette(row, effort_condition), _action_texts_2(row)
    )


def parse_action_response(response_text):
    """Parse JSON ratings with action_0 / action_1 keys (2 actions)."""
    if response_text is None:
        return None
    js = find_json(response_text)
    if js is None:
        return None
    try:
        ratings = json.loads(js)
        expected = {"action_0", "action_1"}
        if expected.issubset(ratings.keys()):
            return {k: float(ratings[k]) for k in expected}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Failed to parse JSON: {e}")
    return None


def run_effort_conditional(client, scenarios_df, output_path):
    """Effort-conditional access + effort ratings.
    One row per (scenario, effort_condition, action) — 64 rows total.

    Resumes from output_path if it exists; flushes after each scenario."""
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
            print(f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} — already scored, skipping.", flush=True)
            continue

        for effort_condition in EFFORT_CONDITIONS:
            print(f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} (effort={effort_condition})", flush=True)

            print("  Getting access ratings (concurrent, structured, effort-conditional)...", flush=True)
            access_ratings, access_failures = get_ratings_concurrent(
                client,
                ACCESS_SYSTEM_PROMPT,
                format_access_prompt(row, effort_condition),
                parse_action_response,
                response_format=numeric_action_schema(2),
                label=f"{scenario}/effort={effort_condition}/access",
            )
            access_agg = aggregate_action_ratings(access_ratings, n_actions=2)

            print("  Getting effort ratings (concurrent, structured)...", flush=True)
            effort_ratings, effort_failures = get_ratings_concurrent(
                client,
                EFFORT_SYSTEM_PROMPT,
                format_effort_prompt(row, effort_condition),
                parse_action_response,
                response_format=numeric_action_schema(2),
                label=f"{scenario}/effort={effort_condition}/effort",
            )
            effort_agg = aggregate_action_ratings(effort_ratings, n_actions=2)

            # LM's action_0 = CSV action_1, LM's action_1 = CSV action_2
            for lm_idx, csv_action in enumerate([1, 2]):
                key = f"action_{lm_idx}"
                a_mean, a_std = access_agg[key]
                e_mean, e_std = effort_agg[key]
                results.append(
                    {
                        "scenario_label": scenario,
                        "effort_condition": effort_condition,
                        "action": csv_action,
                        "access_raw": a_mean,
                        "access_raw_std": a_std,
                        "effort_raw": e_mean,
                        "effort_raw_std": e_std,
                        "access": normalize_access(a_mean) if not np.isnan(a_mean) else np.nan,
                        "effort": normalize_effort(e_mean) if not np.isnan(e_mean) else np.nan,
                        "n_runs_access": len(access_ratings),
                        "n_runs_effort": len(effort_ratings),
                        "n_failures_access": access_failures,
                        "n_failures_effort": effort_failures,
                    }
                )

            acc_str = [f"{access_agg[f'action_{i}'][0]:.1f}" for i in range(2)]
            eff_str = [f"{effort_agg[f'action_{i}'][0]:.1f}" for i in range(2)]
            print(f"  Access (raw, action_1/action_2): {acc_str}", flush=True)
            print(f"  Effort (raw, action_1/action_2): {eff_str}", flush=True)

        # Checkpoint after each scenario.
        pd.DataFrame(results).to_csv(output_path, index=False)

    results_df = pd.DataFrame(results)
    print(f"\nSaved effort-conditional results to {output_path}")

    print("\n=== Conditional summary ===")
    print(f"Total rows: {len(results_df)} (expected 64 = 16 scenarios × 2 effort × 2 actions)")
    for col, target in [("access", "[0, 2]"), ("effort", "[0, 1]")]:
        print(
            f"\n{col.capitalize()} (normalized, target {target}):"
            f"\n  Mean: {results_df[col].mean():.2f}, Std: {results_df[col].std():.2f}"
            f"\n  Range: [{results_df[col].min():.2f}, {results_df[col].max():.2f}]"
        )

    print("\n=== Effort manipulation sanity (action_1 effort: low vs high) ===")
    act1 = results_df[results_df["action"] == 1]
    wide = act1.pivot(index="scenario_label", columns="effort_condition", values="effort")
    wide["delta"] = wide["high"] - wide["low"]
    print(wide.round(3).to_string())
    print(f"\nMean Δ effort(action_1): {wide['delta'].mean():+.3f} "
          f"(positive = manipulation worked)")


def run_marginal_access(client, scenarios_df, output_path):
    """Effort-marginal access ratings — vignette without effort paragraph.
    One row per (scenario, action) — 32 rows total. Resumes per scenario."""
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
            print(f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} — already scored, skipping.", flush=True)
            continue

        print(f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} (effort-marginal access, concurrent)", flush=True)

        access_ratings, access_failures = get_ratings_concurrent(
            client,
            ACCESS_SYSTEM_PROMPT,
            format_access_prompt_marginal(row),
            parse_action_response,
            response_format=numeric_action_schema(2),
            label=f"{scenario}/marginal/access",
        )
        access_agg = aggregate_action_ratings(access_ratings, n_actions=2)

        for lm_idx, csv_action in enumerate([1, 2]):
            key = f"action_{lm_idx}"
            a_mean, a_std = access_agg[key]
            results.append(
                {
                    "scenario_label": scenario,
                    "action": csv_action,
                    "access_raw": a_mean,
                    "access_raw_std": a_std,
                    "access": normalize_access(a_mean) if not np.isnan(a_mean) else np.nan,
                    "n_runs_access": len(access_ratings),
                    "n_failures_access": access_failures,
                }
            )

        acc_str = [f"{access_agg[f'action_{i}'][0]:.1f}" for i in range(2)]
        print(f"  Access (raw, action_1/action_2): {acc_str}", flush=True)

        # Checkpoint after each scenario.
        pd.DataFrame(results).to_csv(output_path, index=False)

    results_df = pd.DataFrame(results)
    print(f"\nSaved effort-marginal access to {output_path}")
    print(f"Total rows: {len(results_df)} (expected 32 = 16 scenarios × 2 actions)")
    print(
        f"\nAccess (normalized, target [0, 2]):"
        f"\n  Mean: {results_df['access'].mean():.2f}, Std: {results_df['access'].std():.2f}"
        f"\n  Range: [{results_df['access'].min():.2f}, {results_df['access'].max():.2f}]"
    )


def main():
    api_key = load_api_key()

    print("Loading effort scenarios...")
    scenarios_df = load_scenarios()
    print(f"Loaded {len(scenarios_df)} scenarios")

    print(f"\nInitializing Together AI client for {MODEL_ID}...")
    client = Together(api_key=api_key)

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    cond_path = output_dir / "lm_scenario_params_effort.csv"
    marg_path = output_dir / "lm_scenario_params_effort_marginal.csv"

    # Both passes resume per-scenario from their respective CSVs, so an
    # already-complete file becomes a no-op (each scenario is logged-and-skipped).
    print("\n=== Effort-conditional pass ===")
    run_effort_conditional(client, scenarios_df, cond_path)

    print("\n=== Effort-marginal access pass ===")
    run_marginal_access(client, scenarios_df, marg_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
