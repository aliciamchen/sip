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
  Effort-conditional access + effort. Used by forw_plan_effort and
  inv_plan_effort, where the actor / observer sees the effort paragraph.
- lm_scenario_params_effort_marginal.csv (32 rows: 16 scenarios × 2 actions)
  Effort-marginal access only (vignette without effort paragraph). Used by
  inv_plan_effort_inferred, where the observer does not see the effort
  paragraph and so must reason about access from the base vignette alone.

Usage:
    uv run python model/lm_scenario_params_effort.py

If the conditional CSV already exists the conditional pass is skipped (useful
when only refreshing the marginal table).

Requires TOGETHER_API_KEY in env or .env.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from together import Together

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

# Reuse helpers from the canonical 4-action script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_scenario_params import (
    MODEL_ID,
    NUM_RUNS,
    TEMPERATURE,
    _find_json,
    _load_api_key,
    normalize_access,
    normalize_effort,
)


EFFORT_CONDITIONS = ["low", "high"]


# System prompts — same rating scale as the canonical 4-action prompts but
# worded for 2 actions instead of 4.

ACCESS_SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read vignettes about two people sharing food in different situations. For each scenario, you will read about two different actions the two people can take.

For each action, evaluate: how much does this action create a direct bodily channel between the two people — a pathway for substances from one person's body to reach the other, or for their bodies to physically contact each other?

Consider concrete things like:
- Does any substance from one person's body (saliva, breath, skin oils) reach the other person or their food?
- Does the action involve direct physical contact between the two people's bodies?
- Does the action involve one person handling food that will then enter the other person's mouth?

Simply eating in the same physical space — for example, two people at the same table with fully separate portions — does NOT by itself create such a channel, and should be rated near zero.

Rate only what the action DOES in this physical sense — not how intimate or awkward it would feel in any particular relationship.

Use this scale from 0 to 6 (continuous values allowed):
0 = No bodily channel between the two people (complete physical separation)
3 = Indirect bodily channel (e.g. eating from the same shared container with separate utensils)
6 = Direct transfer of bodily substances (e.g. sharing the same piece of food that both bite)

Respond with your numerical ratings in this JSON format only, no explanation needed:
{"action_0": 0.5, "action_1": 3.8}"""


EFFORT_SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read vignettes about two people in a food-sharing situation. For each scenario, you will read about two different actions the two people can take.

For each action, evaluate the PHYSICAL AND LOGISTICAL COST of executing the action. Consider:
- How much physical work does the action require (preparing, serving, cutting, pouring, handing over)?
- Does the action need extra items or utensils (plates, napkins, cutlery, containers)?
- Does the action add practical steps beyond simply eating?

Do NOT rate social awkwardness or interpersonal discomfort — only the physical and logistical cost.

Use this scale from 0 to 6 (continuous values allowed):
0 = No effort (acting independently, eating what you already have)
3 = Moderate effort (a few steps, some preparation)
6 = High effort (many steps, substantial setup)

Respond with your numerical ratings in this JSON format only, no explanation needed:
{"action_0": 0.5, "action_1": 3.2}"""


def load_scenarios():
    scenarios_path = get_project_root() / "experiments" / "scenarios_effort.csv"
    return pd.read_csv(scenarios_path)


def format_full_vignette(row, effort_condition):
    """Concatenate the shared vignette with the effort_low or effort_high paragraph."""
    effort_paragraph = row[f"effort_{effort_condition}"]
    return f"{row['vignette']} {effort_paragraph}"


def format_access_prompt(row, effort_condition):
    vignette = format_full_vignette(row, effort_condition)
    return f"""Scenario: {vignette}

Rate how much each action opens each person up to the other — physically, informationally, or both (0-6 scale):

Action 0: {row["action_1"]}
Action 1: {row["action_2"]}"""


def format_access_prompt_marginal(row):
    """Access prompt that omits the effort paragraph. Used to estimate access
    as it would be perceived by an observer who does not see the effort context
    (specifically, the inv_plan_effort_inferred experiment)."""
    return f"""Scenario: {row["vignette"]}

Rate how much each action opens each person up to the other — physically, informationally, or both (0-6 scale):

Action 0: {row["action_1"]}
Action 1: {row["action_2"]}"""


def format_effort_prompt(row, effort_condition):
    vignette = format_full_vignette(row, effort_condition)
    return f"""Scenario: {vignette}

Rate the physical and logistical cost of executing each action — how much physical work, preparation, or extra equipment is required (0-6 scale):

Action 0: {row["action_1"]}
Action 1: {row["action_2"]}"""


def parse_action_response(response_text):
    """Parse JSON ratings with action_0 / action_1 keys (2 actions)."""
    if response_text is None:
        return None
    js = _find_json(response_text.strip())
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


def get_ratings(client, system_prompt, user_prompt, num_runs=NUM_RUNS):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    all_ratings = []
    for run in range(num_runs):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=200,
                temperature=TEMPERATURE,
            )
            ratings = parse_action_response(response.choices[0].message.content)
            if ratings is not None:
                all_ratings.append(ratings)
        except Exception as e:
            print(f"  Run {run + 1} error: {e}")
        time.sleep(0.5)
    return all_ratings


def aggregate_action_ratings(ratings_list):
    if not ratings_list:
        return {f"action_{i}": (np.nan, np.nan) for i in range(2)}
    result = {}
    for i in range(2):
        key = f"action_{i}"
        values = [r[key] for r in ratings_list if key in r]
        result[key] = (np.mean(values), np.std(values)) if values else (np.nan, np.nan)
    return result


def run_effort_conditional(client, scenarios_df, output_path):
    """Effort-conditional access + effort ratings (existing behaviour).
    One row per (scenario, effort_condition, action) — 64 rows total."""
    results = []
    for idx, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        for effort_condition in EFFORT_CONDITIONS:
            print(f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} (effort={effort_condition})")

            print("  Getting access ratings (effort-conditional)...")
            access_ratings = get_ratings(
                client,
                ACCESS_SYSTEM_PROMPT,
                format_access_prompt(row, effort_condition),
            )
            access_agg = aggregate_action_ratings(access_ratings)

            print("  Getting effort ratings...")
            effort_ratings = get_ratings(
                client,
                EFFORT_SYSTEM_PROMPT,
                format_effort_prompt(row, effort_condition),
            )
            effort_agg = aggregate_action_ratings(effort_ratings)

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
                    }
                )

            acc_str = [f"{access_agg[f'action_{i}'][0]:.1f}" for i in range(2)]
            eff_str = [f"{effort_agg[f'action_{i}'][0]:.1f}" for i in range(2)]
            print(f"  Access (raw, action_1/action_2): {acc_str}")
            print(f"  Effort (raw, action_1/action_2): {eff_str}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
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
    One row per (scenario, action) — 32 rows total."""
    results = []
    for idx, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        print(f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} (effort-marginal access)")

        access_ratings = get_ratings(
            client,
            ACCESS_SYSTEM_PROMPT,
            format_access_prompt_marginal(row),
        )
        access_agg = aggregate_action_ratings(access_ratings)

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
                }
            )

        acc_str = [f"{access_agg[f'action_{i}'][0]:.1f}" for i in range(2)]
        print(f"  Access (raw, action_1/action_2): {acc_str}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved effort-marginal access to {output_path}")
    print(f"Total rows: {len(results_df)} (expected 32 = 16 scenarios × 2 actions)")
    print(
        f"\nAccess (normalized, target [0, 2]):"
        f"\n  Mean: {results_df['access'].mean():.2f}, Std: {results_df['access'].std():.2f}"
        f"\n  Range: [{results_df['access'].min():.2f}, {results_df['access'].max():.2f}]"
    )


def main():
    api_key = _load_api_key()

    print("Loading effort scenarios...")
    scenarios_df = load_scenarios()
    print(f"Loaded {len(scenarios_df)} scenarios")

    print(f"\nInitializing Together AI client for {MODEL_ID}...")
    client = Together(api_key=api_key)

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    cond_path = output_dir / "lm_scenario_params_effort.csv"
    marg_path = output_dir / "lm_scenario_params_effort_marginal.csv"

    if cond_path.exists():
        print(f"\nSkipping effort-conditional pass — {cond_path.name} already exists.")
    else:
        print("\n=== Effort-conditional pass ===")
        run_effort_conditional(client, scenarios_df, cond_path)

    print("\n=== Effort-marginal access pass ===")
    run_marginal_access(client, scenarios_df, marg_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
