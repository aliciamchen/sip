#!/usr/bin/env python3
"""
Generate LLM-derived per-(scenario, action) priors π(a|s) for the forward-
planning actor.

Motivation: the current Full model assumes a uniform π(a|s) in the softmax,
so at a fixed risk level all scenario-level variance must come through effort
and the softmax competition with the other 3 actions. The residual diagnostic
in forw-plan-analysis.qmd shows that humans systematically over-pick action 1
(minimal share) and under-pick action 0 (no share) in casual / sharing-
normative scenarios (fair, cooking, birthday, takeout, driving, brunch, hike,
dip) — a pattern consistent with scenario-specific action priors rather than
risk/effort/V mis-calibration. This script elicits those priors from an LLM.

For each of the 16 scenarios, rates the "default-ness" of each of the 4
canonical actions on a 0-6 scale (10 runs aggregated), independent of
relationship intimacy and motivation, then normalizes across actions within a
scenario to get π(a|s) that sums to 1.

Output: model/outputs/lm_action_priors.csv with columns
  scenario_label, action, prior_raw, prior_raw_std, prior, n_runs

Usage:
    uv run python model/lm_action_priors.py

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

# Reuse helpers from lm_scenario_params
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_scenario_params import (
    MODEL_ID,
    NUM_RUNS,
    TEMPERATURE,
    _find_json,
    _load_api_key,
    load_scenarios,
)


ACTION_PRIOR_SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, going off of your intuition.

In this survey, you will read vignettes about two people in a food-sharing situation. For each scenario, you will read about four different actions the two people can take.

For each action, evaluate how NATURAL or EXPECTED the action is as a "default" behavior in this setting — what you'd imagine typically happening given the food, the place, and the social occasion, independent of any specific information about the two people's relationship or how much they want the food.

Consider:
- Does the action fit the social conventions of this kind of setting?
- Is it a typical way people behave here, in general?

Do NOT factor in the specific relationship closeness between the two people, or how much they individually want the food. Just rate whether the action is a natural default for the setting itself.

Use this scale from 0 to 6 (continuous values allowed):
0 = Very unusual or out of place in this setting
3 = Plausible — could happen, neither default nor unusual
6 = Very natural — the typical default behavior in this setting

Respond with your numerical ratings in this JSON format only, no explanation needed:
{"action_0": 0.5, "action_1": 1.2, "action_2": 3.8, "action_3": 5.5}"""


def format_prior_prompt(row):
    return f"""Scenario: {row["vignette"]}

Rate how natural / default each action is in this setting (0-6 scale):

Action 0: {row["action_0"]}
Action 1: {row["action_1"]}
Action 2: {row["action_2"]}
Action 3: {row["action_3"]}"""


def parse_action_response(response_text):
    if response_text is None:
        return None
    js = _find_json(response_text.strip())
    if js is None:
        return None
    try:
        ratings = json.loads(js)
        expected = {"action_0", "action_1", "action_2", "action_3"}
        if expected.issubset(ratings.keys()):
            return {k: float(ratings[k]) for k in expected}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Failed to parse JSON: {e}")
    return None


def get_ratings(client, user_prompt, num_runs=NUM_RUNS):
    messages = [
        {"role": "system", "content": ACTION_PRIOR_SYSTEM_PROMPT},
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


def aggregate(ratings_list):
    if not ratings_list:
        return {f"action_{i}": (np.nan, np.nan) for i in range(4)}
    out = {}
    for i in range(4):
        key = f"action_{i}"
        values = [r[key] for r in ratings_list if key in r]
        out[key] = (np.mean(values), np.std(values)) if values else (np.nan, np.nan)
    return out


def normalize_prior(raw_means, epsilon=0.1):
    """Turn 0-6 ratings into a probability distribution over the 4 actions.

    Uses sum-normalization with a small additive smoothing so no action gets
    probability 0 (keeps the softmax differentiable and avoids knocking actions
    out entirely when the LLM rates them as 0).
    """
    smoothed = np.array(raw_means) + epsilon
    return smoothed / smoothed.sum()


def main():
    api_key = _load_api_key()
    print("Loading scenarios...")
    scenarios_df = load_scenarios()
    print(f"Loaded {len(scenarios_df)} scenarios")

    print(f"\nInitializing Together AI client for {MODEL_ID}...")
    client = Together(api_key=api_key)

    results = []
    for idx, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        print(f"\n[{idx + 1}/{len(scenarios_df)}] {scenario}")

        ratings = get_ratings(client, format_prior_prompt(row))
        agg = aggregate(ratings)

        raw_means = [agg[f"action_{i}"][0] for i in range(4)]
        raw_stds = [agg[f"action_{i}"][1] for i in range(4)]
        if any(np.isnan(m) for m in raw_means):
            print("  Warning: missing ratings, skipping normalization")
            priors = [np.nan] * 4
        else:
            priors = normalize_prior(raw_means)

        for action in range(4):
            results.append({
                "scenario_label": scenario,
                "action": action,
                "prior_raw": raw_means[action],
                "prior_raw_std": raw_stds[action],
                "prior": priors[action],
                "n_runs": len(ratings),
            })

        raw_str = [f"{m:.1f}" for m in raw_means]
        prior_str = [f"{p:.2f}" for p in priors]
        print(f"  Raw (0-6): {raw_str}")
        print(f"  Prior:     {prior_str}  (sum = {sum(priors):.3f})")

    results_df = pd.DataFrame(results)
    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "lm_action_priors.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    print("\n=== Summary ===")
    print(f"Total rows: {len(results_df)}")
    print(f"Prior per action (mean across scenarios):")
    for a in range(4):
        col = results_df[results_df["action"] == a]["prior"]
        print(f"  action {a}: mean={col.mean():.2f}, sd={col.std():.2f}")


if __name__ == "__main__":
    main()
