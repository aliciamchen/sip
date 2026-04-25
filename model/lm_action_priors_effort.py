#!/usr/bin/env python3
"""
Generate LLM-derived per-(scenario, effort_condition, action) priors π(a|s,e)
for the effort-manipulation forward-planning actor.

For each of the 16 scenarios in experiments/scenarios_effort.csv, under each of
the 2 effort conditions (low, high), rates the "default-ness" of each of the 2
actions on a 0-6 scale (10 runs aggregated), independent of relationship
intimacy, then normalizes across actions within (scenario, effort_condition)
to get π(a|s,e) that sums to 1.

Output: model/outputs/lm_action_priors_effort.csv with columns
  scenario_label, effort_condition, action, prior_raw, prior_raw_std, prior, n_runs
64 rows (16 × 2 × 2).

Usage:
    uv run python model/lm_action_priors_effort.py

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_scenario_params import (
    MODEL_ID,
    NUM_RUNS,
    TEMPERATURE,
    _find_json,
    _load_api_key,
)
from lm_scenario_params_effort import (
    EFFORT_CONDITIONS,
    format_full_vignette,
    load_scenarios,
)
from lm_action_priors import normalize_prior


from lm_prompts import system_prompt as build_system_prompt
from lm_prompts import user_prompt as build_user_prompt


ACTION_PRIOR_SYSTEM_PROMPT = build_system_prompt("prior", n_actions=2)


def format_prior_prompt(row, effort_condition):
    vignette = format_full_vignette(row, effort_condition)
    return build_user_prompt("prior", vignette, [row["action_1"], row["action_2"]])


def parse_action_response(response_text):
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
        return {f"action_{i}": (np.nan, np.nan) for i in range(2)}
    out = {}
    for i in range(2):
        key = f"action_{i}"
        values = [r[key] for r in ratings_list if key in r]
        out[key] = (np.mean(values), np.std(values)) if values else (np.nan, np.nan)
    return out


def main():
    api_key = _load_api_key()
    print("Loading effort scenarios...")
    scenarios_df = load_scenarios()
    print(f"Loaded {len(scenarios_df)} scenarios")

    print(f"\nInitializing Together AI client for {MODEL_ID}...")
    client = Together(api_key=api_key)

    results = []
    for idx, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        for effort_condition in EFFORT_CONDITIONS:
            print(f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} (effort={effort_condition})")

            ratings = get_ratings(client, format_prior_prompt(row, effort_condition))
            agg = aggregate(ratings)

            raw_means = [agg[f"action_{i}"][0] for i in range(2)]
            raw_stds = [agg[f"action_{i}"][1] for i in range(2)]
            if any(np.isnan(m) for m in raw_means):
                print("  Warning: missing ratings, skipping normalization")
                priors = [np.nan] * 2
            else:
                priors = normalize_prior(raw_means)

            for lm_idx, csv_action in enumerate([1, 2]):
                results.append({
                    "scenario_label": scenario,
                    "effort_condition": effort_condition,
                    "action": csv_action,
                    "prior_raw": raw_means[lm_idx],
                    "prior_raw_std": raw_stds[lm_idx],
                    "prior": priors[lm_idx],
                    "n_runs": len(ratings),
                })

            raw_str = [f"{m:.1f}" for m in raw_means]
            prior_str = [f"{p:.2f}" for p in priors]
            print(f"  Raw (0-6, action_1/action_2): {raw_str}")
            print(f"  Prior (action_1/action_2):    {prior_str}  (sum = {sum(priors):.3f})")

    results_df = pd.DataFrame(results)
    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "lm_action_priors_effort.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    print("\n=== Summary ===")
    print(f"Total rows: {len(results_df)} (expected 64 = 16 × 2 × 2)")
    print("Prior-sum check by (scenario, effort_condition):")
    sum_check = results_df.groupby(["scenario_label", "effort_condition"])["prior"].sum()
    print(f"  min={sum_check.min():.3f}, max={sum_check.max():.3f} (should all ≈ 1.0)")


if __name__ == "__main__":
    main()
