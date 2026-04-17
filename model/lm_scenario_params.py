#!/usr/bin/env python3
"""
Generate LLM-derived scenario-specific parameters for the access-based models.

Uses Together AI's Llama-3.3-70B-Instruct-Turbo to estimate, for each of the 16
scenarios in experiments/scenarios.csv:

- access(a): physical / informational / spatial exposure per action  (0-6 -> [0, 2])
- effort(a): physical / logistical cost per action                   (0-6 -> [0, 1])

Reward is NOT elicited from the LLM — it's stipulated in `model/model_utils.py`
as a binary goal-satisfaction gate (V=1 iff the action satisfies the active
goal: sharing under HIGH motivation, not-sharing under LOW motivation).

10 runs per parameter-type per scenario, aggregated to mean/std.

Usage:
    uv run python model/lm_scenario_params.py

Requires:
    - TOGETHER_API_KEY environment variable or in .env file
    - `together` Python package (add to pyproject.toml)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from together import Together

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

# Configuration
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
NUM_RUNS = 10
TEMPERATURE = 0.2


# ==============================================================================
# System prompts
# ==============================================================================

ACCESS_SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read vignettes about two people sharing food in different situations. For each scenario, you will read about four different actions the two people can take.

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
{"action_0": 0.5, "action_1": 1.2, "action_2": 3.8, "action_3": 5.5}"""


EFFORT_SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read vignettes about two people in a food-sharing situation. For each scenario, you will read about four different actions the two people can take.

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
{"action_0": 0.5, "action_1": 3.2, "action_2": 2.1, "action_3": 1.5}"""


# Reward is stipulated in model/model_utils.py as a binary goal-satisfaction
# gate (V=1 iff the action satisfies the active goal: sharing under HIGH
# motivation, not-sharing under LOW motivation). Not elicited from the LLM.


# ==============================================================================
# Scenario loading and prompt formatting
# ==============================================================================


def load_scenarios():
    scenarios_path = get_project_root() / "experiments" / "scenarios.csv"
    return pd.read_csv(scenarios_path)


def format_access_prompt(row):
    return f"""Scenario: {row["vignette"]}

Rate how much each action opens each person up to the other — physically, informationally, or both (0-6 scale):

Action 0: {row["action_0"]}
Action 1: {row["action_1"]}
Action 2: {row["action_2"]}
Action 3: {row["action_3"]}"""


def format_effort_prompt(row):
    return f"""Scenario: {row["vignette"]}

Rate the physical and logistical cost of executing each action — how much physical work, preparation, or extra equipment is required (0-6 scale):

Action 0: {row["action_0"]}
Action 1: {row["action_1"]}
Action 2: {row["action_2"]}
Action 3: {row["action_3"]}"""


# ==============================================================================
# Parsers + aggregators
# ==============================================================================


def _find_json(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    return text[start:end] if start != -1 and end > start else None


def parse_action_response(response_text):
    """Parse JSON ratings with action_0..action_3 keys."""
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


def get_ratings(client, system_prompt, user_prompt, parse_fn, num_runs=NUM_RUNS):
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
            ratings = parse_fn(response.choices[0].message.content)
            if ratings is not None:
                all_ratings.append(ratings)
        except Exception as e:
            print(f"  Run {run + 1} error: {e}")
        time.sleep(0.5)
    return all_ratings


def aggregate_action_ratings(ratings_list):
    if not ratings_list:
        return {f"action_{i}": (np.nan, np.nan) for i in range(4)}
    result = {}
    for i in range(4):
        key = f"action_{i}"
        values = [r[key] for r in ratings_list if key in r]
        result[key] = (np.mean(values), np.std(values)) if values else (np.nan, np.nan)
    return result


# ==============================================================================
# Normalization (0-6 LLM scale -> model-native scales)
# ==============================================================================


def normalize_access(value, target_max=2.0):
    """0-6 -> [0, target_max]. Matches the [0, 2] range of the fixed access vector."""
    return value * (target_max / 6.0)


def normalize_effort(value, target_max=1.0):
    return value * (target_max / 6.0)


# ==============================================================================
# Main
# ==============================================================================


def _load_api_key():
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        env_path = get_project_root() / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("TOGETHER_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip('"').strip("'")
                        break
    if not api_key:
        print("Error: set TOGETHER_API_KEY in env or .env")
        sys.exit(1)
    return api_key


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
        print(f"\nProcessing {idx + 1}/{len(scenarios_df)}: {scenario}")

        print("  Getting access ratings...")
        access_ratings = get_ratings(
            client,
            ACCESS_SYSTEM_PROMPT,
            format_access_prompt(row),
            parse_action_response,
        )
        access_agg = aggregate_action_ratings(access_ratings)

        print("  Getting effort ratings...")
        effort_ratings = get_ratings(
            client,
            EFFORT_SYSTEM_PROMPT,
            format_effort_prompt(row),
            parse_action_response,
        )
        effort_agg = aggregate_action_ratings(effort_ratings)

        for action in range(4):
            key = f"action_{action}"
            a_mean, a_std = access_agg[key]
            e_mean, e_std = effort_agg[key]
            results.append(
                {
                    "scenario_label": scenario,
                    "action": action,
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

        acc_str = [f"{access_agg[f'action_{i}'][0]:.1f}" for i in range(4)]
        eff_str = [f"{effort_agg[f'action_{i}'][0]:.1f}" for i in range(4)]
        print(f"  Access (raw): {acc_str}")
        print(f"  Effort (raw): {eff_str}")

    results_df = pd.DataFrame(results)
    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "lm_scenario_params.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    print("\n=== Summary ===")
    print(f"Total rows: {len(results_df)}")
    for col, target in [("access", "[0, 2]"), ("effort", "[0, 1]")]:
        print(
            f"\n{col.capitalize()} (normalized, target {target}):"
            f"\n  Mean: {results_df[col].mean():.2f}, Std: {results_df[col].std():.2f}"
            f"\n  Range: [{results_df[col].min():.2f}, {results_df[col].max():.2f}]"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
