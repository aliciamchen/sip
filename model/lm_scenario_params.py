#!/usr/bin/env python3
"""
Generate LLM-derived scenario-specific parameters (risk, effort, reward) using Together AI.

This script estimates scenario-specific values for the inverse planning model parameters:
- Risk (ρ): Saliva-transfer risk per action (0-6 scale -> normalized to [0, 2])
- Effort (c): Coordination/sharing effort per action (0-6 scale -> normalized to [0, 1])
- Reward (r_0): Base reward for sharing per scenario (0-6 scale -> normalized to ~1)

Usage:
    python lm_scenario_params.py

Requires:
    - TOGETHER_API_KEY environment variable or in .env file
    - pip install together
"""

import json
import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from together import Together

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root

# Configuration
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
NUM_RUNS = 10  # Fewer runs for exploratory script
TEMPERATURE = 0.2

# System prompts for each parameter type
RISK_SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read vignettes about two people sharing different kinds of food in different situations. For each scenario, you will read about four different actions the two people can take.

For each action, evaluate how much saliva is likely to be exchanged between the two people. Consider each option independently.

Use this scale from 0 to 6 (continuous values allowed):
0 = No saliva at all
1 = Very little saliva
2 = Little saliva
3 = Moderate saliva
4 = Considerable saliva
5 = Much saliva
6 = Very much saliva

Consider the physical mechanics of each action - does saliva from one person's mouth have a path to reach the other person's mouth? Direct contact (sharing utensils, biting the same food) transfers more saliva than indirect contact (using separate portions).

Respond with your numerical ratings in this JSON format only, no explanation needed:
{"action_0": 0.5, "action_1": 1.2, "action_2": 3.8, "action_3": 5.5}"""

EFFORT_SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read vignettes about two people in a food-sharing situation. For each scenario, you will read about four different actions the two people can take.

For each action, evaluate how much SOCIAL COORDINATION is required. Consider:
- Does this action require the two people to agree to share?
- Does this action involve commitment to a joint activity?
- Is there social negotiation involved (offering, accepting, coordinating)?

NOT sharing food requires no social coordination.
Any form of sharing requires at least some coordination between the two people.

Use this scale from 0 to 6 (continuous values allowed):
0 = No social coordination (acting independently, not sharing)
1-2 = Minimal coordination (simple sharing arrangement)
3-4 = Moderate coordination (requires agreement and some back-and-forth)
5-6 = High coordination (complex joint activity)

Respond with your numerical ratings in this JSON format only, no explanation needed:
{"action_0": 0.5, "action_1": 3.2, "action_2": 2.1, "action_3": 1.5}"""

REWARD_SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read vignettes about two people in a food-sharing situation. For each scenario, evaluate how rewarding or enjoyable the experience of sharing this particular food in this situation would be.

Consider:
- How appealing is this food?
- How special or enjoyable is this sharing occasion?
- How much social bonding value does this situation have?

Use this scale from 0 to 6 (continuous values allowed):
0 = Not rewarding at all
1 = Very little reward
2 = Little reward
3 = Moderate reward
4 = Considerable reward
5 = Much reward
6 = Very rewarding (special food, meaningful occasion)

Respond with a single numerical rating in this JSON format only:
{"reward": 3.5}"""


def load_scenarios():
    """Load scenario data from scenarios.csv."""
    scenarios_path = get_project_root() / "experiments" / "scenarios.csv"
    return pd.read_csv(scenarios_path)


def format_risk_prompt(row):
    """Format a scenario for the risk rating prompt."""
    return f"""Scenario: {row['vignette']}

Rate how much saliva from one person is likely to end up in the other person's mouth for each action (0-6 scale):

Action 0: {row['action_0']}
Action 1: {row['action_1']}
Action 2: {row['action_2']}
Action 3: {row['action_3']}"""


def format_effort_prompt(row):
    """Format a scenario for the effort rating prompt."""
    return f"""Scenario: {row['vignette']}

Rate how much coordination or logistical effort is required for each action (0-6 scale):

Action 0: {row['action_0']}
Action 1: {row['action_1']}
Action 2: {row['action_2']}
Action 3: {row['action_3']}"""


def format_reward_prompt(row):
    """Format a scenario for the reward rating prompt."""
    return f"""Scenario: {row['vignette']}

Rate how rewarding or enjoyable sharing this food in this situation would be (0-6 scale):"""


def parse_action_response(response_text):
    """Parse JSON ratings with action_0 through action_3 keys."""
    if response_text is None:
        return None

    text = response_text.strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1

    if start_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx:end_idx]
        try:
            ratings = json.loads(json_str)
            expected_keys = {"action_0", "action_1", "action_2", "action_3"}
            if expected_keys.issubset(ratings.keys()):
                return {k: float(ratings[k]) for k in expected_keys}
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Failed to parse JSON: {e}")

    return None


def parse_reward_response(response_text):
    """Parse JSON rating with reward key."""
    if response_text is None:
        return None

    text = response_text.strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1

    if start_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx:end_idx]
        try:
            ratings = json.loads(json_str)
            if "reward" in ratings:
                return float(ratings["reward"])
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Failed to parse JSON: {e}")

    return None


def get_ratings(client, system_prompt, user_prompt, parse_fn, num_runs=NUM_RUNS):
    """Get ratings from the model with multiple runs."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
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
            response_text = response.choices[0].message.content
            ratings = parse_fn(response_text)
            if ratings is not None:
                all_ratings.append(ratings)
        except Exception as e:
            print(f"  Run {run+1} Error: {e}")

        time.sleep(0.5)  # Rate limiting

    return all_ratings


def aggregate_action_ratings(ratings_list):
    """Aggregate action ratings across runs, returning mean and std."""
    if not ratings_list:
        return {f"action_{i}": (np.nan, np.nan) for i in range(4)}

    result = {}
    for i in range(4):
        key = f"action_{i}"
        values = [r[key] for r in ratings_list if key in r]
        if values:
            result[key] = (np.mean(values), np.std(values))
        else:
            result[key] = (np.nan, np.nan)
    return result


def aggregate_reward_ratings(ratings_list):
    """Aggregate reward ratings across runs."""
    if not ratings_list:
        return (np.nan, np.nan)
    return (np.mean(ratings_list), np.std(ratings_list))


def normalize_risk(value, target_max=2.0):
    """Normalize risk from 0-6 scale to [0, target_max]."""
    return value * (target_max / 6.0)


def normalize_effort(value, target_max=1.0):
    """Normalize effort from 0-6 scale to [0, target_max]."""
    return value * (target_max / 6.0)


def normalize_reward(value, target_mean=1.0):
    """Normalize reward from 0-6 scale to center around target_mean."""
    # Map 0-6 to roughly 0.5-1.5 range (centered at 1)
    return 0.5 + value * (1.0 / 6.0)


def main():
    # Get Together API key
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        env_path = get_project_root() / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("TOGETHER_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break
    if not api_key:
        print("Error: Please set TOGETHER_API_KEY environment variable or add to .env file")
        sys.exit(1)

    print("Loading scenarios...")
    scenarios_df = load_scenarios()
    print(f"Loaded {len(scenarios_df)} scenarios")

    print(f"\nInitializing Together AI client for {MODEL_ID}...")
    client = Together(api_key=api_key)

    results = []

    for idx, row in scenarios_df.iterrows():
        scenario = row['scenario_label']
        print(f"\nProcessing {idx+1}/{len(scenarios_df)}: {scenario}")

        # Get risk ratings
        print("  Getting risk ratings...")
        risk_ratings = get_ratings(
            client,
            RISK_SYSTEM_PROMPT,
            format_risk_prompt(row),
            parse_action_response
        )
        risk_agg = aggregate_action_ratings(risk_ratings)

        # Get effort ratings
        print("  Getting effort ratings...")
        effort_ratings = get_ratings(
            client,
            EFFORT_SYSTEM_PROMPT,
            format_effort_prompt(row),
            parse_action_response
        )
        effort_agg = aggregate_action_ratings(effort_ratings)

        # Get reward rating
        print("  Getting reward rating...")
        reward_ratings = get_ratings(
            client,
            REWARD_SYSTEM_PROMPT,
            format_reward_prompt(row),
            parse_reward_response
        )
        reward_mean, reward_std = aggregate_reward_ratings(reward_ratings)

        # Store results for each action
        for action in range(4):
            action_key = f"action_{action}"
            risk_mean, risk_std = risk_agg[action_key]
            effort_mean, effort_std = effort_agg[action_key]

            results.append({
                "scenario_label": scenario,
                "action": action,
                # Raw LLM ratings (0-6 scale)
                "risk_raw": risk_mean,
                "risk_raw_std": risk_std,
                "effort_raw": effort_mean,
                "effort_raw_std": effort_std,
                "reward_raw": reward_mean,
                "reward_raw_std": reward_std,
                # Normalized to model scale
                "risk": normalize_risk(risk_mean) if not np.isnan(risk_mean) else np.nan,
                "effort": normalize_effort(effort_mean) if not np.isnan(effort_mean) else np.nan,
                "reward": normalize_reward(reward_mean) if not np.isnan(reward_mean) else np.nan,
                "n_valid_runs": len(risk_ratings)
            })

        risk_str = [f'{risk_agg[f"action_{i}"][0]:.1f}' for i in range(4)]
        effort_str = [f'{effort_agg[f"action_{i}"][0]:.1f}' for i in range(4)]
        print(f"  Risk (raw): {risk_str}")
        print(f"  Effort (raw): {effort_str}")
        print(f"  Reward (raw): {reward_mean:.1f}")

    # Save results
    results_df = pd.DataFrame(results)
    output_path = get_project_root() / "model" / "lm_scenario_params.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Total rows: {len(results_df)}")
    print(f"\nRisk (normalized, target [0, 2]):")
    print(f"  Mean: {results_df['risk'].mean():.2f}, Std: {results_df['risk'].std():.2f}")
    print(f"  Range: [{results_df['risk'].min():.2f}, {results_df['risk'].max():.2f}]")
    print(f"\nEffort (normalized, target [0, 1]):")
    print(f"  Mean: {results_df['effort'].mean():.2f}, Std: {results_df['effort'].std():.2f}")
    print(f"  Range: [{results_df['effort'].min():.2f}, {results_df['effort'].max():.2f}]")
    print(f"\nReward (normalized, target ~1):")
    print(f"  Mean: {results_df['reward'].mean():.2f}, Std: {results_df['reward'].std():.2f}")
    print(f"  Range: [{results_df['reward'].min():.2f}, {results_df['reward'].max():.2f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
