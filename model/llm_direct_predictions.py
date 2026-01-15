#!/usr/bin/env python3
"""
Generate direct LLM predictions for the forward planning experiment.

This script presents the exact experiment instructions and stimuli to an LLM
and collects probability predictions for each action, to compare against
human data and other cognitive models.

Usage:
    python llm_direct_predictions.py

Requires:
    - TOGETHER_API_KEY environment variable or in .env file
    - pip install together
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from together import Together

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root

# Configuration
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
NUM_RUNS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 200
RATE_LIMIT_DELAY = 0.5

INTIMACY_LEVELS = [0, 50, 75, 100]
MOTIVATION_CONDITIONS = ["low", "high"]

INTIMACY_TEXTS = {
    0: "0 (maximally formal)",
    50: "50 (neither formal nor intimate)",
    75: "75 (somewhat intimate)",
    100: "100 (maximally intimate)",
}

# System prompt - matches experiment instructions from trials.js
SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read vignettes about two people in different kinds of social relationships, sharing different kinds of food in different situations.

Some relationships are formal, like some relationships with an employee, a religious leader, a shopkeeper or a new acquaintance. Other relationships are close and intimate, like some relationships with a romantic partner, sibling or best friend.

For each scenario, you will read about four different actions the two people can take. You will indicate the probability that the two people will choose each action. The probabilities must sum to 100%.

Please pay attention to the social relationship between the two people, and read each of the scenarios and ways of sharing food carefully!

Respond with your probability estimates in this JSON format only, no explanation needed:
{"action_0": 25.0, "action_1": 25.0, "action_2": 25.0, "action_3": 25.0}

The four values must sum to exactly 100.0."""


def load_stimuli():
    """Load scenario stimuli from JSON file."""
    stimuli_path = get_project_root() / "experiments" / "forw_plan" / "json" / "stimuli.json"
    with open(stimuli_path) as f:
        return json.load(f)


def format_trial_prompt(scenario: dict, intimacy: int, motivation: str) -> str:
    """Format a single trial prompt matching the exact experiment presentation."""
    reward_text = scenario["reward_low"] if motivation == "low" else scenario["reward_high"]

    return f"""On a scale from 0 (maximally formal) to 100 (maximally intimate), {scenario['name_0']} and {scenario['name_1']} are in a relationship they would describe as {INTIMACY_TEXTS[intimacy]}.

{scenario['vignette']}

{reward_text}

Please indicate the probability that the two people will choose each action. The probabilities must sum to 100%.

Action 0: {scenario['action_0']}
Action 1: {scenario['action_1']}
Action 2: {scenario['action_2']}
Action 3: {scenario['action_3']}"""


def parse_probability_response(response_text: str) -> dict | None:
    """Parse and validate probability response from LLM."""
    if response_text is None:
        return None

    text = response_text.strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1

    if start_idx == -1 or end_idx <= start_idx:
        return None

    json_str = text[start_idx:end_idx]

    try:
        probs = json.loads(json_str)
        expected_keys = {"action_0", "action_1", "action_2", "action_3"}

        if not expected_keys.issubset(probs.keys()):
            return None

        # Extract and validate probabilities
        values = [float(probs[f"action_{i}"]) for i in range(4)]

        # Check for negative values
        if any(v < 0 for v in values):
            return None

        total = sum(values)

        # Reject if total is too small (all zeros)
        if total < 0.1:
            return None

        # Normalize to sum to 1.0
        normalized = {f"action_{i}": values[i] / total for i in range(4)}

        return normalized

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"    Parse error: {e}")
        return None


def get_llm_predictions(client, user_prompt: str, num_runs: int) -> list:
    """Get predictions from LLM with multiple runs."""
    predictions = []

    for run in range(num_runs):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            response_text = response.choices[0].message.content
            probs = parse_probability_response(response_text)
            if probs is not None:
                predictions.append(probs)
            else:
                print(f"    Run {run+1}: Failed to parse response")
        except Exception as e:
            print(f"    Run {run+1} Error: {e}")

        time.sleep(RATE_LIMIT_DELAY)

    return predictions


def main():
    print("=" * 60)
    print("LLM Direct Predictions for Forward Planning Experiment")
    print("=" * 60)

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

    print(f"\nModel: {MODEL_ID}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Runs per condition: {NUM_RUNS}")

    # Load stimuli
    print("\nLoading stimuli...")
    stimuli = load_stimuli()
    print(f"Loaded {len(stimuli)} scenarios")

    # Initialize API client
    print(f"\nInitializing Together AI client...")
    client = Together(api_key=api_key)

    # Generate predictions for all conditions
    results = []
    total_conditions = len(stimuli) * len(INTIMACY_LEVELS) * len(MOTIVATION_CONDITIONS)
    condition_idx = 0

    for scenario in stimuli:
        for intimacy in INTIMACY_LEVELS:
            for motivation in MOTIVATION_CONDITIONS:
                condition_idx += 1
                print(
                    f"\n[{condition_idx}/{total_conditions}] {scenario['scenario_label']}, "
                    f"intimacy={intimacy}, motivation={motivation}"
                )

                user_prompt = format_trial_prompt(scenario, intimacy, motivation)
                predictions = get_llm_predictions(client, user_prompt, NUM_RUNS)

                print(f"  Valid runs: {len(predictions)}/{NUM_RUNS}")

                # Store results for each action
                for action in range(4):
                    values = [p[f"action_{action}"] for p in predictions]
                    results.append(
                        {
                            "scenario_label": scenario["scenario_label"],
                            "intimacy": intimacy,
                            "motivation": motivation,
                            "action": action,
                            "pred_llm_mean": np.mean(values) if values else np.nan,
                            "pred_llm_std": np.std(values) if values else np.nan,
                            "n_valid_runs": len(values),
                        }
                    )

                # Print summary for this condition
                if predictions:
                    probs_str = [
                        f"{np.mean([p[f'action_{i}'] for p in predictions]):.2f}"
                        for i in range(4)
                    ]
                    print(f"  Predictions: {probs_str}")

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    output_path = get_project_root() / "model" / "llm_direct_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    # Summary statistics
    print(f"\nTotal rows: {len(results_df)}")
    print(f"Valid predictions: {results_df['n_valid_runs'].sum()} / {len(results_df) * NUM_RUNS}")

    # Merge with forward_planning_fits.csv
    print("\n" + "-" * 40)
    print("Merging with forward_planning_fits.csv...")
    print("-" * 40)

    fits_path = get_project_root() / "model" / "forward_planning_fits.csv"
    if fits_path.exists():
        fits_df = pd.read_csv(fits_path)

        # Merge on condition columns
        llm_for_merge = results_df[
            ["scenario_label", "intimacy", "motivation", "action", "pred_llm_mean"]
        ].rename(columns={"pred_llm_mean": "pred_llm"})

        merged = fits_df.merge(
            llm_for_merge,
            on=["scenario_label", "intimacy", "motivation", "action"],
            how="left",
        )

        merged.to_csv(fits_path, index=False)
        print(f"Added pred_llm column to {fits_path}")
    else:
        print(f"Warning: {fits_path} not found, skipping merge")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
