#!/usr/bin/env python3
"""
Generate LM predictions for the risk (saliva transfer) task using Llama-3.3-70B-Instruct.
"""

import json
import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from huggingface_hub import InferenceClient

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root

# Configuration
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
NUM_RUNS = 10
TEMPERATURE = 0.2

# Instructions given to human participants + output formatting
SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read vignettes about two people sharing different kinds of food in different situations. For each scenario, you will read about four different actions the two people can take.

For each action, we will ask you to evaluate how much saliva is likely to be exchanged between the two people, in the context of the scenario. Please consider each option independently.

Use this scale from 0 to 6 (continuous values allowed):
0 = No saliva at all
1 = Very little saliva
2 = Little saliva
3 = Moderate saliva
4 = Considerable saliva
5 = Much saliva
6 = Very much saliva

Consider the physical mechanics of each action - does saliva from one person's mouth have a path to reach the other person's mouth? Direct contact (sharing utensils, biting the same food) transfers more saliva than indirect contact (using separate portions).

Guidelines:
- Shared containers (even with separate utensils) carry indirect contamination risk
- People rarely perceive actions as absolutely 0 or maximum 6
- Consider both direct paths (shared utensils, bites) and indirect paths (shared containers, surfaces)

Respond by first writing out your reasoning for each action, then provide your numerical ratings in the following JSON format:
{"action_0": 0.5, "action_1": 1.2, "action_2": 3.8, "action_3": 5.5}"""


def load_stimuli():
    """Load the risk experiment stimuli."""
    stimuli_path = get_project_root() / "experiments" / "pilots" / "risk" / "json" / "stimuli.json"
    with open(stimuli_path, "r") as f:
        return json.load(f)


def load_human_data():
    """Load human risk ratings for comparison."""
    data_path = get_project_root() / "data" / "pilots" / "risk" / "risk_summary.csv"
    return pd.read_csv(data_path)


def format_user_prompt(stimulus):
    """Format a single scenario into a user prompt."""
    return f"""Scenario: {stimulus['vignette']}

Rate how much saliva from one person is likely to end up in the other person's mouth for each action (0-6 scale, decimals allowed):

Action 0: {stimulus['action_0']}
Action 1: {stimulus['action_1']}
Action 2: {stimulus['action_2']}
Action 3: {stimulus['action_3']}"""


def parse_response(response_text):
    """Parse JSON ratings from model response."""
    if response_text is None:
        return None

    text = response_text.strip()

    # Find JSON in the response
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
            print(f"Failed to parse JSON: {e}")
            print(f"Response was: {text}")

    return None


def get_predictions(stimuli, client):
    """Get LM predictions for all stimuli with multiple runs."""
    predictions = []
    all_explanations = []

    for i, stimulus in enumerate(stimuli):
        print(f"Processing {i+1}/{len(stimuli)}: {stimulus['scenario_label']}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_prompt(stimulus)}
        ]

        # Collect ratings from multiple runs
        run_ratings = {f"action_{j}": [] for j in range(4)}
        run_explanations = []

        for run in range(NUM_RUNS):
            try:
                response = client.chat_completion(
                    messages=messages,
                    max_tokens=500,  # More tokens for explanations
                    temperature=TEMPERATURE,
                )
                response_text = response.choices[0].message.content
                ratings = parse_response(response_text)

                if ratings:
                    for action_idx in range(4):
                        run_ratings[f"action_{action_idx}"].append(ratings[f"action_{action_idx}"])
                    run_explanations.append(response_text)
                    print(f"  Run {run+1}: {ratings}")
                else:
                    print(f"  Run {run+1}: Failed to parse response")

            except Exception as e:
                print(f"  Run {run+1} Error: {e}")

            # Small delay to avoid rate limiting
            time.sleep(0.3)

        # Calculate mean ratings across runs
        for action_idx in range(4):
            action_key = f"action_{action_idx}"
            if run_ratings[action_key]:
                mean_rating = np.mean(run_ratings[action_key])
                std_rating = np.std(run_ratings[action_key])
                n_valid = len(run_ratings[action_key])
            else:
                mean_rating = float("nan")
                std_rating = float("nan")
                n_valid = 0

            predictions.append({
                "scenario_label": stimulus["scenario_label"],
                "action": action_idx,
                "lm_rating": mean_rating,
                "lm_rating_std": std_rating,
                "n_valid_runs": n_valid
            })

        # Save one example explanation per scenario
        if run_explanations:
            all_explanations.append({
                "scenario_label": stimulus["scenario_label"],
                "explanation": run_explanations[0]  # Save first explanation as example
            })

        print(f"  Mean ratings: " + ", ".join([
            f"action_{j}={np.mean(run_ratings[f'action_{j}']):.2f}"
            for j in range(4) if run_ratings[f'action_{j}']
        ]))

    return pd.DataFrame(predictions), pd.DataFrame(all_explanations)


def main():
    # Get HF token from environment or .env file
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        env_path = get_project_root() / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.strip().split("=", 1)[1]
                        break
    if not hf_token:
        print("Error: Please set HF_TOKEN environment variable or add to .env file")
        sys.exit(1)

    print("Loading stimuli...")
    stimuli = load_stimuli()
    print(f"Loaded {len(stimuli)} scenarios")

    print(f"\nInitializing client for {MODEL_ID}...")
    client = InferenceClient(model=MODEL_ID, token=hf_token)

    print(f"\nGetting LM predictions ({NUM_RUNS} runs per scenario, temperature={TEMPERATURE})...")
    predictions_df, explanations_df = get_predictions(stimuli, client)

    # Save predictions
    output_path = get_project_root() / "data" / "pilots" / "risk" / "lm_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")

    # Save explanations
    explanations_path = get_project_root() / "data" / "pilots" / "risk" / "lm_explanations.csv"
    explanations_df.to_csv(explanations_path, index=False)
    print(f"Saved explanations to {explanations_path}")

    # Load human data and merge for comparison
    print("\nLoading human data for comparison...")
    human_df = load_human_data()

    merged = predictions_df.merge(
        human_df[["scenario_label", "action", "mean"]],
        on=["scenario_label", "action"],
        how="left"
    )
    merged = merged.rename(columns={"mean": "human_mean"})

    # Save merged comparison
    comparison_path = get_project_root() / "data" / "pilots" / "risk" / "lm_human_comparison.csv"
    merged.to_csv(comparison_path, index=False)
    print(f"Saved comparison to {comparison_path}")

    # Print correlation
    valid_data = merged.dropna()
    if len(valid_data) > 0:
        correlation = valid_data["lm_rating"].corr(valid_data["human_mean"])
        print(f"\nCorrelation between LM and human ratings: r = {correlation:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
