#!/usr/bin/env python3
"""
Generate LM predictions for the risk (saliva transfer) task using Llama-3.3-70B-Instruct.
"""

import json
import os
import sys
from pathlib import Path
import time
import pandas as pd
from huggingface_hub import InferenceClient

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root

# Configuration
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# Instructions given to human participants + output formatting
SYSTEM_PROMPT = """In this survey, you will read vignettes about two people sharing different kinds of food in different situations. For each scenario, you will read about four different actions the two people can take.

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

Respond ONLY with a JSON object mapping each action to a numeric rating (decimals allowed), like:
{"action_0": 0.5, "action_1": 1.2, "action_2": 3.8, "action_3": 5.5}"""


def load_stimuli():
    """Load the risk experiment stimuli."""
    stimuli_path = get_project_root() / "experiments" / "risk" / "json" / "stimuli.json"
    with open(stimuli_path, "r") as f:
        return json.load(f)


def load_human_data():
    """Load human risk ratings for comparison."""
    data_path = get_project_root() / "data" / "risk" / "risk_summary.csv"
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
    """Get LM predictions for all stimuli."""
    predictions = []

    for i, stimulus in enumerate(stimuli):
        print(f"Processing {i+1}/{len(stimuli)}: {stimulus['scenario_label']}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_prompt(stimulus)}
        ]

        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=100,
                temperature=0.05,
            )
            response_text = response.choices[0].message.content
            ratings = parse_response(response_text)

            if ratings:
                for action_idx in range(4):
                    predictions.append({
                        "scenario_label": stimulus["scenario_label"],
                        "action": action_idx,
                        "lm_rating": ratings[f"action_{action_idx}"]
                    })
                print(f"  Ratings: {ratings}")
            else:
                print(f"  Failed to parse response: {response_text}")
                for action_idx in range(4):
                    predictions.append({
                        "scenario_label": stimulus["scenario_label"],
                        "action": action_idx,
                        "lm_rating": float("nan")
                    })

        except Exception as e:
            print(f"  Error: {e}")
            for action_idx in range(4):
                predictions.append({
                    "scenario_label": stimulus["scenario_label"],
                    "action": action_idx,
                    "lm_rating": float("nan")
                })

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    return pd.DataFrame(predictions)


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

    print("\nGetting LM predictions...")
    predictions_df = get_predictions(stimuli, client)

    # Save predictions
    output_path = get_project_root() / "data" / "risk" / "lm_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")

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
    comparison_path = get_project_root() / "data" / "risk" / "lm_human_comparison.csv"
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
