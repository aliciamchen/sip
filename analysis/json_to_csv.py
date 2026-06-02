#!/usr/bin/env python3
"""
Convert raw jsPsych JSON from an experiment into analysis CSVs.

For each experiment it writes two CSVs under data/<experiment>/:
1. main_trials.csv - one row per prior/posterior rating trial
2. exit_survey.csv - demographics + attention/memory summary, one row per participant

It then builds main_trials_long.csv (the model/analysis input), applying the
standard exclusions (failed attention check or 0 correct memory checks).

Usage:
    python json_to_csv.py <experiment_name>

Active inverse studies (3-action set):
    - food_inv_desire   (Study 1a): infer desire; single continuous 0-100 desire slider
    - food_inv_joint_de (Study 1b): joint desire + effort; two sliders per trial
    - food_inv_intimacy (Study 2a): infer intimacy; single 0-100 intimacy slider
    - food_inv_joint_ie (Study 2b): joint intimacy + effort; two sliders per trial
"""

import argparse
import csv
import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root

# Experiment configurations
EXPERIMENT_CONFIGS = {
    "food_inv_desire": {
        "description": "Study 1a — desire inference under known effort + intimacy (3-action set)",
        "main_trial_fields": [
            "subject_id",
            "scenario_label",
            "action_condition",
            "effort_condition",
            "intimacy_condition",
            "stage",
            "response",
        ],
        "exit_survey_fields": [
            "subject_id",
            "gender",
            "age",
            "understood",
            "comments",
            "attention_passed",
            "memory_correct_count",
            "comprehension_attempt",
        ],
        "has_closeness": False,
        "has_attention_memory": True,
        "has_comprehension": True,
    },
    "food_inv_joint_de": {
        "description": "Study 1b — joint desire + effort inference under known intimacy (3-action set)",
        "main_trial_fields": [
            "subject_id",
            "scenario_label",
            "action_condition",
            "intimacy_condition",
            "stage",
            "desire_rating",
            "effort_rating",
        ],
        "exit_survey_fields": [
            "subject_id",
            "gender",
            "age",
            "understood",
            "comments",
            "attention_passed",
            "memory_correct_count",
            "comprehension_attempt",
        ],
        "has_closeness": False,
        "has_attention_memory": True,
        "has_comprehension": True,
    },
    "food_inv_intimacy": {
        "description": "Study 2a — intimacy inference under known desire + effort (3-action set)",
        "main_trial_fields": [
            "subject_id",
            "scenario_label",
            "action_condition",
            "desire_condition",
            "effort_condition",
            "stage",
            "intimacy_rating",
        ],
        "exit_survey_fields": [
            "subject_id",
            "gender",
            "age",
            "understood",
            "comments",
            "attention_passed",
            "memory_correct_count",
            "comprehension_attempt",
        ],
        "has_closeness": False,
        "has_attention_memory": True,
        "has_comprehension": True,
    },
    "food_inv_joint_ie": {
        "description": "Study 2b — joint intimacy + effort inference under known desire (3-action set)",
        "main_trial_fields": [
            "subject_id",
            "scenario_label",
            "action_condition",
            "desire_condition",
            "stage",
            "intimacy_rating",
            "effort_rating",
        ],
        "exit_survey_fields": [
            "subject_id",
            "gender",
            "age",
            "understood",
            "comments",
            "attention_passed",
            "memory_correct_count",
            "comprehension_attempt",
        ],
        "has_closeness": False,
        "has_attention_memory": True,
        "has_comprehension": True,
    },
}


def process_json_files(input_dir, output_dir, config, experiment_name):
    """
    Process all JSON files in the input directory and create CSV files.

    Args:
        input_dir (str): Path to directory containing JSON files
        output_dir (str): Path to directory where CSV files will be saved
        config (dict): Experiment configuration
        experiment_name (str): Name of the experiment being processed
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Lists to store data for each CSV
    main_trials_data = []
    exit_survey_data = []

    # Dictionary to map original subject IDs to anonymous IDs
    subject_id_mapping = {}

    def generate_deterministic_id(original_id):
        """Generate a deterministic UUID based on the original subject ID."""
        # Use a fixed namespace UUID for this project
        namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        # Create a deterministic UUID based on the original subject ID
        return str(uuid.uuid5(namespace, original_id))

    # Process each JSON file
    json_files = list(input_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process")

    for json_file in json_files:
        print(f"Processing {json_file.name}...")

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract subject_id from the first trial
            original_subject_id = data[0].get("subject_id", "unknown")

            # Create anonymous subject ID if not already mapped
            if original_subject_id not in subject_id_mapping:
                # Generate a deterministic UUID based on the original subject ID
                subject_id_mapping[original_subject_id] = generate_deterministic_id(
                    original_subject_id
                )

            subject_id = subject_id_mapping[original_subject_id]

            # Process each trial in the data
            for trial in data:
                trial_type = trial.get("response_type", "")

                if trial_type == "response":
                    # Extract main trial data
                    scenario_label = trial.get("scenario_label", "")

                    # Handle different experiment types
                    if experiment_name == "food_inv_desire":
                        # Prior/posterior desire slider response under known
                        # effort + intimacy. response is the continuous 0-100
                        # rating ("how much do they want to eat the food?",
                        # not at all → moderately → extremely).
                        trial_data = {
                            "subject_id": subject_id,
                            "scenario_label": scenario_label,
                            "action_condition": trial.get("action_condition", ""),
                            "effort_condition": trial.get("effort_condition", ""),
                            "intimacy_condition": trial.get("intimacy_condition", ""),
                            "stage": trial.get("stage", ""),
                            "response": trial.get("response", ""),
                        }

                    elif experiment_name == "food_inv_joint_de":
                        # Two sliders on one page (survey-html-form); the response
                        # is an object {desire, effort} of 0-100 string values.
                        # desire = continuous "how much do they want the food";
                        # effort = "which effort situation is more likely"
                        # (0 = effort_low ... 100 = effort_high).
                        response = trial.get("response", {}) or {}
                        trial_data = {
                            "subject_id": subject_id,
                            "scenario_label": scenario_label,
                            "action_condition": trial.get("action_condition", ""),
                            "intimacy_condition": trial.get("intimacy_condition", ""),
                            "stage": trial.get("stage", ""),
                            "desire_rating": response.get("desire", ""),
                            "effort_rating": response.get("effort", ""),
                        }

                    elif experiment_name == "food_inv_intimacy":
                        # Single intimacy slider (0-100) under known desire +
                        # effort.
                        trial_data = {
                            "subject_id": subject_id,
                            "scenario_label": scenario_label,
                            "action_condition": trial.get("action_condition", ""),
                            "desire_condition": trial.get("desire_condition", ""),
                            "effort_condition": trial.get("effort_condition", ""),
                            "stage": trial.get("stage", ""),
                            "intimacy_rating": trial.get("response", ""),
                        }

                    elif experiment_name == "food_inv_joint_ie":
                        # Two sliders on one page (survey-html-form); the response
                        # is an object {intimacy, effort} of 0-100 string values.
                        # effort = "which effort situation is more likely"
                        # (0 = effort_low ... 100 = effort_high).
                        response = trial.get("response", {}) or {}
                        trial_data = {
                            "subject_id": subject_id,
                            "scenario_label": scenario_label,
                            "action_condition": trial.get("action_condition", ""),
                            "desire_condition": trial.get("desire_condition", ""),
                            "stage": trial.get("stage", ""),
                            "intimacy_rating": response.get("intimacy", ""),
                            "effort_rating": response.get("effort", ""),
                        }

                    main_trials_data.append(trial_data)

                elif trial_type == "exit_survey":
                    # Extract exit survey data
                    response = trial.get("response", {})

                    gender = response.get("gender", "")
                    age = response.get("age", "")
                    understood = response.get("understood", "")
                    comments = response.get("comments", "")

                    # Build survey data dictionary
                    survey_data = {
                        "subject_id": subject_id,
                        "gender": gender,
                        "age": age,
                        "understood": understood,
                        "comments": comments,
                    }

                    # Add attention and memory data if this experiment has it
                    if config["has_attention_memory"]:
                        survey_data["attention_passed"] = trial.get(
                            "attention_passed", ""
                        )
                        survey_data["memory_correct_count"] = trial.get(
                            "memory_correct_count", ""
                        )

                    # Which attempt (1-3) the participant passed the comprehension
                    # check on. Everyone in the data passed; failers are ended
                    # before the save, so they never produce a raw JSON.
                    if config.get("has_comprehension"):
                        survey_data["comprehension_attempt"] = trial.get(
                            "comprehension_attempt", ""
                        )

                    exit_survey_data.append(survey_data)

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue

    # Write main trials CSV
    main_trials_file = output_path / "main_trials.csv"
    with open(main_trials_file, "w", newline="", encoding="utf-8") as f:
        if main_trials_data:
            writer = csv.DictWriter(f, fieldnames=config["main_trial_fields"])
            writer.writeheader()
            writer.writerows(main_trials_data)
            print(f"Created {main_trials_file} with {len(main_trials_data)} rows")
        else:
            print("No main trial data found")

    # Write exit survey CSV
    exit_survey_file = output_path / "exit_survey.csv"
    with open(exit_survey_file, "w", newline="", encoding="utf-8") as f:
        if exit_survey_data:
            writer = csv.DictWriter(f, fieldnames=config["exit_survey_fields"])
            writer.writeheader()
            writer.writerows(exit_survey_data)
            print(f"Created {exit_survey_file} with {len(exit_survey_data)} rows")
        else:
            print("No exit survey data found")


def create_food_inv_desire_long(output_dir):
    """
    Create main_trials_long.csv for the food_inv_desire experiment.

    Filters out participants who failed attention or got 0 correct on memory.
    Renames effort_condition -> effort and intimacy_condition -> intimacy for
    analysis consistency.
    """
    output_path = Path(output_dir)

    main_trials = pd.read_csv(output_path / "main_trials.csv")
    exit_survey = pd.read_csv(output_path / "exit_survey.csv")

    excluded_subjects = exit_survey[
        (exit_survey["attention_passed"] != True)
        | (exit_survey["memory_correct_count"] == 0)
    ]["subject_id"].tolist()

    n_excluded = len(excluded_subjects)
    n_total = exit_survey["subject_id"].nunique()
    print(
        f"Excluding {n_excluded} of {n_total} participants (failed attention or 0 memory correct)"
    )

    main_trials_filtered = main_trials[
        ~main_trials["subject_id"].isin(excluded_subjects)
    ]

    main_trials_long = main_trials_filtered.rename(
        columns={"effort_condition": "effort", "intimacy_condition": "intimacy"}
    )

    main_trials_long = main_trials_long.sort_values(
        ["subject_id", "scenario_label", "action_condition", "stage"]
    ).reset_index(drop=True)

    output_file = output_path / "main_trials_long.csv"
    main_trials_long.to_csv(output_file, index=False)
    print(
        f"Created {output_file} with {len(main_trials_long)} rows ({main_trials_long['subject_id'].nunique()} participants)"
    )


def create_food_inv_joint_de_long(output_dir):
    """
    Create main_trials_long.csv for the food_inv_joint_de experiment (Study 1b).

    Intimacy is the given condition; desire and effort are jointly inferred via
    two sliders. Renames intimacy_condition -> intimacy and keeps the
    desire_rating / effort_rating slider columns. Filters out participants who
    failed attention or got 0 correct on memory.
    """
    output_path = Path(output_dir)

    main_trials = pd.read_csv(output_path / "main_trials.csv")
    exit_survey = pd.read_csv(output_path / "exit_survey.csv")

    excluded_subjects = exit_survey[
        (exit_survey["attention_passed"] != True)
        | (exit_survey["memory_correct_count"] == 0)
    ]["subject_id"].tolist()

    n_excluded = len(excluded_subjects)
    n_total = exit_survey["subject_id"].nunique()
    print(
        f"Excluding {n_excluded} of {n_total} participants (failed attention or 0 memory correct)"
    )

    main_trials_filtered = main_trials[
        ~main_trials["subject_id"].isin(excluded_subjects)
    ]

    main_trials_long = main_trials_filtered.rename(
        columns={"intimacy_condition": "intimacy"}
    )

    main_trials_long = main_trials_long.sort_values(
        ["subject_id", "scenario_label", "action_condition", "stage"]
    ).reset_index(drop=True)

    output_file = output_path / "main_trials_long.csv"
    main_trials_long.to_csv(output_file, index=False)
    print(
        f"Created {output_file} with {len(main_trials_long)} rows ({main_trials_long['subject_id'].nunique()} participants)"
    )


def create_food_inv_intimacy_long(output_dir):
    """
    Create main_trials_long.csv for the food_inv_intimacy experiment (Study 2a).

    Desire and effort are the given conditions; intimacy is inferred via a single
    slider. Renames desire_condition -> desire and effort_condition -> effort
    and keeps the intimacy_rating column. Filters out participants who failed
    attention or got 0 correct on memory.
    """
    output_path = Path(output_dir)

    main_trials = pd.read_csv(output_path / "main_trials.csv")
    exit_survey = pd.read_csv(output_path / "exit_survey.csv")

    excluded_subjects = exit_survey[
        (exit_survey["attention_passed"] != True)
        | (exit_survey["memory_correct_count"] == 0)
    ]["subject_id"].tolist()

    n_excluded = len(excluded_subjects)
    n_total = exit_survey["subject_id"].nunique()
    print(
        f"Excluding {n_excluded} of {n_total} participants (failed attention or 0 memory correct)"
    )

    main_trials_filtered = main_trials[
        ~main_trials["subject_id"].isin(excluded_subjects)
    ]

    main_trials_long = main_trials_filtered.rename(
        columns={"desire_condition": "desire", "effort_condition": "effort"}
    )

    main_trials_long = main_trials_long.sort_values(
        ["subject_id", "scenario_label", "action_condition", "stage"]
    ).reset_index(drop=True)

    output_file = output_path / "main_trials_long.csv"
    main_trials_long.to_csv(output_file, index=False)
    print(
        f"Created {output_file} with {len(main_trials_long)} rows ({main_trials_long['subject_id'].nunique()} participants)"
    )


def create_food_inv_joint_ie_long(output_dir):
    """
    Create main_trials_long.csv for the food_inv_joint_ie experiment (Study 2b).

    Desire is the given condition; intimacy and effort are jointly inferred via
    two sliders. Renames desire_condition -> desire and keeps the
    intimacy_rating / effort_rating slider columns. Filters out participants who
    failed attention or got 0 correct on memory.
    """
    output_path = Path(output_dir)

    main_trials = pd.read_csv(output_path / "main_trials.csv")
    exit_survey = pd.read_csv(output_path / "exit_survey.csv")

    excluded_subjects = exit_survey[
        (exit_survey["attention_passed"] != True)
        | (exit_survey["memory_correct_count"] == 0)
    ]["subject_id"].tolist()

    n_excluded = len(excluded_subjects)
    n_total = exit_survey["subject_id"].nunique()
    print(
        f"Excluding {n_excluded} of {n_total} participants (failed attention or 0 memory correct)"
    )

    main_trials_filtered = main_trials[
        ~main_trials["subject_id"].isin(excluded_subjects)
    ]

    main_trials_long = main_trials_filtered.rename(
        columns={"desire_condition": "desire"}
    )

    main_trials_long = main_trials_long.sort_values(
        ["subject_id", "scenario_label", "action_condition", "stage"]
    ).reset_index(drop=True)

    output_file = output_path / "main_trials_long.csv"
    main_trials_long.to_csv(output_file, index=False)
    print(
        f"Created {output_file} with {len(main_trials_long)} rows ({main_trials_long['subject_id'].nunique()} participants)"
    )


def main():
    """Main function to run the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert JSON files from experiments to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Active inverse studies (3-action set):
  food_inv_desire     Study 1a -- infer desire (single 0-100 desire slider)
  food_inv_joint_de   Study 1b -- joint desire + effort (two sliders)
  food_inv_intimacy   Study 2a -- infer intimacy (single 0-100 intimacy slider)
  food_inv_joint_ie   Study 2b -- joint intimacy + effort (two sliders)

Examples:
  python json_to_csv.py food_inv_desire
  python json_to_csv.py food_inv_joint_de
  python json_to_csv.py food_inv_intimacy
  python json_to_csv.py food_inv_joint_ie
        """,
    )

    parser.add_argument(
        "experiment",
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="Name of the experiment to process",
    )

    args = parser.parse_args()

    # Get experiment configuration
    config = EXPERIMENT_CONFIGS[args.experiment]

    # Get project root directory
    project_root = get_project_root()

    # Define paths relative to project root
    input_dir = project_root / f"data/{args.experiment}/raw_data"
    output_dir = project_root / f"data/{args.experiment}"

    print(f"Converting JSON files to CSV for {args.experiment} experiment...")
    print(f"Description: {config['description']}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Process the files
    process_json_files(input_dir, output_dir, config, args.experiment)

    # Create long format with exclusions
    if args.experiment == "food_inv_desire":
        print("\nCreating long format with exclusions...")
        create_food_inv_desire_long(output_dir)
    elif args.experiment == "food_inv_joint_de":
        print("\nCreating long format with exclusions...")
        create_food_inv_joint_de_long(output_dir)
    elif args.experiment == "food_inv_intimacy":
        print("\nCreating long format with exclusions...")
        create_food_inv_intimacy_long(output_dir)
    elif args.experiment == "food_inv_joint_ie":
        print("\nCreating long format with exclusions...")
        create_food_inv_joint_ie_long(output_dir)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
