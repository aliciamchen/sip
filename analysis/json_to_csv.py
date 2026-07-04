#!/usr/bin/env python3
"""
Convert raw jsPsych JSON from an experiment into analysis CSVs.

For each experiment it writes two CSVs under data/<experiment>/:
1. main_trials.csv - one row per prior/posterior rating trial
2. exit_survey.csv - demographics + attention/memory summary, one row per participant

It then builds main_trials_long.csv (the model/analysis input), applying the
study's exclusion rule (the config's `exclusion_rule`): Study 1a's preregistered
lax rule excludes only participants who failed the attention check AND answered
0 memory questions correctly; the later studies use the stricter rule (retain
only participants who passed the attention check AND answered at least one
memory question correctly).

If any raw JSON file fails to parse, the script reports every failing file and
exits without writing CSVs, so a corrupt download can't silently drop a
participant. Remove or fix the offending files and re-run.

Usage:
    python json_to_csv.py <experiment_name>

Active inverse studies (3-action set):
    - food_inv_desire      (Study 1a): infer desire; single continuous 0-100 desire slider
    - food_inv_joint_de    (Study 1b): joint desire + effort; two sliders per trial
    - food_inv_intimacy    (Study 2a): infer intimacy; single 0-100 intimacy slider
    - food_inv_joint_ie    (Study 2b): joint intimacy + effort; two sliders per trial
    - nonfood_inv_joint_de (Study 3a): 1b's design on the nonfood scenario set
    - nonfood_inv_joint_ie (Study 3b): 2b's design on the nonfood scenario set
"""

import argparse
import csv
import json
import sys
import uuid
from pathlib import Path

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root


def _to_unit(raw):
    """Normalize a raw 0-100 jsPsych slider value to the 0-1 scale.

    All DV ratings are collected on a 0-100 slider but stored, analyzed, and
    modeled on the 0-1 scale (so belief updates fall in [-1, 1]). Returns "" for
    blank/non-numeric values so empty cells stay empty in the CSV.
    """
    if raw is None or raw == "":
        return ""
    try:
        return float(raw) / 100.0
    except (TypeError, ValueError):
        return ""


# Experiment configurations. Per-study keys that drive the conversion:
#   condition_fields — trial keys copied verbatim into main_trials.csv
#   rating_fields    — output column -> where the rating lives. None means the
#                      trial's "response" is itself the rating (single-slider
#                      studies); a string key indexes into the response object
#                      (two-slider survey-html-form studies). All ratings pass
#                      through _to_unit.
#   long_renames     — given-condition columns renamed to bare factor names in
#                      main_trials_long.csv
EXPERIMENT_CONFIGS = {
    "food_inv_desire": {
        "description": "Study 1a — desire inference under known effort + intimacy (3-action set)",
        # response is the continuous desire rating ("how much do they want to
        # eat the food?", not at all → moderately → extremely).
        "main_trial_fields": [
            "subject_id",
            "scenario_label",
            "action_condition",
            "effort_condition",
            "intimacy_condition",
            "stage",
            "response",
        ],
        "condition_fields": [
            "action_condition",
            "effort_condition",
            "intimacy_condition",
        ],
        "rating_fields": {"response": None},
        "long_renames": {
            "effort_condition": "effort",
            "intimacy_condition": "intimacy",
        },
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
        "has_attention_memory": True,
        "has_comprehension": True,
        # Preregistered lax rule for 1a: exclude only participants who fail the
        # attention check AND answer 0 memory questions correctly.
        "exclusion_rule": "lax",
    },
    "food_inv_joint_de": {
        "description": "Study 1b — joint desire + effort inference under known intimacy (3-action set)",
        # Two sliders on one page; the response is an object {desire, effort}.
        # desire = continuous "how much do they want the food"; effort = "which
        # effort situation is more likely" (0 = effort_low ... 1 = effort_high).
        "main_trial_fields": [
            "subject_id",
            "scenario_label",
            "action_condition",
            "intimacy_condition",
            "stage",
            "desire_rating",
            "effort_rating",
        ],
        "condition_fields": ["action_condition", "intimacy_condition"],
        "rating_fields": {"desire_rating": "desire", "effort_rating": "effort"},
        "long_renames": {"intimacy_condition": "intimacy"},
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
        "has_attention_memory": True,
        "has_comprehension": True,
        # Stricter rule for the post-1a studies: retain only participants who
        # pass the attention check AND answer >=1 memory question correctly.
        "exclusion_rule": "strict",
    },
    "food_inv_intimacy": {
        "description": "Study 2a — intimacy inference under known desire + effort (3-action set)",
        # Single intimacy slider under known desire + effort.
        "main_trial_fields": [
            "subject_id",
            "scenario_label",
            "action_condition",
            "desire_condition",
            "effort_condition",
            "stage",
            "intimacy_rating",
        ],
        "condition_fields": [
            "action_condition",
            "desire_condition",
            "effort_condition",
        ],
        "rating_fields": {"intimacy_rating": None},
        "long_renames": {
            "desire_condition": "desire",
            "effort_condition": "effort",
        },
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
        "has_attention_memory": True,
        "has_comprehension": True,
        # Stricter rule for the post-1a studies: retain only participants who
        # pass the attention check AND answer >=1 memory question correctly.
        "exclusion_rule": "strict",
    },
    "food_inv_joint_ie": {
        "description": "Study 2b — joint intimacy + effort inference under known desire (3-action set)",
        # Two sliders on one page; the response is an object {intimacy, effort}.
        # effort = "which effort situation is more likely" (0 = effort_low ...
        # 1 = effort_high).
        "main_trial_fields": [
            "subject_id",
            "scenario_label",
            "action_condition",
            "desire_condition",
            "stage",
            "intimacy_rating",
            "effort_rating",
        ],
        "condition_fields": ["action_condition", "desire_condition"],
        "rating_fields": {"intimacy_rating": "intimacy", "effort_rating": "effort"},
        "long_renames": {"desire_condition": "desire"},
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
        "has_attention_memory": True,
        "has_comprehension": True,
        # Stricter rule for the post-1a studies: retain only participants who
        # pass the attention check AND answer >=1 memory question correctly.
        "exclusion_rule": "strict",
    },
    # Study 3 (nonfood scenario set): 3a mirrors 1b's conversion config, 3b
    # mirrors 2b's — same trial schema, sliders, and strict exclusion rule.
    "nonfood_inv_joint_de": {
        "description": "Study 3a — joint desire + effort inference under known intimacy (nonfood set)",
        "main_trial_fields": [
            "subject_id",
            "scenario_label",
            "action_condition",
            "intimacy_condition",
            "stage",
            "desire_rating",
            "effort_rating",
        ],
        "condition_fields": ["action_condition", "intimacy_condition"],
        "rating_fields": {"desire_rating": "desire", "effort_rating": "effort"},
        "long_renames": {"intimacy_condition": "intimacy"},
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
        "has_attention_memory": True,
        "has_comprehension": True,
        "exclusion_rule": "strict",
    },
    "nonfood_inv_joint_ie": {
        "description": "Study 3b — joint intimacy + effort inference under known desire (nonfood set)",
        "main_trial_fields": [
            "subject_id",
            "scenario_label",
            "action_condition",
            "desire_condition",
            "stage",
            "intimacy_rating",
            "effort_rating",
        ],
        "condition_fields": ["action_condition", "desire_condition"],
        "rating_fields": {"intimacy_rating": "intimacy", "effort_rating": "effort"},
        "long_renames": {"desire_condition": "desire"},
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
        "has_attention_memory": True,
        "has_comprehension": True,
        "exclusion_rule": "strict",
    },
}


def _extract_main_trial(trial, subject_id, config):
    """Build one main_trials.csv row from a raw rating trial."""
    row = {
        "subject_id": subject_id,
        "scenario_label": trial.get("scenario_label", ""),
        "stage": trial.get("stage", ""),
    }
    for field in config["condition_fields"]:
        row[field] = trial.get(field, "")
    response = trial.get("response", "")
    for out_col, response_key in config["rating_fields"].items():
        if response_key is None:
            row[out_col] = _to_unit(response)
        else:
            row[out_col] = _to_unit((response or {}).get(response_key, ""))
    return row


def _write_csv(path, rows, fieldnames):
    """Write rows to path, refusing to clobber an existing file with nothing."""
    if not rows:
        print(f"No rows for {path.name}; existing file left untouched")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {path} with {len(rows)} rows")


def process_json_files(input_dir, output_dir, config):
    """
    Process all JSON files in the input directory and create CSV files.

    Args:
        input_dir (str): Path to directory containing JSON files
        output_dir (str): Path to directory where CSV files will be saved
        config (dict): Experiment configuration

    Exits non-zero without writing anything if no JSON files are found or any
    file fails to process.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

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
    if not json_files:
        sys.exit(f"No JSON files found in {input_path}; nothing written.")
    print(f"Found {len(json_files)} JSON files to process")

    failures = []
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
                    main_trials_data.append(
                        _extract_main_trial(trial, subject_id, config)
                    )

                elif trial_type == "exit_survey":
                    # Extract exit survey data
                    response = trial.get("response", {})

                    survey_data = {
                        "subject_id": subject_id,
                        "gender": response.get("gender", ""),
                        "age": response.get("age", ""),
                        "understood": response.get("understood", ""),
                        "comments": response.get("comments", ""),
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
            failures.append((json_file.name, e))
            print(f"Error processing {json_file.name}: {e}")
            continue

    if failures:
        print(f"\n{len(failures)} of {len(json_files)} files failed to process:")
        for name, err in failures:
            print(f"  {name}: {err}")
        sys.exit(
            "Aborting without writing CSVs so no participant is silently dropped. "
            "Fix or deliberately remove the files above and re-run."
        )

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    _write_csv(
        output_path / "main_trials.csv", main_trials_data, config["main_trial_fields"]
    )
    _write_csv(
        output_path / "exit_survey.csv", exit_survey_data, config["exit_survey_fields"]
    )


def create_main_trials_long(output_dir, config):
    """
    Create main_trials_long.csv from main_trials.csv and exit_survey.csv.

    Applies the study's exclusion rule (the config's `exclusion_rule`; see the
    module docstring) and renames the given-condition columns to bare factor
    names per the config's long_renames (e.g. effort_condition -> effort).
    """
    output_path = Path(output_dir)

    main_trials = pd.read_csv(output_path / "main_trials.csv")
    exit_survey = pd.read_csv(output_path / "exit_survey.csv")

    # Per-study exclusion rule (see the config's `exclusion_rule` and the
    # manuscript Methods). Study 1a preregistered the lax rule (exclude only
    # participants who fail the attention check AND answer every memory-check
    # question incorrectly); it excluded 0 participants, so the rule was made
    # more stringent for the later studies (retain only participants who pass
    # the attention check AND answer at least one memory-check question
    # correctly). `memory_correct_count` counts questions (three across the two
    # memory checks), not checks.
    rule = config["exclusion_rule"]
    if rule == "lax":
        excluded_mask = (exit_survey["attention_passed"] != True) & (
            exit_survey["memory_correct_count"] == 0
        )
        rule_desc = "failed attention and 0 memory questions correct"
    elif rule == "strict":
        excluded_mask = (exit_survey["attention_passed"] != True) | (
            exit_survey["memory_correct_count"] == 0
        )
        rule_desc = "failed attention or 0 memory questions correct"
    else:
        raise ValueError(f"Unknown exclusion_rule: {rule!r}")
    excluded_subjects = exit_survey[excluded_mask]["subject_id"].tolist()

    n_excluded = len(excluded_subjects)
    n_total = exit_survey["subject_id"].nunique()
    print(f"Excluding {n_excluded} of {n_total} participants ({rule_desc})")

    main_trials_filtered = main_trials[
        ~main_trials["subject_id"].isin(excluded_subjects)
    ]

    main_trials_long = main_trials_filtered.rename(columns=config["long_renames"])

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
  food_inv_desire        Study 1a -- infer desire (single 0-100 desire slider)
  food_inv_joint_de      Study 1b -- joint desire + effort (two sliders)
  food_inv_intimacy      Study 2a -- infer intimacy (single 0-100 intimacy slider)
  food_inv_joint_ie      Study 2b -- joint intimacy + effort (two sliders)
  nonfood_inv_joint_de   Study 3a -- 1b's design on the nonfood scenario set
  nonfood_inv_joint_ie   Study 3b -- 2b's design on the nonfood scenario set

Examples:
  python json_to_csv.py food_inv_desire
  python json_to_csv.py nonfood_inv_joint_de
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
    process_json_files(input_dir, output_dir, config)

    # Create long format with exclusions
    print("\nCreating long format with exclusions...")
    create_main_trials_long(output_dir, config)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
