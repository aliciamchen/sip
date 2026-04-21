#!/usr/bin/env python3
"""Convert scenarios.csv to stimuli.json for each experiment."""

import csv
import json
from pathlib import Path

EXPERIMENTS = ["forw_plan", "inv_plan_intimacy_alt", "inv_plan_desire_alt", "inv_plan_intimacy_noalt"]


def load_scenarios(csv_path):
    """Load scenarios from CSV file."""
    scenarios = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append(row)
    return scenarios


def clean_text(text):
    """Fix known typos and strip whitespace from scenario text."""
    text = text.strip()
    text = text.replace("intruiging", "intriguing")
    text = text.replace("Intruigued", "Intrigued")
    text = text.replace("that that the bar", "that the bar")
    return text


def write_json(scenarios, output_path):
    """Write scenarios to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    script_dir = Path(__file__).parent
    csv_path = script_dir / "scenarios.csv"

    # Load and clean scenarios
    scenarios = load_scenarios(csv_path)
    for scenario in scenarios:
        for key, value in scenario.items():
            scenario[key] = clean_text(value)

    # Write to each experiment
    for exp in EXPERIMENTS:
        output_path = script_dir / exp / "json" / "stimuli.json"
        write_json(scenarios, output_path)
        print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
