#!/usr/bin/env python3
"""Convert scenario CSVs to stimuli.json for each experiment.

Each stimulus CSV maps to a set of experiment directories that consume it:
- scenarios.csv → forw_plan, inv_plan_intimacy_alt, inv_plan_desire_alt, inv_plan_intimacy_noalt
- scenarios_effort.csv → forw_plan_effort, inv_plan_effort
"""

import csv
import json
from pathlib import Path

SOURCES = [
    (
        "scenarios.csv",
        [
            "forw_plan",
            "inv_plan_intimacy_alt",
            "inv_plan_desire_alt",
            "inv_plan_intimacy_noalt",
        ],
    ),
    (
        "scenarios_effort.csv",
        [
            "forw_plan_effort",
            "inv_plan_effort",
        ],
    ),
]


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

    for csv_name, experiments in SOURCES:
        csv_path = script_dir / csv_name
        scenarios = load_scenarios(csv_path)
        for scenario in scenarios:
            for key, value in scenario.items():
                scenario[key] = clean_text(value)

        for exp in experiments:
            output_path = script_dir / exp / "json" / "stimuli.json"
            write_json(scenarios, output_path)
            print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
