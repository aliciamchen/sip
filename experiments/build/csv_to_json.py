#!/usr/bin/env python3
"""Convert scenario CSVs to stimuli.json for each experiment.

Each stimulus CSV maps to a set of experiment directories that consume it; the
full routing is in the SOURCES list below. The active inverse experiments
(Studies 1a, 1b, 2a, 2b) all read scenarios.csv.
"""

import csv
import json
from pathlib import Path

# Each scenario CSV maps to the experiment directories under experiments/ that
# consume it. Active inverse experiments (Studies 1a, 1b, 2a, 2b) read
# scenarios.csv and live at experiments/<slug>/. Legacy experiments live at
# experiments/legacy/<slug>/ and are referenced with a "legacy/" prefix; they're
# kept regenerate-able but are not part of `make all`. A slug is silently skipped
# if its experiment dir has no json/ subdir.
SOURCES = [
    # Active: the 3-action stimulus set (experiments/scenarios.csv).
    (
        "scenarios.csv",
        [
            "food_inv_intimacy",
            "food_inv_desire",
            "food_inv_joint_de",
            "food_inv_joint_ie",
        ],
    ),
    # Legacy stimulus sets live under experiments/legacy/ (4-action canonical,
    # 2-action effort, non-food); kept regenerate-able for the legacy experiments.
    (
        "legacy/scenarios.csv",
        [
            "legacy/food_forw_intimacy_desire",
            "legacy/food_inv_intimacy_desire_noalt",
            "legacy/food_inv_desire_intimacy_noalt",
        ],
    ),
    (
        "legacy/scenarios_effort.csv",
        [
            "legacy/food_forw_intimacy_effort",
        ],
    ),
    (
        "legacy/scenarios_nonfood.csv",
        [
            "legacy/nonfood_forw_intimacy_desire",
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
    """Strip surrounding whitespace from a scenario field.

    No text substitutions: the source-of-truth `.py` files are the place to fix
    typos, so the CSV and the emitted JSON stay identical. (The active scenario
    set is typo-clean; a legacy CSV may still carry its original spellings.)
    """
    return text.strip()


def write_json(scenarios, output_path):
    """Write scenarios to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    # This script lives in experiments/build/; the scenario CSVs and experiment
    # dirs are one level up under experiments/.
    script_dir = Path(__file__).resolve().parent.parent

    for csv_name, experiments in SOURCES:
        csv_path = script_dir / csv_name
        scenarios = load_scenarios(csv_path)
        for scenario in scenarios:
            for key, value in scenario.items():
                scenario[key] = clean_text(value)

        for exp in experiments:
            exp_dir = script_dir / exp
            if not exp_dir.is_dir():
                print(f"Skipped (no experiment dir): {exp}")
                continue
            output_path = exp_dir / "json" / "stimuli.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(scenarios, output_path)
            print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
