#!/usr/bin/env python3
"""Write the identical index.html and experiment.js into each active experiment
directory from a single source, so the entry files never drift apart.

index.html and experiment.js are byte-identical across the active experiments
(the only per-experiment code is trials.js, which experiment.js imports). Edit
the templates below and re-run this to propagate a change (e.g. a jsPsych
version bump, or a new shared plugin) to every experiment at once:

    uv run python experiments/sync_entry_files.py
"""

from pathlib import Path

# Active experiment slugs (mirror Makefile's EXPERIMENTS_INVERSE).
ACTIVE_SLUGS = [
    "food_inv_desire",
    "food_inv_joint_de",
    "food_inv_intimacy",
    "food_inv_joint_ie",
]

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Social interactions survey</title>
    <script src="https://unpkg.com/jspsych@8.2.2"></script>
    <script src="https://unpkg.com/@jspsych/plugin-instructions@2.1.0"></script>
    <script src="https://unpkg.com/@jspsych/plugin-html-keyboard-response@2.1.0"></script>
    <script src="https://unpkg.com/@jspsych/plugin-survey-multi-choice@2.1.0"></script>
    <script src="https://unpkg.com/@jspsych/plugin-survey-html-form@2.0.0"></script>
    <script src="https://unpkg.com/@jspsych-contrib/plugin-pipe@0.5.0"></script>
    <script src="https://unpkg.com/@jspsych/plugin-html-slider-response@2.1.0"></script>
    <link
      href="https://unpkg.com/jspsych@8.2.2/css/jspsych.css"
      rel="stylesheet"
      type="text/css"
    />
    <link rel="stylesheet" href="../_lib/style.css" />
  </head>
  <body>
    <div id="jspsych-content"></div>
    <script type="module" src="experiment.js"></script>
  </body>
</html>
"""

EXPERIMENT_JS = """import { runExperiment } from "../_lib/bootstrap.js";
import { CONFIG, makeStimulusTrials, INSTRUCTIONS_PAGES } from "./trials.js";

runExperiment({
  config: CONFIG,
  makeStimulusTrials,
  instructionsPages: INSTRUCTIONS_PAGES,
  consentTemplate: "food-inverse",
});
"""


def main():
    base = Path(__file__).resolve().parent
    for slug in ACTIVE_SLUGS:
        exp_dir = base / slug
        if not exp_dir.is_dir():
            print(f"Skipped (no experiment dir): {slug}")
            continue
        (exp_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
        (exp_dir / "experiment.js").write_text(EXPERIMENT_JS, encoding="utf-8")
        print(f"Wrote: {slug}/index.html, {slug}/experiment.js")


if __name__ == "__main__":
    main()
