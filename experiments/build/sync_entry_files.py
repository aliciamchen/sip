#!/usr/bin/env python3
"""Write the index.html and experiment.js into each active experiment directory
from a single source, so the entry files never drift apart.

index.html is byte-identical across the active experiments, and experiment.js
differs only in which consent template the study loads (the only other
per-experiment code is trials.js, which experiment.js imports). Edit the
templates below and re-run this to propagate a change (e.g. a jsPsych version
bump, or a new shared plugin) to every experiment at once:

    uv run python experiments/build/sync_entry_files.py
"""

from pathlib import Path

# Active experiment slugs -> consent template under _lib/consent/ (mirror the
# Makefile's experiment roster). The food studies share the food-inverse
# consent; the nonfood studies (Study 3) use the domain-general one.
ACTIVE_SLUGS = {
    "food_inv_desire": "food-inverse",
    "food_inv_joint_de": "food-inverse",
    "food_inv_intimacy": "food-inverse",
    "food_inv_joint_ie": "food-inverse",
    "nonfood_inv_joint_de": "nonfood-inverse",
    "nonfood_inv_joint_ie": "nonfood-inverse",
}

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
import {
  CONFIG,
  makeStimulusTrials,
  INSTRUCTIONS_PAGES,
  COMPREHENSION_QUESTIONS,
} from "./trials.js";

runExperiment({
  config: CONFIG,
  makeStimulusTrials,
  instructionsPages: INSTRUCTIONS_PAGES,
  comprehensionQuestions: COMPREHENSION_QUESTIONS,
  consentTemplate: "{consent_template}",
});
"""


def main():
    # This script lives in experiments/build/; the experiment dirs are one level
    # up under experiments/.
    base = Path(__file__).resolve().parent.parent
    for slug, consent_template in ACTIVE_SLUGS.items():
        exp_dir = base / slug
        if not exp_dir.is_dir():
            print(f"Skipped (no experiment dir): {slug}")
            continue
        (exp_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
        (exp_dir / "experiment.js").write_text(
            EXPERIMENT_JS.replace("{consent_template}", consent_template),
            encoding="utf-8",
        )
        print(f"Wrote: {slug}/index.html, {slug}/experiment.js")


if __name__ == "__main__":
    main()
