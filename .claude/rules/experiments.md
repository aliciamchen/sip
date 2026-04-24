---
paths:
  - "experiments/**/*"
---

# Experiment structure

Each experiment folder contains:
- `index.html` - Entry point
- `experiment.js` - jsPsych 8.x experiment logic
- `trials.js` - Trial configuration

Experiments collect data via jsPsych-contrib/pipe plugin to `data/<experiment_name>/raw_data/`.

Each different experiment needs a new datapipe ID but they all have the same prolific completion link.

## Scenario CSVs are generated from Python

The scenario CSVs (`scenarios.csv`, `scenarios_nonfood.csv`) are generated artifacts. Their sources of truth are the corresponding `.py` files (`scenarios.py`, `scenarios_nonfood.py`), which hold the scenario data as Python dicts and write the CSVs when run. Edit the `.py` file and regenerate with `uv run python experiments/<file>.py` — never edit the CSVs directly, since the next regeneration will overwrite the edits.
