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

The scenario CSVs (`scenarios.csv`, `scenarios_nonfood.csv`, `scenarios_effort.csv`) are generated artifacts. Their sources of truth are the corresponding `.py` files (`scenarios.py`, `scenarios_nonfood.py`, `scenarios_effort.py`), which hold the scenario data as Python dicts and write the CSVs when run. Edit the `.py` file and regenerate with `uv run python experiments/<file>.py` — never edit the CSVs directly, since the next regeneration will overwrite the edits.

Each experiment directory also has a per-experiment `json/stimuli.json` that is generated from one of the scenario CSVs. The routing from CSV to experiment dirs is in `experiments/csv_to_json.py` — run that after editing any scenario `.py` file to propagate changes into the experiments that consume it.
