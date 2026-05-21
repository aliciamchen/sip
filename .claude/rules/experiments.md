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

The scenario CSVs (`scenarios.csv`, `scenarios_effort.csv`, `scenarios_3act.csv`, `scenarios_nonfood.csv`) are generated artifacts. Their sources of truth are the corresponding `.py` files (`scenarios.py`, `scenarios_effort.py`, `scenarios_3act.py`, `scenarios_nonfood.py`), which hold the scenario data as Python dicts and write the CSVs when run. Edit the `.py` file and regenerate with `uv run python experiments/<file>.py` — never edit the CSVs directly, since the next regeneration will overwrite the edits.

`scenarios_3act.csv` is the 3-action canonical set introduced in May 2026 for the new inverse-planning experiments (Studies 2, 3a, 3b, 4a, 4b). It merges the effort_low/effort_high paragraphs from `scenarios_effort.csv` into the canonical scenarios so all three latent variables — desire, effort, intimacy — can be manipulated alongside the observed action.

Each experiment directory also has a per-experiment `json/stimuli.json` that is generated from one of the scenario CSVs. The routing from CSV to experiment dirs is in `experiments/csv_to_json.py` — run that after editing any scenario `.py` file to propagate changes into the experiments that consume it.
