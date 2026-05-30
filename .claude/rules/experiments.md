---
paths:
  - "experiments/**/*"
---

# Experiment structure

Each experiment folder contains:
- `index.html` - Entry point
- `experiment.js` - thin call to `runExperiment()` from `_lib/bootstrap.js`
- `trials.js` - per-experiment `CONFIG`, instruction text, and prior/posterior trial rendering

All shared jsPsych boilerplate (consent + instructions screens, attention check, memory checks, exit survey, save, thank-you, the stylesheet, and the consent + exit-survey HTML templates) lives in [`experiments/_lib/`](../../experiments/_lib/). Each experiment references it via `../_lib/`, so the active experiments are not standalone folders — `_lib/` must be deployed alongside them.

Experiments collect data via jsPsych-contrib/pipe plugin to `data/<experiment_name>/raw_data/`.

Each different experiment needs a new datapipe ID but they all have the same prolific completion link.

## Deploy

Deploys go through [`bin/deploy-experiment`](../../bin/deploy-experiment), which rsyncs an experiment + `_lib/` to athena (`~/www/sip/experiments/`). Usage: `bin/deploy-experiment <slug>`. The script only accepts the four active slugs (Makefile's `EXPERIMENTS_INVERSE`): `food_inv_desire`, `food_inv_joint_de`, `food_inv_intimacy`, `food_inv_joint_ie`. See [experiments/README.md](../../experiments/README.md#deploying-experiments) for the full reference.

## Scenario CSVs are generated from Python

The scenario CSVs (`scenarios.csv`, `scenarios_effort.csv`, `scenarios_3act.csv`, `scenarios_nonfood.csv`) are generated artifacts. Their sources of truth are the corresponding `.py` files (`scenarios.py`, `scenarios_effort.py`, `scenarios_3act.py`, `scenarios_nonfood.py`), which hold the scenario data as Python dicts and write the CSVs when run. Edit the `.py` file and regenerate with `uv run python experiments/<file>.py` — never edit the CSVs directly, since the next regeneration will overwrite the edits.

`scenarios_3act.csv` is the 3-action canonical set introduced in May 2026 for the inverse-planning experiments (Studies 1a, 1b, 2a, 2b). It merges the effort_low/effort_high paragraphs from `scenarios_effort.csv` into the canonical scenarios so all three latent variables — desire, effort, intimacy — can be manipulated alongside the observed action.

Each experiment directory also has a per-experiment `json/stimuli.json` that is generated from one of the scenario CSVs. The routing from CSV to experiment dirs is in `experiments/csv_to_json.py` — run that after editing any scenario `.py` file to propagate changes into the experiments that consume it.
