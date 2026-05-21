# food_inv_intimacy_3act

Study 2 — Inverse intimacy.

## Design

**Factor crossing**: 2 (desire: low/high) × 2 (effort: low/high) × 3 (observed action: action_0/1/2)

**Known to participant**: reward_condition, effort_condition

**Inferred by participant**: intimacy (one slider, 0 = maximally formal, 100 = maximally intimate)

## Trial structure

Each trial shows: vignette + reward paragraph + effort paragraph + observed action (at posterior). Goes straight to the prior slider; no intimacy-descriptor preamble since intimacy is the inferred variable.

Slider responses: one intimacy slider (0–100), prior and posterior.

Each participant sees 16 trials (one per scenario) with cells balanced across participants. Cell space: 12 cells. The counterbalancing script (`python/generate_counterbalancing.py`) produces 192 sequences (12 rounds × 16 rotations), each a 16-trial assignment of factor cells to the 16 scenarios.

The trial flow follows the "noalt" pattern from `food_inv_intimacy_desire_noalt` / `food_inv_desire_intimacy_noalt`: no candidate action list shown to the participant, only the single observed action at the posterior stage.

## Stimulus source

Loads stimuli from `experiments/scenarios_3act.csv` via the routing in `experiments/csv_to_json.py`. Regenerate `json/stimuli.json` with:

```bash
uv run python experiments/csv_to_json.py
```

## Files in this directory

- `index.html` — entry point
- `experiment.js` — jsPsych 8.x boilerplate; spreads the sequence item's factor fields onto each stimulus
- `trials.js` — trial logic (instructions, attention check, scenario presentation, sliders, memory checks, exit survey, save)
- `python/generate_counterbalancing.py` — produces `json/full_counterbalancing.json`
- `json/stimuli.json` — generated from `scenarios_3act.csv`; do not edit by hand
- `json/full_counterbalancing.json` — generated; one sequence per condition_assignment

## Before running pilots

- Replace `PIPE_EXPERIMENT_ID` in `trials.js` `CONFIG` with the real DataPipe experiment ID.
- Replace `PROLIFIC_COMPLETION_URL` with the real Prolific completion URL.
- Open `index.html` locally and walk through a few trials per cell to confirm the UI renders and the slider endpoints / observed actions look right.

## Deploy

```bash
bin/deploy-experiment food_inv_intimacy_3act
```

The deploy script pushes this directory and `experiments/_lib/` to athena; see [experiments/README.md](../README.md#deploying-experiments) for details. After deploy the experiment is reachable at:

https://web.mit.edu/aliciach/www/sip/experiments/food_inv_intimacy_3act/
