# food_inv_desire

Study 1a — Desire inference.

## Design

**Factor crossing**: 2 (effort: low/high) × 4 (intimacy: 0/50/75/100) × 3 (observed action)

**Known to participant**: effort_condition, intimacy_condition

**Inferred by participant**: reward_condition (one slider). The two reward_low/reward_high paragraphs are the slider endpoints.

## Trial structure

Each trial shows: intimacy-descriptor preamble page, then vignette + effort paragraph + observed action (at posterior). Reward paragraph is hidden — its low/high values anchor the slider instead.

Slider responses: one slider (0–100). Endpoints = reward_low / reward_high paragraphs..

Each participant sees 16 trials (one per scenario) with cells balanced across participants. Cell space: 24 cells (each participant samples 16 of 24). The counterbalancing script (`python/generate_counterbalancing.py`) produces 192 sequences (12 rounds × 16 rotations), each a 16-trial assignment of factor cells to the 16 scenarios.

The trial flow follows the "noalt" pattern from `food_inv_intimacy_desire_noalt` / `food_inv_desire_intimacy_noalt`: no candidate action list shown to the participant, only the single observed action at the posterior stage. The participant sees an intimacy-descriptor preamble page first (since intimacy is a known frame for this study), then the prior slider.

## Stimulus source

Loads stimuli from `experiments/scenarios.csv` via the routing in `experiments/csv_to_json.py`. Regenerate `json/stimuli.json` with:

```bash
uv run python experiments/csv_to_json.py
```

## Files in this directory

- `index.html` — entry point
- `experiment.js` — jsPsych 8.x boilerplate; spreads the sequence item's factor fields onto each stimulus
- `trials.js` — trial logic (instructions, attention check, scenario presentation, sliders, memory checks, exit survey, save)
- `python/generate_counterbalancing.py` — produces `json/full_counterbalancing.json`
- `json/stimuli.json` — generated from `scenarios.csv`; do not edit by hand
- `json/full_counterbalancing.json` — generated; one sequence per condition_assignment

## Before running pilots

- Replace `PIPE_EXPERIMENT_ID` in `trials.js` `CONFIG` with the real DataPipe experiment ID.
- Replace `PROLIFIC_COMPLETION_URL` with the real Prolific completion URL.
- Open `index.html` locally and walk through a few trials per cell to confirm the UI renders and the slider endpoints / observed actions look right.

## Deploy

```bash
bin/deploy-experiment food_inv_desire
```

The deploy script pushes this directory and `experiments/_lib/` to athena; see [experiments/README.md](../README.md#deploying-experiments) for details. After deploy the experiment is reachable at:

https://web.mit.edu/aliciach/www/sip/experiments/food_inv_desire/
