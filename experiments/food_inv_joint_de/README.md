# food_inv_joint_de

Study 1b — Joint inference over desire and effort.

## Design

**Factor crossing**: 4 (intimacy: 0/50/75/100) × 3 (observed action)

**Known to participant**: intimacy_condition

**Inferred by participant**: reward_condition AND effort_condition (two sliders per trial — a continuous 0–100 desire slider with "not at all" / "extremely" endpoints, and an effort slider with effort_low/high paragraph endpoints)

## Trial structure

Each trial shows: intimacy-descriptor preamble page, then vignette + observed action (at posterior). Neither reward nor effort paragraph is shown in the vignette.

Slider responses: two sliders per phase (prior + posterior = 4 sliders total). A continuous 0–100 desire slider ("how much do they want to eat the food?", endpoints "not at all" / "extremely"); an effort slider with effort_low/high paragraph endpoints. The `response_target` field on each data row indicates which slider it was.

Each participant sees 16 trials (one per scenario) with cells balanced across participants. Cell space: 12 cells. The counterbalancing script (`python/generate_counterbalancing.py`) produces 192 sequences (12 rounds × 16 rotations), each a 16-trial assignment of factor cells to the 16 scenarios.

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

- Set this experiment's real DataPipe experiment ID in `experiments/_lib/config.js` (the `DATAPIPE_IDS` map, keyed by slug). The Prolific completion URL is shared across experiments and already set there.
- Open `index.html` locally and walk through a few trials per cell to confirm the UI renders and the slider endpoints / observed actions look right.

## Deploy

```bash
bin/deploy-experiment food_inv_joint_de
```

The deploy script pushes this directory and `experiments/_lib/` to athena; see [experiments/README.md](../README.md#deploying-experiments) for details. After deploy the experiment is reachable at:

https://web.mit.edu/aliciach/www/sip/experiments/food_inv_joint_de/
