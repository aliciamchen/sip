# food_inv_joint_ie

Study 2b — Joint inference over intimacy and effort, given desire.

## Design

**Factor crossing**: 2 (desire: low/high) × 3 (observed action)

**Known to participant**: reward_condition (shown as a paragraph preamble describing how much the characters want the food)

**Inferred by participant**: intimacy AND effort_condition (two sliders per phase — intimacy on a 0–100 numeric scale, effort with `effort_low` / `effort_high` paragraph endpoints)

## Trial structure

Each trial shows: vignette + reward (desire) paragraph as preamble, then prior sliders, then observed action, then posterior sliders. Neither the intimacy descriptor nor the effort paragraph is shown directly — both are inferred from the action.

Slider responses: two sliders per phase (prior + posterior = 4 sliders total). Intimacy slider 0–100 with maximally formal / maximally intimate labels; effort slider with effort_low / effort_high paragraph endpoints. The `response_target` field on each data row indicates which slider it was.

Each participant sees 16 trials (one per scenario) with cells balanced across participants. Cell space: 6 cells. The counterbalancing script (`python/generate_counterbalancing.py`) produces 96 sequences (6 rounds × 16 rotations), each a 16-trial assignment of factor cells to the 16 scenarios.

The trial flow follows the "noalt" pattern: no candidate action list shown to the participant, only the single observed action at the posterior stage.

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
bin/deploy-experiment food_inv_joint_ie
```

The deploy script pushes this directory and `experiments/_lib/` to athena; see [experiments/README.md](../README.md#deploying-experiments) for details. After deploy the experiment is reachable at:

https://web.mit.edu/aliciach/www/sip/experiments/food_inv_joint_ie/
