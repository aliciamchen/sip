# food_forw_intimacy_effort

Forward planning experiment parallel to `food_forw_intimacy_desire`, using the `scenarios_effort.csv` stimulus set. Manipulates **intimacy (4 levels) × relative effort (2 levels)** within-subject. Reward is held fixed at high (integrated into the vignette), and the action space is collapsed to two actions per scenario: `action_1` (non-saliva-sharing) and `action_2` (saliva-sharing). Participants rate the probability that the two people will take each action.

Condition counts: 4 × 2 = 8 combinations per scenario, doubled to 16 so each participant sees each combination twice across 16 scenarios.

Before running the experiment for the first time, set `CONFIG.PIPE_EXPERIMENT_ID` in `trials.js` to a new datapipe ID (currently `TODO_SET_DATAPIPE_ID`).

Regenerate the counterbalancing JSON and stimuli JSON from the project root:

```bash
uv run python experiments/csv_to_json.py
uv run python experiments/food_forw_intimacy_effort/python/generate_counterbalancing.py
```

Deploy:

```bash
bin/deploy-experiment food_forw_intimacy_effort
```

The deploy script pushes this directory and `experiments/_lib/` to athena; see [experiments/README.md](../README.md#deploying-experiments) for details. After deploy the experiment is reachable at:

https://web.mit.edu/aliciach/www/sip/experiments/food_forw_intimacy_effort/
