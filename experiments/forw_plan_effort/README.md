# forw_plan_effort

Forward planning experiment parallel to `forw_plan`, using the `scenarios_effort.csv` stimulus set. Manipulates **intimacy (4 levels) × relative effort (2 levels)** within-subject. Reward is held fixed at high (integrated into the vignette), and the action space is collapsed to two actions per scenario: `action_1` (non-saliva-sharing) and `action_2` (saliva-sharing). Participants rate the probability that the two people will take each action.

Condition counts: 4 × 2 = 8 combinations per scenario, doubled to 16 so each participant sees each combination twice across 16 scenarios.

Before running the experiment for the first time, set `CONFIG.PIPE_EXPERIMENT_ID` in `trials.js` to a new datapipe ID (currently `TODO_SET_DATAPIPE_ID`).

Regenerate the counterbalancing JSON and stimuli JSON from the project root:

```bash
uv run python experiments/csv_to_json.py
uv run python experiments/forw_plan_effort/python/generate_counterbalancing.py
```

Deploy:

```bash
ssh aliciach@athena.dialup.mit.edu "mkdir -p ~/www/food-sharing"
cd experiments
rsync -av --delete forw_plan_effort/ aliciach@athena.dialup.mit.edu:~/www/food-sharing/forw_plan_effort
```

URL: https://web.mit.edu/aliciach/www/food-sharing/forw_plan_effort
