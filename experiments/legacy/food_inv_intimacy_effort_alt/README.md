# food_inv_intimacy_effort_alt

Inverse planning experiment parallel to `food_inv_intimacy_desire_alt`, using the `scenarios_effort.csv` stimulus set. Participants see both candidate actions and the one the actors took, and infer relationship intimacy. Manipulates **observed action (2 levels) × relative effort (2 levels)** within-subject.

For each scenario, participants first see the vignette and both candidate actions (`action_1` non-saliva, `action_2` saliva) and rate intimacy as a prior. The trial then reveals which action the actors took, and participants re-rate intimacy as a posterior.

Condition counts: 2 × 2 = 4 combinations per scenario, quadrupled to 16 so each participant sees each combination four times across 16 scenarios.

Before running the experiment for the first time, set `CONFIG.PIPE_EXPERIMENT_ID` in `trials.js` to a new datapipe ID (currently `TODO_SET_DATAPIPE_ID`).

Regenerate the counterbalancing JSON and stimuli JSON from the project root:

```bash
uv run python experiments/csv_to_json.py
uv run python experiments/food_inv_intimacy_effort_alt/python/generate_counterbalancing.py
```

Deploy:

```bash
ssh aliciach@athena.dialup.mit.edu "mkdir -p ~/www/food-sharing"
cd experiments
rsync -av --delete food_inv_intimacy_effort_alt/ aliciach@athena.dialup.mit.edu:~/www/food-sharing/food_inv_intimacy_effort_alt
```

URL: https://web.mit.edu/aliciach/www/food-sharing/food_inv_intimacy_effort_alt
