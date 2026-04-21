---
paths:
  - "model/**/*"
---

# Model structure

Models are built using the `memo` DSL with JAX backend. Both actor (forward planning) and observer (inverse planning) models use the canonical access utility.

## Canonical utility

```
U(a|s, I) = w_v · V(a|s)
          − w_d · access(a) · (1 − I)
          − w_e · effort(a)
```

Intimacy `I` scales the access-discomfort term (bodily/spatial/informational exposure): at high intimacy the `−w_d · access · (1 − I)` penalty shrinks toward zero, so higher-access actions become relatively more attractive. `V(a|s)` is the food-sharing reward (not scaled by intimacy). Three ablations are fit and compared:

- **access_full** — the full utility above (the main model)
- **access_only** — only the access-discomfort term (`−w_d · access · (1 − I)`); drops food reward and effort
- **no_access** — `w_v · V − w_e · effort` (the base model)

Parameters: `w_v` (food-reward weight), `w_d` (access-discomfort weight), `w_e` (effort weight), plus `alpha` (actor softmax temperature) and `alpha_observer` (observer softmax temperature). Each ablation uses only the subset of weights its utility requires.

## Where the utility values come from

`V(a|s)` is **stipulated** in `model_utils.py` as a binary goal-satisfaction gate via `get_stipulated_reward(action, reward_condition)`: V=1 iff the action satisfies the active goal. Under HIGH motivation the goal is to eat/share, so V=1 for sharing actions (`action != 0`); under LOW motivation the goal is to not eat, so V=1 for `action == 0`. V=0 otherwise. No LLM call; reward is a closed-form function of motivation × action.

`access(a)` and `effort(a)` are **LLM-generated per scenario** by `model/lm_scenario_params.py` (Llama-3.3-70B via Together AI, 10 runs averaged), saved to `model/outputs/lm_scenario_params.csv`. On import, `model_utils.py` loads these into `LLM_TABLES` — a dict of two JAX arrays: `access` (16×4) and `effort` (16×4). If the CSV is missing, import fails with `FileNotFoundError` — always regenerate it before running fits.

Every memo model takes the scenario tables as arguments (`access_table: ...`, `effort_table: ...`) and has `scenario_idx: Scenarios` as a dimension, so predictions vary by scenario. Reward is computed inline inside the utility functions; no `reward_table` argument.

## Core files

- `model_utils.py` — utility functions, `Scenarios` enum, `LLM_TABLES`, and memo models (forward actors, discrete/continuous inverse-planning actors, intimacy/reward observers)
- `lm_scenario_params.py` — LLM-calls Together AI to generate the scenario CSV
- `fit_forward_planning.py` — fits the three actor ablations to `data/forw_plan/` (output: `forward_planning_fit_results.csv`, `forward_planning_fits.csv`)
- `fit_inverse_planning.py` — fits `alpha_observer` for each ablation with frozen actor params (output: `inverse_planning_fit_results.csv`)
- `generate_inverse_planning_preds.py` — emits per-scenario posterior predictions (output: `inv_plan_{intimacy,desire}_preds_{full,summary}.csv`)
- `test_model_compliance.py` — validation tests

Model outputs are saved to `model/outputs/`. Preregistration documents are in `model/preregs/`. Legacy/experimental code is in `model/sandbox/`.
