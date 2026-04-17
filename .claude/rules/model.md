---
paths:
  - "model/**/*"
---

# Model structure

Models are built using the `memo` DSL with JAX backend. Both actor (forward planning) and observer (inverse planning) models use the canonical access utility.

## Canonical utility

```
U(a|s, I) = w_v · V(a|s)
          + w_r · access(a) · I
          − w_d · access(a) · (1 − I)
          − w_e · effort(a)
```

Intimacy `I` scales `access(a)` (bodily/spatial/informational exposure). `V(a|s)` is the intrinsic food reward (not scaled by intimacy). Three ablations are fit and compared:

- **access_full** — the full utility above (the main model)
- **access_only** — only the two access terms (`w_r·access·I − w_d·access·(1−I)`); drops food reward and effort
- **no_access** — `w_v·V − w_e·effort` (the base model)

Parameters: `w_v` (food-reward weight), `w_r` (positive-access weight), `w_d` (negative-access weight), `w_e` (effort weight), plus `alpha` (actor softmax temperature) and `alpha_observer` (observer softmax temperature). Each ablation uses only the subset of weights its utility requires.

## Per-scenario values from an LLM

`access(a)`, `effort(a)`, and the scenario-level `V(s)` are generated per-scenario by an LLM. `model/lm_scenario_params.py` uses Llama-3.3-70B via Together AI (10 runs averaged, mean ± std saved) and writes `model/outputs/lm_scenario_params.csv`. On import, `model_utils.py` loads these into `LLM_TABLES` — a dict of three JAX arrays: `access` (16×4), `effort` (16×4), `reward` (16,). If the CSV is missing, import fails with `FileNotFoundError` — always regenerate it before running fits.

Every memo model takes the scenario tables as arguments (`access_table: ...`, `effort_table: ...`, `reward_table: ...`) and has `scenario_idx: Scenarios` as a dimension, so predictions vary by scenario.

## Core files

- `model_utils.py` — utility functions, `Scenarios` enum, `LLM_TABLES`, and memo models (forward actors, discrete/continuous inverse-planning actors, intimacy/reward observers)
- `lm_scenario_params.py` — LLM-calls Together AI to generate the scenario CSV
- `fit_forward_planning.py` — fits the three actor ablations to Exp 1 data (output: `forward_planning_fit_results.csv`, `forward_planning_fits.csv`)
- `fit_inverse_planning.py` — fits `alpha_observer` for each ablation with frozen actor params (output: `inverse_planning_fit_results.csv`)
- `generate_inverse_planning_preds.py` — emits per-scenario posterior predictions (output: `inv_plan_{intimacy,desire}_preds_{full,summary}.csv`)
- `test_model_compliance.py` — validation tests

Model outputs are saved to `model/outputs/`. Preregistration documents are in `model/preregs/`. Legacy/experimental code is in `model/sandbox/`.
