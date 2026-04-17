---
paths:
  - "model/**/*"
---

# Model structure

Models are built using the `memo` DSL with JAX backend. Both actor (forward planning) and observer (inverse planning) models come in two families.

## Pre-registered family
1. **full_model (Social planning)**: reward scaled by intimacy and motivation, discomfort scaled by intimacy, and sharing cost
2. **vanilla**: reward, discomfort, and cost without intimacy scaling
3. **discomfort_only**: only discomfort from saliva transfer (scaled by intimacy)

Parameters: `alpha`, `alpha_observer`, `w_r` (reward), `w_d` (discomfort/risk), `w_c` (cost/effort), `beta` (reward-intimacy scaling, modified models only).

## Access-based family (canonical reformulation)

Canonical utility:
```
U(a|s, I) = w_v · V(a|s)
          + w_r · access(a) · I
          − w_d · access(a) · (1 − I)
          − w_e · effort(a)
```

Intimacy scales `access(a)` (bodily/spatial exposure) rather than the food reward `V(a|s)`. Three variants are fit and compared:
- **access_full**: the full utility above — food reward, both access terms, and effort (the main model)
- **access_only**: only the two access terms (`w_r·access·I − w_d·access·(1−I)`); drops food reward and effort
- **no_access**: `w_v·V − w_e·effort` (the Base model / baseline)

Parameters: `w_v` (food-reward weight), `w_r` (positive-access weight), `w_d` (negative-access weight), `w_e` (effort weight), plus `alpha` / `alpha_observer`. Each variant uses only the subset of weights its utility requires.

Fixed (stipulated) vectors: `access(a) = [0, 0.3, 1, 2]` (graded exposure; saliva transfer is a qualitatively bigger step than just eating together), `effort(a) = [0, 1, 1, 1]`. `V(a|s)` reuses `get_reward_base` from the pre-registered family (0 for action 0 or LOW motivation; 1 otherwise).

## LLM-parameterized variants (`*_llm`)

`model/lm_scenario_params.py` uses Llama-3.3-70B via Together AI to generate scenario-specific values for `access(a)`, `effort(a)`, and a scenario-level reward `V(s)`, saving them to `model/outputs/lm_scenario_params.csv`. On import, `model_utils.py` loads these into `LLM_TABLES` (a dict of three JAX arrays: `access` (16×4), `effort` (16×4), `reward` (16,)).

For each of the three access variants there is a `*_llm` companion that reads from these tables instead of the fixed vectors: `actor_forw_access_full_llm`, `actor_discrete_access_only_llm`, `observer_intimacy_no_access_llm`, etc. The scenario-specific tables are passed as memo arguments (`access_table`, `effort_table`, `reward_table`), and `scenario_idx: Scenarios` is a new memo dimension.

If `model/outputs/lm_scenario_params.csv` is missing, `LLM_TABLES` is `None` and the fit + prediction scripts skip the `*_llm` variants gracefully.

Core model files:
- `model_utils.py` - Actor and observer models with discrete (4-level) and continuous (0-1) intimacy variants
- `fit_forward_planning.py` - Fit actor models to Exp 1 data
- `fit_inverse_planning.py` - Fit observer models to Exp 2a/2b data
- `generate_inverse_planning_preds.py` - Generate model predictions
- `test_model_compliance.py` - Validation tests

Model outputs are saved to `model/outputs/`. Preregistration documents are in `model/preregs/`. Legacy/experimental code is in `model/sandbox/`.
