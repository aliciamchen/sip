---
paths:
  - "model/**/*"
---

# Model structure

Models are built using the `memo` DSL with JAX backend. Three model types exist for both actor (forward planning) and observer (inverse planning):

1. **full_model (Social planning)**: Considers reward scaled by intimacy and motivation, discomfort scaled by intimacy, and sharing cost
2. **vanilla**: Considers reward, discomfort, and cost without intimacy scaling
3. **discomfort_only**: Only considers discomfort from saliva transfer (scaled by intimacy)

Key model parameters:
- `alpha`: inverse temperature (actor rationality)
- `alpha_observer`: inverse temperature for observer's inference
- `w_r`: reward weight
- `w_d`: discomfort/risk weight
- `w_c`: cost/effort weight
- `beta`: reward-intimacy scaling parameter (modified models only)

Core model files:
- `model_utils.py` - Actor and observer models with discrete (4-level) and continuous (0-1) intimacy variants
- `fit_forward_planning.py` - Fit actor models to Exp 1 data
- `fit_inverse_planning.py` - Fit observer models to Exp 2a/2b data
- `generate_inverse_planning_preds.py` - Generate model predictions
- `test_model_compliance.py` - Validation tests

Model outputs are saved to `model/outputs/`. Preregistration documents are in `model/preregs/`. Legacy/experimental code is in `model/sandbox/`.
