---
paths:
  - "model/**/*"
---

# Model structure

Models are built using the `memo` DSL with JAX backend. Both actor (forward planning) and observer (inverse planning) models use the canonical access utility with an LM-derived action prior.

## Canonical actor

```
P(a | s, I) ∝ π(a|s)^β · exp( U(a|s, I) )

U(a|s, I) = w_v · V(a|s)
          − w_d · access(a) · (1 − I)
          − w_e · effort(a)
```

`π(a|s)` is a scenario-specific prior over the four actions (how "default" each action feels in the scenario's setting). `β` tempers the prior's weight; fitted as a free parameter alongside the utility weights. Intimacy `I` scales the access-discomfort term (bodily/spatial/informational exposure): at high intimacy the `−w_d · access · (1 − I)` penalty shrinks toward zero, so higher-access actions become relatively more attractive. `V(a|s)` is the signed food-utility (not scaled by intimacy). Three ablations are fit and compared (all share the same softmax-with-prior structure and each fits `β`):

- **access_full_prior** — the full utility above (the main Full model)
- **access_only_prior** — only the access-discomfort term (`−w_d · access · (1 − I)`); drops food utility and effort (Discomfort-only)
- **no_access_prior** — `w_v · V − w_e · effort`; no relational structure (Base model)

Uniform-prior variants (`access_full`, `access_only`, `no_access`) are still fit for internal comparison but are no longer the canonical display models.

Parameters: `w_v` (food-utility weight), `w_d` (access-discomfort weight), `w_e` (effort weight), `beta_prior` (prior-tempering weight), plus `alpha` (actor softmax temperature, fixed to 1) and `alpha_observer` (observer softmax temperature). Each ablation uses only the subset of weights its utility requires.

## Where the utility values come from

`V(a|s)` is **stipulated** in `model_utils.py` as a signed food-utility via `get_stipulated_reward(action, reward_condition)`. Under HIGH motivation the food is wanted, so `V = +1` for actions that involve eating (`action != 0`) and `V = 0` for action 0. Under LOW motivation the food is unwanted, so `V = −1` for eating actions and `V = 0` for action 0. No LLM call; V is a closed-form function of motivation × action. Mathematically equivalent under softmax to the older binary goal-satisfaction gate (V ∈ {0, 1}); the signed form is just a cleaner framing.

`access(a)` and `effort(a)` are **LLM-generated per scenario** by `model/lm_scenario_params.py` (Llama-3.3-70B via Together AI, 10 runs averaged), saved to `model/outputs/lm_scenario_params.csv`.

`π(a|s)` is **LLM-generated per scenario** by `model/lm_action_priors.py` (same LLM, same 10-run averaging). The prompt asks the LLM to rate how natural / expected each action is as a "default" in the scenario's setting, independent of any relationship or motivation information. Ratings are sum-normalized (with small additive smoothing, ε=0.1, so no action gets probability zero) to yield a proper distribution. Saved to `model/outputs/lm_action_priors.csv`.

On import, `model_utils.py` loads these into `LLM_TABLES` — a dict of JAX arrays: `access` (16×4), `effort` (16×4), and `action_prior` (16×4). If `lm_scenario_params.csv` is missing, import fails with `FileNotFoundError`. If `lm_action_priors.csv` is missing, the `_prior` variants are silently skipped at fit/predict time but the uniform-prior variants still run. Always regenerate both CSVs before running fits if you've changed the pipeline.

Every memo model takes the scenario tables as arguments (`access_table: ...`, `effort_table: ...`, and for prior variants `prior_table: ...`) and has `scenario_idx: Scenarios` as a dimension, so predictions vary by scenario. Reward is computed inline inside the utility functions; no `reward_table` argument.

## Core files

- `model_utils.py` — utility functions, `Scenarios` enum, `LLM_TABLES`, and memo models (forward actors, discrete/continuous inverse-planning actors, intimacy/reward observers; both uniform-prior and LM-prior variants)
- `lm_scenario_params.py` — LLM-calls Together AI to generate per-scenario access and effort
- `lm_action_priors.py` — LLM-calls Together AI to generate per-scenario action priors π(a|s)
- `fit_forward_planning.py` — fits all six actor variants to `data/forw_plan/` (output: `forward_planning_fit_results.csv`, `forward_planning_fits.csv`)
- `fit_inverse_planning.py` — fits `alpha_observer` for each variant with frozen actor params (output: `inverse_planning_fit_results.csv`)
- `generate_inverse_planning_preds.py` — emits per-scenario posterior predictions (output: `inv_plan_{intimacy,desire}_preds_{full,summary}.csv`)
- `test_model_compliance.py` — validation tests

Model outputs are saved to `model/outputs/`. Preregistration documents are in `model/preregs/`. Legacy/experimental code is in `model/sandbox/`.
