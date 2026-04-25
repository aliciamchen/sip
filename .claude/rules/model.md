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
- `fit_inverse_planning.py` — alt-shown observers; fits only `alpha_observer` with frozen actor params (output: `inverse_planning_fit_results.csv`)
- `fit_inverse_planning_noalt.py` — no-alt observer; **jointly fits all actor weights + α_observer** on no-alt data (not frozen from Exp 1, because the padded observer's variable-length action space differs from Exp 1's fixed 4-action space). Output: `inverse_planning_noalt_fit_results.csv`
- `generate_inverse_planning_preds.py` — emits per-scenario posterior predictions for alt-shown (`inv_plan_{intimacy,desire}_preds_{full,summary}.csv`)
- `generate_inverse_planning_noalt_preds.py` — same for no-alt, using the joint-fit weights from `inverse_planning_noalt_fit_results.csv`
- `test_model_compliance.py` — validation tests

### Effort-experiment parallel pipeline

A second, parallel pipeline mirrors the canonical scripts on the effort stimulus set (`scenarios_effort.csv`): 16 scenarios × 2 actions × 2 effort conditions (low / high), with reward held fixed at HIGH so V is constant across actions and `w_v` is non-identified under the softmax (it's kept in the utility for parallelism with the canonical pipeline but stays near initialization). Scenario labels are shared with the canonical 16, so `Scenarios` / `SCENARIO_TO_IDX` are reused; effort adds a separate `EffortConditions` IntEnum and `EFFORT_CONDITION_TO_IDX` map.

- `model_utils_effort.py` — effort-experiment utility functions and memo models (2-action actors and intimacy observers, with an `effort_condition` covariate). Loads `LLM_TABLES_EFFORT` (`access`, `effort`, `action_prior`, all shape 16×2×2) at import.
- `lm_scenario_params_effort.py` — LM-prompts the full vignette + effort paragraph so the manipulation lands in the ratings. Output: `lm_scenario_params_effort.csv` (64 rows).
- `lm_action_priors_effort.py` — same idea for π(a|s,e). Output: `lm_action_priors_effort.csv` (64 rows).
- `fit_forward_planning_effort.py` — fits all six actor variants to `data/forw_plan_effort/`. Outputs: `forward_planning_effort_fit_results.csv`, `forward_planning_effort_fits.csv`.
- `fit_inverse_planning_effort.py` — fits only `alpha_observer` for `inv_plan_effort`, with actor weights frozen from `forward_planning_effort_fit_results.csv` (NOT the canonical `forw_plan` fit, because the effort actor's 2-action softmax doesn't transplant).
- `generate_inverse_planning_effort_preds.py` — emits `inv_plan_effort_preds_{full,summary}.csv`.

## Cross-validation

All model-vs-human correlations reported in the analysis qmds are out-of-sample, from leave-one-scenario-out (LOSO) CV. The analysis qmds load CV-prediction CSVs (`cv_loso_*`) as the source for all plots.

- `cv/loso_forward.py` — Exp 1; refits $w_v, w_d, w_e, \beta$ per fold. Outputs: `cv_loso_forward.csv`, `cv_loso_preds.csv`
- `cv/loso_inverse_alt.py` — Exp 2a intimacy + 2b desire; refits only $\alpha_{\mathrm{obs}}$ per fold (actor frozen from all-data Exp 1 fit, same 4-action space). Outputs: `cv_loso_inv_plan_{intimacy,desire}_alt_preds_summary.csv`, `cv_loso_inverse_alt_folds.csv`
- `cv/loso_inverse_noalt.py` — Exp 2c no-alt; joint LOSO refit of all actor weights + $\alpha_{\mathrm{obs}}$ per fold. Outputs: `cv_loso_inv_plan_intimacy_noalt_preds_summary.csv`, `cv_loso_inverse_noalt_folds.csv`
- `cv/loso_forward_effort.py` — `forw_plan_effort`; refits $w_v, w_d, w_e, \beta$ per fold (note `w_v` is non-identified). Outputs: `cv_loso_forward_effort.csv`, `cv_loso_preds_effort.csv`
- `cv/loso_inverse_effort.py` — `inv_plan_effort`; refits only $\alpha_{\mathrm{obs}}$ per fold (actor frozen from the effort all-data forward fit). Outputs: `cv_loso_inv_plan_effort_preds_summary.csv`, `cv_loso_inverse_effort_folds.csv`

The non-CV `fit_*` / `generate_*` pipelines still produce all-data fits; AIC and fitted-parameter tables in the qmds use the all-data fit, but all model-vs-human displays use the CV predictions.

Model outputs are saved to `model/outputs/`. Preregistration documents are in `model/preregs/`. Legacy/experimental code is in `model/sandbox/`.
