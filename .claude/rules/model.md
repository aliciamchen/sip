---
paths:
  - "model/**/*"
---

# Model structure

Models are built using the `memo` DSL with JAX backend. Both actor (forward planning) and observer (inverse planning) models use the canonical access utility with a uniform action prior.

## Canonical actor

```
P(a | s, I) ∝ exp( U(a|s, I) )

U(a|s, I) = w_v · V(a|s)
          − w_d · access(a) · (1 − I)
          − w_e · effort(a)
```

Intimacy `I` scales the access-discomfort term (bodily/spatial/informational exposure): at high intimacy the `−w_d · access · (1 − I)` penalty shrinks toward zero, so higher-access actions become relatively more attractive. `V(a|s)` is the signed food-utility (not scaled by intimacy). Three ablations are fit and compared:

- **access_full** — the full utility above (the main Full model)
- **access_only** — only the access-discomfort term (`−w_d · access · (1 − I)`); drops food utility and effort (Discomfort-only)
- **no_access** — `w_v · V − w_e · effort`; no relational structure (Base model)

Parameters: `w_v` (food-utility weight), `w_d` (access-discomfort weight), `w_e` (effort weight), plus `alpha` (actor softmax temperature, fixed to 1) and `alpha_observer` (observer softmax temperature). Each ablation uses only the subset of weights its utility requires.

## Where the utility values come from

`V(a|s)` is **stipulated** in `model_utils.py` as a signed food-utility via `get_stipulated_reward(action, reward_condition)`. Under HIGH motivation the food is wanted, so `V = +1` for actions that involve eating (`action != 0`) and `V = 0` for action 0. Under LOW motivation the food is unwanted, so `V = −1` for eating actions and `V = 0` for action 0. No LLM call; V is a closed-form function of motivation × action. Mathematically equivalent under softmax to the older binary goal-satisfaction gate (V ∈ {0, 1}); the signed form is just a cleaner framing.

`access(a)` and `effort(a)` are **LLM-generated per scenario** by `model/lm_scenario_params.py` (Llama-3.3-70B via Together AI, 10 runs averaged), saved to `model/outputs/lm_scenario_params.csv`.

On import, `model_utils.py` loads these into `LLM_TABLES` — a dict of JAX arrays: `access` (16×4) and `effort` (16×4). If `lm_scenario_params.csv` is missing, import fails with `FileNotFoundError`.

Every memo model takes the scenario tables as arguments (`access_table: ...`, `effort_table: ...`) and has `scenario_idx: Scenarios` as a dimension, so predictions vary by scenario. Reward is computed inline inside the utility functions; no `reward_table` argument.

## Core files

- `model_utils.py` — utility functions, `Scenarios` enum, `LLM_TABLES`, and memo models (forward actors, discrete/continuous inverse-planning actors, intimacy/reward observers)
- `lm_scenario_params.py` — LLM-calls Together AI to generate per-scenario access and effort
- `fit_forward_planning.py` — fits the three actor ablations to `data/forw_plan/` (output: `forward_planning_fit_results.csv`, `forward_planning_fits.csv`)
- `fit_inverse_planning.py` — alt-shown observers; fits only `alpha_observer` with frozen actor params (output: `inverse_planning_fit_results.csv`)
- `fit_inverse_planning_noalt.py` — no-alt observer; **jointly fits all actor weights + α_observer** on no-alt data (not frozen from Exp 1, because the padded observer's variable-length action space differs from Exp 1's fixed 4-action space). Output: `inverse_planning_noalt_fit_results.csv`
- `generate_inverse_planning_preds.py` — emits per-scenario posterior predictions for alt-shown (`inv_plan_{intimacy,desire}_preds_{full,summary}.csv`)
- `generate_inverse_planning_noalt_preds.py` — same for no-alt, using the joint-fit weights from `inverse_planning_noalt_fit_results.csv`
- `test_model_compliance.py` — validation tests

### Effort-experiment parallel pipeline

A second, parallel pipeline mirrors the canonical scripts on the effort stimulus set (`scenarios_effort.csv`): 16 scenarios × 2 actions × 2 effort conditions (low / high), with reward held fixed at HIGH so V is constant across actions and `w_v` is non-identified under the softmax (it's kept in the utility for parallelism with the canonical pipeline but stays near initialization). Scenario labels are shared with the canonical 16, so `Scenarios` / `SCENARIO_TO_IDX` are reused; effort adds a separate `EffortConditions` IntEnum and `EFFORT_CONDITION_TO_IDX` map.

- `model_utils_effort.py` — effort-experiment utility functions and memo models (2-action actors and intimacy observers, with an `effort_condition` covariate). Loads `LLM_TABLES_EFFORT` (`access`, `effort`, both shape 16×2×2; plus `access_marg` shape 16×2×2 — the effort-marginal access broadcast across the effort_condition dimension) at import.
- `lm_scenario_params_effort.py` — produces two CSVs: (1) `lm_scenario_params_effort.csv` (64 rows: 16 scenarios × 2 effort × 2 actions) — effort-conditional access + effort, where the LM is prompted with the full vignette + effort paragraph so the manipulation lands in the ratings; (2) `lm_scenario_params_effort_marginal.csv` (32 rows: 16 scenarios × 2 actions) — effort-marginal access only, where the LM is prompted with just the base vignette. The marginal pass is needed because the inv_plan_effort_inferred observer does not see the effort paragraph and so must reason about access from the base vignette alone.
- `fit_forward_planning_effort.py` — fits the three actor ablations to `data/forw_plan_effort/`. Outputs: `forward_planning_effort_fit_results.csv`, `forward_planning_effort_fits.csv`.
- `fit_inverse_planning_effort.py` — fits only `alpha_observer` for `inv_plan_effort`, with actor weights frozen from `forward_planning_effort_fit_results.csv` (NOT the canonical `forw_plan` fit, because the effort actor's 2-action softmax doesn't transplant).
- `generate_inverse_planning_effort_preds.py` — emits `inv_plan_effort_preds_{full,summary}.csv`.
- `fit_inverse_planning_effort_inferred.py` — flips the inference target: observer infers effort condition (latent) given observed action × intimacy. Uses `observer_effort_inferred_*` from `model_utils_effort.py` and binary cross-entropy NLL (slider 0–100 = P(effort_high)·100). Actor weights frozen from `forward_planning_effort_fit_results.csv`, but the actor's utility is evaluated with **effort-marginal access** (`LLM_TABLES_EFFORT['access_marg']`) instead of the effort-conditional table — because the observer in this experiment does not see the effort paragraph and so cannot perceive any effort-induced setting differences in the access of an action. The effort term itself remains effort-conditional (the observer does compute likelihoods under each candidate effort condition). Output: `inverse_planning_effort_inferred_fit_results.csv`.
- `generate_inverse_planning_effort_inferred_preds.py` — emits `inv_plan_effort_inferred_preds_{full,summary}.csv`. The `summary` CSV's column `p_effort_high` is what the slider response 0-100 encodes.

## Cross-validation

All model-vs-human correlations reported in the analysis qmds are out-of-sample, from leave-one-scenario-out (LOSO) CV. The analysis qmds load CV-prediction CSVs (`cv_loso_*`) as the source for all plots.

- `cv/loso_forward.py` — Exp 1; refits $w_v, w_d, w_e$ per fold. Outputs: `cv_loso_forward.csv`, `cv_loso_preds.csv`
- `cv/loso_inverse_alt.py` — Exp 2a intimacy + 2b desire; refits only $\alpha_{\mathrm{obs}}$ per fold (actor frozen from all-data Exp 1 fit, same 4-action space). Outputs: `cv_loso_inv_plan_{intimacy,desire}_alt_preds_summary.csv`, `cv_loso_inverse_alt_folds.csv`
- `cv/loso_inverse_noalt.py` — Exp 2c no-alt; joint LOSO refit of all actor weights + $\alpha_{\mathrm{obs}}$ per fold. Outputs: `cv_loso_inv_plan_intimacy_noalt_preds_summary.csv`, `cv_loso_inverse_noalt_folds.csv`
- `cv/loso_forward_effort.py` — `forw_plan_effort`; refits $w_v, w_d, w_e$ per fold (note `w_v` is non-identified). Outputs: `cv_loso_forward_effort.csv`, `cv_loso_preds_effort.csv`
- `cv/loso_inverse_effort.py` — `inv_plan_effort`; refits only $\alpha_{\mathrm{obs}}$ per fold (actor frozen from the effort all-data forward fit). Outputs: `cv_loso_inv_plan_effort_preds_summary.csv`, `cv_loso_inverse_effort_folds.csv`
- `cv/loso_inverse_effort_inferred.py` — `inv_plan_effort_inferred`; refits only $\alpha_{\mathrm{obs}}$ per fold (actor frozen from the effort all-data forward fit). Outputs: `cv_loso_inv_plan_effort_inferred_preds_summary.csv`, `cv_loso_inverse_effort_inferred_folds.csv`

The non-CV `fit_*` / `generate_*` pipelines still produce all-data fits; AIC and fitted-parameter tables in the qmds use the all-data fit, but all model-vs-human displays use the CV predictions.

Model outputs are saved to `model/outputs/`. Preregistration documents are in `model/preregs/`. Legacy/experimental code is in `model/sandbox/`.

## Commands

LLM-derived scenario parameters (prerequisite for all fits; requires `TOGETHER_API_KEY` in `.env`; Llama-3.3-70B via Together AI, 10 runs averaged):

```bash
uv run python model/lm_scenario_params.py                     # canonical food: 16×4 access + effort → lm_scenario_params.csv
uv run python model/lm_scenario_params.py --domain nonfood    # nonfood: → lm_scenario_params_nonfood.csv (uses scenarios_nonfood.csv)
uv run python model/lm_scenario_params_effort.py              # effort: 64-row conditional (lm_scenario_params_effort.csv) + 32-row marginal (lm_scenario_params_effort_marginal.csv)
```

Forward-planning fits (3 ablations: Base / Discomfort-only / Full):

```bash
uv run python model/fit_forward_planning.py                   # canonical food
uv run python model/fit_forward_planning.py --domain nonfood  # nonfood (writes *_nonfood.csv outputs)
uv run python model/fit_forward_planning_effort.py            # effort
```

Inverse-planning fits + prediction generators:

```bash
uv run python model/fit_inverse_planning.py                            # alt-shown (intimacy + desire), α_observer only
uv run python model/generate_inverse_planning_preds.py
uv run python model/fit_inverse_planning_noalt.py                      # no-alt, joint fit (all weights + α_observer)
uv run python model/generate_inverse_planning_noalt_preds.py
uv run python model/fit_inverse_planning_effort.py                     # inv_plan_effort, α_observer only
uv run python model/generate_inverse_planning_effort_preds.py
uv run python model/fit_inverse_planning_effort_inferred.py            # inv_plan_effort_inferred, α_observer only
uv run python model/generate_inverse_planning_effort_inferred_preds.py
```

LOSO cross-validation (16 folds × 3 variants per experiment; the analysis qmds plot from these CSVs):

```bash
uv run python model/cv/loso_forward.py                     # refits w_v, w_d, w_e, β per fold (food)
uv run python model/cv/loso_forward.py --domain nonfood    # nonfood (writes *_nonfood.csv outputs)
uv run python model/cv/loso_inverse_alt.py                 # refits only α_observer per fold
uv run python model/cv/loso_inverse_noalt.py               # joint fit per fold
uv run python model/cv/loso_forward_effort.py              # refits w_d, w_e, β per fold (w_v non-identified)
uv run python model/cv/loso_inverse_effort.py              # refits only α_observer per fold
uv run python model/cv/loso_inverse_effort_inferred.py     # refits only α_observer per fold
```

`fit_forward_planning.py` and `cv/loso_forward.py` accept `--domain food|nonfood`. Food is the default and writes the canonical filenames (`forward_planning_*.csv`, `cv_loso_forward.csv`, `cv_loso_preds.csv`); nonfood writes `*_nonfood.csv` siblings. Both branches share the same memo models in `model_utils.py` — only the scenario-label↔index map and the LLM tables differ (see `load_domain_assets`).

CV outputs (in `model/outputs/`):
- `cv_loso_forward.csv` / `cv_loso_preds.csv` — per-fold fits + per-trial held-out forward predictions
- `cv_loso_forward_nonfood.csv` / `cv_loso_preds_nonfood.csv` — same, nonfood
- `cv_loso_inv_plan_intimacy_alt_preds_summary.csv` / `cv_loso_inv_plan_desire_alt_preds_summary.csv` / `cv_loso_inverse_alt_folds.csv`
- `cv_loso_inv_plan_intimacy_noalt_preds_summary.csv` / `cv_loso_inverse_noalt_folds.csv`
- `cv_loso_forward_effort.csv` / `cv_loso_preds_effort.csv`
- `cv_loso_inv_plan_effort_preds_summary.csv` / `cv_loso_inverse_effort_folds.csv`
- `cv_loso_inv_plan_effort_inferred_preds_summary.csv` / `cv_loso_inverse_effort_inferred_folds.csv`

The non-CV `fit_*` / `generate_*` pipelines still produce all-data fits — AIC and fitted-parameter tables in the qmds use the all-data fit, but all model-vs-human displays use the CV predictions.

Tests:

```bash
uv run python model/test_model_compliance.py
```
