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
          − w_d · access(a) · (1 − I)^γ
          − w_e · effort(a)
```

Intimacy `I` scales the access-discomfort term (bodily/spatial/informational exposure) through a power-law modulator `(1 − I)^γ`: at high intimacy the penalty shrinks toward zero, so higher-access actions become relatively more attractive. The exponent γ is a free parameter (initialized at 1.0; γ = 1 reproduces the linear-intimacy special case). Empirically food prefers γ < 1 (late relaxation) and non-food prefers γ > 1 (early relaxation). `V(a|s, m)` is the signed valence of the action with respect to the actor's motivational state (in [-1, +1]; positive = serves the state, negative = actively counterproductive). Three ablations are fit and compared:

- **full** — the full utility above (the main Full model)
- **discomfort_only** — only the access-discomfort term (`−w_d · access · (1 − I)^γ`); drops V and effort (Discomfort-only)
- **base** — `w_v · V − w_e · effort`; no relational structure (Base model). Has no intimacy term, so γ does not apply.

Parameters: `w_v` (V weight), `w_d` (access-discomfort weight), `w_e` (effort weight), `gamma` (intimacy power-law exponent, free, init 1.0, clipped ≥ 1e-6 by the optimizer's clip), plus `alpha` (actor softmax temperature, fixed to 1) and `alpha_observer` (observer softmax temperature). Each ablation uses only the subset of weights its utility requires; full and discomfort_only fit γ, base does not.

## Three stimulus sets, three LM table families

The manuscript covers three different stimulus structures, each backed by its own LM-elicited parameter tables:

- **4-action canonical** (Study 1a, `food_forw_intimacy_desire`) — `scenarios.csv` with 4 actions per scenario. Tables in `LLM_TABLES` (access/effort, shape (16, 4)) and `load_lm_v()` (V, (16, 4, 2)).
- **2-action effort** (Study 1b, `food_forw_intimacy_effort`) — `scenarios_effort.csv` with 2 actions and an `effort_low`/`effort_high` paragraph. Tables in `LLM_TABLES_EFFORT` (access/effort indexed by `(scenario, effort_condition, action)` of shape (16, 2, 2); V is stipulated to 1 in `utility.py:get_stipulated_reward_effort`).
- **3-action canonical** (Studies 2, 3a, 3b, 4a, 4b — all `food_inv_*_3act` slugs) — `scenarios_3act.csv` merges the effort paragraphs into the canonical scenarios. Tables in `LLM_TABLES_3ACT` (access/effort, (16, 2, 3)) and `load_lm_v_3act()` (V, (16, 3, 2)). Study 3a additionally needs an effort-marginal access table (broadcast across effort_condition) since that observer doesn't see the effort paragraph; it's stored as `LLM_TABLES_3ACT["access_marg"]`.

All three LM table loaders return `None` when their CSV is missing, so imports stay clean before elicitation has been run.

The LM call infrastructure goes through `model/lm/client.py`, which fans NUM_RUNS calls across a thread pool, constrains output to a JSON schema via `response_format`, retries transient errors at the SDK layer, and checkpoints per-scenario; new LM call sites should reuse `get_ratings_concurrent` + the schema helpers (`numeric_action_schema`, `alternatives_array_schema`) rather than calling Together directly. CSV outputs include both `n_runs_*` and `n_failures_*` columns.

Every memo model takes the scenario tables as arguments (`access_table: ...`, `effort_table: ...`, and `v_table: ...` for the full and base variants) and has `scenario_idx: Scenarios` as a dimension, so predictions vary by scenario. The `discomfort_only` ablation is V-independent and doesn't take `v_table`.

## Layout

`model/` is organized so that every script's name tells you what it does. Per-experiment scripts live in `forward/`, `inverse/`, and `cv/` subfolders, named after the experiment slug (e.g. `forward/fit_food_forw_intimacy_desire.py`). LM-elicitation scripts live in `lm/`, named after the output they produce. Shared math is in four core modules at the top of `model/`.

### Core math (one copy, shared across all experiments)

- `tables.py` — enums (`Scenarios`, `RewardConditions`, `RelationshipConditions`, `EffortConditions`, `PaddedActionSlots`), action arrays (`actions`, `actions_effort`, `actions_3act`), `SCENARIO_LABELS`, LM table loaders for all three stimulus structures (`LLM_TABLES`, `LLM_TABLES_EFFORT`, `LLM_TABLES_3ACT`, `load_lm_v`, `load_lm_v_3act`), and the legacy padded-table loaders.
- `utility.py` — jit-compiled utility functions. Three families: `get_utility_full/discomfort_only/base` (canonical 4-action), `get_utility_effort_*` (2-action effort), and `get_utility_3act_*` (3-action; includes `_disc` siblings for discrete relationship). The 3-action variants index access/effort like the effort experiment (`access[scen, effort_condition, action]`) and V like the canonical pipeline (`v[scen, action, reward]`).
- `actors.py` — actor memos: `actor_forw_*` (forward, canonical), `actor_forw_effort_*` (forward, effort experiment), `actor_forw_3act_*` (forward 3-action), `actor_continuous_3act_*` and `actor_discrete_3act_*` (3-action inverse actors used inside Studies 2/3/4 observer `thinks[...]` blocks). Legacy `actor_discrete_*`, `actor_continuous_*`, and their `_padded` / `_padded_rel` variants remain for the archived inverse experiments.
- `observers.py` — observer memos. New roster (3-action, single observed action — no candidate alternatives shown to participants):
    - **Study 2** — `observer_intimacy_3act_*` (infers intimacy given reward + effort)
    - **Study 3a** — `observer_effort_3act_*` (infers effort given reward + intimacy)
    - **Study 3b** — `observer_reward_3act_*` (infers reward given effort + intimacy)
    - **Study 4a** — `observer_joint_de_3act_*` (joint posterior over reward × effort given intimacy)
    - **Study 4b** — `observer_joint_di_3act_*` (joint posterior over reward × intimacy given effort)
  Each comes in `_full`, `_discomfort_only`, `_base`. The joint observers use memo's multi-choice syntax `chooses(x in X, y in Y, ...)` for the actor draw and `Pr[a, b]` to return the joint posterior; downstream code marginalizes for per-slider predictions.
- `test_model_compliance.py` — validation tests.

### LM elicitation (`model/lm/`)

- `client.py` — shared LM-call infrastructure: `get_ratings_concurrent` (thread-pooled fan-out + SDK retries), schema helpers, JSON parsing helpers, `load_api_key`. Used by every LM script.
- `prompts.py` — prompt templates shared across food + nonfood pipelines.
- `_features_dispatcher.py` — internal multi-mode helper for the canonical scripts. Not run directly.
- `score_canonical_features.py` (`--domain food|nonfood`) — access + effort per (scenario, action). Used by Study 1a. → `outputs/lm/lm_scenario_params{,_nonfood}.csv`.
- `score_canonical_v.py` (`--domain food|nonfood`) — signed-valence V per (scenario, action, motivation). Used by Study 1a. → `outputs/lm/lm_scenario_v{,_nonfood}.csv`.
- `score_effort_features.py` — Study 1b's access + effort and an effort-marginal access table. → `outputs/lm/lm_scenario_params_effort{,_marginal}.csv`.
- `score_3act_features.py` — Studies 2/3/4 access + effort (16 × 2 effort × 3 actions = 96 cells) plus a 48-row effort-marginal access table for Study 3a. → `outputs/lm/lm_scenario_params_3act{,_marginal}.csv`.
- `score_3act_v.py` (`--domain food|nonfood`) — Studies 2/3/4 signed-valence V (16 × 3 × 2). → `outputs/lm/lm_scenario_v_3act{,_nonfood}.csv`.

### Forward planning (`model/forward/`)

- `_shared.py` — NLL/AIC/BIC, `_fit_with_adam`, prediction/fit helpers (`predict_canonical_*` / `fit_canonical_*` / `predict_effort_*` / `fit_effort_*`), data loaders, `build_canonical_cells(scenario_labels, n_actions)` for enumerating prediction cells, and `run_fit_and_save_results` / `run_predict_and_save_fits` orchestration helpers.
- `fit_<slug>.py` — fits the three actor ablations; writes `outputs/<slug>/fit_results.csv`. One per forward experiment: `food_forw_intimacy_desire`, `food_forw_intimacy_effort`, `nonfood_forw_intimacy_desire`.
- `predict_<slug>.py` — reads the fit results, computes per-cell p_action, writes `outputs/<slug>/preds.csv`.

### Inverse planning (`model/inverse/`)

Five active inverse experiments, all on the 3-action set: `food_inv_intimacy_3act`, `food_inv_effort_3act`, `food_inv_desire_3act`, `food_inv_joint_de_3act`, `food_inv_joint_di_3act`. Each has its own `fit_<slug>.py` + `predict_<slug>.py` pair.

- `_helpers.py` — `_fit_alpha_observer`, `compute_intimacy_nll`, `compute_reward_nll`. Single-target 3-action fit wrappers: `fit_intimacy_3act_observer`, `fit_effort_3act_observer`, `fit_desire_3act_observer` — each composes `_fit_alpha_observer` with a posterior slicer appropriate to the 5-D table shape `(action, scenario, intimacy/rel, reward, effort)`. Joint fit helpers `fit_joint_de_3act_observer` and `fit_joint_di_3act_observer` marginalize the joint observer table to each slider judgment and sum the two binary cross-entropies. Plus table-kwargs helpers (`intimacy_3act_table_kwargs`, `effort_3act_table_kwargs`, `desire_3act_table_kwargs`, `joint_3act_table_kwargs`) and `load_fitted_params` / `load_fitted_alpha_observer`.
- `fit_<slug>.py` — fits the three observer ablations with frozen actor weights from `food_forw_intimacy_desire/fit_results.csv` and writes `outputs/<slug>/fit_results.csv`. Study 3a uses effort-marginal access since its observer doesn't see the effort paragraph.
- `predict_<slug>.py` — reads the fit results, runs the observer on a per-scenario grid, writes `outputs/<slug>/preds_<variant>.npy` (the raw table per variant) and a summary CSV.

### Cross-validation (`model/cv/`)

All model-vs-human correlations reported in the analysis qmds are out-of-sample, from leave-one-scenario-out (LOSO) CV. The analysis qmds load CV-prediction CSVs (`outputs/<slug>/cv_preds[_summary].csv`) as the source for all plots.

- `_forward_dispatcher.py` — joint LOSO logic for the 3 forward experiments (config-driven).
- `_alt_dispatcher.py` — joint LOSO logic for legacy alt-shown inverse experiments. The 3-action CV scripts (`cv_<slug>.py` for the 5 new inverse experiments) are currently stubs pending the full LOSO loop.
- `cv_<slug>.py` — one per experiment, runs LOSO and writes `outputs/<slug>/cv_folds.csv` + `cv_preds.csv` (forward) or `cv_preds_summary.csv` (inverse).

Forward CV refits actor weights ($w_v, w_d, w_e, \gamma$) per fold. Inverse CV (when the 3-action loop lands) will refit only $\alpha_\mathrm{obs}$, freezing actor weights from the forward fit.

The non-CV `fit_*` / `predict_*` pipelines still produce all-data fits; AIC and fitted-parameter tables in the qmds use the all-data fit, but all model-vs-human displays use the CV predictions.

### Outputs (`model/outputs/`)

Grouped by experiment slug. For every experiment, look in `outputs/<slug>/` for:
- `fit_results.csv` — fitted parameters + AIC/BIC/r per ablation.
- `preds.csv` — forward only — per-cell predictions.
- `preds_<variant>.npy`, `preds_summary.csv` — inverse only — per-variant prediction arrays and a summary CSV.
- `cv_folds.csv` — per-fold fit results from LOSO CV.
- `cv_preds.csv` — forward only — per-trial held-out predictions.
- `cv_preds_summary.csv` — inverse only — held-out per-condition summary.

LM-elicited tables live in `outputs/lm/` (`lm_scenario_params*.csv`, `lm_scenario_v*.csv`).

Preregistration documents are in `model/preregs/`. Sandboxed/experimental code is in `model/sandbox/`.

## Commands

LLM-derived scenario parameters (prerequisite for all fits; requires `TOGETHER_API_KEY` in `.env`; Llama-3.3-70B via Together AI, 10 runs averaged):

```bash
uv run python model/lm/score_canonical_features.py                # → lm_scenario_params.csv (Study 1a access+effort)
uv run python model/lm/score_canonical_features.py --domain nonfood
uv run python model/lm/score_canonical_v.py                       # → lm_scenario_v.csv (Study 1a signed-valence V)
uv run python model/lm/score_canonical_v.py --domain nonfood
uv run python model/lm/score_effort_features.py                   # → lm_scenario_params_effort{,_marginal}.csv (Study 1b)
uv run python model/lm/score_3act_features.py                     # → lm_scenario_params_3act{,_marginal}.csv (Studies 2/3/4)
uv run python model/lm/score_3act_v.py                            # → lm_scenario_v_3act.csv (Studies 2/3/4 V)
```

Forward-planning fits + predictions (Studies 1a, 1b, plus non-food forward):

```bash
uv run python model/forward/fit_food_forw_intimacy_desire.py
uv run python model/forward/predict_food_forw_intimacy_desire.py
uv run python model/forward/fit_food_forw_intimacy_effort.py
uv run python model/forward/predict_food_forw_intimacy_effort.py
uv run python model/forward/fit_nonfood_forw_intimacy_desire.py
uv run python model/forward/predict_nonfood_forw_intimacy_desire.py
```

Inverse-planning fits + predictions (Studies 2, 3a, 3b, 4a, 4b):

```bash
uv run python model/inverse/fit_food_inv_intimacy_3act.py   # Study 2
uv run python model/inverse/predict_food_inv_intimacy_3act.py
uv run python model/inverse/fit_food_inv_effort_3act.py     # Study 3a
uv run python model/inverse/predict_food_inv_effort_3act.py
uv run python model/inverse/fit_food_inv_desire_3act.py     # Study 3b
uv run python model/inverse/predict_food_inv_desire_3act.py
uv run python model/inverse/fit_food_inv_joint_de_3act.py   # Study 4a
uv run python model/inverse/predict_food_inv_joint_de_3act.py
uv run python model/inverse/fit_food_inv_joint_di_3act.py   # Study 4b
uv run python model/inverse/predict_food_inv_joint_di_3act.py
```

LOSO cross-validation (forward only for now; inverse 3-action CV scripts are stubs):

```bash
uv run python model/cv/cv_food_forw_intimacy_desire.py
uv run python model/cv/cv_food_forw_intimacy_effort.py
uv run python model/cv/cv_nonfood_forw_intimacy_desire.py
```

Tests:

```bash
uv run python model/test_model_compliance.py
```
