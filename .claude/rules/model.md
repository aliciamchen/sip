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

## Where the utility values come from

All three components — V, access, effort — are LLM-elicited per scenario by `model/lm/score_canonical_features.py` (Llama-3.3-70B via Together AI, 10 runs averaged). The Together calls themselves go through `model/lm/client.py`, which fans NUM_RUNS calls across a thread pool, constrains output to a JSON schema via `response_format`, retries transient errors at the SDK layer, and checkpoints per-scenario; new LM call sites should reuse `get_ratings_concurrent` + the schema helpers (`numeric_action_schema`, `alternatives_array_schema`) rather than calling Together directly. CSV outputs include both `n_runs_*` and `n_failures_*` columns.

- **V**: `--feature v` mode produces `lm_scenario_v.csv` (16 × 4 × 2 — scenario × action × motivation, signed [-1, +1]).
- **access**, **effort**: default mode produces `lm_scenario_params.csv` (16 × 4 each, normalized [0, 2] and [0, 1]).
- **alternatives V** (no-alt observers only): `--feature v_alternatives` mode produces `lm_alternatives_v_food_inv-intimacy_desire_noalt.csv`, scoring V for each LM-generated alternative under both motivation states.

`tables.py` loads these into `LLM_TABLES` (access/effort canonical) at import; `load_lm_v(domain)` lazily loads the V table; `load_padded_lm_tables()` builds the (16, 4, 2, MAX_ACTIONS) padded tables (access, effort, v, prior) used by the no-alt observers. If any required CSV is missing, the loader raises FileNotFoundError or returns None.

Every memo model takes the scenario tables as arguments (`access_table: ...`, `effort_table: ...`, and `v_table: ...` for the full and base variants) and has `scenario_idx: Scenarios` as a dimension, so predictions vary by scenario. The `discomfort_only` ablation is V-independent and doesn't take `v_table`.

## Layout

`model/` is organized so that every script's name tells you what it does. Per-experiment scripts live in `forward/`, `inverse/`, and `cv/` subfolders, named after the experiment slug (e.g. `forward/fit_food_forw_intimacy_desire.py`). LM-elicitation scripts live in `lm/`, named after the output they produce. Shared math is in four core modules at the top of `model/`.

### Core math (one copy, shared across all experiments)

- `tables.py` — `Scenarios` / `RewardConditions` / `RelationshipConditions` / `EffortConditions` / `PaddedActionSlots` enums, `SCENARIO_LABELS`, `LLM_TABLES`, `LLM_TABLES_EFFORT`, padded-table loaders (`load_padded_lm_tables`, `load_padded_lm_tables_relationship`), domain-asset loader, V table loader.
- `utility.py` — jit-compiled utility functions: `get_utility_full / discomfort_only / base` (with `_disc`, `_padded`, `_padded_rel` siblings) plus the effort-experiment counterparts. Dimension-agnostic — used by both canonical 4-action and effort 2-action actors.
- `actors.py` — actor memo models: `actor_forw_*` (forward), `actor_discrete_*` and `actor_continuous_*` (inverse), `actor_continuous_*_padded` and `_padded_rel` (no-alt), plus `actor_forw_effort_*` and `actor_continuous_effort_*` (effort experiment).
- `observers.py` — observer memo models: `observer_intimacy_*` and `observer_reward_*` (alt-shown), `observer_intimacy_*_padded` and `observer_reward_*_padded_rel` (no-alt), `observer_intimacy_effort_*` and `observer_effort_intimacy_*` (effort experiment).
- `test_model_compliance.py` — validation tests.

### LM elicitation (`model/lm/`)

- `client.py` — shared LM-call infrastructure: `get_ratings_concurrent` (thread-pooled fan-out + SDK retries), schema helpers (`numeric_action_schema`, `alternatives_array_schema`), JSON parsing helpers, `load_api_key`. Used by every LM script.
- `prompts.py` — prompt templates shared across food + nonfood pipelines.
- `_features_dispatcher.py`, `_alternatives_dispatcher.py` — internal multi-mode helpers that the per-output scripts below call into. Not run directly.
- `score_canonical_features.py` (`--domain food|nonfood`) — access + effort per (scenario, action). → `outputs/lm/lm_scenario_params{,_nonfood}.csv`.
- `score_canonical_v.py` (`--domain food|nonfood`) — signed-valence V per (scenario, action, motivation). → `outputs/lm/lm_scenario_v{,_nonfood}.csv`.
- `score_alternative_features.py` (`--conditioning motivation|relationship`) — access + effort for LM-generated alternatives. → `outputs/lm/lm_alternatives_features_food_inv-intimacy_desire_noalt.csv` or `lm_alternatives_features_food_inv-desire_intimacy_noalt.csv`.
- `score_alternative_v.py` (`--conditioning motivation|relationship`) — V for those alternatives. → `outputs/lm/lm_alternatives_v_food_inv-intimacy_desire_noalt.csv` or `lm_alternatives_v_food_inv-desire_intimacy_noalt.csv`.
- `score_effort_features.py` — produces both `lm_scenario_params_effort.csv` (effort-conditional access + effort) and `lm_scenario_params_effort_marginal.csv` (effort-marginal access only, used by `food_inv-effort_intimacy_alt`).
- `generate_alternatives_motivation.py` — motivation-conditioned LM alternatives. → `outputs/lm/lm_alternatives_food_inv-intimacy_desire_noalt.csv`.
- `generate_alternatives_relationship.py` — relationship-conditioned LM alternatives. → `outputs/lm/lm_alternatives_food_inv-desire_intimacy_noalt.csv`.

### Forward planning (`model/forward/`)

- `_shared.py` — NLL/AIC/BIC, `_fit_with_adam`, `predict_canonical_*` / `fit_canonical_*` / `predict_effort_*` / `fit_effort_*`, data loaders, `run_fit_and_save_results` / `run_predict_and_save_fits` orchestration helpers.
- `fit_<slug>.py` — fits the three actor ablations for that experiment; writes `outputs/<slug>/fit_results.csv`. One per forward experiment: `food_forw_intimacy_desire`, `food_forw_intimacy_effort`, `nonfood_forw_intimacy_desire`.
- `predict_<slug>.py` — reads `outputs/<slug>/fit_results.csv`, computes per-cell p_action for each `(scenario, action, intimacy, IV)` cell, writes `outputs/<slug>/preds.csv`. Predictions are per-cell because the model's prediction for a given cell is identical across subjects in that cell.

### Inverse planning (`model/inverse/`)

- `_helpers.py` — `_fit_alpha_observer`, `compute_intimacy_nll`, `compute_reward_nll`, padded joint fitters (`fit_padded_joint_intimacy`, `fit_padded_joint_desire`), data loaders for all 6 inverse experiments, `load_fitted_params` (forward → inverse), `load_fitted_alpha_observer` (inverse fit → predict), variant registries (`ACCESS_VARIANTS`, `PADDED_VARIANTS_INTIMACY`, `PADDED_VARIANTS_REWARD`, `ACCESS_VARIANTS_EFFORT`, `ACCESS_VARIANTS_EFFORT_INFERRED`), table-kwargs helpers (`alt_table_kwargs`, `effort_table_kwargs`, `effort_marginal_table_kwargs`).
- `fit_<slug>.py` — fits the three observer ablations for that experiment. Alt-shown experiments fit only `alpha_observer` (frozen actor weights from upstream forward fit). No-alt experiments **jointly fit all actor weights + α_observer** because the padded action space differs from the alt-shown 4-action space and Exp 1's weights don't transplant. Effort experiments fit only `alpha_observer` with actor weights frozen from `food_forw_intimacy_effort/fit_results.csv`. The `food_inv-effort_intimacy_alt` observer uses **effort-marginal access** (`LLM_TABLES_EFFORT['access_marg']`) because that observer doesn't see the effort paragraph.
- `predict_<slug>.py` — reads `outputs/<slug>/fit_results.csv`, runs the observer on the per-scenario grid, writes `outputs/<slug>/preds_full.csv` and `outputs/<slug>/preds_summary.csv`. The summary's column depends on the experiment: `expected_intimacy` for intimacy-inference, `p_high_reward` (×100) for desire-inference, `p_effort_high` (×100) for effort-inference.

Six inverse experiments: `food_inv-intimacy_desire_alt`, `food_inv-desire_intimacy_alt`, `food_inv-intimacy_desire_noalt`, `food_inv-desire_intimacy_noalt`, `food_inv-intimacy_effort_alt`, `food_inv-effort_intimacy_alt`. Each gets its own `fit_<slug>.py` + `predict_<slug>.py` pair.

### Cross-validation (`model/cv/`)

All model-vs-human correlations reported in the analysis qmds are out-of-sample, from leave-one-scenario-out (LOSO) CV. The analysis qmds load CV-prediction CSVs (`outputs/<slug>/cv_preds[_summary].csv`) as the source for all plots.

- `_forward_dispatcher.py` — joint LOSO logic for the 3 forward experiments (config-driven). Imported by the per-experiment forward CV scripts.
- `_alt_dispatcher.py` — joint LOSO logic for the 2 alt-shown inverse experiments. Provides `main_intimacy_alt()` and `main_desire_alt()`.
- `cv_<slug>.py` — one per experiment, runs LOSO and writes `outputs/<slug>/cv_folds.csv` + `cv_preds.csv` (forward) or `cv_preds_summary.csv` (inverse).

Forward CV refits actor weights ($w_v, w_d, w_e, \gamma$) per fold. Inverse alt CV refits only $\alpha_\mathrm{obs}$ (actor frozen from all-data forward fit). Inverse no-alt CV jointly refits all actor weights + $\alpha_\mathrm{obs}$ per fold (same justification as the all-data joint fits).

The non-CV `fit_*` / `predict_*` pipelines still produce all-data fits; AIC and fitted-parameter tables in the qmds use the all-data fit, but all model-vs-human displays use the CV predictions.

### Outputs (`model/outputs/`)

Grouped by experiment slug. For every experiment, look in `outputs/<slug>/` for:
- `fit_results.csv` — fitted parameters + AIC/BIC/r per ablation.
- `preds.csv` — forward only — per-cell predictions (one row per `(scenario, action, intimacy, IV)` cell).
- `preds_full.csv`, `preds_summary.csv` — inverse only — per-(scenario, condition) posteriors and summary scalars.
- `cv_folds.csv` — per-fold fit results from LOSO CV.
- `cv_preds.csv` — forward only — per-trial held-out predictions.
- `cv_preds_summary.csv` — inverse only — held-out per-condition summary.

LM-elicited tables live in `outputs/lm/` (e.g. `lm_scenario_params.csv`, `lm_scenario_v.csv`, `lm_alternatives*.csv`).

Preregistration documents are in `model/preregs/`. Legacy/experimental code is in `model/sandbox/`.

## Commands

LLM-derived scenario parameters (prerequisite for all fits; requires `TOGETHER_API_KEY` in `.env`; Llama-3.3-70B via Together AI, 10 runs averaged):

```bash
uv run python model/lm/score_canonical_features.py                                  # → model/outputs/lm/lm_scenario_params.csv (food access+effort)
uv run python model/lm/score_canonical_features.py --domain nonfood                 # → model/outputs/lm/lm_scenario_params_nonfood.csv
uv run python model/lm/score_canonical_v.py                                         # → model/outputs/lm/lm_scenario_v.csv (food signed-valence V)
uv run python model/lm/score_canonical_v.py --domain nonfood                        # → model/outputs/lm/lm_scenario_v_nonfood.csv
uv run python model/lm/generate_alternatives_motivation.py                          # → model/outputs/lm/lm_alternatives_food_inv-intimacy_desire_noalt.csv
uv run python model/lm/score_alternative_features.py                                # → model/outputs/lm/lm_alternatives_features_food_inv-intimacy_desire_noalt.csv
uv run python model/lm/score_alternative_v.py                                       # → model/outputs/lm/lm_alternatives_v_food_inv-intimacy_desire_noalt.csv
uv run python model/lm/generate_alternatives_relationship.py                        # → model/outputs/lm/lm_alternatives_food_inv-desire_intimacy_noalt.csv
uv run python model/lm/score_alternative_features.py --conditioning relationship    # → model/outputs/lm/lm_alternatives_features_food_inv-desire_intimacy_noalt.csv
uv run python model/lm/score_alternative_v.py --conditioning relationship           # → model/outputs/lm/lm_alternatives_v_food_inv-desire_intimacy_noalt.csv
uv run python model/lm/score_effort_features.py                                     # → model/outputs/lm/lm_scenario_params_effort{,_marginal}.csv
```

Forward-planning fits + predictions:

```bash
uv run python model/forward/fit_food_forw_intimacy_desire.py
uv run python model/forward/predict_food_forw_intimacy_desire.py
uv run python model/forward/fit_food_forw_intimacy_effort.py
uv run python model/forward/predict_food_forw_intimacy_effort.py
uv run python model/forward/fit_nonfood_forw_intimacy_desire.py
uv run python model/forward/predict_nonfood_forw_intimacy_desire.py
```

Inverse-planning fits + predictions (one fit + one predict per experiment):

```bash
uv run python "model/inverse/fit_food_inv-intimacy_desire_alt.py"
uv run python "model/inverse/predict_food_inv-intimacy_desire_alt.py"
uv run python "model/inverse/fit_food_inv-desire_intimacy_alt.py"
uv run python "model/inverse/predict_food_inv-desire_intimacy_alt.py"
uv run python "model/inverse/fit_food_inv-intimacy_desire_noalt.py"           # joint fit (all weights + α_observer)
uv run python "model/inverse/predict_food_inv-intimacy_desire_noalt.py"
uv run python "model/inverse/fit_food_inv-desire_intimacy_noalt.py"           # joint fit (relationship-keyed)
uv run python "model/inverse/predict_food_inv-desire_intimacy_noalt.py"
uv run python "model/inverse/fit_food_inv-intimacy_effort_alt.py"
uv run python "model/inverse/predict_food_inv-intimacy_effort_alt.py"
uv run python "model/inverse/fit_food_inv-effort_intimacy_alt.py"
uv run python "model/inverse/predict_food_inv-effort_intimacy_alt.py"
```

(Hyphens in the inverse experiment slugs require quoting on the shell.)

LOSO cross-validation (16 folds × 3 variants per experiment):

```bash
uv run python model/cv/cv_food_forw_intimacy_desire.py
uv run python model/cv/cv_food_forw_intimacy_effort.py
uv run python model/cv/cv_nonfood_forw_intimacy_desire.py
uv run python "model/cv/cv_food_inv-intimacy_desire_alt.py"
uv run python "model/cv/cv_food_inv-desire_intimacy_alt.py"
uv run python "model/cv/cv_food_inv-intimacy_desire_noalt.py"
uv run python "model/cv/cv_food_inv-desire_intimacy_noalt.py"
uv run python "model/cv/cv_food_inv-intimacy_effort_alt.py"
uv run python "model/cv/cv_food_inv-effort_intimacy_alt.py"
```

Tests:

```bash
uv run python model/test_model_compliance.py
```
