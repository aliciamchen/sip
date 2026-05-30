# Model outputs codebook

Outputs are grouped by experiment slug, with legacy experiments under `legacy/`:

```
outputs/
├── lm/                                # LM-elicited tables (lm_*.csv)
├── <active_slug>/                     # one folder per active experiment
│   ├── fit_results.csv
│   ├── preds.csv                      # forward only — per-cell predictions (gitignored)
│   ├── preds_<variant>.npy            # inverse only — raw posterior arrays per variant (gitignored)
│   ├── preds_summary.csv              # inverse only — summary scalars (gitignored)
│   ├── cv_folds.csv                   # per-fold fit results from LOSO CV
│   ├── cv_preds.csv                   # forward only — per-trial held-out predictions
│   └── cv_preds_summary.csv           # inverse only — held-out per-condition summary
└── legacy/<slug>/                     # same layout, for legacy experiments
```

Active experiment slugs (4 inverse, all on the 3-action set):
`food_inv_desire` (Study 1a), `food_inv_joint_de` (1b), `food_inv_intimacy` (2a),
`food_inv_joint_ie` (2b). None have collected data yet, so their output folders are
created on first fit.

`outputs/legacy/` holds outputs for the legacy forward experiments
(`food_forw_intimacy_desire`, `food_forw_intimacy_effort`, `nonfood_forw_intimacy_desire`),
the six pre-3-action inverse experiments (`food_inv_*_alt`, `food_inv_*_noalt`), and the
Study 1a pilot (`food_inv_desire_pilot`). The forward and `_noalt` scripts write here via
their per-slug `make fit-/predict-/cv-` targets.

`preds.csv`, `preds_<variant>.npy`, and `preds_summary.csv` are the all-data (non-CV) predictions. They're written by the predict scripts but not read anywhere downstream — analysis qmds and any other consumers use `cv_preds.csv` / `cv_preds_summary.csv` instead, since reported correlations are out-of-sample. The non-CV files are gitignored (regenerate locally for debugging via `make predict-<slug>`).

The sections below document the columns of each output type.

## Terminology note

Internal variable names use "reward" (e.g., `p_high_reward`, `reward_condition`) or "motivation" rather than "desire" — the public manuscript uses "desire" but the data column names were fixed before that rename.

## LM-elicited tables (`outputs/lm/`)

### `lm_scenario_params.csv` — Study 1a access + effort

LLM-generated per-scenario values for the 4-action canonical set. Produced by `model/lm/score_canonical_features.py`. Reward (V) is stipulated in `model/utility.py`, not in this file.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0–3) |
| `access_raw`, `access_raw_std` | Mean and std of raw access ratings (0–6 scale, 10 runs) |
| `effort_raw`, `effort_raw_std` | Same for effort |
| `access` | Normalized access ([0, 2]) |
| `effort` | Normalized effort ([0, 1]) |
| `n_runs_access`, `n_runs_effort` | Number of successful LLM runs |
| `n_failures_access`, `n_failures_effort` | Number of LLM calls that did not parse |

`lm_scenario_params_nonfood.csv` has the same schema for the non-food set.

### `lm_scenario_v.csv` — Study 1a signed-valence V

LLM-generated signed valence per (scenario, action, motivation). Normalized to [-1, +1]. Produced by `model/lm/score_canonical_v.py`.

| Column | Description |
|--------|-------------|
| `scenario_label`, `action`, `motivation` | Identifiers |
| `v_raw`, `v_raw_std` | Mean and std of raw V ratings (signed -3..+3 scale) |
| `v` | Normalized V ([-1, +1]) |
| `n_runs`, `n_failures` | Run diagnostics |

`lm_scenario_v_nonfood.csv` has the same schema for non-food.

### `lm_scenario_params_effort.csv` — Study 1b access + effort

Same schema as `lm_scenario_params.csv` plus an `effort_condition` column ("low"/"high"). 64 rows (16 × 2 × 2). The LM is prompted with the vignette plus the matching effort paragraph so the manipulation lands in the ratings.

### `lm_scenario_params_effort_marginal.csv` — Study 3a-equivalent for the 2-action set

Effort-marginal access ratings (no effort paragraph shown). 32 rows (16 × 2). Used by the legacy `food_inv_effort_intimacy_alt` observer. Same column schema as the conditional version minus `effort_condition`.

### `lm_scenario_params_3act.csv` — Studies 2/3/4 access + effort

Same schema as `lm_scenario_params_effort.csv` but for the 3-action set. 96 rows (16 × 2 effort × 3 actions).

### `lm_scenario_params_3act_marginal.csv` — Study 3a effort-marginal access

48 rows (16 × 3 actions). Used as `LLM_TABLES_3ACT["access_marg"]` by the Study 3a observer, which doesn't see the effort paragraph.

### `lm_scenario_v_3act.csv` — Studies 2/3/4 signed-valence V

96 rows (16 × 3 × 2). Same column schema as `lm_scenario_v.csv`. `lm_scenario_v_3act_nonfood.csv` is the parallel non-food file (pending).

## Forward-planning outputs

### `<forward_slug>/preds.csv`

Per-cell forward-planning predictions. One row per `(scenario, action, intimacy, IV)` cell — predictions are identical across subjects in the same cell, so per-cell is the natural granularity. Cell counts: 16 × 4 × 4 × 2 = 512 for canonical 4-action (Study 1a); 16 × 2 × 4 × 2 = 256 for effort 2-action (Study 1b).

| Column | Description |
|--------|-------------|
| `scenario_label`, `scenario_idx` | Scenario identifier and index |
| `action` | Action index |
| `intimacy`, `intimacy_scaled` | Intimacy at 0/50/75/100 and its [0, 1] rescaling |
| `motivation`/`motivation_idx` (1a) or `effort`/`effort_idx` (1b) | Manipulated IV |
| `pred_full`, `pred_discomfort_only`, `pred_base` | Predicted probability per ablation |

### `<forward_slug>/fit_results.csv`

Summary of fitted forward planning models (3 rows — one per ablation).

| Column | Description |
|--------|-------------|
| `model` | `full`, `discomfort_only`, or `base` |
| `nll`, `n_params`, `aic`, `bic` | Fit diagnostics |
| `r`, `r_ci_lower`, `r_ci_upper` | Pearson correlation with human data + 95% CI |
| `param_alpha`, `param_w_v`, `param_w_d`, `param_w_e`, `param_gamma` | Fitted parameters (NaN where not applicable per ablation) |

## Inverse-planning outputs (3-action experiments)

### `<inverse_slug>/fit_results.csv`

Summary of fitted observer models. Actor weights are frozen from `food_forw_intimacy_desire/fit_results.csv`; only `alpha_observer` is fit.

| Column | Description |
|--------|-------------|
| `model` | `full`, `discomfort_only`, or `base` |
| `experiment` | Slug (e.g., `food_inv_intimacy_3act`) |
| `alpha_observer` | Fitted observer inverse temperature |
| `nll`, `n_params` | Fit diagnostics |

### `<inverse_slug>/preds_<variant>.npy`

Raw observer-table array, one file per ablation. Saved as numpy `.npy` because the joint observers (Studies 4a, 4b) return high-dimensional joint posteriors that don't flatten cleanly into a CSV. Shapes:

- Studies 2 / 3a / 3b: `(action=3, scenario=16, intimacy_axis, reward=2, effort=2)`. `intimacy_axis = 101` for Study 2 (continuous intimacy is inferred); `intimacy_axis = 4` for Studies 3a/3b (intimacy is one of 4 discrete levels).
- Studies 4a / 4b: same shape as their single-target counterparts, but the inferred axes hold the joint posterior.

### `<inverse_slug>/preds_summary.csv`

Lightweight summary of the prediction array per variant — currently logs the shape and sum as a sanity check. Full per-cell prediction tables will be added once data lands and a downstream consumer needs them.

## Cross-validation CSVs

All model-vs-human correlations reported in the analysis qmds are **out-of-sample**, pooled from leave-one-scenario-out (LOSO) CV. Each forward-planning analysis qmd loads a `cv_preds.csv` per slug as the source for model plots; inverse-planning qmds will load `cv_preds_summary.csv` once the 3-action CV loop is implemented (currently a stub).

### `<forward_slug>/cv_folds.csv`

Per-fold forward-planning LOSO results. 48 rows (16 folds × 3 ablations).

| Column | Description |
|--------|-------------|
| `fold`, `held_out_scenario` | Held-out scenario index and label |
| `variant` | `full`, `discomfort_only`, or `base` |
| `train_nll`, `test_nll`, `train_nll_per_trial`, `test_nll_per_trial`, `n_train`, `n_test` | Fit diagnostics |
| `param_alpha`, `param_w_v`, `param_w_d`, `param_w_e`, `param_gamma` | Per-fold fitted params (NaN where not applicable) |
| `test_cell_r` | Pearson r at cell means on the held-out scenario |

### `<forward_slug>/cv_preds.csv`

Per-trial held-out forward-planning predictions, pooled across the 16 LOSO folds. One row per (trial, variant). Consumed by the forward analysis qmds.

| Column | Description |
|--------|-------------|
| `fold`, `held_out_scenario`, `variant` | Fold metadata |
| `subject_id`, `intimacy`, `motivation`/`effort`, `action` | Trial identifiers |
| `p_action` | Human response |
| `p_action_pred` | Model prediction (fit on the other 15 scenarios) |
