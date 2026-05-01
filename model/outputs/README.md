# Model outputs codebook

Outputs are grouped by experiment slug:

```
outputs/
├── lm/                                # LM-elicited tables (lm_*.csv)
└── <experiment_slug>/                 # one folder per experiment
    ├── fit_results.csv
    ├── preds.csv                      # forward only — per-cell predictions (one row per (scenario, action, intimacy, IV) cell)
    ├── preds_full.csv, preds_summary.csv  # inverse only — per-condition posteriors
    ├── cv_folds.csv                   # per-fold fit results from LOSO CV
    ├── cv_preds.csv                   # forward only — per-trial held-out predictions
    └── cv_preds_summary.csv           # inverse only — held-out per-condition summary
```

The 9 experiment slugs: `food_forw_intimacy_desire`, `food_forw_intimacy_effort`, `nonfood_forw_intimacy_desire`, `food_inv-intimacy_desire_alt`, `food_inv-desire_intimacy_alt`, `food_inv-intimacy_desire_noalt`, `food_inv-desire_intimacy_noalt`, `food_inv-intimacy_effort_alt`, `food_inv-effort_intimacy_alt`.

The sections below document the columns of each output type.

## Terminology note

In the inverse-planning experiments, internal variable names use "reward" (e.g., `p_high_reward`, `reward_condition`) or "motivation" rather than "desire" — we changed the terminology to "desire" after running the experiments, for clarity.

## lm/lm_scenario_params.csv

LLM-generated per-scenario values for access and effort. Produced by `model/lm/score_canonical_features.py`. Reward (V) is stipulated in `model/utility.py`, not in this file.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `access_raw`, `access_raw_std` | Mean and std of raw access ratings (0-6 scale, 10 runs) |
| `effort_raw`, `effort_raw_std` | Same for effort |
| `access` | Normalized access ([0, 2]) |
| `effort` | Normalized effort ([0, 1]) |
| `n_runs_access`, `n_runs_effort` | Number of successful LLM runs |

Note: the CSV currently in the repo also has `reward_low_raw`, `reward_high_raw`, `reward_low`, `reward_high`, and `n_runs_reward_*` columns from a previous schema; these are ignored by `tables.load_lm_scenario_params` and will be dropped the next time `lm/score_canonical_features.py` is run.

## <forward_slug>/preds.csv

Per-cell model predictions for forward planning. One row per `(scenario, action, intimacy, IV)` cell — predictions are identical across subjects in the same cell, so the per-cell format is the natural granularity. Cell counts: 16 × 4 × 4 × 2 = 512 for canonical 4-action; 16 × 2 × 4 × 2 = 256 for effort 2-action.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `scenario_idx` | Numeric scenario index (0-15) |
| `action` | Action index (0-3 for canonical; 0-1 for effort, with `action_csv` giving the 1-2 label that matches the data CSV) |
| `intimacy` | Intimacy level (0, 50, 75, 100) |
| `intimacy_scaled` | Intimacy normalized to [0, 1] |
| `motivation` (canonical) or `effort` (effort) | The contextual IV ("low" or "high") |
| `motivation_idx` (canonical) or `effort_idx` (effort) | Integer index for the IV |
| `pred_full` | Predicted probability from the Full model |
| `pred_discomfort_only` | Predicted probability from the Discomfort-only ablation |
| `pred_base` | Predicted probability from the Base ablation |

## <forward_slug>/fit_results.csv

Summary of fitted forward planning models (3 rows — one per ablation).

| Column | Description |
|--------|-------------|
| `model` | Model name (`full`, `discomfort_only`, `base`) |
| `nll` | Negative log-likelihood |
| `n_params` | Number of free parameters |
| `aic` | Akaike Information Criterion |
| `bic` | Bayesian Information Criterion |
| `r` | Pearson correlation with human data |
| `r_ci_lower`, `r_ci_upper` | 95% CI bounds for correlation |
| `param_alpha` | Fitted softmax inverse temperature (fixed to 1 during fitting) |
| `param_w_v` | Fitted food-utility weight (Full + Base variants) |
| `param_w_d` | Fitted access-discomfort weight (Full + Discomfort-only variants) |
| `param_w_e` | Fitted effort weight (Full + Base variants) |

## <inverse_slug>/fit_results.csv (alt-shown)

Summary of fitted inverse planning (observer) models (6 rows — 3 ablations × 2 experiments). Observer parameters (α_observer) are fit with frozen actor weights from `<slug>/fit_results.csv`.

| Column | Description |
|--------|-------------|
| `model` | Model name (`full`, `discomfort_only`, `base`) |
| `experiment` | Experiment (`intimacy` or `reward`) |
| `alpha_observer` | Fitted observer inverse temperature |
| `nll` | Negative log-likelihood |
| `n_params` | Number of free parameters |

## food_inv-intimacy_desire_alt/preds_summary.csv

Summarized model predictions for the alt-shown intimacy-inference experiment (`data/food_inv-intimacy_desire_alt/`). One row per scenario × action × motivation × model.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `reward_condition` | Motivation condition ("low" or "high") |
| `model` | Model name |
| `expected_intimacy` | Model's expected intimacy (0-100) |

## food_inv-intimacy_desire_alt/preds_full.csv

Full posterior distributions for alt-shown intimacy inference (101 intimacy levels per row set).

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `reward_condition` | Motivation condition |
| `intimacy` | Intimacy value (continuous 0-1) |
| `density` | Posterior density at this intimacy value |
| `model` | Model name |

## food_inv-desire_intimacy_alt/preds_summary.csv

Summarized model predictions for the alt-shown desire-inference experiment (`data/food_inv-desire_intimacy_alt/`).

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `intimacy_condition` | Intimacy level (0, 50, 75, 100) |
| `p_high_reward` | Model's predicted probability of high desire (0-100) |
| `model` | Model name |

## food_inv-desire_intimacy_alt/preds_full.csv

Full posterior distributions for alt-shown desire inference.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `intimacy_condition` | Intimacy level |
| `reward_condition` | Desire state ("low" or "high") |
| `density` | Posterior probability of this desire state |
| `model` | Model name |

## food_inv-intimacy_desire_noalt/fit_results.csv (and similar for desire_noalt)

Summary of jointly-fitted parameters for the no-alternatives-shown intimacy-inference experiment (`data/food_inv-intimacy_desire_noalt/`), using the padded observer with LM-generated counterfactual alternatives. One row per utility ablation. Unlike the alt-shown pipeline, actor weights are **not** frozen from the forward-planning fit — the padded observer reasons over a variable-length action set whose softmax competition structure differs from Exp 1's fixed four-action space, so all actor weights are refit jointly with α_observer on the no-alt data.

| Column | Description |
|--------|-------------|
| `model` | Model name (e.g., `full_padded`) |
| `experiment` | Always `intimacy_noalt` |
| `param_alpha` | Actor softmax temperature (fixed at 1) |
| `param_w_v`, `param_w_d`, `param_w_e` | Fitted utility weights (NaN where not applicable per ablation) |
| `alpha_observer` | Fitted observer inverse temperature |
| `nll` | Negative log-likelihood on all-data fit |
| `n_params` | Number of free parameters |

## food_inv-intimacy_desire_noalt/preds_summary.csv and preds_full.csv

No-alt intimacy-inference predictions. `_summary.csv` has one row per (scenario, observed_action, motivation, model) with `expected_intimacy` (0-100). `_full.csv` has one row per (scenario, observed_action, motivation, intimacy_level, model) with the posterior `density`.

## lm_alternatives.csv

LM-generated counterfactual action sets used by the no-alt padded observer. One row per (scenario, observed_action, motivation, alt_idx).

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `observed_action` | Canonical action that was observed (e.g., `action_0`) |
| `motivation` | Stipulated motivational state ("low" or "high") |
| `alt_idx` | Alternative index within the cell (0-based) |
| `action_text` | Text of the LM-generated alternative action |
| `is_share` | Binary tag: 1 if the alternative involves both characters eating the shared food, else 0 |

## lm_alternatives_features.csv

Access and effort features scored by the LM for each alternative in `lm_alternatives.csv`. One row per (scenario, observed_action, motivation, alt_idx) with the same identifier columns plus `access` (normalized to [0, 2]) and `effort` (normalized to [0, 1]).

## Cross-validation CSVs (cv_*)

All model-vs-human correlations reported in the analysis qmds are **out-of-sample**, pooled from leave-one-scenario-out (LOSO) CV. Each inverse-planning qmd loads a `cv_preds_summary.csv` per slug as the source for model plots. Per-fold fitted parameters are in the corresponding `*_folds.csv` files.

### <forward_slug>/cv_folds.csv

Per-fold forward-planning LOSO results. 48 rows (16 folds × 3 ablations).

| Column | Description |
|--------|-------------|
| `fold` | Held-out scenario index (0–15) |
| `held_out_scenario` | Held-out scenario label |
| `variant` | `full`, `discomfort_only`, `base` |
| `train_nll`, `test_nll`, `train_nll_per_trial`, `test_nll_per_trial`, `n_train`, `n_test` | Fit diagnostics |
| `param_alpha`, `param_w_v`, `param_w_d`, `param_w_e` | Per-fold fitted params (NaN where not applicable) |
| `test_cell_r` | Pearson r at (intimacy, motivation, action) cell-means on the held-out scenario |

### <forward_slug>/cv_preds.csv

Per-trial held-out forward-planning predictions, pooled across the 16 LOSO folds. One row per (trial, variant). Consumed by `analysis/food-forw-intimacy-desire-analysis.qmd` for all model-vs-human displays.

| Column | Description |
|--------|-------------|
| `fold`, `held_out_scenario`, `variant` | Fold metadata |
| `subject_id`, `intimacy`, `motivation`, `action` | Trial identifiers |
| `p_action` | Human response |
| `p_action_pred` | Model prediction (fit on the other 15 scenarios) |

### <inverse_slug>/cv_preds_summary.csv

Out-of-sample cell-mean predictions, same schema as the corresponding non-CV `inv_plan_*_preds_summary.csv` files. Populated by pooling held-out predictions across 16 folds. The alt-shown CVs refit only α_observer per fold (actor frozen from all-data Exp 1 fit); the no-alt CV refits all actor weights + α_observer jointly per fold.

### <inverse_slug>/cv_folds.csv

Per-fold fitted parameters for the inverse-planning CVs.

`cv_folds.csv` columns: `experiment` (intimacy or desire), `variant`, `fold`, `held_out_scenario`, `alpha_observer`, `train_nll`, `test_nll`, `n_train`, `n_test`.

`food_inv-intimacy_desire_noalt/cv_folds.csv` adds `param_w_v`, `param_w_d`, `param_w_e` columns (NaN where not applicable per ablation) — these are the jointly-refit utility weights per fold.

## Effort-experiment outputs

A parallel set of CSVs covers the effort-manipulation experiments (`food_forw_intimacy_effort/`, `food_inv-intimacy_effort_alt/`). The schemas mirror the canonical pipeline above, with two action indices (1 = non-saliva, 2 = saliva) instead of four and an `effort_condition` column ("low" or "high") in place of `reward_condition` / `motivation`. Reward is held fixed at high so V is constant across actions and `param_w_v` is non-identified — it appears in the fit-result tables for parallelism but stays near its initialization.

### lm_scenario_params_effort.csv

LLM-generated access and effort per (scenario, effort_condition, action). 64 rows (16 × 2 × 2). The LM is prompted with the full vignette plus the matching effort paragraph, so the manipulation lands in the ratings (the non-saliva action's effort rating should rise from `low` to `high`). Same column schema as `lm_scenario_params.csv`, with `effort_condition` added.

### food_forw_intimacy_effort/preds.csv and fit_results.csv

Per-cell predictions (256 rows) and per-variant fit summaries for `data/food_forw_intimacy_effort/`. Same column schema as the canonical `<forward_slug>/preds.csv` and `<forward_slug>/fit_results.csv`, with `effort` / `effort_idx` in place of `motivation` / `motivation_idx` and only two action indices (with `action_csv` giving the 1-2 label that matches the data CSV). `param_w_v` is non-identified and may print as the initial value.

### inverse_planning_intimacy_effort_fit_results.csv

Per-variant α_observer for `data/food_inv-intimacy_effort_alt/`. Same columns as `<inverse_slug>/fit_results.csv`; the `experiment` column is `intimacy_effort`. Actor weights are frozen from `food_forw_intimacy_effort/fit_results.csv` (NOT the canonical `food_forw_intimacy_desire` fit).

### food_inv-intimacy_effort_alt_preds_summary.csv / food_inv-intimacy_effort_alt_preds_full.csv

Posterior predictions for `food_inv-intimacy_effort_alt`. `_summary.csv` has one row per (scenario, action, effort_condition, model) with `expected_intimacy` (0-100). `_full.csv` adds the `intimacy` axis with the posterior `density` at each level.

### cv_loso_forward_effort.csv / cv_loso_preds_effort.csv

LOSO CV outputs for the effort forward planning. Same schema as `cv_loso_forward.csv` / `cv_loso_preds.csv`, with `effort` / `effort_condition` in place of `motivation`.

### food_inv-intimacy_effort_alt/cv_preds_summary.csv and cv_folds.csv

LOSO CV outputs for the effort inverse planning. `_preds_summary.csv` has the same schema as `food_inv-intimacy_effort_alt_preds_summary.csv` but populated by pooling held-out predictions across 16 folds. `_folds.csv` has per-fold α_observer and NLL — same schema as `cv_folds.csv` with `experiment = intimacy_effort`.
