# Model outputs codebook

## Terminology note

In the inverse-planning experiments, internal variable names use "reward" (e.g., `p_high_reward`, `reward_condition`) or "motivation" rather than "desire" — we changed the terminology to "desire" after running the experiments, for clarity.

## lm_scenario_params.csv

LLM-generated per-scenario values for access and effort. Produced by `model/lm_scenario_params.py`. Reward (V) is stipulated in `model_utils.py`, not in this file.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `access_raw`, `access_raw_std` | Mean and std of raw access ratings (0-6 scale, 10 runs) |
| `effort_raw`, `effort_raw_std` | Same for effort |
| `access` | Normalized access ([0, 2]) |
| `effort` | Normalized effort ([0, 1]) |
| `n_runs_access`, `n_runs_effort` | Number of successful LLM runs |

Note: the CSV currently in the repo also has `reward_low_raw`, `reward_high_raw`, `reward_low`, `reward_high`, and `n_runs_reward_*` columns from a previous schema; these are ignored by `model_utils.load_lm_scenario_params` and will be dropped the next time `lm_scenario_params.py` is run.

## lm_action_priors.csv

LLM-generated per-scenario action priors π(a|s) — how natural / expected each of the four actions is as a "default" in the scenario's setting, independent of the dyad's relationship or motivation. Produced by `model/lm_action_priors.py` (same Llama-3.3-70B pipeline as `lm_scenario_params.py`, 10 runs averaged). Consumed by the `_prior` variants of the actor/observer memos.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `prior_raw`, `prior_raw_std` | Mean and std of raw default-ness ratings (0-6 scale, 10 runs) |
| `prior` | Per-scenario distribution over the 4 actions: `(prior_raw + 0.1) / sum` (sums to 1 per scenario) |
| `n_runs` | Number of successful LLM runs |

## forward_planning_fits.csv

Per-trial model predictions for forward planning (`data/forw_plan/`). One row per subject × scenario × condition × action.

| Column | Description |
|--------|-------------|
| `subject_id` | Participant UUID |
| `scenario_label` | Scenario identifier |
| `intimacy` | Intimacy condition (0, 50, 75, 100) |
| `motivation` | Motivation condition ("low" or "high") |
| `action` | Action index (0-3) |
| `p_action` | Observed probability for this action |
| `intimacy_scaled` | Intimacy normalized to 0-1 scale |
| `reward_condition` | Binary reward condition (0 or 1) |
| `scenario_idx` | Numeric scenario index |
| `pred_access_full` | Predicted probability from the full model (uniform prior) |
| `pred_access_only` | Predicted probability from the discomfort-only ablation (uniform prior) |
| `pred_no_access` | Predicted probability from the base ablation (uniform prior) |
| `pred_access_full_prior` | Predicted probability from the full model with β-tempered LM prior (canonical Full model) |
| `pred_access_only_prior` | Predicted probability from the discomfort-only ablation with LM prior (canonical Discomfort-only) |
| `pred_no_access_prior` | Predicted probability from the base ablation with LM prior (canonical Base model) |

## forward_planning_fit_results.csv

Summary of fitted forward planning models (6 rows — 3 ablations × 2 prior types).

| Column | Description |
|--------|-------------|
| `model` | Model name (`access_full`, `access_only`, `no_access`, and `_prior` variants) |
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
| `param_beta_prior` | Fitted prior-tempering weight (LM-prior variants only) |

## inverse_planning_fit_results.csv

Summary of fitted inverse planning (observer) models (12 rows — 6 ablations × 2 experiments). Observer parameters (α_observer) are fit with frozen actor weights from `forward_planning_fit_results.csv`.

| Column | Description |
|--------|-------------|
| `model` | Model name (`access_full`, `access_only`, `no_access`, and `_prior` variants) |
| `experiment` | Experiment (`intimacy` or `reward`) |
| `alpha_observer` | Fitted observer inverse temperature |
| `nll` | Negative log-likelihood |
| `n_params` | Number of free parameters |

## inv_plan_intimacy_alt_preds_summary.csv

Summarized model predictions for the alt-shown intimacy-inference experiment (`data/inv_plan_intimacy_alt/`). One row per scenario × action × motivation × model.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `reward_condition` | Motivation condition ("low" or "high") |
| `model` | Model name |
| `expected_intimacy` | Model's expected intimacy (0-100) |

## inv_plan_intimacy_alt_preds_full.csv

Full posterior distributions for alt-shown intimacy inference (101 intimacy levels per row set).

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `reward_condition` | Motivation condition |
| `intimacy` | Intimacy value (continuous 0-1) |
| `density` | Posterior density at this intimacy value |
| `model` | Model name |

## inv_plan_desire_alt_preds_summary.csv

Summarized model predictions for the alt-shown desire-inference experiment (`data/inv_plan_desire_alt/`).

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `intimacy_condition` | Intimacy level (0, 50, 75, 100) |
| `p_high_reward` | Model's predicted probability of high desire (0-100) |
| `model` | Model name |

## inv_plan_desire_alt_preds_full.csv

Full posterior distributions for alt-shown desire inference.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `intimacy_condition` | Intimacy level |
| `reward_condition` | Desire state ("low" or "high") |
| `density` | Posterior probability of this desire state |
| `model` | Model name |

## inverse_planning_noalt_fit_results.csv

Summary of jointly-fitted parameters for the no-alternatives-shown intimacy-inference experiment (`data/inv_plan_intimacy_noalt/`), using the padded observer with LM-generated counterfactual alternatives. One row per utility ablation. Unlike the alt-shown pipeline, actor weights are **not** frozen from the forward-planning fit — the padded observer reasons over a variable-length action set whose softmax competition structure differs from Exp 1's fixed four-action space, so all actor weights are refit jointly with α_observer on the no-alt data.

| Column | Description |
|--------|-------------|
| `model` | Model name (e.g., `access_full_padded`) |
| `experiment` | Always `intimacy_noalt` |
| `param_alpha` | Actor softmax temperature (fixed at 1) |
| `param_w_v`, `param_w_d`, `param_w_e` | Fitted utility weights (NaN where not applicable per ablation) |
| `alpha_observer` | Fitted observer inverse temperature |
| `nll` | Negative log-likelihood on all-data fit |
| `n_params` | Number of free parameters |

## inv_plan_intimacy_noalt_preds_summary.csv / inv_plan_intimacy_noalt_preds_full.csv

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

## Cross-validation CSVs (cv_loso_*)

All model-vs-human correlations reported in the analysis qmds are **out-of-sample**, pooled from leave-one-scenario-out (LOSO) CV. Each inverse-planning qmd loads a `cv_loso_*_preds_summary.csv` as the source for model plots. Per-fold fitted parameters are in the corresponding `*_folds.csv` files.

### cv_loso_forward.csv

Per-fold forward-planning LOSO results. 48 rows (16 folds × 3 prior variants).

| Column | Description |
|--------|-------------|
| `fold` | Held-out scenario index (0–15) |
| `held_out_scenario` | Held-out scenario label |
| `variant` | `access_full_prior`, `access_only_prior`, `no_access_prior` |
| `train_nll`, `test_nll`, `train_nll_per_trial`, `test_nll_per_trial`, `n_train`, `n_test` | Fit diagnostics |
| `param_alpha`, `param_w_v`, `param_w_d`, `param_w_e`, `param_beta_prior` | Per-fold fitted params (NaN where not applicable) |
| `test_cell_r` | Pearson r at (intimacy, motivation, action) cell-means on the held-out scenario |

### cv_loso_preds.csv

Per-trial held-out forward-planning predictions, pooled across the 16 LOSO folds. One row per (trial, variant). Consumed by `analysis/forw-plan-analysis.qmd` for all model-vs-human displays.

| Column | Description |
|--------|-------------|
| `fold`, `held_out_scenario`, `variant` | Fold metadata |
| `subject_id`, `intimacy`, `motivation`, `action` | Trial identifiers |
| `p_action` | Human response |
| `p_action_pred` | Model prediction (fit on the other 15 scenarios) |

### cv_loso_inv_plan_{intimacy_alt,desire_alt,intimacy_noalt}_preds_summary.csv

Out-of-sample cell-mean predictions, same schema as the corresponding non-CV `inv_plan_*_preds_summary.csv` files. Populated by pooling held-out predictions across 16 folds. The alt-shown CVs refit only α_observer per fold (actor frozen from all-data Exp 1 fit); the no-alt CV refits all actor weights + α_observer jointly per fold.

### cv_loso_inverse_alt_folds.csv / cv_loso_inverse_noalt_folds.csv

Per-fold fitted parameters for the inverse-planning CVs.

`cv_loso_inverse_alt_folds.csv` columns: `experiment` (intimacy or desire), `variant`, `fold`, `held_out_scenario`, `alpha_observer`, `train_nll`, `test_nll`, `n_train`, `n_test`.

`cv_loso_inverse_noalt_folds.csv` adds `param_w_v`, `param_w_d`, `param_w_e` columns (NaN where not applicable per ablation) — these are the jointly-refit utility weights per fold.
