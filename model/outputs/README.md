# Model outputs codebook

## Terminology note

In the inverse-planning experiments, internal variable names use "reward" (e.g., `p_high_reward`, `reward_condition`) or "motivation" rather than "desire" — we changed the terminology to "desire" after running the experiments, for clarity.

## lm_scenario_params.csv

LLM-generated per-scenario values for access and effort. Produced by `model/lm_scenario_params.py`. Reward is stipulated in `model_utils.py`, not in this file.

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
| `pred_access_full` | Predicted probability from the full access model |
| `pred_access_only` | Predicted probability from the access-only ablation |
| `pred_no_access` | Predicted probability from the no-access (base) ablation |

## forward_planning_fit_results.csv

Summary of fitted forward planning models (3 rows — one per ablation).

| Column | Description |
|--------|-------------|
| `model` | Model name (`access_full`, `access_only`, `no_access`) |
| `nll` | Negative log-likelihood |
| `n_params` | Number of free parameters |
| `aic` | Akaike Information Criterion |
| `bic` | Bayesian Information Criterion |
| `r` | Pearson correlation with human data |
| `r_ci_lower`, `r_ci_upper` | 95% CI bounds for correlation |
| `param_alpha` | Fitted softmax inverse temperature (fixed to 1 during fitting) |
| `param_w_v` | Fitted food-reward weight (access_full, no_access) |
| `param_w_d` | Fitted access-discomfort weight (access_full, access_only) |
| `param_w_e` | Fitted effort weight (access_full, no_access) |

## inverse_planning_fit_results.csv

Summary of fitted inverse planning (observer) models (6 rows — 3 ablations × 2 experiments).

| Column | Description |
|--------|-------------|
| `model` | Model name (`access_full`, `access_only`, `no_access`) |
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

Summary of fitted observer parameters for the no-alternatives-shown intimacy-inference experiment (`data/inv_plan_intimacy_noalt/`), using the padded observer with LM-generated counterfactual alternatives. One row per utility ablation.

| Column | Description |
|--------|-------------|
| `model` | Model name (e.g., `access_full_padded`) |
| `experiment` | Always `intimacy_noalt` |
| `alpha_observer` | Fitted observer inverse temperature |
| `nll` | Negative log-likelihood |
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
