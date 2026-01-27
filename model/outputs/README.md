# Model Outputs Codebook

## forward_planning_fits.csv

Per-trial model predictions for Experiment 1.

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
| `pred_full` | Predicted probability from full (social planning) model |
| `pred_vanilla` | Predicted probability from vanilla model |
| `pred_discomfort_only` | Predicted probability from discomfort-only model |

## forward_planning_fit_results.csv

Summary of fitted forward planning models.

| Column | Description |
|--------|-------------|
| `model` | Model name ("full", "vanilla", "discomfort_only") |
| `nll` | Negative log-likelihood |
| `n_params` | Number of free parameters |
| `aic` | Akaike Information Criterion |
| `bic` | Bayesian Information Criterion |
| `r` | Pearson correlation with human data |
| `r_ci_lower`, `r_ci_upper` | 95% CI bounds for correlation |
| `param_alpha` | Fitted inverse temperature parameter |
| `param_w_r` | Fitted reward weight |
| `param_w_d` | Fitted discomfort weight |
| `param_w_c` | Fitted cost weight |

## inverse_planning_fit_results.csv

Summary of fitted inverse planning (observer) models.

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `experiment` | Experiment ("intimacy" or "desire") |
| `alpha_observer` | Fitted observer inverse temperature |
| `beta` | Fitted reward-intimacy scaling (modified models only) |
| `nll` | Negative log-likelihood |
| `n_params` | Number of free parameters |

## inv_plan_intimacy_preds_summary.csv

Model predictions for Experiment 2a (intimacy inference).

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `reward_condition` | Motivation condition ("low" or "high") |
| `model` | Model name |
| `param_source` | Parameter source ("stipulated" or "fitted") |
| `expected_intimacy` | Model's predicted expected intimacy (0-100) |

## inv_plan_intimacy_preds_full.csv

Full posterior distributions for intimacy inference.

| Column | Description |
|--------|-------------|
| `action` | Action index (0-3) |
| `reward_condition` | Motivation condition |
| `intimacy` | Intimacy value (continuous 0-1) |
| `density` | Posterior density at this intimacy value |
| `model` | Model name |
| `param_source` | Parameter source |
| `scenario_label` | Scenario identifier |

## inv_plan_desire_preds_summary.csv

Model predictions for Experiment 2b (desire inference).

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier |
| `action` | Action index (0-3) |
| `intimacy_condition` | Intimacy level (0, 50, 75, 100) |
| `p_high_reward` | Model's predicted probability of high desire (0-100) |
| `model` | Model name |
| `param_source` | Parameter source |

## inv_plan_desire_preds_full.csv

Full posterior distributions for desire inference.

| Column | Description |
|--------|-------------|
| `action` | Action index (0-3) |
| `intimacy_condition` | Intimacy level |
| `reward_condition` | Desire state ("low" or "high") |
| `density` | Posterior probability of this desire state |
| `model` | Model name |
| `param_source` | Parameter source |
| `scenario_label` | Scenario identifier |
