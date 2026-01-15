# Sandbox - Legacy Model Development Files

This folder contains older model development code from September-October 2025, prior to the final model implementation in `model_utils.py` and `fit_forward_planning.py`.

## Files

### Core Model Files

**`model.py`**
- Early scenario-specific model fitting code using the `memo` DSL
- Loads pilot data (risk, effort, discomfort, priors) from `data/pilots/`
- Defines two actor models:
  - `vanilla_actor`: Utility based on reward and risk (no relationship scaling)
  - `relationship_actor`: Utility with relationship-dependent cost scaling
- Used with `fit_planning-1.ipynb` for fitting to pilot data

**`fit_planning-1.ipynb`**
- Fits `vanilla` and `relationship` models to pilot planning data (`data/pilots/planning-1/`)
- Uses gradient descent (Adam optimizer) to minimize negative log-likelihood
- Outputs predictions to `main_trials_tidy_with_preds.csv`

### Prediction Generation

**`generate_model_preds.ipynb`**
- Generates model predictions using functions from `model_utils.py` (parent directory)
- Outputs three CSV files with predictions for toy parameter values (all weights = 1):
  - `actor_preds.csv` - Forward planning predictions
  - `observer_intimacy_preds.csv` - Inverse planning (infer intimacy) predictions
  - `observer_reward_preds.csv` - Inverse planning (infer reward) predictions

### Output CSVs

**`actor_preds.csv`**
- Actor model predictions for 3 model types: `discomfort_only`, `vanilla_inv_plan`, `full_model`
- Columns: alpha, w_e, w_r, w_c, action, intimacy, reward, p_action, model

**`observer_intimacy_preds.csv`**
- Observer model predictions for inferring intimacy from observed actions
- Contains posterior density over intimacy levels (0-100) for each action/reward condition

**`observer_reward_preds.csv`**
- Observer model predictions for inferring reward from observed actions
- Contains posterior probability of high vs low reward for each action/intimacy condition

### Early Development Notebooks

**`2025-09_actor-model.ipynb`**
- Initial development of forward planning (actor) models
- Explores vanilla vs relationship-aware utility functions
- Precursor to the models in `model_utils.py`

**`2025-09_observer-model.ipynb`**
- Initial development of inverse planning (observer) models
- Implements Bayesian inference over intimacy and reward
- Defines observer models that reason about actor's choices

## Relationship to Current Code

These files were superseded by:
- `model/model_utils.py` - Final model implementations with standardized API
- `model/fit_forward_planning.py` - Production model fitting for main experiment data
- `analysis/exp-1-data-analysis.qmd` - Main analysis using fitted models

The pilot data analysis using these files is in `analysis/2025-10_pilots.Rmd`.
