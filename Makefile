# Makefile for Saliva Inverse Planning Project
#
# This pipeline works without raw JSON data - processed CSVs are included in the repo.
# Data processing targets (data-*) are optional and require raw JSON files.

.PHONY: all help clean test data data-forw data-intimacy data-reward \
        fit fit-forward fit-inverse predictions \
        analysis analysis-exp1 analysis-exp2a analysis-exp2b analysis-combined

# Default target
all: fit predictions analysis

help:
	@echo "Available targets:"
	@echo ""
	@echo "  Main Pipeline (works without raw JSON):"
	@echo "    all              - Run full pipeline: fit models, generate predictions, render analysis"
	@echo "    fit              - Fit all models (forward + inverse planning)"
	@echo "    fit-forward      - Fit forward planning models (Exp 1)"
	@echo "    fit-inverse      - Fit inverse planning models (Exp 2a/2b)"
	@echo "    predictions      - Generate inverse planning predictions"
	@echo "    analysis         - Render all Quarto analysis documents"
	@echo ""
	@echo "  Individual Analysis Targets:"
	@echo "    analysis-exp1    - Render Exp 1 (forward planning) analysis"
	@echo "    analysis-exp2a   - Render Exp 2a (intimacy inference) analysis"
	@echo "    analysis-exp2b   - Render Exp 2b (reward inference) analysis"
	@echo "    analysis-combined - Render combined correlation analysis"
	@echo ""
	@echo "  Data Processing (requires raw JSON - for internal use):"
	@echo "    data             - Process all raw JSON to CSV"
	@echo "    data-forw        - Process forward planning data"
	@echo "    data-intimacy    - Process intimacy inference data"
	@echo "    data-reward      - Process reward inference data"
	@echo ""
	@echo "  Utilities:"
	@echo "    test             - Run model compliance tests"
	@echo "    clean            - Remove generated model outputs"

# =============================================================================
# Data Processing (optional - requires raw JSON files)
# =============================================================================

data: data-forw data-intimacy data-reward

data-forw:
	uv run python analysis/json_to_csv.py forw_plan

data-intimacy:
	uv run python analysis/json_to_csv.py inv_plan_intimacy

data-reward:
	uv run python analysis/json_to_csv.py inv_plan_reward

# =============================================================================
# Model Fitting
# =============================================================================

fit: fit-forward fit-inverse

# Forward planning model fitting
# Depends on processed CSV (included in repo)
model/outputs/forward_planning_fit_results.csv model/outputs/forward_planning_fits.csv: data/forw_plan/main_trials_long.csv model/fit_forward_planning.py model/model_utils.py
	uv run python model/fit_forward_planning.py

fit-forward: model/outputs/forward_planning_fit_results.csv

# Inverse planning model fitting
# Depends on forward planning results and processed CSVs
model/outputs/inverse_planning_fit_results.csv: model/outputs/forward_planning_fit_results.csv \
                                        data/inv_plan_intimacy/main_trials_long.csv \
                                        data/inv_plan_reward/main_trials_long.csv \
                                        model/fit_inverse_planning.py \
                                        model/model_utils.py
	uv run python model/fit_inverse_planning.py

fit-inverse: model/outputs/inverse_planning_fit_results.csv

# =============================================================================
# Model Predictions
# =============================================================================

model/outputs/inv_plan_intimacy_preds_summary.csv model/outputs/inv_plan_reward_preds_summary.csv: \
        model/outputs/forward_planning_fit_results.csv \
        model/outputs/inverse_planning_fit_results.csv \
        model/generate_inverse_planning_preds.py \
        model/model_utils.py
	uv run python model/generate_inverse_planning_preds.py

predictions: model/outputs/inv_plan_intimacy_preds_summary.csv

# =============================================================================
# Analysis
# =============================================================================

analysis: analysis-exp1 analysis-exp2a analysis-exp2b analysis-combined

analysis-exp1: model/outputs/forward_planning_fits.csv
	quarto render analysis/exp-1-analysis.qmd

analysis-exp2a: model/outputs/inv_plan_intimacy_preds_summary.csv
	quarto render analysis/exp-2a-inv-plan-intimacy-analysis.qmd

analysis-exp2b: model/outputs/inv_plan_reward_preds_summary.csv
	quarto render analysis/exp-2b-inv-plan-reward-analysis.qmd

analysis-combined: model/outputs/inv_plan_intimacy_preds_summary.csv model/outputs/inv_plan_reward_preds_summary.csv
	quarto render analysis/exp-2-combined-correlation.qmd

# =============================================================================
# Utilities
# =============================================================================

test:
	uv run python model/test_model_compliance.py

clean:
	rm -f model/outputs/forward_planning_fits.csv
	rm -f model/outputs/forward_planning_fit_results.csv
	rm -f model/outputs/inverse_planning_fit_results.csv
	rm -f model/outputs/inv_plan_intimacy_preds_full.csv
	rm -f model/outputs/inv_plan_intimacy_preds_summary.csv
	rm -f model/outputs/inv_plan_reward_preds_full.csv
	rm -f model/outputs/inv_plan_reward_preds_summary.csv
