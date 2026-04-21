# Makefile for Saliva Inverse Planning Project
#
# This pipeline works without raw JSON data - processed CSVs are included in the repo.
# Data processing targets (data-*) are optional and require raw JSON files.

.PHONY: all help clean test data data-forw data-intimacy-alt data-intimacy-noalt data-desire-alt \
        fit fit-forward fit-inverse predictions \
        analysis analysis-forw-plan analysis-inv-plan-intimacy-alt analysis-inv-plan-desire-alt \
        analysis-inv-plan-intimacy-noalt analysis-inv-plan-combined

# Default target
all: fit predictions analysis

help:
	@echo "Available targets:"
	@echo ""
	@echo "  Main Pipeline (works without raw JSON):"
	@echo "    all              - Run full pipeline: fit models, generate predictions, render analysis"
	@echo "    fit              - Fit all models (forward + inverse planning)"
	@echo "    fit-forward      - Fit forward planning actor models"
	@echo "    fit-inverse      - Fit alt-shown inverse planning observer models"
	@echo "    predictions      - Generate alt-shown inverse planning predictions"
	@echo "    analysis         - Render all Quarto analysis documents"
	@echo ""
	@echo "  Individual Analysis Targets:"
	@echo "    analysis-forw-plan               - Render forward planning analysis"
	@echo "    analysis-inv-plan-intimacy-alt   - Render alt-shown intimacy inference analysis"
	@echo "    analysis-inv-plan-desire-alt     - Render alt-shown desire inference analysis"
	@echo "    analysis-inv-plan-intimacy-noalt - Render no-alt intimacy inference analysis"
	@echo "    analysis-inv-plan-combined       - Render combined correlation analysis"
	@echo ""
	@echo "  Data Processing (requires raw JSON - for internal use):"
	@echo "    data                   - Process all raw JSON to CSV"
	@echo "    data-forw              - Process forward planning data"
	@echo "    data-intimacy-alt      - Process alt-shown intimacy inference data"
	@echo "    data-intimacy-noalt    - Process no-alt intimacy inference data"
	@echo "    data-desire-alt        - Process alt-shown desire inference data"
	@echo ""
	@echo "  Utilities:"
	@echo "    test             - Run model compliance tests"
	@echo "    clean            - Remove generated model outputs"

# =============================================================================
# Data Processing (optional - requires raw JSON files)
# =============================================================================

data: data-forw data-intimacy-alt data-intimacy-noalt data-desire-alt

data-forw:
	uv run python analysis/json_to_csv.py forw_plan

data-intimacy-alt:
	uv run python analysis/json_to_csv.py inv_plan_intimacy_alt

data-intimacy-noalt:
	uv run python analysis/json_to_csv.py inv_plan_intimacy_noalt

data-desire-alt:
	uv run python analysis/json_to_csv.py inv_plan_desire_alt

# =============================================================================
# Model Fitting
# =============================================================================

fit: fit-forward fit-inverse

# Forward planning model fitting
# Depends on processed CSV (included in repo)
model/outputs/forward_planning_fit_results.csv model/outputs/forward_planning_fits.csv: data/forw_plan/main_trials_long.csv model/fit_forward_planning.py model/model_utils.py
	uv run python model/fit_forward_planning.py

fit-forward: model/outputs/forward_planning_fit_results.csv

# Inverse planning model fitting (alt-shown)
# Depends on forward planning results and processed CSVs
model/outputs/inverse_planning_fit_results.csv: model/outputs/forward_planning_fit_results.csv \
                                        data/inv_plan_intimacy_alt/main_trials_long.csv \
                                        data/inv_plan_desire_alt/main_trials_long.csv \
                                        model/fit_inverse_planning.py \
                                        model/model_utils.py
	uv run python model/fit_inverse_planning.py

fit-inverse: model/outputs/inverse_planning_fit_results.csv

# =============================================================================
# Model Predictions
# =============================================================================

model/outputs/inv_plan_intimacy_alt_preds_summary.csv model/outputs/inv_plan_desire_alt_preds_summary.csv: \
        model/outputs/forward_planning_fit_results.csv \
        model/outputs/inverse_planning_fit_results.csv \
        model/generate_inverse_planning_preds.py \
        model/model_utils.py
	uv run python model/generate_inverse_planning_preds.py

predictions: model/outputs/inv_plan_intimacy_alt_preds_summary.csv

# =============================================================================
# Analysis
# =============================================================================

analysis: analysis-forw-plan analysis-inv-plan-intimacy-alt analysis-inv-plan-desire-alt analysis-inv-plan-combined

analysis-forw-plan: model/outputs/forward_planning_fits.csv
	quarto render analysis/forw-plan-analysis.qmd

analysis-inv-plan-intimacy-alt: model/outputs/inv_plan_intimacy_alt_preds_summary.csv
	quarto render analysis/inv-plan-intimacy-alt-analysis.qmd

analysis-inv-plan-desire-alt: model/outputs/inv_plan_desire_alt_preds_summary.csv
	quarto render analysis/inv-plan-desire-alt-analysis.qmd

analysis-inv-plan-intimacy-noalt: model/outputs/inv_plan_intimacy_noalt_preds_summary.csv
	quarto render analysis/inv-plan-intimacy-noalt-analysis.qmd

analysis-inv-plan-combined: model/outputs/inv_plan_intimacy_alt_preds_summary.csv model/outputs/inv_plan_desire_alt_preds_summary.csv
	quarto render analysis/inv-plan-combined-correlation.qmd

# =============================================================================
# Utilities
# =============================================================================

test:
	uv run python model/test_model_compliance.py

clean:
	rm -f model/outputs/forward_planning_fits.csv
	rm -f model/outputs/forward_planning_fit_results.csv
	rm -f model/outputs/inverse_planning_fit_results.csv
	rm -f model/outputs/inv_plan_intimacy_alt_preds_full.csv
	rm -f model/outputs/inv_plan_intimacy_alt_preds_summary.csv
	rm -f model/outputs/inv_plan_desire_alt_preds_full.csv
	rm -f model/outputs/inv_plan_desire_alt_preds_summary.csv
