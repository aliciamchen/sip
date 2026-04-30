# Makefile for Saliva Inverse Planning Project
#
# This pipeline works without raw JSON data - processed CSVs are included in the repo.
# Data processing targets (data-*) are optional and require raw JSON files.

.PHONY: all help clean test data data-forw data-intimacy-alt data-intimacy-noalt data-desire-alt \
        data-nonfood-forw \
        fit fit-forward fit-inverse fit-forward-nonfood fit-forward-nonfood-ext fit-forward-food-ext predictions \
        cv-forward-nonfood cv-forward-nonfood-ext cv-forward-food-ext \
        lm-v lm-v-food lm-v-nonfood lm-v-alternatives \
        analysis analysis-forw-plan analysis-inv-plan-intimacy-alt analysis-inv-plan-desire-alt \
        analysis-inv-plan-intimacy-noalt analysis-inv-plan-combined analysis-nonfood-forw-plan

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
	@echo "    analysis-nonfood-forw-plan       - Render non-food forward planning analysis"
	@echo ""
	@echo "  Non-food pipeline (parallels canonical food pipeline):"
	@echo "    data-nonfood-forw         - Process non-food forward planning raw JSON"
	@echo "    fit-forward-nonfood       - Fit non-food forward planning models"
	@echo "    cv-forward-nonfood        - LOSO CV for non-food forward planning"
	@echo "    fit-forward-nonfood-ext   - Fit non-food extensions (Full + power-law gamma)"
	@echo "    cv-forward-nonfood-ext    - LOSO CV for non-food extensions"
	@echo "    fit-forward-food-ext      - Fit food gamma extension (cross-domain comparison)"
	@echo "    cv-forward-food-ext       - LOSO CV for food gamma extension"
	@echo ""
	@echo "  LM-V tables (signed-valence V is the canonical V; required for forward + alt-shown fits):"
	@echo "    lm-v                    - Generate LM-V tables for both food and non-food"
	@echo "    lm-v-food / lm-v-nonfood - Generate LM-V table for one domain"
	@echo "    lm-v-alternatives       - Generate V for LM-generated alternatives (food, required for no-alt observers)"
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

data-nonfood-forw:
	uv run python analysis/json_to_csv.py nonfood_forw_plan

# =============================================================================
# Model Fitting
# =============================================================================

fit: fit-forward fit-inverse

# Forward planning model fitting
# Depends on processed CSV (included in repo) plus LM-V table
model/outputs/forward_planning_fit_results.csv model/outputs/forward_planning_fits.csv: data/forw_plan/main_trials_long.csv model/outputs/lm_scenario_v.csv model/fit_forward_planning.py model/model_utils.py
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

# Non-food forward planning fit
# Requires lm_scenario_params_nonfood.csv and lm_scenario_v_nonfood.csv
model/outputs/forward_planning_fit_results_nonfood.csv model/outputs/forward_planning_fits_nonfood.csv: \
        data/nonfood_forw_plan/main_trials_long.csv \
        model/outputs/lm_scenario_params_nonfood.csv \
        model/outputs/lm_scenario_v_nonfood.csv \
        model/fit_forward_planning.py model/model_utils.py
	uv run python model/fit_forward_planning.py --domain nonfood

fit-forward-nonfood: model/outputs/forward_planning_fit_results_nonfood.csv

# Non-food LOSO CV
model/outputs/cv_loso_preds_nonfood.csv model/outputs/cv_loso_forward_nonfood.csv: \
        model/outputs/forward_planning_fit_results_nonfood.csv \
        model/cv/loso_forward.py model/model_utils.py
	uv run python model/cv/loso_forward.py --domain nonfood

cv-forward-nonfood: model/outputs/cv_loso_preds_nonfood.csv

# Non-food extensions: Full + power-law intimacy ((1 - I)^gamma).
# Lives in nonfood_ext files so the canonical food pipeline is not touched.
model/outputs/forward_planning_fit_results_nonfood_ext.csv model/outputs/forward_planning_fits_nonfood_ext.csv: \
        data/nonfood_forw_plan/main_trials_long.csv \
        model/outputs/lm_scenario_params_nonfood.csv \
        model/outputs/lm_scenario_v_nonfood.csv \
        model/fit_forward_planning_nonfood_ext.py \
        model/model_utils_nonfood_ext.py \
        model/model_utils.py
	uv run python model/fit_forward_planning_nonfood_ext.py

fit-forward-nonfood-ext: model/outputs/forward_planning_fit_results_nonfood_ext.csv

model/outputs/cv_loso_preds_nonfood_ext.csv model/outputs/cv_loso_forward_nonfood_ext.csv: \
        model/outputs/forward_planning_fit_results_nonfood_ext.csv \
        model/cv/loso_forward_nonfood_ext.py \
        model/fit_forward_planning_nonfood_ext.py \
        model/model_utils_nonfood_ext.py \
        model/model_utils.py
	uv run python model/cv/loso_forward_nonfood_ext.py

cv-forward-nonfood-ext: model/outputs/cv_loso_preds_nonfood_ext.csv

# Food gamma extension (cross-domain comparison; canonical food fits in
# fit_forward_planning.py are NOT touched).
model/outputs/forward_planning_fit_results_ext.csv model/outputs/forward_planning_fits_ext.csv: \
        data/forw_plan/main_trials_long.csv \
        model/outputs/lm_scenario_params.csv \
        model/outputs/lm_scenario_v.csv \
        model/fit_forward_planning_nonfood_ext.py \
        model/model_utils_nonfood_ext.py \
        model/model_utils.py
	uv run python model/fit_forward_planning_nonfood_ext.py --domain food

fit-forward-food-ext: model/outputs/forward_planning_fit_results_ext.csv

model/outputs/cv_loso_preds_ext.csv model/outputs/cv_loso_forward_ext.csv: \
        model/outputs/forward_planning_fit_results_ext.csv \
        model/cv/loso_forward_nonfood_ext.py \
        model/fit_forward_planning_nonfood_ext.py \
        model/model_utils_nonfood_ext.py \
        model/model_utils.py
	uv run python model/cv/loso_forward_nonfood_ext.py --domain food

cv-forward-food-ext: model/outputs/cv_loso_preds_ext.csv

# LM-V tables (canonical signed-valence V — required for all forward + alt-shown fits)
model/outputs/lm_scenario_v.csv: experiments/scenarios.csv model/lm_scenario_params.py model/lm_prompts.py
	uv run python model/lm_scenario_params.py --feature v --domain food

model/outputs/lm_scenario_v_nonfood.csv: experiments/scenarios_nonfood.csv model/lm_scenario_params.py model/lm_prompts.py
	uv run python model/lm_scenario_params.py --feature v --domain nonfood

# V for LM-generated alternatives — required only for no-alt observer fits
model/outputs/lm_alternatives_v.csv: model/outputs/lm_alternatives.csv model/lm_scenario_params.py model/lm_prompts.py
	uv run python model/lm_scenario_params.py --feature v_alternatives --domain food

lm-v-food: model/outputs/lm_scenario_v.csv
lm-v-nonfood: model/outputs/lm_scenario_v_nonfood.csv
lm-v: lm-v-food lm-v-nonfood
lm-v-alternatives: model/outputs/lm_alternatives_v.csv

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

analysis-nonfood-forw-plan: model/outputs/cv_loso_preds_nonfood.csv
	quarto render analysis/nonfood-forw-plan-analysis.qmd

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
