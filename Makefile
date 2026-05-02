# Makefile for saliva-inverse-planning
#
# Pipeline: data → LM elicitation → fit → predict → CV → analysis (qmds)
#
# Processed CSVs are checked into the repo, so the model + analysis stages
# work without re-running data processing or LM elicitation.

EXPERIMENTS_FORWARD := food_forw_intimacy_desire food_forw_intimacy_effort nonfood_forw_intimacy_desire
EXPERIMENTS_INVERSE := food_inv_intimacy_desire_alt food_inv_desire_intimacy_alt \
                       food_inv_intimacy_desire_noalt food_inv_desire_intimacy_noalt \
                       food_inv_intimacy_effort_alt food_inv_effort_intimacy_alt
EXPERIMENTS_ALL := $(EXPERIMENTS_FORWARD) $(EXPERIMENTS_INVERSE)

ANALYSIS_QMDS := \
  food-forw-intimacy-desire-analysis \
  food-forw-intimacy-effort-analysis \
  nonfood-forw-intimacy-desire-analysis \
  food-inv-intimacy-desire-alt-analysis \
  food-inv-desire-intimacy-alt-analysis \
  food-inv-intimacy-desire-noalt-analysis \
  food-inv-desire-intimacy-noalt-analysis \
  food-inv-intimacy-effort-alt-analysis \
  food-inv-effort-intimacy-alt-analysis \
  inv-plan-combined-correlation \
  inv-plan-combined-correlation-by-scenario \
  cv-loso-forward

.PHONY: all help test clean \
        data lm lm-canonical lm-effort lm-alternatives \
        fit fit-forward fit-inverse \
        predict predict-forward predict-inverse \
        cv cv-forward cv-inverse \
        analysis \
        $(addprefix data-,$(EXPERIMENTS_ALL)) \
        $(addprefix fit-,$(EXPERIMENTS_ALL)) \
        $(addprefix predict-,$(EXPERIMENTS_ALL)) \
        $(addprefix cv-,$(EXPERIMENTS_ALL)) \
        $(addprefix analysis-,$(ANALYSIS_QMDS))

all: fit predict cv analysis

help:
	@echo "Saliva inverse planning pipeline"
	@echo ""
	@echo "Aggregates:"
	@echo "  all        - fit + predict + cv + analysis"
	@echo "  fit        - fit all 9 experiments (3 forward + 6 inverse)"
	@echo "  predict    - generate predictions for all 9 experiments"
	@echo "  cv         - leave-one-scenario-out CV for all 9 experiments"
	@echo "  analysis   - render all 12 quarto analysis qmds"
	@echo "  lm         - regenerate all LM-elicited CSVs (needs TOGETHER_API_KEY)"
	@echo "  data       - process raw JSON to CSV for all 9 experiments"
	@echo "  test       - model compliance tests"
	@echo "  clean      - remove fit/predict/CV outputs"
	@echo ""
	@echo "Per-stage aggregates:"
	@echo "  fit-forward, fit-inverse"
	@echo "  predict-forward, predict-inverse"
	@echo "  cv-forward, cv-inverse"
	@echo "  lm-canonical, lm-effort, lm-alternatives"
	@echo ""
	@echo "Per-experiment (substitute slug):"
	@echo "  fit-<slug>, predict-<slug>, cv-<slug>, data-<slug>"
	@echo "  e.g. make fit-food_forw_intimacy_desire"
	@echo ""
	@echo "Per-qmd:"
	@echo "  analysis-<name>  (without .qmd suffix)"
	@echo "  e.g. make analysis-inv-plan-combined-correlation"
	@echo ""
	@echo "Forward slugs: $(EXPERIMENTS_FORWARD)"
	@echo "Inverse slugs: $(EXPERIMENTS_INVERSE)"

# =============================================================================
# Data: raw JSON → CSV. Only useful if raw JSON in data/<slug>/raw_data/ exists;
# otherwise the checked-in CSVs are already current.
# =============================================================================

data: $(addprefix data-,$(EXPERIMENTS_ALL))

$(addprefix data-,$(EXPERIMENTS_ALL)): data-%:
	uv run python analysis/json_to_csv.py $*

# =============================================================================
# LM elicitation (Llama-3.3-70B via Together AI; needs TOGETHER_API_KEY in .env)
# =============================================================================

lm: lm-canonical lm-effort lm-alternatives

lm-canonical:
	uv run python model/lm/score_canonical_features.py
	uv run python model/lm/score_canonical_features.py --domain nonfood
	uv run python model/lm/score_canonical_v.py
	uv run python model/lm/score_canonical_v.py --domain nonfood

lm-effort:
	uv run python model/lm/score_effort_features.py

lm-alternatives:
	uv run python model/lm/generate_alternatives_motivation.py
	uv run python model/lm/score_alternative_features.py
	uv run python model/lm/score_alternative_v.py
	uv run python model/lm/generate_alternatives_relationship.py
	uv run python model/lm/score_alternative_features.py --conditioning relationship
	uv run python model/lm/score_alternative_v.py --conditioning relationship

# =============================================================================
# Fits → outputs/<slug>/fit_results.csv
# =============================================================================

fit: fit-forward fit-inverse
fit-forward: $(addprefix fit-,$(EXPERIMENTS_FORWARD))
fit-inverse: $(addprefix fit-,$(EXPERIMENTS_INVERSE))

$(addprefix fit-,$(EXPERIMENTS_FORWARD)): fit-%:
	uv run python model/forward/fit_$*.py

$(addprefix fit-,$(EXPERIMENTS_INVERSE)): fit-%:
	uv run python model/inverse/fit_$*.py

# =============================================================================
# Predicts → outputs/<slug>/preds.csv (forward) or preds_full.csv +
# preds_summary.csv (inverse)
# =============================================================================

predict: predict-forward predict-inverse
predict-forward: $(addprefix predict-,$(EXPERIMENTS_FORWARD))
predict-inverse: $(addprefix predict-,$(EXPERIMENTS_INVERSE))

$(addprefix predict-,$(EXPERIMENTS_FORWARD)): predict-%:
	uv run python model/forward/predict_$*.py

$(addprefix predict-,$(EXPERIMENTS_INVERSE)): predict-%:
	uv run python model/inverse/predict_$*.py

# =============================================================================
# Leave-one-scenario-out CV → outputs/<slug>/cv_folds.csv +
# cv_preds.csv (forward) or cv_preds_summary.csv (inverse)
# =============================================================================

cv: cv-forward cv-inverse
cv-forward: $(addprefix cv-,$(EXPERIMENTS_FORWARD))
cv-inverse: $(addprefix cv-,$(EXPERIMENTS_INVERSE))

$(addprefix cv-,$(EXPERIMENTS_FORWARD)): cv-%:
	uv run python model/cv/cv_$*.py

$(addprefix cv-,$(EXPERIMENTS_INVERSE)): cv-%:
	uv run python model/cv/cv_$*.py

# =============================================================================
# Analysis: quarto render
# =============================================================================

analysis: $(addprefix analysis-,$(ANALYSIS_QMDS))

$(addprefix analysis-,$(ANALYSIS_QMDS)): analysis-%:
	quarto render analysis/$*.qmd

# =============================================================================
# Utilities
# =============================================================================

test:
	uv run python model/test_model_compliance.py

clean:
	rm -f model/outputs/*/fit_results.csv
	rm -f model/outputs/*/preds*.csv
	rm -f model/outputs/*/cv_*.csv
