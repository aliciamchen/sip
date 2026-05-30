# Makefile for saliva-inverse-planning
#
# Pipeline: data → LM elicitation → fit → predict → CV → analysis (qmds)
#
# Processed CSVs are checked into the repo, so the model + analysis stages
# work without re-running data processing or LM elicitation.

# The active roster is four inverse-planning studies, all on the 3-action set:
#   food_inv_desire    (Study 1a — infer desire)
#   food_inv_joint_de  (Study 1b — joint desire + effort)
#   food_inv_intimacy  (Study 2a — infer intimacy)
#   food_inv_joint_ie  (Study 2b — joint intimacy + effort)
EXPERIMENTS_INVERSE := food_inv_desire food_inv_joint_de \
                       food_inv_intimacy food_inv_joint_ie
# Legacy forward-planning experiments: real data archived under data/legacy/,
# reachable via per-slug targets (e.g. `make fit-food_forw_intimacy_desire`) but
# not part of the default `make all` pipeline.
LEGACY_FORWARD := food_forw_intimacy_desire food_forw_intimacy_effort nonfood_forw_intimacy_desire
# Legacy 4-action inverse experiments; data under data/legacy/, reachable via
# per-slug targets (e.g. `make fit-food_inv_intimacy_desire_noalt`).
LEGACY_INVERSE := food_inv_intimacy_desire_noalt food_inv_desire_intimacy_noalt
EXPERIMENTS_ALL := $(EXPERIMENTS_INVERSE)
EXPERIMENTS_REGISTERED := $(EXPERIMENTS_ALL) $(LEGACY_FORWARD) $(LEGACY_INVERSE)

ANALYSIS_QMDS := \
  food-inv-desire-analysis \
  food-inv-joint-de-analysis \
  food-inv-intimacy-analysis \
  food-inv-joint-ie-analysis
LEGACY_ANALYSIS_QMDS := \
  food-forw-intimacy-desire-analysis \
  food-forw-intimacy-effort-analysis \
  nonfood-forw-intimacy-desire-analysis \
  cv-loso-forward \
  food-inv-intimacy-desire-noalt-analysis \
  food-inv-desire-intimacy-noalt-analysis
ANALYSIS_QMDS_REGISTERED := $(ANALYSIS_QMDS) $(LEGACY_ANALYSIS_QMDS)

.PHONY: all help test clean \
        data lm lm-3act lm-alternatives lm-canonical lm-effort \
        fit fit-forward fit-inverse \
        predict predict-forward predict-inverse \
        cv cv-forward cv-inverse \
        analysis \
        $(addprefix data-,$(EXPERIMENTS_ALL)) \
        $(addprefix fit-,$(EXPERIMENTS_REGISTERED)) \
        $(addprefix predict-,$(EXPERIMENTS_REGISTERED)) \
        $(addprefix cv-,$(EXPERIMENTS_REGISTERED)) \
        $(addprefix analysis-,$(ANALYSIS_QMDS_REGISTERED))

all: fit predict cv analysis

help:
	@echo "Saliva inverse planning pipeline"
	@echo ""
	@echo "Aggregates (active experiments only):"
	@echo "  all        - fit + predict + cv + analysis"
	@echo "  fit        - fit all active experiments"
	@echo "  predict    - generate predictions for all active experiments"
	@echo "  cv         - leave-one-scenario-out CV for all active experiments"
	@echo "  analysis   - render all active quarto analysis qmds"
	@echo "  lm         - regenerate all LM-elicited CSVs (needs TOGETHER_API_KEY)"
	@echo "  data       - process raw JSON to CSV for all active experiments"
	@echo "  test       - model compliance tests"
	@echo "  clean      - remove fit/predict/CV outputs"
	@echo ""
	@echo "Per-stage aggregates:"
	@echo "  fit-inverse, predict-inverse, cv-inverse        (active)"
	@echo "  fit-forward, predict-forward, cv-forward        (legacy forwards)"
	@echo "  lm-3act, lm-alternatives, lm-canonical, lm-effort"
	@echo ""
	@echo "Per-experiment (substitute slug):"
	@echo "  fit-<slug>, predict-<slug>, cv-<slug>  (works for active and legacy)"
	@echo "  data-<slug>                            (active only; legacy CSVs already processed)"
	@echo "  e.g. make fit-food_inv_desire"
	@echo "  e.g. make fit-food_forw_intimacy_desire        (legacy forward)"
	@echo "  e.g. make fit-food_inv_intimacy_desire_noalt   (legacy inverse)"
	@echo ""
	@echo "Per-qmd:"
	@echo "  analysis-<name>  (without .qmd suffix)"
	@echo "  e.g. make analysis-food-inv-desire-analysis"
	@echo ""
	@echo "Active inverse slugs:   $(EXPERIMENTS_INVERSE)"
	@echo "Legacy forward slugs:   $(LEGACY_FORWARD)"
	@echo "Legacy inverse slugs:   $(LEGACY_INVERSE)"

# =============================================================================
# Data: raw JSON → CSV. Only useful if raw JSON in data/<slug>/raw_data/ exists;
# otherwise the checked-in CSVs are already current.
# =============================================================================

data: $(addprefix data-,$(EXPERIMENTS_ALL))

$(addprefix data-,$(EXPERIMENTS_ALL)): data-%:
	uv run python analysis/json_to_csv.py $*

# Legacy data CSVs are already processed and live under data/legacy/<slug>/;
# re-processing them from raw JSON is not part of the active pipeline. To
# regenerate one, point analysis/json_to_csv.py at the legacy location manually.

# =============================================================================
# LM elicitation (Llama-3.3-70B via Together AI; needs TOGETHER_API_KEY in .env)
# =============================================================================

# `lm` regenerates the tables for the active 3-action pipeline. The legacy
# canonical (4-action) and effort (2-action) tables back the legacy forward
# experiments and are regenerated separately via lm-canonical / lm-effort.
lm: lm-3act lm-alternatives

# Fixed-action 3-action tables (access/effort/V), used by the studies still on
# the fixed-action pipeline.
lm-3act:
	uv run python model/lm/score_3act_features.py
	uv run python model/lm/score_3act_v.py

# LM-generated alternative actions + merged scoring for the padded-action
# pipeline. Study 1a (food_inv_desire) is migrated; the other active studies'
# --study entries are added as they are migrated.
lm-alternatives:
	uv run python model/lm/generate_alternatives_3act.py --study food_inv_desire
	uv run python model/lm/score_3act_merged.py          --study food_inv_desire

# Legacy LM tables (4-action canonical + 2-action effort) for the legacy
# forward experiments.
lm-canonical:
	uv run python model/lm/score_canonical_features.py
	uv run python model/lm/score_canonical_features.py --domain nonfood
	uv run python model/lm/score_canonical_v.py
	uv run python model/lm/score_canonical_v.py --domain nonfood

lm-effort:
	uv run python model/lm/score_effort_features.py

# =============================================================================
# Fits → outputs/<slug>/fit_results.csv
# =============================================================================

fit: fit-inverse
fit-inverse: $(addprefix fit-,$(EXPERIMENTS_INVERSE))
# Legacy forwards: per-slug only (e.g. `make fit-food_forw_intimacy_desire`).
fit-forward: $(addprefix fit-,$(LEGACY_FORWARD))

$(addprefix fit-,$(LEGACY_FORWARD)): fit-%:
	uv run python model/forward/fit_$*.py

$(addprefix fit-,$(EXPERIMENTS_INVERSE) $(LEGACY_INVERSE)): fit-%:
	uv run python model/inverse/fit_$*.py

# =============================================================================
# Predicts → outputs/<slug>/preds.csv (forward) or preds_full.csv +
# preds_summary.csv (inverse)
# =============================================================================

predict: predict-inverse
predict-inverse: $(addprefix predict-,$(EXPERIMENTS_INVERSE))
# Legacy forwards: per-slug only.
predict-forward: $(addprefix predict-,$(LEGACY_FORWARD))

$(addprefix predict-,$(LEGACY_FORWARD)): predict-%:
	uv run python model/forward/predict_$*.py

$(addprefix predict-,$(EXPERIMENTS_INVERSE) $(LEGACY_INVERSE)): predict-%:
	uv run python model/inverse/predict_$*.py

# =============================================================================
# Leave-one-scenario-out CV → outputs/<slug>/cv_folds.csv +
# cv_preds.csv (forward) or cv_preds_summary.csv (inverse)
# =============================================================================

cv: cv-inverse
cv-inverse: $(addprefix cv-,$(EXPERIMENTS_INVERSE))
# Legacy forwards: per-slug only.
cv-forward: $(addprefix cv-,$(LEGACY_FORWARD))

$(addprefix cv-,$(LEGACY_FORWARD) $(EXPERIMENTS_INVERSE) $(LEGACY_INVERSE)): cv-%:
	uv run python model/cv/cv_$*.py

# =============================================================================
# Analysis: quarto render
# =============================================================================

analysis: $(addprefix analysis-,$(ANALYSIS_QMDS))

$(addprefix analysis-,$(ANALYSIS_QMDS) $(LEGACY_ANALYSIS_QMDS)): analysis-%:
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
