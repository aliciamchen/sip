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
EXPERIMENTS_ALL := $(EXPERIMENTS_INVERSE)

ANALYSIS_QMDS := \
  food-inv-desire-analysis \
  food-inv-joint-de-analysis \
  food-inv-intimacy-analysis \
  food-inv-joint-ie-analysis

.PHONY: all help test clean \
        data lm lm-alternatives \
        fit fit-inverse \
        predict predict-inverse \
        cv cv-inverse \
        analysis \
        $(addprefix data-,$(EXPERIMENTS_ALL)) \
        $(addprefix lm-,$(EXPERIMENTS_INVERSE)) \
        $(addprefix fit-,$(EXPERIMENTS_INVERSE)) \
        $(addprefix predict-,$(EXPERIMENTS_INVERSE)) \
        $(addprefix cv-,$(EXPERIMENTS_INVERSE)) \
        $(addprefix analysis-,$(ANALYSIS_QMDS))

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
	@echo "Experiment assets (jsPsych build, run before deploying):"
	@echo "  experiments       - regenerate stimuli + counterbalancing + entry files"
	@echo "  stimuli           - scenarios.py -> scenarios.csv -> per-experiment stimuli.json"
	@echo "  counterbalancing  - regenerate every active full_counterbalancing.json"
	@echo "  entry-files       - sync index.html + experiment.js across active experiments"
	@echo "  preview           - serve the trial-preview page locally (open /preview/)"
	@echo "  deploy-preview    - publish the trial-preview page to athena"
	@echo "  deploy-all        - publish _lib/ + all experiments + preview to athena (one login)"
	@echo ""
	@echo "Per-stage aggregates:"
	@echo "  fit-inverse, predict-inverse, cv-inverse"
	@echo "  lm, lm-alternatives   (lm-alternatives does all 4 studies;"
	@echo "                         'make -j4 lm-alternatives SCENARIO_WORKERS=1' runs them in parallel)"
	@echo ""
	@echo "Per-experiment (substitute slug):"
	@echo "  lm-<slug>, fit-<slug>, predict-<slug>, cv-<slug>, data-<slug>"
	@echo "  e.g. make fit-food_inv_desire"
	@echo ""
	@echo "Per-qmd:"
	@echo "  analysis-<name>  (without .qmd suffix)"
	@echo "  e.g. make analysis-food-inv-desire-analysis"
	@echo ""
	@echo "Active inverse slugs:   $(EXPERIMENTS_INVERSE)"

# =============================================================================
# Experiment assets (jsPsych): regenerate what a deploy needs from source.
# Run before bin/deploy-experiment when scenarios, counterbalancing, or the
# shared entry files change. Not part of `make all`.
# =============================================================================

.PHONY: experiments stimuli counterbalancing entry-files preview \
        deploy-preview deploy-all \
        $(addprefix counterbalancing-,$(EXPERIMENTS_INVERSE))

experiments: stimuli counterbalancing entry-files

# Trial-preview page (experiments/preview/): a static page to show collaborators
# what any study/scenario/condition looks like to a participant. It uses ES-module
# imports + fetch, so it must be served over HTTP (a file:// path won't work).
# `make preview` serves the experiments/ tree so the page's ../_lib/ and
# ../food_inv_*/ imports resolve, then open http://localhost:8000/preview/.
preview:
	@echo "Serving experiments/ at http://localhost:8000/  (open /preview/)"
	cd experiments && python3 -m http.server 8000

# Publish the preview page to athena (assumes the four experiments are deployed).
deploy-preview:
	bin/deploy-experiment preview

# Publish everything to athena in one pass — _lib/, all four experiments, and the
# preview page — entering the athena password once. Use this when experiment code
# (not just the preview) has changed. Build the assets first with `make experiments`
# if scenarios/counterbalancing/entry files changed.
deploy-all:
	bin/deploy-experiment --all

# scenarios.py (source of truth) -> scenarios.csv -> per-experiment stimuli.json.
stimuli:
	uv run python experiments/scenarios.py
	uv run python experiments/build/csv_to_json.py

# Per-participant condition sequences (each json/full_counterbalancing.json),
# all from one registry-driven generator in experiments/build/.
counterbalancing:
	uv run python experiments/build/counterbalancing.py

$(addprefix counterbalancing-,$(EXPERIMENTS_INVERSE)): counterbalancing-%:
	uv run python experiments/build/counterbalancing.py --study $*

# Byte-identical index.html + experiment.js across the active experiments.
entry-files:
	uv run python experiments/build/sync_entry_files.py

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

lm: lm-alternatives

# Per-study LM-generated alternatives + merged scoring for the padded-action
# pipeline. `make lm-alternatives` runs all four studies; the per-study
# `lm-<slug>` targets let you run one, and `make -j4 lm-alternatives` runs the
# four studies as PARALLEL processes (each writes to its own outputs/lm/<slug>/
# folder, so no contention). Within a study, generation must finish before
# scoring, so those stay ordered.
#
# SCENARIO_WORKERS controls how many scenarios score_merged scores concurrently
# (in-flight requests ≈ SCENARIO_WORKERS × NUM_RUNS). When parallelizing studies
# with -j, lower it so 4 × SCENARIO_WORKERS × NUM_RUNS stays under your Together
# tier's limit, e.g.  make -j4 lm-alternatives SCENARIO_WORKERS=1
SCENARIO_WORKERS ?= 4

lm-alternatives: $(addprefix lm-,$(EXPERIMENTS_INVERSE))

$(addprefix lm-,$(EXPERIMENTS_INVERSE)): lm-%:
	uv run python model/lm/generate_alternatives.py --study $*
	uv run python model/lm/score_merged.py --study $* --scenario-workers $(SCENARIO_WORKERS)

# =============================================================================
# Fits → outputs/<slug>/fit_results.csv
# =============================================================================

fit: fit-inverse
fit-inverse: $(addprefix fit-,$(EXPERIMENTS_INVERSE))

$(addprefix fit-,$(EXPERIMENTS_INVERSE)): fit-%:
	uv run python model/inverse/fit_$*.py

# =============================================================================
# Predicts → outputs/<slug>/preds_<variant>.npy + preds_summary.csv
# =============================================================================

predict: predict-inverse
predict-inverse: $(addprefix predict-,$(EXPERIMENTS_INVERSE))

$(addprefix predict-,$(EXPERIMENTS_INVERSE)): predict-%:
	uv run python model/inverse/predict_$*.py

# =============================================================================
# Leave-one-scenario-out CV → outputs/<slug>/cv_folds.csv + cv_preds_summary.csv
# =============================================================================

cv: cv-inverse
cv-inverse: $(addprefix cv-,$(EXPERIMENTS_INVERSE))

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
