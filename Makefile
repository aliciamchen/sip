# Makefile for saliva-inverse-planning
#
# Pipeline: data → LM elicitation → fit → CV → model comparison → figures
# (CV is the sole source of model predictions — all reported model-vs-human
#  numbers are out-of-sample; there is no separate in-sample predict stage.)
#
# Processed CSVs are checked into the repo, so the model and figure stages
# work without re-running data processing or LM elicitation.
#
# fit / CV / model-comparison are a real file-target dependency graph (not just
# phony task aliases), so `make` rebuilds only what is out of date; the phony
# aggregate names (`make fit`, `make cv`, `make all`, `make fit-<slug>`, …) still
# work and simply no-op when their outputs are already current.

# The active roster is six inverse-planning studies, all on the 3-action set:
#   food_inv_desire       (Study 1a — infer desire)
#   food_inv_joint_de     (Study 1b — joint desire + effort)
#   food_inv_intimacy     (Study 2a — infer intimacy)
#   food_inv_joint_ie     (Study 2b — joint intimacy + effort)
#   nonfood_inv_joint_de  (Study 3a — 1b's design on the nonfood scenarios)
#   nonfood_inv_joint_ie  (Study 3b — 2b's design on the nonfood scenarios)
#
# All six are data-complete, so EXPERIMENTS_INVERSE holds the whole roster
# and drives the data-dependent aggregate stages (data / fit / cv /
# model-comparison / lm-alternatives). EXPERIMENTS_NONFOOD is kept (empty) so the roster-sync test's
# disjoint-lists invariant still has both variables to read, and as the holding
# list for any future not-yet-data-complete study family; per-study targets
# (lm-/fit-/cv-/data-<slug>) cover the roster either way.
EXPERIMENTS_INVERSE := food_inv_desire food_inv_joint_de \
                       food_inv_intimacy food_inv_joint_ie \
                       nonfood_inv_joint_de nonfood_inv_joint_ie
EXPERIMENTS_NONFOOD :=
EXPERIMENTS_ALL := $(EXPERIMENTS_INVERSE) $(EXPERIMENTS_NONFOOD)

# Studies that get a relationship-free base-model alternative set (the `--base`
# elicitation mode): the given-relationship studies only (2a/2b/3b infer
# intimacy, so they never show a relationship paragraph).
EXPERIMENTS_BASE := food_inv_desire food_inv_joint_de nonfood_inv_joint_de

.PHONY: all help test clean \
        data lm lm-alternatives lm-base lm-diag lm-base-diag \
        fit fit-inverse \
        cv cv-inverse model-comparison \
        figures-lm-si figures-panels figures-nonfood-domains \
        figures-si-scenarios \
        figures-si-prior-posterior figures-si-prereg-predictions \
        $(addprefix data-,$(EXPERIMENTS_ALL)) \
        $(addprefix lm-,$(EXPERIMENTS_ALL)) \
        $(addprefix lm-base-,$(EXPERIMENTS_BASE)) \
        $(addprefix lm-diag-,$(EXPERIMENTS_ALL)) \
        $(addprefix lm-base-diag-,$(EXPERIMENTS_BASE)) \
        $(addprefix fit-,$(EXPERIMENTS_ALL)) \
        $(addprefix cv-,$(EXPERIMENTS_ALL))

# The pipeline stages must run strictly in order even under `make -j`. CV
# warm-starts each fold from the full-data fit, so a cv running concurrently with
# fit silently produces different results, and the figure prerequisites are
# wildcard-resolved at parse time, so they cannot order themselves behind a CV
# that runs in the same make invocation. Keeping `all` as sequential sub-makes
# guarantees the ordering; each sub-make is incremental (a no-op when its
# stage's outputs are already current) and still parallelizes internally.
all:
	$(MAKE) fit
	$(MAKE) cv
	$(MAKE) model-comparison
	$(MAKE) figures-panels

help:
	@echo "Saliva inverse planning pipeline"
	@echo ""
	@echo "Aggregates (active experiments only; incremental -- rebuild only what is stale):"
	@echo "  all        - fit -> cv -> model-comparison -> figures-panels, in order"
	@echo "  fit        - fit all active experiments"
	@echo "  cv         - leave-one-scenario-out CV for all active experiments (the predictions)"
	@echo "  model-comparison - held-out LL differences + correlations with bootstrap CIs"
	@echo "  transfer         - cross-study parameter transfer over the designed pairs (exploratory)"
	@echo "  pooled           - one shared utility per domain / across all six (exploratory)"
	@echo "  generalization-primary - score the transfer/pooled runs on the reported metrics"
	@echo "  lm         - regenerate the LM-elicited JSONL tables (needs TOGETHER_API_KEY)"
	@echo "  data       - process raw JSON to CSV for all active experiments"
	@echo "  test       - model compliance + elicitation-guard + data-converter + experiment-list tests"
	@echo "  clean      - remove fit, CV, and model-comparison outputs"
	@echo "  freshen-outputs - restamp the committed outputs after a fresh clone (see README)"
	@echo "  figures-lm-si        - render the SI LM-elicitation validation figures into figures/si/"
	@echo "  figures-panels       - render the Illustrator results panels + legends into figures/panels/"
	@echo "  figures-nonfood-domains - render the Study 3 human panels split by sharing domain into figures/panels/"
	@echo "  figures-schematic    - render the method-figure schematic panels into figures/panels/"
	@echo "  figures-si-scenarios - render the per-scenario SI facet grids (one per study) into figures/si/"
	@echo "  figures-si-prior-posterior - SI prior/posterior rating levels + distributions into figures/si/"
	@echo "  figures-si-prereg-predictions - both models' per-cell predictions beside humans, all six studies"
	@echo "  sync-journal-figures - copy curated figures/ PDFs into SIP_journal/ (Overleaf)"
	@echo "  results-latex        - regenerate the results macros + table bodies in SIP_journal/"
	@echo ""
	@echo "Experiment assets (jsPsych build):"
	@echo "  experiments       - regenerate stimuli + counterbalancing + entry files"
	@echo "  check-experiments - regenerate + fail if any asset was out of sync with source"
	@echo "  stimuli           - scenarios.py -> scenarios.csv -> per-experiment stimuli.json"
	@echo "  counterbalancing  - regenerate every active full_counterbalancing.json"
	@echo "  entry-files       - sync index.html + experiment.js across active experiments"
	@echo "  preview           - serve the trial-preview page locally (open /preview/)"
	@echo "  deploy-preview    - publish the trial-preview page to athena"
	@echo "  deploy-all        - publish _lib/ + all experiments + preview to athena (one login)"
	@echo "  (deploys auto-run check-experiments first, so stale assets can never ship)"
	@echo ""
	@echo "Model viz (interactive prediction explorer; source kept local in model_viz_nonfit/):"
	@echo "  explorer          - build the explorer page from the precomputed grid"
	@echo "  explorer-grid     - recompute the parameter-grid predictions (~13 min)"
	@echo "  deploy-explorer   - build + publish the explorer to athena (one login)"
	@echo ""
	@echo "Per-stage aggregates:"
	@echo "  fit-inverse, cv-inverse   (all six studies in EXPERIMENTS_INVERSE)"
	@echo "  lm, lm-alternatives   (lm-alternatives does all six studies;"
	@echo "                         'make -j3 lm-alternatives SCENARIO_WORKERS=3 CELL_WORKERS=12' runs them in parallel)"
	@echo "  lm-base               (relationship-free alternatives for the base model;"
	@echo "                         given-relationship studies only; smoke with K_RUNS=1)"
	@echo "  lm-diag, lm-base-diag (K-run-safe diagnostic vintages; never overwrite canonical tables)"
	@echo ""
	@echo "Per-experiment (substitute slug):"
	@echo "  lm-<slug>, fit-<slug>, cv-<slug>, data-<slug>, counterbalancing-<slug>"
	@echo "  fit-<slug> / cv-<slug> take NO_REWEIGHTING=1 (fits the PREREGISTERED model:"
	@echo "    no comparison-set reweighting; writes model/outputs/<slug>/alt/<tag>/ instead of <slug>/)"
	@echo "  lm-base-<slug>   (given-relationship studies only)"
	@echo "  e.g. make fit-food_inv_desire"
	@echo ""
	@echo "Inverse study slugs:  $(EXPERIMENTS_INVERSE)"

# =============================================================================
# Experiment assets (jsPsych): regenerate what a deploy needs from source.
# `bin/deploy-experiment` runs `check-experiments` automatically before every
# deploy, so a stale asset can never reach the server; run `check-experiments`
# yourself before committing experiment changes. Not part of `make all`.
# =============================================================================

.PHONY: experiments check-experiments stimuli counterbalancing entry-files preview \
        deploy-preview deploy-all \
        $(addprefix counterbalancing-,$(EXPERIMENTS_ALL))

experiments: stimuli counterbalancing entry-files

# Regenerate every generated asset from source and fail if that changed anything
# — i.e. a stimuli.json / full_counterbalancing.json / entry file had drifted
# from scenarios.py or the build/ generators. This is the guard against shipping
# stale stimuli; bin/deploy-experiment runs it before every deploy.
check-experiments:
	bin/deploy-experiment --check-artifacts

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
# (not just the preview) has changed. The deploy regenerates assets from source
# and aborts if any had drifted, so no manual `make experiments` is needed first.
deploy-all:
	bin/deploy-experiment --all

# scenarios.py / scenarios_nonfood.py (sources of truth) -> the scenario CSVs
# -> per-experiment stimuli.json.
stimuli:
	uv run python experiments/scenarios.py
	uv run python experiments/scenarios_nonfood.py
	uv run python experiments/build/csv_to_json.py

# Per-participant condition sequences (each json/full_counterbalancing.json),
# all from one registry-driven generator in experiments/build/.
counterbalancing:
	uv run python experiments/build/counterbalancing.py

$(addprefix counterbalancing-,$(EXPERIMENTS_ALL)): counterbalancing-%:
	uv run python experiments/build/counterbalancing.py --study $*

# Byte-identical index.html + experiment.js across the active experiments.
entry-files:
	uv run python experiments/build/sync_entry_files.py

# =============================================================================
# Data: raw JSON → CSV. Only useful if raw JSON in data/<slug>/raw_data/ exists;
# otherwise the checked-in CSVs are already current. `make data` loops the whole
# roster (all six now have raw_data/); json_to_csv.py aborts on a slug with no
# raw_data/, so a study whose collection hasn't started must stay out of
# EXPERIMENTS_INVERSE. The per-study data-<slug> targets cover the roster too.
# =============================================================================

data: $(addprefix data-,$(EXPERIMENTS_INVERSE))

$(addprefix data-,$(EXPERIMENTS_ALL)): data-%:
	uv run python analysis/json_to_csv.py $*

# =============================================================================
# LM elicitation (Llama-3.3-70B via Together AI; needs TOGETHER_API_KEY in .env)
# =============================================================================

lm: lm-alternatives

# Per-study LM-generated alternatives + merged scoring for the padded-action
# pipeline. `make lm-alternatives` runs all six studies; the per-study
# `lm-<slug>` targets let you run one, and `make -jN lm-alternatives` runs the
# studies as PARALLEL processes (each writes to its own outputs/lm/<slug>/
# folder, so no contention). Within a study, generation must finish before
# scoring, so those stay ordered.
#
# SCENARIO_WORKERS controls how many (scenario, run) units score_merged scores
# concurrently; each unit itself fans out its 4 feature calls (+2 desire-scalar
# calls in the given-desire studies), so in-flight requests ≈ 4-6× this value.
# CELL_WORKERS is the generation-side analogue (one call per (cell, run), so
# pool size = in-flight calls). Together's serverless rate limits are dynamic
# per-org (they grow with sustained traffic; bursts past them get 429s, which
# the SDK retries with backoff) — if a run prints repeated 429 / rate-limit
# errors, lower these. When parallelizing studies with -j, remember total
# in-flight = (studies) × (per-study in-flight), e.g.
#   make -j4 lm-alternatives SCENARIO_WORKERS=2 CELL_WORKERS=8
SCENARIO_WORKERS ?= 8
CELL_WORKERS ?= 32
# K_RUNS = elicitation runs per cell (the simulated-observer mixture components);
# ALT_T = generation temperature (nonzero, so runs explore different alternatives).
# A K=1 smoke test before the full paid run:  make lm-alternatives K_RUNS=1
K_RUNS ?= 20
ALT_T ?= 0.7

# `lm-alternatives` re-elicits the whole roster (all six studies). It is always
# an explicit, paid step, never a side effect of `make fit`/`cv` (those depend
# on the committed lm_runs.jsonl but have no rule to regenerate it).
lm-alternatives: $(addprefix lm-,$(EXPERIMENTS_INVERSE))

# score_merged.py takes no K_RUNS: it scores whatever runs the alternatives
# file contains, so the run count is set once at generation time.
$(addprefix lm-,$(EXPERIMENTS_ALL)): lm-%:
	K_RUNS=$(K_RUNS) ALT_T=$(ALT_T) CELL_WORKERS=$(CELL_WORKERS) uv run python model/lm/generate_alternatives.py --study $*
	uv run python model/lm/score_merged.py --study $* --scenario-workers $(SCENARIO_WORKERS)

# Base-model alternatives: same two-stage pipeline with --base, so the LM is NOT
# shown the relationship description (the base ablation has no intimacy term). Writes
# lm_alternatives_base.jsonl / lm_runs_base.jsonl alongside the relationship-
# conditioned files; the base fit/CV loads them via desire_table_kwargs(base=True).
# `make lm-base K_RUNS=1` smoke first, then `make lm-base` for the full K=20.
lm-base: $(addprefix lm-base-,$(EXPERIMENTS_BASE))

$(addprefix lm-base-,$(EXPERIMENTS_BASE)): lm-base-%:
	K_RUNS=$(K_RUNS) ALT_T=$(ALT_T) CELL_WORKERS=$(CELL_WORKERS) uv run python model/lm/generate_alternatives.py --study $* --base
	uv run python model/lm/score_merged.py --study $* --base --scenario-workers $(SCENARIO_WORKERS)

# Diagnostic main/base vintages for K=1 prompt and pipeline smokes. These are
# gitignored and never feed fits, so they can be cleared and replaced without
# disturbing the canonical K=20 tables.
lm-diag: $(addprefix lm-diag-,$(EXPERIMENTS_INVERSE))

$(addprefix lm-diag-,$(EXPERIMENTS_ALL)): lm-diag-%:
	K_RUNS=$(K_RUNS) ALT_T=$(ALT_T) CELL_WORKERS=$(CELL_WORKERS) uv run python model/lm/generate_alternatives.py --study $* --arm-output-only
	uv run python model/lm/score_merged.py --study $* --arm --scenario-workers $(SCENARIO_WORKERS)

lm-base-diag: $(addprefix lm-base-diag-,$(EXPERIMENTS_BASE))

$(addprefix lm-base-diag-,$(EXPERIMENTS_BASE)): lm-base-diag-%:
	K_RUNS=$(K_RUNS) ALT_T=$(ALT_T) CELL_WORKERS=$(CELL_WORKERS) uv run python model/lm/generate_alternatives.py --study $* --base --arm-output-only
	uv run python model/lm/score_merged.py --study $* --base --arm --scenario-workers $(SCENARIO_WORKERS)

# =============================================================================
# Per-study file-target graph (the incremental core, roster-driven)
#
# One $(eval) per slug over EXPERIMENTS_ALL turns fit and CV into real file
# targets, so `make` rebuilds only what is out of date. The rules are generated
# from the roster — add a slug to EXPERIMENTS_ALL and its fit/cv nodes appear
# automatically, with no per-study block to hand-maintain.
#
# Two deliberate "source leaf" choices in the prerequisites:
#   * LM tables (lm_runs.jsonl, + lm_runs_base.jsonl for the base-model studies in
#     EXPERIMENTS_BASE) are expensive, committed artifacts. We depend ON them so
#     editing them re-triggers a refit, but give them NO build rule: `make fit`
#     must never fire a paid re-elicitation as a side effect. If a study's
#     lm_runs.jsonl is genuinely missing, make stops with a clear "No rule to make
#     target" error — the signal to run `make lm-<slug>` — instead of eliciting.
#   * The processed data CSV is likewise a committed input; regenerate it with the
#     explicit `make data-<slug>` step (json_to_csv writes three CSVs at once and
#     needs raw_data/, which most studies don't have yet, so it stays an explicit
#     stage rather than an auto-rebuild rule). It is listed via $(wildcard ...) so
#     a study whose data hasn't been collected yet (CSV absent) doesn't error the
#     graph — it just has no data edge and its fit script fails at runtime exactly
#     as today. When the CSV is committed, editing it re-triggers fit and cv.
#
# cv depends on the study's fit_results.json (each fold warm-starts from the
# full-data fit), so fit-before-cv is enforced by the graph itself, even under -j.
# fit_results.json / cv_trial_ll.jsonl each stand in for their co-written siblings
# (fit_restarts + fit_manifest; cv_preds_summary + cv_folds + cv_manifest) — one
# recipe writes the whole set atomically, and `clean` removes it as a set.
# =============================================================================

# Run-config passthrough for fit-/cv- targets (the reported config when unset):
#   make fit-food_inv_joint_de NO_REWEIGHTING=1     # the preregistered model
# With the var empty CONFIG_FLAGS is empty, so the recipes stay the reported
# invocation (the comparison-set reweighting, outputs/<slug>/). The non-default
# config routes the fit/CV to outputs/<slug>/alt/<tag>/ instead.
#
# Note that the file targets below are the study-root paths, so a non-default
# config's outputs are invisible to make: `make fit-<slug> NO_REWEIGHTING=1`
# re-runs every time and, worse, is a no-op when the ROOT output is already
# current. Drive multi-study non-default runs from a script that calls the fit/CV
# modules directly (see bin/) rather than through these targets.
NO_REWEIGHTING ?=
CONFIG_FLAGS = $(if $(NO_REWEIGHTING),--no-reweighting)

define MODEL_PIPELINE_RULES
model/outputs/$(1)/fit_results.json: \
    $$(wildcard data/$(1)/main_trials_long.csv) \
    model/outputs/lm/$(1)/lm_runs.jsonl \
    $$(if $$(filter $(1),$$(EXPERIMENTS_BASE)),model/outputs/lm/$(1)/lm_runs_base.jsonl)
	uv run python model/inverse/fit_$(1).py $$(CONFIG_FLAGS)

model/outputs/$(1)/cv_trial_ll.jsonl: \
    $$(wildcard data/$(1)/main_trials_long.csv) \
    model/outputs/lm/$(1)/lm_runs.jsonl \
    $$(if $$(filter $(1),$$(EXPERIMENTS_BASE)),model/outputs/lm/$(1)/lm_runs_base.jsonl) \
    model/outputs/$(1)/fit_results.json
	CV_WORKERS=$$(CV_WORKERS) CV_WORKER_THREADS=$$(CV_WORKER_THREADS) uv run python model/cv/cv_$(1).py $$(CONFIG_FLAGS)

# Phony aliases keep `make fit-<slug>` / `make cv-<slug>` working by name; the
# recipe lives on the file target, so they no-op when the output is current.
fit-$(1): model/outputs/$(1)/fit_results.json
cv-$(1): model/outputs/$(1)/cv_trial_ll.jsonl
endef

$(foreach s,$(EXPERIMENTS_ALL),$(eval $(call MODEL_PIPELINE_RULES,$(s))))

# =============================================================================
# Fits → outputs/<slug>/fit_results.json (+ fit_restarts.jsonl, fit_manifest.json)
# =============================================================================

fit: fit-inverse
fit-inverse: $(addprefix fit-,$(EXPERIMENTS_INVERSE))

# =============================================================================
# Leave-one-scenario-out CV → outputs/<slug>/cv_trial_ll.jsonl (primary metric)
# + cv_preds_summary.json + cv_folds.jsonl (+ cv_manifest.json). CV is the sole
# source of model predictions — every reported model-vs-human number is
# out-of-sample.
# =============================================================================

# CV_WORKERS: how many of a study's (variant × fold) refits run as parallel
# worker processes (48 jobs per study; the refits are independent and
# deterministic, so the outputs are identical to a sequential run — worker and
# thread counts change wall-clock only, never results). Left empty, each study
# uses its observer family's default from the dispatcher's _FAMILIES registry
# (the single source of truth) — currently 8 single-threaded workers for every
# family, now that the fast joint observers keep a worker at ~1.5 GB. Setting
# CV_WORKERS explicitly applies to every family; lower it when parallelizing
# studies with -j so (studies × CV_WORKERS × threads) stays ≲ the machine's
# cores. CV_RESTARTS and CV_PATIENCE pass through the environment as before.
#
# A CV run can be interrupted freely: completed fold refits land in
# outputs/<slug>/cv_checkpoint.jsonl as they finish, and the next run of the
# same study resumes from them (the checkpoint is fingerprint-guarded against
# any change to the data, LM tables, warm-start fit, refit config, or the
# model-math source files).
CV_WORKERS ?=
CV_WORKER_THREADS ?=

cv: cv-inverse
cv-inverse: $(addprefix cv-,$(EXPERIMENTS_INVERSE))

# =============================================================================
# Model comparison → outputs/<slug>/cv_model_comparison.json: the paper's
# primary statistic (full − ablation per-trial held-out LL with participant-
# bootstrap 95% CIs) plus the secondary model-vs-human correlations.
#
# model_comparison.py runs ONCE, reading every EXPERIMENTS_INVERSE study's CV
# outputs and writing each study's cv_model_comparison.json in a single pass.
# GNU make 3.81 has no grouped-target (`&:`) support, so we track one witness
# output — the first roster study's cv_model_comparison.json — whose prereqs are
# every study's primary CV output (cv_trial_ll.jsonl, co-written with the
# cv_preds_summary.json the script also reads). The one recipe regenerates them
# all; `clean` removes them together and the script always rewrites the full set,
# so they never drift apart. Because the prereqs are the CV outputs, a stale or
# absent one rebuilds fit→cv first — so `make model-comparison` is correct in
# isolation too (from clean it builds the whole chain).
# =============================================================================

CMP_WITNESS := model/outputs/$(firstword $(EXPERIMENTS_INVERSE))/cv_model_comparison.json

$(CMP_WITNESS): $(foreach s,$(EXPERIMENTS_INVERSE),model/outputs/$(s)/cv_trial_ll.jsonl)
	uv run python model/cv/model_comparison.py

model-comparison: $(CMP_WITNESS)

# =============================================================================
# Cross-study parameter transfer (exploratory; in no preregistration): score one
# study's fitted utility on another over the designed pairs, in two arms —
# `frozen` (zero free parameters) and `refit` (utility frozen, alpha_observer /
# sigma / eta re-estimated). Depends on every study's own CV, which supplies
# both the donor parameters and the ceiling each arm is read against.
#
# Not a file target: outputs land per (recipient, donor, arm) under
# outputs/<slug>/alt/transfer-*/, and the script skips pairs already on disk
# (pass ARGS='--force' to redo them).
# =============================================================================

.PHONY: transfer
transfer: $(foreach s,$(EXPERIMENTS_INVERSE),model/outputs/$(s)/cv_trial_ll.jsonl)
	uv run python model/cv/transfer.py $(ARGS)

# =============================================================================
# Pooled cross-experiment fits (exploratory; in no preregistration): one utility
# shared across a group of experiments, with each keeping its own response
# parameters. Rung 3 pools by stimulus domain, rung 4 pools all six; the
# reported per-experiment fits are rung 1. Depends on every study's own fit and
# CV, which supply the starting vectors and the per-experiment ceiling.
#
# Not a file target: outputs land per (experiment, group) under
# outputs/<slug>/alt/pooled-<group>/, and the script skips groups already on
# disk (pass ARGS='--force' to redo them).
# =============================================================================

.PHONY: pooled
pooled: $(foreach s,$(EXPERIMENTS_INVERSE),model/outputs/$(s)/cv_trial_ll.jsonl)
	uv run python model/cv/pooled.py $(ARGS)

# =============================================================================
# The same generalization arms on the PRIMARY metric. `transfer` and `pooled`
# score a shared utility by held-out log-likelihood, which the paper demotes for
# being insensitive to the modulation these studies test; this scores the arms
# they already ran on the condition-averaged correlation and the recovered
# modulation instead. Refits nothing -- it reads their CV predictions -- so it is
# cheap and must be re-run after either of them.
# =============================================================================

.PHONY: generalization-primary
generalization-primary:
	uv run python model/cv/generalization_primary.py $(ARGS)

# =============================================================================
# Results LaTeX (SIP_journal/): every number the results section states, as
# generated macros plus the table bodies. Depends on the comparison JSONs,
# so it rebuilds fit -> cv -> model-comparison first when any is stale, and the
# exporter itself verifies both manifests before emitting anything. Not a file
# target: SIP_journal/ is a separate gitignored repo that may not be present.
# =============================================================================

.PHONY: results-latex
results-latex: $(CMP_WITNESS)
	uv run python model/export_results_latex.py

# =============================================================================
# SI LM-elicitation validation figures (no API calls — read the persisted
# lm_runs.jsonl tables and CV outputs, and write PDFs to figures/si/). Each
# figure spans all six active studies.
# =============================================================================

figures-lm-si:
	uv run python figures/scripts/plot_si_validation.py
	uv run python figures/scripts/figure_si_consolidated.py

# =============================================================================
# Figure build inputs, shared by the panel and SI targets.
# =============================================================================

FIG_SCRIPTS := figures/scripts
FIG_SI := figures/si
# Shared code every figure depends on (a change here rebuilds them all). _agg.py
# belongs here even though only some figures draw a correlation panel: it is
# imported by figure_paper_panels.py, so leaving it out made an _agg.py-only edit
# a silent no-op for `make figures-panels`.
FIG_SHARED := $(FIG_SCRIPTS)/_data.py $(FIG_SCRIPTS)/_panels.py $(FIG_SCRIPTS)/_points.py \
              $(FIG_SCRIPTS)/_agg.py \
              plot_style.py study_registry.py model/cv/model_comparison.py

# The data + model outputs a study contributes to a figure. Deliberately only
# the CV *side-outputs* that are NOT make targets -- the data CSV and
# cv_preds_summary.json -- never cv_trial_ll.jsonl or cv_model_comparison.json
# (both targets), since depending on a target would try to rebuild the whole
# fit->CV chain whenever it looks stale. Wildcard-guarded so a study whose
# outputs don't exist yet simply drops out of the prereqs.
fig_inputs = $(wildcard data/$(1)/main_trials_long.csv) \
             $(wildcard model/outputs/$(1)/cv_preds_summary.json)

# =============================================================================
# Illustrator panels (figures/panels/): the manuscript's results figures are
# assembled by hand, so this writes the components — one four-column row per
# sub-study into results/, and the four shared legends into legends/, as PDFs
# whose text stays editable in Illustrator. Witness on the first panel, since
# the script writes all ten files in one pass.
# =============================================================================

PANEL_DIR := figures/panels
PANEL_WITNESS := $(PANEL_DIR)/results/panel_study1a.pdf

$(PANEL_WITNESS): $(FIG_SCRIPTS)/figure_paper_panels.py $(FIG_SHARED) \
    $(foreach s,$(EXPERIMENTS_ALL),$(call fig_inputs,$(s)))
	uv run python $(FIG_SCRIPTS)/figure_paper_panels.py

figures-panels: $(PANEL_WITNESS)

# =============================================================================
# Non-food domain panels: the Humans column of the Study 3 rows, split into the
# three sharing domains the non-food scenarios span (bodily access / shared
# exposure / private access) instead of averaged over all 16. Human data only,
# so this depends on the two non-food data CSVs rather than any model output.
# Witness on 3a's file; the script writes both.
# =============================================================================

NONFOOD_DOMAIN_WITNESS := $(PANEL_DIR)/results/panel_study3a_domains.pdf
EXPERIMENTS_NONFOOD_SET := $(filter nonfood_%,$(EXPERIMENTS_INVERSE))

$(NONFOOD_DOMAIN_WITNESS): $(FIG_SCRIPTS)/figure_nonfood_domains.py \
    $(FIG_SCRIPTS)/figure_paper_panels.py $(FIG_SHARED) \
    experiments/scenarios_nonfood.csv \
    $(foreach s,$(EXPERIMENTS_NONFOOD_SET),data/$(s)/main_trials_long.csv)
	uv run python $(FIG_SCRIPTS)/figure_nonfood_domains.py

figures-nonfood-domains: $(NONFOOD_DOMAIN_WITNESS)

# =============================================================================
# Schematic panels (figures/panels/schematic/): the method-figure components,
# rebuilt from the cached inputs in figures/figure_data/ (no API calls, no
# model outputs needed).
# =============================================================================

.PHONY: figures-schematic
figures-schematic:
	uv run python $(FIG_SCRIPTS)/figure_schematic_plots.py

# =============================================================================
# SI per-scenario figures: one 4x4 scenario facet grid per study, showing the
# cell means the main figures average over. Witness on the first study's file,
# since the script writes all six in one pass.
# =============================================================================

SI_SCENARIO_WITNESS := $(FIG_SI)/si_scenarios_study1a.pdf

$(SI_SCENARIO_WITNESS): $(FIG_SCRIPTS)/figure_si_scenarios.py $(FIG_SHARED) \
    $(foreach s,$(EXPERIMENTS_ALL),$(call fig_inputs,$(s)))
	uv run python $(FIG_SCRIPTS)/figure_si_scenarios.py

figures-si-scenarios: $(SI_SCENARIO_WITNESS)

# =============================================================================
# SI prior/posterior figures: the rating LEVELS behind the belief-update DV,
# from the human data only (no model outputs, no CV). Two figures rather than
# one two-panel figure, so each lands at \textwidth without being shrunk. Reads
# main_trials_long.csv, which is already long on `stage`, so these depend on the
# data CSVs alone. Witness on the first file; the script writes both.
# =============================================================================

SI_PRIORPOST := $(FIG_SI)/si_prior_posterior_levels.pdf

$(SI_PRIORPOST): $(FIG_SCRIPTS)/figure_si_prior_posterior.py $(FIG_SHARED) \
    $(foreach s,$(EXPERIMENTS_ALL),$(wildcard data/$(s)/main_trials_long.csv))
	uv run python $(FIG_SCRIPTS)/figure_si_prior_posterior.py

figures-si-prior-posterior: $(SI_PRIORPOST)

# =============================================================================
# SI preregistration-deviation figure: what the comparison-set reweighting bought
# per study, shown in the space the models actually predict in — both models'
# per-cell predictions beside the human means, all six studies. Reads the two
# configs' cv_preds_summary.json directly, so it needs only the CV runs (the
# preregistered fit + CV under alt/uniform-noreweight/ come from
# bin/prereg-eta0.sh).
# =============================================================================
SI_PREREG_PREDS := $(FIG_SI)/si_prereg_predictions.pdf

$(SI_PREREG_PREDS): $(FIG_SCRIPTS)/figure_si_prereg_predictions.py $(FIG_SHARED) \
    $(foreach s,$(EXPERIMENTS_INVERSE),model/outputs/$(s)/cv_preds_summary.json) \
    $(foreach s,$(EXPERIMENTS_INVERSE),\
      $(wildcard model/outputs/$(s)/alt/uniform-noreweight/cv_preds_summary.json))
	uv run python $(FIG_SCRIPTS)/figure_si_prereg_predictions.py

figures-si-prereg-predictions: $(SI_PREREG_PREDS)

# =============================================================================
# Manuscript figures: copy a curated set of generated PDFs from figures/si/
# into the journal Overleaf repo (SIP_journal/, its own git repo). Overleaf needs
# real, committed files — symlinks don't sync — so this physically copies them.
# After syncing, commit + push SIP_journal/ to Overleaf. Edit JOURNAL_FIGURES as
# the paper's figure set changes; each entry is  <name-in-figures-outputs/>:<name-in-main.tex>
# (so the analysis keeps descriptive names and the paper gets its own).
# =============================================================================

.PHONY: sync-journal-figures

JOURNAL_DIR := SIP_journal
JOURNAL_FIG_DIR := $(JOURNAL_DIR)/figures
# Only finished figures from figures/si/ are synced. The results figures are not
# here on purpose: they are assembled by hand in Illustrator from the components
# in figures/panels/, so the assembled artwork is placed into the manuscript
# directly rather than generated and copied.
# Only the figures main.tex actually includes; the other generated SI figures
# (run-spread, mixture-check, base-vs-full, choice-set-sizes, semantic-space,
# composition, set-similarity) were dropped from the sync on 2026-08-22 after
# the manuscript's shortening cut their sections. They are still generated into
# figures/si/ -- re-add a pair here if a revision brings one back.
JOURNAL_FIGURES := \
  si_lm_feature_structure_all.pdf:si-lm-feature-structure.pdf \
  si_lm_manipulation_checks_all.pdf:si-lm-manipulation-checks.pdf \
  si_lm_observed_scatter_all.pdf:si-lm-observed-scatter.pdf \
  si_lm_action_sets_combined.pdf:si-lm-action-sets-combined.pdf \
  si_lm_variability_checks.pdf:si-lm-variability-checks.pdf \
  si_scenarios_study1a.pdf:si-scenarios-study1a.pdf \
  si_scenarios_study1b.pdf:si-scenarios-study1b.pdf \
  si_scenarios_study2a.pdf:si-scenarios-study2a.pdf \
  si_scenarios_study2b.pdf:si-scenarios-study2b.pdf \
  si_scenarios_study3a.pdf:si-scenarios-study3a.pdf \
  si_scenarios_study3b.pdf:si-scenarios-study3b.pdf \
  si_prior_posterior_levels.pdf:si-prior-posterior-levels.pdf \
  si_prereg_predictions.pdf:si-prereg-predictions.pdf

sync-journal-figures:
	@test -d $(JOURNAL_DIR) || { echo "$(JOURNAL_DIR)/ not found (the Overleaf repo)"; exit 1; }
	@mkdir -p $(JOURNAL_FIG_DIR)
	@for pair in $(JOURNAL_FIGURES); do \
	  src=figures/si/$${pair%%:*}; dst=$(JOURNAL_FIG_DIR)/$${pair##*:}; \
	  if [ -f "$$src" ]; then cp "$$src" "$$dst" && echo "  $$src -> $$dst"; \
	  else echo "MISSING: $$src — render the analysis that generates it first" >&2; exit 1; fi; \
	done
	@echo "Synced $(words $(JOURNAL_FIGURES)) figure(s) to $(JOURNAL_FIG_DIR)/"

# =============================================================================
# Model visualization: interactive prediction explorer
#
# A self-contained HTML page (sliders over the utility weights -> the full
# model's predicted belief updates per study, no fitting). The source is kept
# local in the gitignored model/model_viz_nonfit/explorer/ (like the rest of
# model_viz_nonfit/); these targets build and publish it to the same MIT web
# space as the experiments, exactly like `make deploy-preview`.
# =============================================================================

.PHONY: explorer explorer-grid deploy-explorer

EXPLORER_DIR := model/model_viz_nonfit/explorer
# Same athena destination convention as bin/deploy-experiment; override per-run
# with `EXPLORER_DEST=... make deploy-explorer` if needed.
EXPLORER_DEST ?= aliciach@athena.dialup.mit.edu:~/www/sip/explorer

# Recompute the parameter-grid predictions by running the observer over the grid
# (~13 min; needs each study's lm_runs.jsonl). Only needed when the params/ranges
# in precompute_grid.py change -> $(EXPLORER_DIR)/predictions_grid.json.
explorer-grid:
	@test -d $(EXPLORER_DIR) || { echo "$(EXPLORER_DIR)/ not found (gitignored explorer source; kept on the primary machine only)"; exit 1; }
	uv run python $(EXPLORER_DIR)/precompute_grid.py $(EXPLORER_DIR)/predictions_grid.json

# Build the self-contained page from the template + precomputed grid (fast).
explorer:
	@test -d $(EXPLORER_DIR) || { echo "$(EXPLORER_DIR)/ not found (gitignored explorer source; kept on the primary machine only)"; exit 1; }
	uv run python $(EXPLORER_DIR)/build_explorer.py $(EXPLORER_DIR)/predictions_grid.json $(EXPLORER_DIR)/site/index.html

# Build, then publish to athena (enter the Athena password once). Served at
#   https://web.mit.edu/aliciach/www/sip/explorer/
deploy-explorer: explorer
	rsync -av --delete "$(EXPLORER_DIR)/site/" "$(EXPLORER_DEST)/"
	@echo "done. URL: https://web.mit.edu/aliciach/www/sip/explorer/"

# =============================================================================
# Utilities
# =============================================================================

# Git does not preserve file timestamps, so on a fresh clone `make all` may
# consider the committed fit/CV outputs out of date and start refitting instead
# of no-oping. This restamps the committed pipeline files in dependency order
# (metadata only — no file contents change); afterwards every pipeline target
# is current. Safe to run at any time. $(wildcard ...) restricts each touch to
# files that exist, so nothing is ever created.
.PHONY: freshen-outputs
freshen-outputs:
	touch $(wildcard data/*/main_trials_long.csv model/outputs/lm/*/lm_runs*.jsonl)
	touch $(wildcard model/outputs/*/fit_results.json model/outputs/*/fit_restarts.jsonl \
	  model/outputs/*/fit_manifest.json model/outputs/*/alt/*/fit_results.json \
	  model/outputs/*/alt/*/fit_restarts.jsonl model/outputs/*/alt/*/fit_manifest.json)
	touch $(wildcard model/outputs/*/cv_trial_ll.jsonl model/outputs/*/cv_preds_summary.json \
	  model/outputs/*/cv_folds.jsonl model/outputs/*/cv_manifest.json \
	  model/outputs/*/cv_run_deltas.json model/outputs/*/alt/*/cv_trial_ll.jsonl \
	  model/outputs/*/alt/*/cv_preds_summary.json model/outputs/*/alt/*/cv_folds.jsonl \
	  model/outputs/*/alt/*/cv_manifest.json)
	touch $(wildcard model/outputs/*/cv_model_comparison.json \
	  model/outputs/*/alt/compare_*.json model/outputs/group_correlations.json)
	touch $(wildcard figures/panels/*/*.pdf figures/si/*.pdf)

test:
	uv run python model/test_model_compliance.py
	uv run python model/test_run_config.py
	uv run python model/test_fit_protocol.py
	uv run python model/cv/test_checkpoint.py
	uv run python model/cv/test_model_comparison.py
	uv run python model/cv/test_contrast_tests.py
	uv run python model/cv/test_transfer.py
	uv run python model/cv/test_pooled.py
	uv run python model/lm/test_elicitation_guards.py
	uv run python analysis/test_json_to_csv.py
	uv run python test_roster_sync.py

clean:
	rm -f model/outputs/*/fit_results.json model/outputs/*/fit_restarts.jsonl model/outputs/*/fit_manifest.json
	rm -f model/outputs/*/cv_trial_ll.jsonl model/outputs/*/cv_preds_summary.json model/outputs/*/cv_folds.jsonl model/outputs/*/cv_manifest.json
	rm -f model/outputs/*/cv_model_comparison.json
	rm -f model/outputs/*/cv_checkpoint.jsonl
