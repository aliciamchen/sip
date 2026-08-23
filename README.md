# Inverse planning in the context of sociological structure

Cognitive science research on social inference and food sharing: how people decide to share saliva-transferring food with someone given the relationship between them, and how observers infer the relationship, desire, or physical effort from observed actions.

The current manuscript organizes the work into six inverse-planning studies, all built on a 3-action stimulus structure (no sharing / low-risk sharing / high-risk sharing) that lets the three latent variables — desire, effort, and intimacy — be crossed and inferred in different combinations. In each study the observer sees a single action and infers one or two of the latent variables: Study 1a infers desire, Study 1b jointly infers desire and effort, Study 2a infers intimacy, and Study 2b jointly infers intimacy and effort, all on a 16-scenario food stimulus set. Study 3 tests whether the same model generalizes beyond food to other forms of interpersonal vulnerability (bodily access, shared physical exposure, and private access) using a matched 16-scenario non-food set: Study 3a repeats 1b's joint desire + effort design and Study 3b repeats 2b's joint intimacy + effort design on those scenarios.

## Quick start

```bash
uv sync                 # Python deps (uv-managed .venv)
# R deps: open R from the project root, then run renv::restore()
make all                # full pipeline: fit → CV → model comparison → render qmds
make help               # list all targets
```

Rendering the analysis documents also needs Quarto. See [Dependencies](#dependencies) below for the full install details (uv, the pip alternative, the required R version, and Quarto).

Everything `make all` needs is checked into the repo: the processed experiment CSVs (`data/`) and the LM-elicited scenario tables (`model/outputs/lm/`), so the fits, cross-validation, and analyses reproduce from a fresh clone without raw participant JSON or a Together AI key. Rendered analysis documents land in `_output/analysis/`; figures land in `figures/panels/` and `figures/si/`.

## Pipeline at a glance

```
jsPsych experiments (experiments/) → JSON → json_to_csv.py → processed CSVs (data/)
                                                              ↓
LM elicitation (model/lm/) → scenario tables (model/outputs/lm/<slug>/)
                                                              ↓
                     fit (model/inverse/) → leave-one-scenario-out CV (model/cv/)
                                                              ↓
                     model comparison (model/cv/model_comparison.py) + analysis (analysis/)
```

The `Makefile` exposes per-stage and per-experiment targets (`make fit-<slug>`, `make cv-<slug>`, etc.); `make help` lists them. The underlying script invocations are in [Reproducing the results](#reproducing-the-results) below.

## Repository structure

```
analysis/          Data processing plus R/Quarto demographics and data-check documents
bin/               Helper scripts: the experiment deploy script and a git-worktree environment setup
data/              Processed experiment data (one folder per experiment slug)
docs/              Design notes recording modeling decisions
experiments/       jsPsych experiment code + scenario definitions
model/             Computational models
  inverse/         Per-experiment inverse-planning fit scripts
  cv/              Per-experiment LOSO cross-validation + model comparison
  lm/              LM-elicitation scripts (Together AI)
  outputs/         Fitted parameters and CV results (predictions, all out-of-sample)
    lm/            LM-elicited scenario tables
    <slug>/        Per-experiment outputs
preregs/           AsPredicted-format preregistration documents (all six studies)
figures/           Paper figures
  scripts/         Figure-generation scripts (results panels, SI LM-validation, schematic panels)
  panels/          Illustrator components, never assembled figures (PDF)
    results/       One four-column row per sub-study, plus the model-vs-humans panel
    legends/       The four shared legends, one artboard each
    schematic/     Schematic sub-panels
  si/              Finished figures that go straight into the manuscript (PDF)
  figure_data/     Cached inputs for the schematic panels
  model-eqs/       Hand-authored equation glyphs
```

See the [data codebook](data/README.md), [experiments README](experiments/README.md), [model README](model/README.md), and [model outputs codebook](model/outputs/README.md) for details on each directory.

## Experiments

The stable identifier for each experiment is its directory slug in `data/` and `experiments/` (paper-level numbers change as the writeup evolves).

On each trial the participant reads a vignette plus whichever condition paragraphs the design reveals, rates their beliefs, sees a single observed action, and rates again; the dependent variable is the belief update (posterior minus prior rating). All ratings are continuous 0–100 sliders, stored on the 0–1 scale.

- **Study 1a — desire inference** (`food_inv_desire/`) — known: effort + intimacy. Inferred: desire. Design: 2 × 4 × 3.
- **Study 1b — joint desire + effort** (`food_inv_joint_de/`) — known: intimacy. Inferred jointly: desire and the physical world state. Design: 4 × 3, two sliders per trial.
- **Study 2a — intimacy inference** (`food_inv_intimacy/`) — known: desire + effort. Inferred: intimacy. Design: 2 × 2 × 3.
- **Study 2b — joint intimacy + effort** (`food_inv_joint_ie/`) — known: desire. Inferred jointly: intimacy and the physical world state. Design: 2 × 3, two sliders per trial.
- **Study 3a — joint desire + effort, non-food** (`nonfood_inv_joint_de/`) — Study 1b's design on the non-food scenario set.
- **Study 3b — joint intimacy + effort, non-food** (`nonfood_inv_joint_ie/`) — Study 2b's design on the non-food scenario set.

The [experiments README](experiments/README.md) documents the shared jsPsych infrastructure, the counterbalancing scheme, the comprehension/attention/memory checks, and a standalone trial-preview page (`make preview`). Data from earlier, superseded experiments is archived under `data/legacy/` and described in [data/legacy/README.md](data/legacy/README.md).

### Scenarios

The Python scripts in `experiments/` are the source of truth for the stimuli; each writes a `.csv` artifact next to it ([`scenarios.py`](experiments/scenarios.py) → `scenarios.csv` for the food set of Studies 1–2, [`scenarios_nonfood.py`](experiments/scenarios_nonfood.py) → `scenarios_nonfood.csv` for the non-food set of Study 3). Edit the `.py` file and regenerate; the scenario tables in the paper's supplementary material are rendered from these CSVs by [`export_scenarios_latex.py`](experiments/export_scenarios_latex.py).

## Utility model

A joint actor chooses an action by softmaxing over the utility:

```
P(a | s, I, d) ∝ exp( U(a | s, I, d) )

U(a | s, I, d) = w_v · d · g(a)
               − w_d · risk(a) · (1 − I)^γ
               − w_e · effort(a)
```

The reward term multiplies **desire** `d ∈ [0, 1]` (how much the dyad wants the outcome) by the desire-free **goal-satisfaction** `g(a) ∈ [0, 1]` (how fully the action delivers the outcome). **`risk(a)`** measures how much an action opens the actors up to each other — bodily, informationally, or spatially — and is scaled by intimacy `I ∈ [0, 1]` through the power-law modulator `(1 − I)^γ` (γ is a free parameter): at high intimacy the risk penalty shrinks toward zero, so higher-risk actions become relatively more attractive. **`effort(a)`** is the total executional cost of carrying out the action across the dyad, counting physical work, preparation, equipment, time, and utterance-production work performed by either person.

An observer inverts this actor model with Bayesian inference to recover whichever variables the study leaves latent — desire is a continuous latent in Studies 1a/1b and given context in 2a/2b, and vice versa for intimacy. Three ablations are fit and compared:

- **Full** — the complete utility above (the main model).
- **Discomfort-only** — only the risk-discomfort term; asks whether the relational risk signal alone can account for behavior.
- **Base** — reward and effort with no relational structure.

## LM-elicited model components

The model's action set and utility features are not hand-specified; they come from a language model (Llama-3.3-70B via Together AI), which plays the role of context-specific retrieval in open-ended scenarios. The elicitation has two steps, and both are repeated for K = 20 independent runs per scenario × condition cell:

1. **Generate alternatives** (`model/lm/generate_alternatives.py --study <slug>`): given the vignette, the observer-visible condition paragraphs, and the observed action, the LM proposes the plausible counterfactual actions the pair could have taken. The observer's actor then chooses from `{observed action} ∪ alternatives`. Participants never see these alternatives; they exist only inside the observer's model of how the actor chose.
2. **Score features** (`model/lm/score_merged.py --study <slug>`): the observed actions and that run's alternatives are scored together — one shared comparative frame — on `risk`, `effort`, and `g`, each on a 0–6 scale normalized to [0, 1]. Variables that are given rather than inferred in a study (the desire condition in 2a/2b, the relationship level in 1a/1b) are also rated by the LM, per run, on a 0–100 scale normalized to [0, 1].

Each run supplies its own alternatives and feature scores, and the run-to-run spread becomes part of the model's predicted response distribution through the elicitation-sample mixture below. Because the base ablation has no intimacy term, its alternatives are elicited without the relationship description (`make lm-base`), so its choice set — and its predictions — cannot depend on the relationship. That design makes the preregistered base differ from the full model in two ways at once, both the missing discomfort term and the different choice set, so the paper reports as "Base" a variant that keeps base's utility but scores it against the full model's relationship-conditioned choice set. The comparison isolates the discomfort term, and the preregistered version is reported alongside it in the preregistration-deviation section. Which variant each study reports is decided in one place, `study_registry.reported_base`.

The prompts live in [`model/lm/prompts.py`](model/lm/prompts.py) and are reproduced verbatim in the paper's supplementary material via [`export_prompts_latex.py`](model/lm/export_prompts_latex.py). The elicited tables are committed under `model/outputs/lm/<slug>/` (see the [outputs codebook](model/outputs/README.md)); regenerating them requires `TOGETHER_API_KEY` in `.env`. Implementation details — cell grids, table shapes, loaders — are documented in the code and in [`.claude/rules/model.md`](.claude/rules/model.md).

## Model fitting and comparison

Each study jointly fits its ablations' utility weights, an observer softmax temperature `α_obs`, and a response-noise scale `σ` from its own belief-update data by maximum likelihood. No parameters are shared or transferred between studies in any reported fit; the exploratory transfer analysis described below is the one place where a study is scored under another study's parameters, and it writes to its own output directories. A participant's belief update `u` is scored under the K-component elicitation-sample mixture `(1/K) Σ_k N(u | δ_k, σ²)`, where `δ_k` is run k's predicted update; the joint studies score their two sliders together under a bivariate version.

All reported predictions are out-of-sample, from leave-one-scenario-out cross-validation: for each of the 16 scenarios, all parameters are refit on the other 15 and used to predict the held-out one. The paper's model-comparison statistics come from `model/cv/model_comparison.py` (`make model-comparison`): the difference between the full model and each ablation in per-trial held-out log-likelihood, with 95% confidence intervals from bootstrap resampling of participants (1,000 resamples), plus the secondary condition-averaged model-vs-human correlations. Results are written to `model/outputs/<slug>/cv_model_comparison.json`.

That primary metric is a global fit index, which is a low-powered test of the specific claim these studies make. The paper's hypothesis is about a modulation -- how the condition that is *given* to the observer changes what an action reveals about the variables that are not -- and that modulation accounts for only 1-3% of the variance in individual trials, most of which is response noise no model can predict. Two further sets of statistics, computed by `model/cv/contrast_tests.py` and written into the same JSON file, complement it: a variance decomposition that measures where a study's trial-level variance actually lives -- the within-cell share no model can predict, the observed action's share, and the action and given condition taken jointly, which is what the paper's SI table reports -- and a gradient test that scores the modulation itself, comparing how much each model variant's held-out predictions change across the given condition's levels against how much human judgments do. Both are secondary to the preregistered comparison; the variance decomposition is reported in the SI, while the gradient test remains a computed diagnostic after the section quoting it was cut in the manuscript's shortening.

The fits and cross-validation default to the configuration the paper reports — uniform priors over the inferred latent variables, plus the surprise-weighted reweighting of the comparison set — and this default reproduces the paper's numbers, writing its outputs to `model/outputs/<slug>/` byte-for-byte as before. So that alternative configurations can be evaluated without disturbing that baseline, both `make fit-<slug>` and `make cv-<slug>` accept run-configuration variables. `PRIORS=informative` (or `PRIORS=informative:<latents>` for a subset) switches the observer onto informative LM-elicited priors. `NO_REWEIGHTING=1` drops the comparison-set reweighting, which recovers the preregistered model exactly: the reweighting is a deviation from what was registered, so the paper has to be able to report the preregistered model's held-out numbers beside the reported ones, and `bin/prereg-eta0.sh` runs that comparison across all six studies.

Any non-default configuration writes its outputs to a separate `model/outputs/<slug>/alt/<tag>/` directory whose tag encodes the configuration, so the reported baseline and each variant coexist on disk and can be compared on matched trials with `model/cv/model_comparison.py --compare-configs <a> <b>`. This keeps exploratory model development running alongside the reported analysis rather than overwriting it. Note that the Makefile's file targets are the study-root paths, so multi-study runs of a non-default configuration should be driven from a script rather than through `make`.

### Cross-study parameter transfer

Because every study is fit on its own data, six studies give six parameter sets, and nothing in the reported analysis says whether the utility function is one stable psychological object or six separately flexible fits. `model/cv/transfer.py` (`make transfer`) asks that question directly: it takes one study's fitted utility weights, puts them in another study's model, and scores the recipient's data out of sample under the same leave-one-scenario-out protocol as every reported number. This analysis is exploratory and appears in no preregistration.

It runs on designed pairs rather than as a single leave-one-study-out number, because the studies differ in more than one way at a time and a single average would hide which ones are compatible. Studies 1a and 2a, and Studies 1b and 2b, share their stimuli and differ only in which latent variable the observer infers, so they test whether the actor's utility is invariant to the question the observer is asked. Studies 1b and 3a, and 2b and 3b, are design-matched across the food and nonfood stimulus sets, so they test whether the utility survives a change of domain. Each pair is run in both directions.

Two arms separate two things that a naive transfer test would confound. The `frozen` arm estimates nothing on the recipient at all: the donor's whole vector, including `α_obs` and `σ`, is used verbatim. That is the strictest test, but `α_obs` is not really comparable across studies, since it sharpens a posterior over different latent spaces and trades off against the overall scale of the weights. The `refit` arm therefore freezes only the utility weights and re-estimates the response layer (`α_obs`, `σ`, and the reweighting gain `η`) on the recipient, which is the arm to read for the psychological claim. Both are compared against the recipient's own cross-validation as a ceiling; because held-out likelihood carries no parameter penalty, a transfer with no free parameters can in principle beat a study's own fit, which would mean that fit overfits.

The reweighting gain `η` is treated as recipient-side rather than transferred, because its scope is defined per study by which questions are contrastive-only there, and Study 1a has none at all. Outputs land in `model/outputs/<slug>/alt/transfer-<donor>-<arm>/` in the standard cross-validation file set, with the summary in `model/outputs/transfer/transfer_summary.json`.

### Pooled fits: how many utilities do the data require?

The transfer analysis asks whether one experiment's weights happen to work on another. `model/cv/pooled.py` (`make pooled`) asks the question directly, by fitting a single shared utility across a group of experiments and cross-validating it. Three groupings are compared, each nesting in the one above: one utility per experiment (the reported model), one per stimulus domain, and one across all six. Only the utility weights are shared — each experiment keeps its own `α_obs`, `σ` and `η`, for the reasons the transfer analysis established. Like the transfer analysis, this is exploratory and supplements the per-experiment fits rather than replacing them.

The two steps ask different questions. Going from per-experiment to per-domain asks whether one utility serves every observer task within a domain; since the utility describes the actor and the inference problem is a property of the observer's task, that is closer to a coherence requirement than a hypothesis, with the caveat that the elicited comparison sets also differ across experiments. Going from per-domain to a single utility asks whether one utility serves both domains, where "risk" means saliva-sharing discomfort on one side and disclosure risk on the other. That is a substantive claim, and a loss there is a finding rather than a defect.

It runs in two stages so that no scoring code is duplicated: each fold's shared utility is fitted on the group's training trials, and then each experiment's held-out trials are scored by handing its slice of that fold's vector to the ordinary cross-validation machinery with nothing left free. Fold *k* holds out scenario index *k* in every experiment, so all three groupings have the same sixteen folds and identical held-out trials and can be compared on matched trials. Results are always reported per experiment rather than as a pooled total, because the joint objective weights experiments by trial count and by how many sliders they use.

### Scoring the generalization arms on the primary metric

Both analyses above compare models by held-out log-likelihood, which is the preregistered metric but not the one the paper reports. That measure is insensitive by construction to the effect these experiments test, because the manipulated condition accounts for only a few percent of trial-level variance, so a shared utility could keep most of the predicted response while losing the relationship or desire modulation and the difference would still look small. Since the generalization analyses make an equivalence claim, that matters: a null on an insensitive measure is weak evidence.

`model/cv/generalization_primary.py` (`make generalization-primary`) therefore scores the same arms on the condition-averaged model-vs-human correlation the paper reports and on the fraction of participants' modulation each arm recovers. Only the second is actually sensitive to the modulation: measured across the six studies, the full model and the ablation that differs from it only in relationship-sensitivity differ by 0.001-0.014 in correlation but by 30-90 percentage points in recovered gradient, so the correlation is a descriptive index of overall fit rather than a test of the claim. The paper's SI table reports the correlation only; the recovered-gradient numbers stay in `generalization_primary.json`. Nothing is refitted, since the transfer and pooled runs already wrote standard cross-validation output sets; this reads their held-out predictions, so it is cheap and needs re-running whenever either of them is. The three measures do not always rank the arms the same way, which is the expected consequence of a modulation worth a few percent of trial variance.

## Dependencies

### Python (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if needed
uv sync                                            # install deps + create .venv
```

To use pip instead: `python3 -m venv .venv && source .venv/bin/activate && pip install .`. Run scripts with plain `python` rather than `uv run python` in that case.

### R (renv)

R version 4.5.2 is required (specified in the lockfile). Open R from the project root — `.Rprofile` will bootstrap renv automatically, then run `renv::restore()` to install all packages. If the auto-bootstrap fails: `install.packages("renv"); renv::activate(); renv::restore()`.

### Quarto

[Quarto](https://quarto.org/docs/get-started/) is needed to render the analysis documents.

## Reproducing the results

The Makefile is the recommended way to run the pipeline; the steps below show the underlying invocations for reference.

```bash
# Raw jsPsych JSON → anonymized CSVs (raw data is not included in the repository):
uv run python analysis/json_to_csv.py <experiment_slug>

# LM elicitation — only to REgenerate the committed tables (requires TOGETHER_API_KEY);
# see "LM-elicited model components" above for what these two steps do:
uv run python model/lm/generate_alternatives.py --study food_inv_desire
uv run python model/lm/score_merged.py          --study food_inv_desire

# Fit → CV → model comparison (per study; CV produces the out-of-sample predictions).
# The CV's independent (variant × fold) refits run as parallel worker processes
# (8 single-threaded workers by default). The outputs are identical to a
# sequential run, so CV_WORKERS / CV_WORKER_THREADS only change the wall-clock
# time. A CV run that is interrupted resumes from its completed folds on the
# next invocation, via a checkpoint file that is discarded automatically
# whenever the study's inputs, fitting configuration, or model code change:
uv run python model/inverse/fit_food_inv_desire.py
uv run python model/cv/cv_food_inv_desire.py
uv run python model/cv/model_comparison.py

# Render the main results figures (each script skips studies whose data or
# CV predictions don't exist yet, so this runs at any pipeline stage):
make figures-panels

# Render an analysis document (demographics + data checks):
quarto render analysis/food-inv-desire-analysis.qmd

# Tests (`make test`): model compliance + CV checkpoint + JSON→CSV converter + roster sync
uv run python model/test_model_compliance.py
uv run python model/cv/test_checkpoint.py
uv run python analysis/test_json_to_csv.py
uv run python test_roster_sync.py
```

### Manuscript figures

The paper's figures are generated by the Python plotting scripts in `figures/scripts/`, all styled through the shared `plot_style.py`, which is the single source of truth for palettes, fonts, and the output directories. Output is split by who consumes it: `figures/panels/` holds the individual components that are assembled by hand in Illustrator, and `figures/si/` holds finished figures that go straight into the manuscript. That split is the rule to preserve when adding a script -- `panels/` should never accumulate an assembled multi-panel figure.

`make figures-panels` writes the results components: one four-column row per sub-study into `panels/results/`, plus the four legends they share into `panels/legends/`, as PDFs whose text stays editable in Illustrator. `make figures-lm-si` renders the SI LM-validation figures. The analysis qmds report demographics and data checks rather than producing any figure.

The non-food scenarios span three domains of interpersonal vulnerability (bodily access, shared physical exposure, and private access), which the Study 3 rows average over. `make figures-nonfood-domains` draws the human side of those rows one domain at a time instead: a three-column panel per non-food study (`panel_study3a_domains.pdf` and `panel_study3b_domains.pdf`), in the same encoding and at the same scale as the results panels, so it can be placed beside them in Illustrator with the same legends.

Because the main results figures average over the 16 scenarios, `make figures-si-scenarios` renders the supplementary per-scenario view: one 4x4 facet grid per study (`si_scenarios_study1a.pdf` and friends), where each facet is a single scenario's human cell means with bootstrap CIs. These carry no legend, since they are assembled in Illustrator alongside the panel legends.

The journal manuscript is a separate Overleaf-synced git repository (`SIP_journal/`, not part of this repo), and Overleaf needs the figure files committed inside it rather than referenced from here. The `make sync-journal-figures` target copies a curated set of figure PDFs into `SIP_journal/figures/`, renaming each to the name the manuscript uses; the mapping is the `JOURNAL_FIGURES` list in the `Makefile`.
