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

Everything `make all` needs is checked into the repo: the processed experiment CSVs (`data/`) and the LM-elicited scenario tables (`model/outputs/lm/`), so the fits, cross-validation, and analyses reproduce from a fresh clone without raw participant JSON or a Together AI key. Rendered analysis documents land in `_output/analysis/`; figures land in `figures/outputs/`.

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
  scripts/         Figure-generation scripts (main results, SI LM-validation, schematic panels)
  outputs/         Generated figures (PDF + PNG preview) written by the scripts
  schematic_panels/  Illustrator-linked schematic sub-panels (SVG + PDF)
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

The reward term multiplies **desire** `d ∈ [0, 1]` (how much the dyad wants the outcome) by the desire-free **goal-satisfaction** `g(a) ∈ [0, 1]` (how fully the action delivers the outcome). **`risk(a)`** measures how much an action opens the actors up to each other — bodily, informationally, or spatially — and is scaled by intimacy `I ∈ [0, 1]` through the power-law modulator `(1 − I)^γ` (γ is a free parameter): at high intimacy the risk penalty shrinks toward zero, so higher-risk actions become relatively more attractive. **`effort(a)`** is the physical cost of executing the action.

An observer inverts this actor model with Bayesian inference to recover whichever variables the study leaves latent — desire is a continuous latent in Studies 1a/1b and given context in 2a/2b, and vice versa for intimacy. Three ablations are fit and compared:

- **Full** — the complete utility above (the main model).
- **Discomfort-only** — only the risk-discomfort term; asks whether the relational risk signal alone can account for behavior.
- **Base** — reward and effort with no relational structure.

## LM-elicited model components

The model's action set and utility features are not hand-specified; they come from a language model (Llama-3.3-70B via Together AI), which plays the role of context-specific retrieval in open-ended scenarios. The elicitation has two steps, and both are repeated for K = 20 independent runs per scenario × condition cell:

1. **Generate alternatives** (`model/lm/generate_alternatives.py --study <slug>`): given the vignette, the observer-visible condition paragraphs, and the observed action, the LM proposes the plausible counterfactual actions the pair could have taken. The observer's actor then chooses from `{observed action} ∪ alternatives`. Participants never see these alternatives; they exist only inside the observer's model of how the actor chose.
2. **Score features** (`model/lm/score_merged.py --study <slug>`): the observed actions and that run's alternatives are scored together — one shared comparative frame — on `risk`, `effort`, and `g`, each on a 0–6 scale normalized to [0, 1]. Variables that are given rather than inferred in a study (the desire condition in 2a/2b, the relationship level in 1a/1b) are also rated by the LM, per run, on a 0–100 scale normalized to [0, 1].

Each run supplies its own alternatives and feature scores, and the run-to-run spread becomes part of the model's predicted response distribution (the simulated-observer mixture below). Because the base ablation has no intimacy term, its alternatives are elicited without the relationship description (`make lm-base`), so its choice set — and its predictions — cannot depend on the relationship.

The prompts live in [`model/lm/prompts.py`](model/lm/prompts.py) and are reproduced verbatim in the paper's supplementary material via [`export_prompts_latex.py`](model/lm/export_prompts_latex.py). The elicited tables are committed under `model/outputs/lm/<slug>/` (see the [outputs codebook](model/outputs/README.md)); regenerating them requires `TOGETHER_API_KEY` in `.env`. Implementation details — cell grids, table shapes, loaders — are documented in the code and in [`.claude/rules/model.md`](.claude/rules/model.md).

## Model fitting and comparison

Each study jointly fits its ablations' utility weights, an observer softmax temperature `α_obs`, and a response-noise scale `σ` from its own belief-update data by maximum likelihood (no parameters are shared or transferred between studies). A participant's belief update `u` is scored under the K-component simulated-observer mixture `(1/K) Σ_k N(u | δ_k, σ²)`, where `δ_k` is run k's predicted update; the joint studies score their two sliders together under a bivariate version.

All reported predictions are out-of-sample, from leave-one-scenario-out cross-validation: for each of the 16 scenarios, all parameters are refit on the other 15 and used to predict the held-out one. The paper's model-comparison statistics come from `model/cv/model_comparison.py` (`make model-comparison`): the difference between the full model and each ablation in per-trial held-out log-likelihood, with 95% confidence intervals from bootstrap resampling of participants (1,000 resamples), plus the secondary condition-averaged model-vs-human correlations. Results are written to `model/outputs/<slug>/cv_model_comparison.json`.

The fits and cross-validation default to the preregistered configuration — uniform priors over the inferred latent variables and the committed alternatives tables — and this default reproduces the paper's numbers, writing its outputs to `model/outputs/<slug>/` byte-for-byte as before. So that revisions of the model can be evaluated without disturbing that registered baseline, both `make fit-<slug>` and `make cv-<slug>` accept run-configuration variables that switch the observer onto informative LM-elicited priors (`PRIORS=informative`) or onto a different alternatives vintage (`ALTS_SUFFIX=...`). Any non-default configuration writes its outputs to a separate `model/outputs/<slug>/alt/<tag>/` directory whose tag encodes the configuration, so the preregistered baseline and each experimental variant coexist on disk and can be compared on matched trials. This keeps exploratory model development running alongside the registered analysis rather than overwriting it.

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
make figures-results

# Render an analysis document (demographics + data checks):
quarto render analysis/food-inv-desire-analysis.qmd

# Tests (`make test`): model compliance + CV checkpoint + JSON→CSV converter + roster sync
uv run python model/test_model_compliance.py
uv run python model/cv/test_checkpoint.py
uv run python analysis/test_json_to_csv.py
uv run python test_roster_sync.py
```

### Manuscript figures

The paper's figures are generated by the Python plotting scripts in `figures/scripts/`, all styled through the shared `plot_style.py` (the single source of truth for palettes and fonts) and written to `figures/outputs/`: `make figures-results` renders the main results figures and `make figures-lm-si` the SI LM-validation figures. The analysis qmds report demographics and data checks rather than producing any figure.

The journal manuscript is a separate Overleaf-synced git repository (`SIP_journal/`, not part of this repo), and Overleaf needs the figure files committed inside it rather than referenced from here. The `make sync-journal-figures` target copies a curated set of figure PDFs into `SIP_journal/figures/`, renaming each to the name the manuscript uses; the mapping is the `JOURNAL_FIGURES` list in the `Makefile`.
