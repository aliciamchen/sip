# Inverse planning in the context of sociological structure

This repository holds the experiments, data, model code, and figure pipeline behind a manuscript on social inference from sharing decisions: how people decide to share saliva-transferring food (and analogous non-food resources) with someone given the relationship between them, and how observers invert that decision to infer the relationship, desire, or physical effort from a single observed action. The manuscript is the source of truth for the studies' motivation, designs, and results; this README explains what is in the repo and how to reproduce the reported numbers and figures from it.

## Quick start

```bash
uv sync                 # Python deps (uv-managed .venv)
make all                # full pipeline: fit → CV → model comparison → figures
make help               # list all targets
```

Everything `make all` needs is checked into the repo: the processed experiment CSVs (`data/`) and the LM-elicited scenario tables (`model/outputs/lm/`), so the fits, cross-validation, and figures reproduce from a fresh clone without raw participant JSON or a Together AI key (see [Dependencies](#dependencies) for the uv install and the pip alternative). Figures land in `figures/panels/` and `figures/si/`.

One caveat on a fresh clone: git does not preserve file timestamps, so `make` may consider the committed fit and CV outputs out of date and start recomputing them. Run `make freshen-outputs` first — it restamps the committed files in dependency order (metadata only), after which `make all` is a no-op until an input actually changes. If a recompute does start, interrupting it is safe: the CV stage checkpoints and resumes.

## Pipeline at a glance

```
jsPsych experiments (experiments/) → JSON → json_to_csv.py → processed CSVs (data/)
                                                              ↓
LM elicitation (model/lm/) → scenario tables (model/outputs/lm/<slug>/)
                                                              ↓
                     fit (model/inverse/) → leave-one-scenario-out CV (model/cv/)
                                                              ↓
                model comparison (model/cv/model_comparison.py) → figures (figures/scripts/)
```

The `Makefile` exposes per-stage and per-experiment targets (`make fit-<slug>`, `make cv-<slug>`, etc.); `make help` lists them. The underlying script invocations are in [Reproducing the results](#reproducing-the-results) below.

## Repository structure

```
analysis/          Raw-data conversion (jsPsych JSON → anonymized CSVs)
bin/               Helper scripts: the experiment deploy script and the preregistered-model runner
data/              Processed experiment data (one folder per experiment slug)
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
  si/              Finished figures that go straight into the manuscript (PDF)
  figure_data/     Cached inputs for the schematic panels
  model-eqs/       Hand-authored equation glyphs
utils.py           Project-root path helper used by every script
study_registry.py  Per-study metadata (conditions, inferred latents, labels), shared by the model and figure code
plot_style.py      Shared matplotlib style and palettes for every figure script
test_roster_sync.py  Consistency test across the experiment lists (Makefile, registry, deploy script)
```

See the [data codebook](data/README.md), [experiments README](experiments/README.md), [figures README](figures/README.md), [model README](model/README.md), and [model outputs codebook](model/outputs/README.md) for details on each directory. Files prefixed with an underscore (`_data.py`, `_helpers.py`, …) are internal helpers imported by the entry-point scripts, not scripts to run. The `.claude/`, `.codex/`, and `.agents/` directories and `AGENTS.md` are configuration for the AI coding assistants used during development; they are not part of the pipeline and can be ignored.

## Experiments

The stable identifier for each experiment is its directory slug in `data/` and `experiments/`; paper-level study numbers can shift as the writeup evolves, so the code and data are organized by slug. The six studies (designs and procedures are in the manuscript):

| Slug | Study | Infers | Stimulus set |
|---|---|---|---|
| `food_inv_desire` | 1a | desire | food (16 scenarios) |
| `food_inv_joint_de` | 1b | desire + effort | food |
| `food_inv_intimacy` | 2a | intimacy | food |
| `food_inv_joint_ie` | 2b | intimacy + effort | food |
| `nonfood_inv_joint_de` | 3a | desire + effort | non-food (16 scenarios) |
| `nonfood_inv_joint_ie` | 3b | intimacy + effort | non-food |

The dependent variable throughout is the belief update (posterior minus prior slider rating, stored on the 0–1 scale). The [experiments README](experiments/README.md) documents the shared jsPsych infrastructure, counterbalancing, the comprehension/attention/memory checks, and a standalone trial-preview page (`make preview`).

The Python scripts in `experiments/` are the source of truth for the stimuli; each writes a `.csv` artifact next to it ([`scenarios.py`](experiments/scenarios.py) → `scenarios.csv` for the food set, [`scenarios_nonfood.py`](experiments/scenarios_nonfood.py) → `scenarios_nonfood.csv` for the non-food set). Edit the `.py` file and regenerate; the scenario tables in the paper's supplementary material are rendered from these CSVs by [`export_scenarios_latex.py`](experiments/export_scenarios_latex.py).

## Utility model

The actor model the observers invert (defined and motivated in the manuscript) is a softmax choice over

```
U(a | s, I, d) = w_v · d · g(a)  −  w_d · risk(a) · (1 − I)^γ  −  w_e · effort(a)
```

where `g`, `risk`, and `effort` are LM-elicited per-action features and desire `d` and intimacy `I` are each either given context or the inferred latent, depending on the study. The four fitted weights appear in the outputs as `param_w_v`, `param_w_d`, `param_w_e`, and `param_gamma` (the reward-term weight is named `w_v`; `w_d` is the risk weight). Three ablations are fit and compared — `full`, `discomfort_only` (risk term only), and `base` (no relational structure) — and which base variant each study's "Base" column reports is decided in one place, `study_registry.reported_base`.

## LM-elicited model components

The model's comparison sets and utility features come from a language model (Llama-3.3-70B via Together AI) rather than hand-stipulation, in two steps run for K = 20 independent runs per scenario × condition cell: [`generate_alternatives.py`](model/lm/generate_alternatives.py) proposes the counterfactual actions the pair could have taken, and [`score_merged.py`](model/lm/score_merged.py) scores the observed action and that run's alternatives together on `risk`, `effort`, and `g` (plus the given-condition scalars where a study needs them). The prompts are in [`prompts.py`](model/lm/prompts.py) and are reproduced verbatim in the supplementary material via [`export_prompts_latex.py`](model/lm/export_prompts_latex.py).

The elicited tables are committed under `model/outputs/lm/<slug>/` (see the [outputs codebook](model/outputs/README.md)), so nothing needs re-eliciting to reproduce the results; regenerating them requires `TOGETHER_API_KEY` in `.env`. Implementation details — cell grids, table shapes, loaders — are documented in the code and in [`.claude/rules/model.md`](.claude/rules/model.md).

## Model fitting and comparison

Each study fits its own parameters from its own data, and every reported prediction is out-of-sample from leave-one-scenario-out cross-validation — CV is the sole source of model predictions. The paper's model-comparison statistics come from `model/cv/model_comparison.py` (`make model-comparison`), with the complementary variance decomposition and condition-gradient statistics from `model/cv/contrast_tests.py`; both write into `model/outputs/<slug>/cv_model_comparison.json`. The likelihood, priors, and comparison construction are specified in the manuscript.

The default configuration reproduces the paper's numbers and writes to `model/outputs/<slug>/`. The one alternative configuration is the preregistered model: `NO_REWEIGHTING=1` drops the comparison-set reweighting (a declared deviation from the preregistration), writes to `model/outputs/<slug>/alt/uniform-noreweight/`, and is run for all six studies by `bin/prereg-eta0.sh`; `model/cv/model_comparison.py --compare-configs` then compares the two on matched trials.

Three exploratory analyses reported in the supplement live in `model/cv/`: `transfer.py` (`make transfer`), `pooled.py` (`make pooled`), and `generalization_primary.py` (`make generalization-primary`). Their outputs land under `model/outputs/<slug>/alt/` with summaries in `model/outputs/{transfer,pooled}/` and `model/outputs/generalization_primary.json`.

## Dependencies

Everything runs on Python, managed with [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if needed
uv sync                                            # install deps + create .venv
```

To use pip instead: `python3 -m venv .venv && source .venv/bin/activate && pip install .`. Run scripts with plain `python` rather than `uv run python` in that case.

## Reproducing the results

The Makefile is the recommended way to run the pipeline; the steps below show the underlying invocations for reference.

```bash
# Raw jsPsych JSON → anonymized CSVs (raw data is not included in the repository):
uv run python analysis/json_to_csv.py <experiment_slug>

# LM elicitation — only to REgenerate the committed tables (requires TOGETHER_API_KEY):
uv run python model/lm/generate_alternatives.py --study food_inv_desire
uv run python model/lm/score_merged.py          --study food_inv_desire

# Fit → CV → model comparison (per study; CV produces the out-of-sample predictions).
# The CV's independent (variant × fold) refits run as parallel worker processes;
# the outputs are identical to a sequential run, and an interrupted CV resumes
# from its completed folds on the next invocation:
uv run python model/inverse/fit_food_inv_desire.py
uv run python model/cv/cv_food_inv_desire.py
uv run python model/cv/model_comparison.py

# Render the main results figures (each script skips studies whose data or
# CV predictions don't exist yet, so this runs at any pipeline stage):
make figures-panels

# Run the full test suite (model compliance, fit/CV protocol, the statistics
# modules, data conversion, elicitation guards, experiment-list consistency):
make test
```

### Manuscript figures

The paper's figures are generated by the Python plotting scripts in `figures/scripts/`, all styled through the shared `plot_style.py`, which is the single source of truth for palettes, fonts, and the output directories. Output is split by who consumes it: `figures/panels/` holds the individual components that are assembled by hand in Illustrator (`make figures-panels` for the results rows and legends, `make figures-schematic` for the method-figure panels, `make figures-nonfood-domains` for the Study 3 per-domain view), and `figures/si/` holds finished figures that go straight into the manuscript (`make figures-lm-si` for the LM-validation figures, `make figures-si-scenarios` for the per-scenario grids, plus the prior/posterior and preregistration-comparison figures listed by `make help`). The figures use the Arial Nova font when it is installed and fall back to DejaVu Sans otherwise, so regenerated figures can differ from the paper's in font alone. The journal manuscript itself is a separate repository, so it is not part of this one.
