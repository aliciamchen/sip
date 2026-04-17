# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a cognitive science research project investigating inverse planning and social inference in food-sharing scenarios involving saliva transfer. The project explores how people make decisions about sharing food based on relationship closeness (intimacy) and motivation (reward), and how observers infer relationship closeness or desire from observed actions.

Right now the project includes three experiments:
- **Experiment 1 (Forward planning)**: Actors choose actions based on intimacy and motivation
- **Experiment 2a (Intimacy inference)**: Observers infer relationship closeness from observed actions
- **Experiment 2b (Desire inference)**: Observers infer the actor's desire (motivation) from observed actions

## Intermediate conference submission

We are currently deciding how to extend this project for submission to a journal. An early version of this project has been submitted to a conference; the paper is in the `cogsci-2026` folder. This folder is gitignored but there is a git repo in that folder that syncs to Overleaf. The reviews for this submission are in `cogsci-2026/cogsci-2026-reviews.md`.

## README.md

`README.md` in the project root is the public-facing documentation — reviewers and the public read it to understand the repo and how to run the code. When prompted to change CLAUDE.md or rules files, update the README too with relevant information. 

## Project instructions

Always use Context7 when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

## Environment setup

```bash
uv sync
```

This creates a `.venv` virtual environment with all dependencies. To run Python scripts:
```bash
uv run python script.py
```

Or activate the environment directly:
```bash
source .venv/bin/activate
```

Key dependencies: JAX, memo-lang (probabilistic modeling DSL), pandas, numpy, optax.

R dependencies are managed with `renv` (`renv.lock` at the repo root).

## Common commands

A `Makefile` at the repo root wraps the most common commands. Run `make help` for a list of targets. The sections below document the underlying commands directly.

### Data pipeline
Convert experiment JSON output to CSV:
```bash
uv run python analysis/json_to_csv.py <experiment_name>
# Available experiments: forw_plan, inv_plan_intimacy, inv_plan_desire
```

For pilot experiments (in `analysis/legacy/`), use `json_to_csv_old_pilots.py`.

### Model fitting

**Forward planning (Exp 1)**:
```bash
uv run python model/fit_forward_planning.py
```
Outputs (in `model/outputs/`):
- `forward_planning_fits.csv` - Predictions for each data point
- `forward_planning_fit_results.csv` - Fitted parameters and NLL

**Inverse planning (Exp 2a & 2b)**:
```bash
uv run python model/fit_inverse_planning.py
```
Outputs (in `model/outputs/`):
- `inverse_planning_fit_results.csv` - Fitted observer parameters

**Generate inverse planning predictions**:
```bash
uv run python model/generate_inverse_planning_preds.py
```
Outputs (in `model/outputs/`):
- `inv_plan_intimacy_preds_full.csv` / `inv_plan_intimacy_preds_summary.csv`
- `inv_plan_desire_preds_full.csv` / `inv_plan_desire_preds_summary.csv`

### Running analysis

Analysis files are Quarto documents (.qmd). Open in RStudio and render, or use:
```bash
quarto render analysis/exp-1-analysis.qmd
quarto render analysis/exp-2a-inv-plan-intimacy-analysis.qmd
quarto render analysis/exp-2b-inv-plan-desire-analysis.qmd
quarto render analysis/exp-2-combined-correlation.qmd
```

### Model tests
```bash
uv run python model/test_model_compliance.py
```

## Workflow

```
jsPsych experiments (experiments/) → JSON → json_to_csv.py → CSV (data/)
                                                              ↓
                                                    R analysis (analysis/)
                                                              ↓
                                            Compare with model predictions (model/outputs/)
```

## Repository layout

Topic-specific details load on demand via `.claude/rules/` when Claude reads files in these areas:

- `analysis/` — R/Quarto analysis scripts (see `.claude/rules/analysis.md`)
- `data/` — processed experiment data (see `.claude/rules/data.md`)
- `experiments/` — jsPsych experiment code (see `.claude/rules/experiments.md`)
- `model/` — computational models (see `.claude/rules/model.md`)

Other top-level directories (kept here because they're cross-cutting or small):

- `cogsci-2026/` - CogSci 2026 paper source (LaTeX: `main.tex`, `cogsci.sty`, bibliography, figures). This is gitignored
- `submissions/` - Compiled paper PDFs for submission
- `figures/` - Rendered figures used in the paper
- `LM_evals/` - Language-model evaluation code (`experiments/`, `providers/`)
- `_quarto.yml` / `_output/` - Quarto project configuration and rendered output

## Utility functions

- `utils.py` - Provides `get_project_root()` for constructing paths relative to project root
- `analysis/utils.R` - Shared R functions: `setup_analysis()`, `boot_cor()`, `calculate_belief_update()`
