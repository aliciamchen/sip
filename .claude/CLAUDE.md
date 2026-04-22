# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a cognitive science research project investigating inverse planning and social inference in food-sharing scenarios involving saliva transfer. The project explores how people make decisions about sharing food based on relationship closeness (intimacy) and motivation (reward), and how observers infer relationship closeness or desire from observed actions.

The project comprises several experimental variants. Paper-level experiment numbering shifts as the writeup evolves, so the stable identifier for each variant is its data directory in `data/`.

- **Forward planning** (`data/forw_plan/`) — actors choose actions based on intimacy and motivation.
- **Intimacy inference, alternatives shown** (`data/inv_plan_intimacy_alt/`) — observers see the scenario plus all four candidate actions, then infer relationship closeness from the one the actor took.
- **Desire inference, alternatives shown** (`data/inv_plan_desire_alt/`) — observers see all four candidate actions, then infer the actor's desire from the one they took.
- **Intimacy inference, no alternatives shown** (`data/inv_plan_intimacy_noalt/`) — observers see only the single action the actor took and infer relationship closeness; on the model side, counterfactual alternatives are LM-generated.

## Intermediate conference submission

We are currently deciding how to extend this project for submission to a journal. An early version of this project has been submitted to a conference; the paper is in the `cogsci-2026` folder. The reviews for this submission are in `cogsci-2026/cogsci-2026-reviews.md`. The model in this submission is an old version that we are not using for the journal version. 

The journal version of the paper is in the `SIP_journal` folder. This is the most up-to-date version of the writeup. 

Both of these folders are gitignored but there are git repos within these folders that sync to Overleaf. 

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
# Available experiments: forw_plan, inv_plan_intimacy_alt, inv_plan_intimacy_noalt, inv_plan_desire_alt
```

For pilot experiments (in `analysis/legacy/`), use `json_to_csv_old_pilots.py`.

### LLM-derived scenario parameters (prerequisite for fits)

Both scripts hit Together AI (Llama-3.3-70B, 10 runs each). Requires `TOGETHER_API_KEY` in `.env`.

```bash
uv run python model/lm_scenario_params.py   # access + effort per (scenario, action) → lm_scenario_params.csv
uv run python model/lm_action_priors.py     # π(a|s) per (scenario, action) → lm_action_priors.csv
```

The `_prior` actor/observer variants require `lm_action_priors.csv`; without it they're silently skipped at fit/predict time.

### Model fitting

**Forward planning**:
```bash
uv run python model/fit_forward_planning.py
```
Fits 6 variants — 3 ablations (Base model / Discomfort-only / Full model) × {uniform prior, β-tempered LM prior}. The `_prior` variants are the canonical models; the uniform-prior variants are kept for comparison.

Outputs (in `model/outputs/`):
- `forward_planning_fits.csv` - Predictions for each data point (one `pred_*` column per variant)
- `forward_planning_fit_results.csv` - Fitted parameters (including `param_beta_prior` for the `_prior` variants) and NLL

**Alt-shown inverse planning (intimacy + desire inference)**:
```bash
uv run python model/fit_inverse_planning.py
```
Outputs (in `model/outputs/`):
- `inverse_planning_fit_results.csv` - Fitted observer parameters

**Generate alt-shown inverse planning predictions**:
```bash
uv run python model/generate_inverse_planning_preds.py
```
Outputs (in `model/outputs/`):
- `inv_plan_intimacy_alt_preds_full.csv` / `inv_plan_intimacy_alt_preds_summary.csv`
- `inv_plan_desire_alt_preds_full.csv` / `inv_plan_desire_alt_preds_summary.csv`

**No-alt intimacy inference** (LM-generated counterfactual alternatives):
```bash
uv run python model/fit_inverse_planning_noalt.py
uv run python model/generate_inverse_planning_noalt_preds.py
```
Outputs: `inverse_planning_noalt_fit_results.csv`, `inv_plan_intimacy_noalt_preds_full.csv`, `inv_plan_intimacy_noalt_preds_summary.csv`.

### Running analysis

Analysis files are Quarto documents (.qmd). Open in RStudio and render, or use:
```bash
quarto render analysis/forw-plan-analysis.qmd
quarto render analysis/inv-plan-intimacy-alt-analysis.qmd
quarto render analysis/inv-plan-desire-alt-analysis.qmd
quarto render analysis/inv-plan-intimacy-noalt-analysis.qmd
quarto render analysis/inv-plan-combined-correlation.qmd
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
