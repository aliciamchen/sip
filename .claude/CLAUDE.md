# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a cognitive science research project investigating inverse planning and social inference in food-sharing scenarios involving saliva transfer. The project explores how people make decisions about sharing food based on relationship closeness (intimacy) and motivation (reward), and how observers infer relationship closeness or desire from observed actions.

The project comprises several experimental variants. Paper-level experiment numbering shifts as the writeup evolves, so the stable identifier for each variant is its data directory in `data/`.

- **Forward planning** (`data/forw_plan/`) — actors choose actions based on intimacy and motivation.
- **Intimacy inference, alternatives shown** (`data/inv_plan_intimacy_alt/`) — observers see the scenario plus all four candidate actions, then infer relationship closeness from the one the actor took.
- **Desire inference, alternatives shown** (`data/inv_plan_desire_alt/`) — observers see all four candidate actions, then infer the actor's desire from the one they took.
- **Intimacy inference, no alternatives shown** (`data/inv_plan_intimacy_noalt/`) — observers see only the single action the actor took and infer relationship closeness; on the model side, counterfactual alternatives are LM-generated.

A parallel pair of experiments manipulates **relative effort** instead of reward, using a different stimulus set (`experiments/scenarios_effort.csv`) in which each scenario has two actions and reward is held fixed at high. No data has been collected yet — only stimuli and jsPsych code exist. No `data/` directories, model fits, or analysis qmds have been added for these.

- **Forward planning, effort** (`experiments/forw_plan_effort/`) — intimacy (4 levels) × relative effort (2 levels); 2-slider linked-probability response.
- **Inverse planning, effort** (`experiments/inv_plan_effort/`) — observed action (2 levels) × relative effort (2 levels); prior/posterior intimacy sliders, both candidate actions shown.

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
Fits only `α_observer` per variant; actor weights are frozen from the all-data forward fit (same 4-action space as Exp 1, so weights transplant cleanly).
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
Unlike alt-shown, this pipeline jointly fits **all** actor weights + α_observer on the no-alt data. Actor weights are not frozen from Exp 1 because the padded observer reasons over a variable-length action set (different softmax competition structure than Exp 1's fixed 4 actions).
Outputs: `inverse_planning_noalt_fit_results.csv` (with per-variant `param_w_v`/`param_w_d`/`param_w_e`/`alpha_observer`), `inv_plan_intimacy_noalt_preds_full.csv`, `inv_plan_intimacy_noalt_preds_summary.csv`.

### Cross-validation

All model-vs-human correlations reported in the analysis qmds are **out-of-sample**, pooled from leave-one-scenario-out (LOSO) cross-validation. 16 folds × 3 variants per experiment.

```bash
uv run python model/cv/loso_forward.py         # Exp 1 forward planning (refits w_v, w_d, w_e, β per fold)
uv run python model/cv/loso_inverse_alt.py     # Exp 2a intimacy + 2b desire (refits only α_observer per fold)
uv run python model/cv/loso_inverse_noalt.py   # Exp 2c intimacy (joint fit — refits all weights per fold)
```

Outputs (in `model/outputs/`):
- `cv_loso_forward.csv` / `cv_loso_preds.csv` — per-fold fits + per-trial held-out forward predictions
- `cv_loso_inv_plan_intimacy_alt_preds_summary.csv` / `cv_loso_inv_plan_desire_alt_preds_summary.csv` — held-out per-cell predictions; `cv_loso_inverse_alt_folds.csv` for fitted α_observer per fold
- `cv_loso_inv_plan_intimacy_noalt_preds_summary.csv` / `cv_loso_inverse_noalt_folds.csv` — held-out per-cell predictions + per-fold joint-fit weights

The main analysis qmds load these CV CSVs as the source for all model plots; the non-CV `generate_inverse_planning_*_preds.py` CSVs are still generated (for anyone wanting the all-data fit) but are not what's displayed.

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
