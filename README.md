# saliva-inverse-planning

## Experiments: 

- Experiment 1: Forward planning
    - Preregistration
- Experiment 2a: Inverse planning (intimacy)
- Experiment 2b: Inverse planning (reward)

## Dependencies

### Python Environment (using uv)

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync
```

### R Environment

The R packages are managed by `renv`. **R version 4.5.2 is required** (specified in the lockfile).

On a fresh clone, open R from the project directory. The first time you open R, renv will automatically bootstrap itself via `.Rprofile`. Then install all packages from the lockfile:

```r
renv::restore()
```

If the auto-bootstrap fails or you see "renv not found", manually bootstrap:
```r
install.packages("renv")
renv::activate()
renv::restore()
```

### System Dependencies (macOS)

For font rendering in plots:
```bash
brew install graphviz
brew install --cask xquartz
```

## Quick Start

Run the full pipeline using Make:

```bash
make all       # Run full pipeline: fit models, generate predictions, render analysis
```

Other useful targets:

```bash
make help      # Show all available targets
make fit       # Fit forward + inverse planning models
make predictions  # Generate model predictions
make analysis  # Render all Quarto analysis documents
make test      # Run model compliance tests
make clean     # Remove generated model outputs
```

The processed data CSVs are included in the repository, so `make all` works without raw JSON data.

## Manual Pipeline Steps

Convert raw data (not included in the repository) to csv format with anonymized participant ids:

```bash
uv run python analysis/json_to_csv.py forw_plan
uv run python analysis/json_to_csv.py inv_plan_intimacy
uv run python analysis/json_to_csv.py inv_plan_reward
```

Fit forward planning models

```bash
uv run python model/fit_forward_planning.py
```

Use forward planning parameters to fit inverse planning models and generate predictions
```bash
uv run python model/fit_inverse_planning.py
uv run python model/generate_inverse_planning_preds.py
```

Analyze data and generate plots

```bash
cd analysis
quarto render exp-1-analysis.qmd
quarto render exp-2a-inv-plan-intimacy-analysis.qmd
quarto render exp-2b-inv-plan-reward-analysis.qmd
quarto render exp-2-combined-correlation.qmd
```
