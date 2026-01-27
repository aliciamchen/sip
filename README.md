# Inverse planning in the context of sociological structure

## Experiments: 

- Experiment 1: Forward planning
- Experiment 2a: Inverse planning (given desire, infer intimacy)
- Experiment 2b: Inverse planning (given intimacy, infer desire)

### Scenarios

See [scenarios here](experiments/scenarios.csv)

## Directory structure

```
├── analysis/          # R/Quarto analysis scripts and data processing
│   ├── *.qmd          # Quarto analysis documents
│   ├── json_to_csv.py # Raw data processing
│   └── utils.R        # Shared R utilities
├── data/              # Processed experiment data (see [codebook](data/README.md))
│   ├── forw_plan/     # Experiment 1
│   ├── inv_plan_intimacy/  # Experiment 2a
│   └── inv_plan_desire/    # Experiment 2b
├── experiments/       # jsPsych experiment code (see [README](experiments/README.md))
│   └── scenarios.csv  # Scenario definitions
├── model/             # Computational models (see [README](model/README.md))
│   ├── model_utils.py # Actor and observer model definitions
│   ├── fit_*.py       # Model fitting scripts
│   └── outputs/       # Fitted parameters and predictions
└── figures/           # Generated figures
```

## Dependencies

### Python environment

#### Option 1: Using uv (recommended)

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync
```

#### Option 2: Using pip

If you don't have uv, you can use pip with the standard `pyproject.toml`:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install .
```

Then run scripts with `python` instead of `uv run python`.

### R environment

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

### Quarto

[Quarto](https://quarto.org/docs/get-started/) is needed to render the analysis documents.

## Quick start

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
make clean     # Remove generated model outputs
```

The processed data CSVs are included in the repository, so `make all` works without raw JSON data.

## Manual pipeline steps

Convert raw data (not included in the repository) to csv format with anonymized participant ids:

```bash
uv run python analysis/json_to_csv.py forw_plan
uv run python analysis/json_to_csv.py inv_plan_intimacy
uv run python analysis/json_to_csv.py inv_plan_desire
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
quarto render exp-2b-inv-plan-desire-analysis.qmd
quarto render exp-2-combined-correlation.qmd
```

The plots are saved in the `figures/` directory.
