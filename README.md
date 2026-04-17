# Inverse planning in the context of sociological structure

## Experiments

- Experiment 1: Forward planning — actors choose actions based on intimacy and desire
- Experiment 2a: Inverse planning — observers see an action (with the actor's desire known) and infer intimacy
- Experiment 2b: Inverse planning — observers see an action (with the actor's intimacy known) and infer desire

### Scenarios

See [scenarios here](experiments/scenarios.csv)

## Utility model

The canonical utility function is:

```
U(a|s, I) = w_v · V(a|s)
          + w_r · access(a) · I
          − w_d · access(a) · (1 − I)
          − w_e · effort(a)
```

Here `V(a|s)` is the base reward of the food (not scaled by intimacy), and `access(a)` is a graded measure of how much an action opens the actor up to the other person — their body, private information, and physical space. Intimacy `I` converts access into positive reward (close relationships) or negative discomfort (distant relationships). The fixed vectors are `access(a) = [0, 0.3, 1, 2]` across the four actions and `effort(a) = [0, 1, 1, 1]`. The jump between action 1 and action 2 is larger than the other steps because crossing into saliva transfer is a qualitatively bigger increment in bodily exposure than merely sharing a meal.

Three variants are fit and compared for both the actor (Experiment 1) and observer (Experiments 2a/2b) models:
- **access_full** — the full utility above: food reward, both access terms, and effort (the main model)
- **access_only** — only the two access terms; drops food reward and physical effort to ask whether the access signal alone can account for behavior
- **no_access** — `w_v · V − w_e · effort` (the Base model)

The pre-registered models (full / vanilla / discomfort_only, which scale food reward by intimacy) are retained for comparison. Fit and comparison results for the three access variants are in `model/outputs/access_model_forward_fit_results.csv` and `model/outputs/access_model_inverse_fit_results.csv`.

### LLM-derived scenario-specific values

By default, `access(a)`, `effort(a)`, and `V(a|s)` are stipulated vectors that are the same across all 16 scenarios. As an alternative, `model/lm_scenario_params.py` uses Llama-3.3-70B via Together AI to generate scenario-specific values for each of these parameters (10 runs averaged, mean ± std saved). The prompts ask the LLM to rate, on a 0-6 scale:

- `access`: how much each action opens each person up to the other — physically (bodily substance transfer, skin contact, spatial proximity), informationally, or both
- `effort`: how much social-coordination effort each action requires
- `reward`: how enjoyable sharing the food in this situation would be (scenario-level)

Running the script writes `model/outputs/lm_scenario_params.csv`; the fitting and prediction scripts then produce `_llm` companions to each access variant (`access_full_llm`, `access_only_llm`, `no_access_llm`) that use the LLM values. Running this requires `TOGETHER_API_KEY` in `.env` and costs a few Together API calls.

## Directory structure

```
├── analysis/          # R/Quarto analysis scripts and data processing
│   ├── *.qmd          # Quarto analysis documents
│   ├── json_to_csv.py # Raw data processing
│   └── utils.R        # Shared R utilities
├── data/              # Processed experiment data
│   ├── forw_plan/     # Experiment 1
│   ├── inv_plan_intimacy/  # Experiment 2a
│   └── inv_plan_desire/    # Experiment 2b
├── experiments/       # jsPsych experiment code
│   └── scenarios.csv  # Scenario definitions
├── model/             # Computational models
│   ├── model_utils.py # Actor and observer model definitions
│   ├── fit_*.py       # Model fitting scripts
│   └── outputs/       # Fitted parameters and predictions
└── figures/           # Generated figures
```

See the [data codebook](data/README.md), [experiments README](experiments/README.md), and [model outputs README](model/outputs/README.md) for details on each directory.

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

If you don't have uv, you can use pip with the standard `pyproject.toml`. With this setup, run scripts with `python` rather than `uv run python`:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install .
```

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

Convert raw data (not included in the repository) to CSV format with anonymized participant ids:

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
