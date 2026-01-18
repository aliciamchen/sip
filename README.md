# saliva-inverse-planning

## Experiments: 

- Experiment 1: Forward planning
    - Preregistration
- Experiment 2a: Inverse planning (intimacy)
- Experiment 2b: Inverse planning (reward)

## Dependencies

Python packages:

```bash
conda env create -f environment.yaml
conda activate saliva-inverse-planning
```

The R packages are managed by `renv`, the R version is `4.5.2`.

In R, run: 
```r
renv::restore()
```

You also need xquartz for the font in the plots to render correctly: 
```bash
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
python analysis/json_to_csv.py forw_plan
python analysis/json_to_csv.py inv_plan_intimacy
python analysis/json_to_csv.py inv_plan_reward
```

Fit forward planning models

```bash
cd model
python fit_forward_planning.py
```

Use forward planning parameters to fit inverse planning models and generate predictions
```bash
cd model
python fit_inverse_planning.py
python generate_inverse_planning_preds.py
```

Analyze data and generate plots

```bash
cd analysis
quarto render exp-1-analysis.qmd
quarto render exp-2a-inv-plan-intimacy-analysis.qmd
quarto render exp-2b-inv-plan-reward-analysis.qmd
quarto render exp-2-combined-correlation.qmd
```
