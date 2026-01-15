# saliva-inverse-planning

## Experiments: 

- Experiment 1: Forward planning
- Experiment 2: Inverse planning (intimacy)
- Experiment 3: Inverse planning (reward)
- Experiment 4: Planning with communicative goals

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


## How to run analyses

Convert raw data (not included in the repository) to csv format with anonymized participant ids:

```bash
python analysis/json_to_csv.py <experiment_name> # available experiments: forw_plan, inv_plan_intimacy, inv_plan_reward
```

Fit forward planning models

```bash
cd model
python fit_forward_planning.py
```

Analyze data and generate plots

```bash
cd analysis
quarto render exp-1-data-analysis.qmd
```
