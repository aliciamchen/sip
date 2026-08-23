---
paths:
  - "data_prep/**/*"
---

# Data-prep structure

`data_prep/` holds only the raw-data conversion. There are no notebooks or R
here: every figure is made by the Python scripts in `figures/scripts/` (styled
by `figures/scripts/plot_style.py` and written to `figures/panels/` and
`figures/si/`), the paper's model-comparison statistics come from
`model/cv/model_comparison.py` (`make model-comparison`), and the manuscript's
demographics numbers come from `model/export_results_latex.py`. The former
R/Quarto qmds (demographics and data-check documents) and `utils.R` were
removed for the public release; any local R exploration files are gitignored.

- `json_to_csv.py` — Data processing pipeline; converts jsPsych raw JSON to anonymized CSVs. Applies each study's exclusion rule from its config: 1a's preregistered lax rule (exclude only failed-attention AND 0 memory questions), strict for the later studies (retain only passed-attention AND >=1 memory question). This script is the single source of truth for exclusion rules. It stops with a clear error on bad raw data (unparseable files, missing or duplicate subject IDs, trial/exit-survey subject mismatches, zero parsed rows) and normalizes the legacy pre-2026-06-19 `neither` intimacy label to `somewhat_formal` at parse time.
- `test_json_to_csv.py` — Offline tests for the converter on synthetic fixtures (`uv run python data_prep/test_json_to_csv.py`).

## Commands

Convert experiment JSON output to CSV (needs `data/<slug>/raw_data/`, which is
gitignored; the committed CSVs are already current):

```bash
uv run python data_prep/json_to_csv.py <experiment_name>
# active experiments: food_inv_desire, food_inv_joint_de, food_inv_intimacy,
#   food_inv_joint_ie, nonfood_inv_joint_de, nonfood_inv_joint_ie
#   (the two-slider studies joint_de/joint_ie split the survey-html-form
#    response into desire_rating/intimacy_rating + effort_rating)
```
