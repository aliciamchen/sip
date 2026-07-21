---
paths:
  - "analysis/**/*"
---

# Analysis structure

The qmds are demographics/data-check documents only. All figures — the main
results figures and the SI figures — are made by the Python scripts in
`figures/scripts/` (styled by the repo-root `plot_style.py`, which is now the
visual source of truth for palettes, and written to `figures/outputs/`), and
the paper's model-comparison statistics come from `model/cv/model_comparison.py`
(`make model-comparison`), not from the qmds. The R plotting layer (ggplot
scales, themes, palettes, `save_figure`, the bootstrap helpers) was removed
from `utils.R` in July 2026 when the plots moved to Python; the Python
equivalents are in `figures/scripts/_data.py`, which reuses
`model_comparison.py`'s loaders rather than re-deriving the belief updates.

Core analysis files (named after their data folder, not paper experiment number):

- `utils.R` — Shared R helpers for the qmds and exploratory scripts:
  `report_demographics()` (with `data/legacy/` fallback), `calculate_belief_update()`,
  the model JSON/JSONL readers, and the `INTIMACY_LEVELS` / `ACTION_LEVELS`
  factor orders. `report_demographics()` reads the retained-after-exclusions N
  off the study's `main_trials_long.csv` rather than re-implementing the
  exclusion rules in R — `json_to_csv.py` is the single source of truth for
  those. The local-only `signature_tests.R` (exploratory, non-preregistered)
  also sources this file; keep `calculate_belief_update`, `ACTION_LEVELS`, and
  `INTIMACY_LEVELS` available for it.
- `json_to_csv.py` — Data processing pipeline; converts jsPsych raw JSON to anonymized CSVs. Applies each study's exclusion rule from its config: 1a's preregistered lax rule (exclude only failed-attention AND 0 memory questions), strict for the later studies (retain only passed-attention AND >=1 memory question). This script is the single source of truth for exclusion rules. It fails fast on bad raw data (unparseable files, missing or duplicate subject IDs, trial/exit-survey subject mismatches, zero parsed rows) and normalizes the legacy pre-2026-06-19 `neither` intimacy label to `somewhat_formal` at parse time.
- `test_json_to_csv.py` — Offline tests for the converter on synthetic fixtures (`uv run python analysis/test_json_to_csv.py`).

### Active analysis qmds

One qmd per study (all six), each reporting demographics and a data glimpse behind
`have_data <- file.exists(...)` guards (not `exists("df")`, which is always
TRUE because of `stats::df`). The figures come from Python either way:

- `food-inv-desire-analysis.qmd` — Study 1a: infer desire under known effort + intimacy (continuous 0–100 DV).
- `food-inv-joint-de-analysis.qmd` — Study 1b: joint over desire × effort given intimacy.
- `food-inv-intimacy-analysis.qmd` — Study 2a: infer intimacy under known desire + effort.
- `food-inv-joint-ie-analysis.qmd` — Study 2b: joint over intimacy × effort given desire.
- `nonfood-inv-joint-de-analysis.qmd` — Study 3a: 1b's design on the non-food set.
- `nonfood-inv-joint-ie-analysis.qmd` — Study 3b: 2b's design on the non-food set.

## Commands

Convert experiment JSON output to CSV:

```bash
uv run python analysis/json_to_csv.py <experiment_name>
# active experiments: food_inv_desire, food_inv_joint_de, food_inv_intimacy,
#   food_inv_joint_ie, nonfood_inv_joint_de, nonfood_inv_joint_ie
#   (all six have configs; the two-slider studies joint_de/joint_ie split the
#    survey-html-form response into desire_rating/intimacy_rating + effort_rating)
```

Render active analysis qmds (or open them in RStudio):

```bash
quarto render analysis/food-inv-desire-analysis.qmd
quarto render analysis/food-inv-joint-de-analysis.qmd
quarto render analysis/food-inv-intimacy-analysis.qmd
quarto render analysis/food-inv-joint-ie-analysis.qmd
quarto render analysis/nonfood-inv-joint-de-analysis.qmd
quarto render analysis/nonfood-inv-joint-ie-analysis.qmd
```

Or via the Makefile: `make analysis` renders the active set; `make analysis-<name>` runs a single qmd. The results figures are `make figures-results` (see the Makefile's figure section and `figures/scripts/`).
