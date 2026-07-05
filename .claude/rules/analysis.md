---
paths:
  - "analysis/**/*"
---

# Analysis structure

The qmds are working/visualization documents. The paper's figures will be made in Python (via the shared `plot_style.py`), and the paper's model-comparison statistics come from `model/cv/model_comparison.py` (`make model-comparison`), not from the qmds.

Core analysis files (named after their data folder, not paper experiment number):

- `utils.R` — Shared utility functions (theme setup, bootstrap correlation, cluster-bootstrap cell means, belief update calculation, demographics reporting with `data/legacy/` fallback). `report_demographics()` reads the retained-after-exclusions N off the study's `main_trials_long.csv` rather than re-implementing the exclusion rules in R — `json_to_csv.py` is the single source of truth for those. Its palettes are the visual source of truth; the Python figure module `plot_style.py` (repo root) copies these hexes so R and Python figures match. Keep the two in sync when a palette changes. Known drift to reconcile: the R LM-elicitation notebooks still color the three actions with an older green/gold/red scheme, whereas `plot_style.py` now uses blue / green / amber (`no_share` / `low_risk_share` / `high_risk_share`).
- `json_to_csv.py` — Data processing pipeline; converts jsPsych raw JSON to anonymized CSVs. Applies each study's exclusion rule from its config: 1a's preregistered lax rule (exclude only failed-attention AND 0 memory questions), strict for the later studies (retain only passed-attention AND >=1 memory question). This script is the single source of truth for exclusion rules. It fails fast on bad raw data (unparseable files, missing or duplicate subject IDs, trial/exit-survey subject mismatches, zero parsed rows) and normalizes the legacy pre-2026-06-19 `neither` intimacy label to `somewhat_formal` at parse time.
- `test_json_to_csv.py` — Offline tests for the converter on synthetic fixtures (`uv run python analysis/test_json_to_csv.py`).

### Active analysis qmds

The Study 1a qmd (`food-inv-desire-analysis.qmd`) runs on the current partial sample (~half the target N), including the CV model-vs-human panels. The joint qmds (1b, 2b) have data and run their demographics + belief-update panels; their model sections are still `eval=FALSE` pending CV predictions. The 2a qmd has data but its belief-update and model sections are still `eval=FALSE` stubs. All four gracefully handle missing data + missing CV predictions via file-existence flags (`have_data <- file.exists(...)` set where the CSV is read, guarding the dependent chunks — not `exists("df")`, which is always TRUE because of `stats::df`). The nonfood studies (3a/3b) have no qmds yet — their data collection hasn't started, and the paper's figures come from Python anyway:

- `food-inv-desire-analysis.qmd` — Study 1a: infer desire under known effort + intimacy (continuous 0–100 DV).
- `food-inv-joint-de-analysis.qmd` — Study 1b: joint over desire × effort given intimacy.
- `food-inv-intimacy-analysis.qmd` — Study 2a: infer intimacy under known desire + effort.
- `food-inv-joint-ie-analysis.qmd` — Study 2b: joint over intimacy × effort given desire.

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
```

Or via the Makefile: `make analysis` renders the active set; `make analysis-<name>` runs a single qmd.
