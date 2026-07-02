---
paths:
  - "analysis/**/*"
---

# Analysis structure

Core analysis files (named after their data folder, not paper experiment number):

- `utils.R` — Shared utility functions (theme setup, bootstrap correlation, belief update calculation, demographics reporting with `data/legacy/` fallback). Its palettes are the visual source of truth; the Python figure module `plot_style.py` (repo root) copies these hexes so R and Python figures match. Keep the two in sync when a palette changes. Known drift to reconcile: the R LM-elicitation notebooks still color the three actions with an older green/gold/red scheme, whereas `plot_style.py` now uses blue / green / amber (`no_share` / `low_risk_share` / `high_risk_share`).
- `json_to_csv.py` — Data processing pipeline; converts jsPsych raw JSON to anonymized CSVs.

### Active analysis qmds

The active roster is four inverse-planning studies on the 3-action set, all currently structure-only stubs (they gracefully handle missing data + missing CV predictions via `file.exists()` guards; TODO blocks mark where belief-update + model-vs-human panels go once pilots land):

- `food-inv-desire-analysis.qmd` — Study 1a: infer desire under known effort + intimacy (continuous 0–100 DV).
- `food-inv-joint-de-analysis.qmd` — Study 1b: joint over desire × effort given intimacy.
- `food-inv-intimacy-analysis.qmd` — Study 2a: infer intimacy under known desire + effort.
- `food-inv-joint-ie-analysis.qmd` — Study 2b: joint over intimacy × effort given desire.

## Commands

Convert experiment JSON output to CSV:

```bash
uv run python analysis/json_to_csv.py <experiment_name>
# active experiments: food_inv_desire, food_inv_joint_de, food_inv_intimacy, food_inv_joint_ie
#   (all four have configs; the two-slider studies joint_de/joint_ie split the
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
