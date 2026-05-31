---
paths:
  - "analysis/**/*"
---

# Analysis structure

Core analysis files (named after their data folder, not paper experiment number):

- `utils.R` — Shared utility functions (theme setup, bootstrap correlation, belief update calculation, demographics reporting with `data/legacy/` fallback).
- `json_to_csv.py` — Data processing pipeline; converts jsPsych raw JSON to anonymized CSVs.

### Active analysis qmds

The active roster is four inverse-planning studies on the 3-action set, all currently structure-only stubs (they gracefully handle missing data + missing CV predictions via `file.exists()` guards; TODO blocks mark where belief-update + model-vs-human panels go once pilots land — patterns live in the surviving `_noalt` qmds):

- `food-inv-desire-analysis.qmd` — Study 1a: infer desire under known effort + intimacy (continuous 0–100 DV).
- `food-inv-joint-de-analysis.qmd` — Study 1b: joint over desire × effort given intimacy.
- `food-inv-intimacy-analysis.qmd` — Study 2a: infer intimacy under known desire + effort.
- `food-inv-joint-ie-analysis.qmd` — Study 2b: joint over intimacy × effort given desire.

### Legacy

Legacy analysis qmds live under `analysis/legacy/` (not part of `make all`; renderable via `make analysis-<name>`, which the Makefile renders from `analysis/legacy/` for everything in `LEGACY_ANALYSIS_QMDS`; their data is in `data/legacy/` and outputs in `model/outputs/legacy/`):

- Forward: `food-forw-intimacy-desire-analysis.qmd`, `food-forw-intimacy-effort-analysis.qmd`, `nonfood-forw-intimacy-desire-analysis.qmd`, `cv-loso-forward.qmd`.
- 4-action inverse: `food-inv-intimacy-desire-noalt-analysis.qmd`, `food-inv-desire-intimacy-noalt-analysis.qmd`.
- `food-inv-desire-pilot-analysis.qmd` — the original Study 1a pilot (0–100 DV; data in `data/legacy/food_inv_desire_pilot/`). Earlier pilots (`2025-*_pilots.Rmd`) are also here.

## Commands

Convert experiment JSON output to CSV:

```bash
uv run python analysis/json_to_csv.py <experiment_name>
# active experiments: food_inv_desire, food_inv_joint_de, food_inv_intimacy, food_inv_joint_ie
#   (all four have configs; the two-slider studies joint_de/joint_ie split the
#    survey-html-form response into desire_rating/intimacy_rating + effort_rating)
# legacy forwards still processable per-slug: food_forw_intimacy_desire, etc.
```

For pilot experiments (in `analysis/legacy/`), use `json_to_csv_old_pilots.py`.

Render active analysis qmds (or open them in RStudio):

```bash
quarto render analysis/food-inv-desire-analysis.qmd
quarto render analysis/food-inv-joint-de-analysis.qmd
quarto render analysis/food-inv-intimacy-analysis.qmd
quarto render analysis/food-inv-joint-ie-analysis.qmd
```

Or via the Makefile: `make analysis` renders the active set; `make analysis-<name>` runs a single qmd (active or legacy).
