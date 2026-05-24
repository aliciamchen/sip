---
paths:
  - "analysis/**/*"
---

# Analysis structure

Core analysis files (named after their data folder, not paper experiment number):

- `utils.R` — Shared utility functions (theme setup, bootstrap correlation, belief update calculation, demographics reporting with `data/legacy/` fallback).
- `json_to_csv.py` — Data processing pipeline; converts jsPsych raw JSON to anonymized CSVs.

### Active analysis qmds

Forward planning (Studies 1a, 1b, and the non-food forward):

- `food-forw-intimacy-desire-analysis.qmd` — Study 1a (4-action canonical, reward + intimacy manipulated).
- `food-forw-intimacy-effort-analysis.qmd` — Study 1b (2-action effort experiment).
- `nonfood-forw-intimacy-desire-analysis.qmd` — Non-food forward (parallels Study 1a on `scenarios_nonfood.csv`).
- `cv-loso-forward.qmd` — LOSO CV summary across the three forward experiments.

Inverse planning (Studies 2, 3a, 3b, 4a, 4b — all on the 3-action set):

- `food-inv-intimacy-3act-analysis.qmd` — Study 2: infer intimacy under known desire + effort.
- `food-inv-effort-3act-analysis.qmd` — Study 3a: infer effort under known desire + intimacy.
- `food-inv-desire-3act-analysis.qmd` — Study 3b: infer desire under known effort + intimacy.
- `food-inv-joint-de-3act-analysis.qmd` — Study 4a: joint over desire × effort given intimacy.
- `food-inv-joint-di-3act-analysis.qmd` — Study 4b: joint over desire × intimacy given effort.

The five inverse qmds are currently stubs that gracefully handle missing data and missing CV predictions via `file.exists()` guards. TODO blocks mark where belief-update plots and model-vs-human correlation panels need to be filled in once pilots land; the patterns to follow live in the surviving `_noalt` qmds.

### Legacy

Legacy pilot analysis files are in `analysis/legacy/`. The two surviving 4-action inverse qmds (`food-inv-intimacy-desire-noalt-analysis.qmd`, `food-inv-desire-intimacy-noalt-analysis.qmd`) still live alongside the active set; their data paths point to `data/legacy/<slug>/`, and they remain renderable. They are not part of `make all` and will be folded into the 3-action pipeline when the corresponding inverse experiments are migrated to the Study 3b template.

## Commands

Convert experiment JSON output to CSV (active experiments only):

```bash
uv run python analysis/json_to_csv.py <experiment_name>
# active experiments:
#   food_forw_intimacy_desire, food_forw_intimacy_effort, nonfood_forw_intimacy_desire,
#   food_inv_intimacy_3act, food_inv_effort_3act, food_inv_desire_3act,
#   food_inv_joint_de_3act, food_inv_joint_di_3act
```

For pilot experiments (in `analysis/legacy/`), use `json_to_csv_old_pilots.py`.

Render active analysis qmds (or open them in RStudio):

```bash
quarto render analysis/food-forw-intimacy-desire-analysis.qmd
quarto render analysis/food-forw-intimacy-effort-analysis.qmd
quarto render analysis/nonfood-forw-intimacy-desire-analysis.qmd
quarto render analysis/cv-loso-forward.qmd
quarto render analysis/food-inv-intimacy-3act-analysis.qmd
quarto render analysis/food-inv-effort-3act-analysis.qmd
quarto render analysis/food-inv-desire-3act-analysis.qmd
quarto render analysis/food-inv-joint-de-3act-analysis.qmd
quarto render analysis/food-inv-joint-di-3act-analysis.qmd
```

Or via the Makefile: `make analysis` renders the active set; `make analysis-<name>` runs a single qmd.
