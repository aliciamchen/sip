---
paths:
  - "analysis/**/*"
---

# Analysis structure

Core analysis files (named after their data folder, not paper experiment number):
- `utils.R` — Shared utility functions (theme setup, bootstrap correlation, belief update calculation)
- `food-forw-intimacy-desire-analysis.qmd` — Forward planning analysis
- `food-inv-intimacy-desire-alt-analysis.qmd` — Intimacy inference, alternatives shown
- `food-inv-intimacy-desire-noalt-analysis.qmd` — Intimacy inference, no alternatives shown (LM-generated counterfactuals on the model side)
- `food-inv-desire-intimacy-alt-analysis.qmd` — Desire inference, alternatives shown
- `food-inv-desire-intimacy-noalt-analysis.qmd` — Desire inference, no alternatives shown (relationship-keyed action space; predictions from LOSO CV)
- `inv-plan-combined-correlation.qmd` — Combined correlation across the alt-shown inverse planning experiments
- `food-forw-intimacy-effort-analysis.qmd` — Forward planning, effort manipulation (parallels `food-forw-intimacy-desire-analysis.qmd` on the 2-action effort stimulus set)
- `food-inv-intimacy-effort-alt-analysis.qmd` — Inverse planning, effort manipulation (parallels `food-inv-intimacy-desire-alt-analysis.qmd`)
- `food-inv-effort-intimacy-alt-analysis.qmd` — Inverse planning, effort inferred (parallels `food-inv-intimacy-effort-alt-analysis.qmd` but flips the latent: action × intimacy → P(effort_high))
- `nonfood-forw-intimacy-desire-analysis.qmd` — Non-food forward planning (parallels `food-forw-intimacy-desire-analysis.qmd`; reads `cv_loso_preds_nonfood.csv` etc.)
- `json_to_csv.py` — Data processing pipeline

Legacy pilot analysis files are in `analysis/legacy/`.

## Commands

Convert experiment JSON output to CSV:

```bash
uv run python analysis/json_to_csv.py <experiment_name>
# experiments: food_forw_intimacy_desire, food_inv-intimacy_desire_alt, food_inv-intimacy_desire_noalt, food_inv-desire_intimacy_alt, food_inv-desire_intimacy_noalt, food_forw_intimacy_effort, food_inv-intimacy_effort_alt, food_inv-effort_intimacy_alt
```

For pilot experiments (in `analysis/legacy/`), use `json_to_csv_old_pilots.py`.

Render analysis qmds (or open them in RStudio):

```bash
quarto render analysis/food-forw-intimacy-desire-analysis.qmd
quarto render analysis/food-inv-intimacy-desire-alt-analysis.qmd
quarto render analysis/food-inv-desire-intimacy-alt-analysis.qmd
quarto render analysis/food-inv-desire-intimacy-noalt-analysis.qmd
quarto render analysis/food-inv-intimacy-desire-noalt-analysis.qmd
quarto render analysis/inv-plan-combined-correlation.qmd
quarto render analysis/food-forw-intimacy-effort-analysis.qmd
quarto render analysis/food-inv-intimacy-effort-alt-analysis.qmd
quarto render analysis/nonfood-forw-intimacy-desire-analysis.qmd
```
