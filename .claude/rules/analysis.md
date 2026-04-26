---
paths:
  - "analysis/**/*"
---

# Analysis structure

Core analysis files (named after their data folder, not paper experiment number):
- `utils.R` — Shared utility functions (theme setup, bootstrap correlation, belief update calculation)
- `forw-plan-analysis.qmd` — Forward planning analysis
- `inv-plan-intimacy-alt-analysis.qmd` — Intimacy inference, alternatives shown
- `inv-plan-intimacy-noalt-analysis.qmd` — Intimacy inference, no alternatives shown (LM-generated counterfactuals on the model side)
- `inv-plan-desire-alt-analysis.qmd` — Desire inference, alternatives shown
- `inv-plan-desire-noalt-analysis.qmd` — Desire inference, no alternatives shown (data-only viz; no model yet)
- `inv-plan-combined-correlation.qmd` — Combined correlation across the alt-shown inverse planning experiments
- `forw-plan-effort-analysis.qmd` — Forward planning, effort manipulation (parallels `forw-plan-analysis.qmd` on the 2-action effort stimulus set)
- `inv-plan-effort-analysis.qmd` — Inverse planning, effort manipulation (parallels `inv-plan-intimacy-alt-analysis.qmd`)
- `inv-plan-effort-inferred-analysis.qmd` — Inverse planning, effort inferred (parallels `inv-plan-effort-analysis.qmd` but flips the latent: action × intimacy → P(effort_high))
- `nonfood-forw-plan-analysis.qmd` — Non-food forward planning (parallels `forw-plan-analysis.qmd`; reads `cv_loso_preds_nonfood.csv` etc.)
- `json_to_csv.py` — Data processing pipeline

Legacy pilot analysis files are in `analysis/legacy/`.

## Commands

Convert experiment JSON output to CSV:

```bash
uv run python analysis/json_to_csv.py <experiment_name>
# experiments: forw_plan, inv_plan_intimacy_alt, inv_plan_intimacy_noalt, inv_plan_desire_alt, inv_plan_desire_noalt, forw_plan_effort, inv_plan_effort, inv_plan_effort_inferred
```

For pilot experiments (in `analysis/legacy/`), use `json_to_csv_old_pilots.py`.

Render analysis qmds (or open them in RStudio):

```bash
quarto render analysis/forw-plan-analysis.qmd
quarto render analysis/inv-plan-intimacy-alt-analysis.qmd
quarto render analysis/inv-plan-desire-alt-analysis.qmd
quarto render analysis/inv-plan-desire-noalt-analysis.qmd
quarto render analysis/inv-plan-intimacy-noalt-analysis.qmd
quarto render analysis/inv-plan-combined-correlation.qmd
quarto render analysis/forw-plan-effort-analysis.qmd
quarto render analysis/inv-plan-effort-analysis.qmd
quarto render analysis/nonfood-forw-plan-analysis.qmd
```
