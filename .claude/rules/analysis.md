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
- `inv-plan-combined-correlation.qmd` — Combined correlation across the alt-shown inverse planning experiments
- `forw-plan-effort-analysis.qmd` — Forward planning, effort manipulation (parallels `forw-plan-analysis.qmd` on the 2-action effort stimulus set)
- `inv-plan-effort-analysis.qmd` — Inverse planning, effort manipulation (parallels `inv-plan-intimacy-alt-analysis.qmd`)
- `inv-plan-effort-inferred-analysis.qmd` — Inverse planning, effort inferred (parallels `inv-plan-effort-analysis.qmd` but flips the latent: action × intimacy → P(effort_high))
- `json_to_csv.py` — Data processing pipeline

Legacy pilot analysis files are in `analysis/legacy/`.
