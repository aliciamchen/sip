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
- `json_to_csv.py` — Data processing pipeline

Legacy pilot analysis files are in `analysis/legacy/`.
