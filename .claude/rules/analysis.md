---
paths:
  - "analysis/**/*"
---

# Analysis structure

Core analysis files:
- `utils.R` - Shared utility functions (theme setup, bootstrap correlation, belief update calculation)
- `exp-1-analysis.qmd` - Forward planning analysis (Exp 1)
- `exp-2a-inv-plan-intimacy-analysis.qmd` - Intimacy inference analysis (Exp 2a)
- `exp-2b-inv-plan-desire-analysis.qmd` - Desire inference analysis (Exp 2b)
- `exp-2-combined-correlation.qmd` - Combined correlation analysis (Exp 2a + 2b)
- `json_to_csv.py` - Data processing pipeline

Legacy pilot analysis files are in `analysis/legacy/`.
