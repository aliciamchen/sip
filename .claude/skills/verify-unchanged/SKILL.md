---
name: verify-unchanged
description: Use when verifying that a refactor, dependency change, rename, or cleanup did not alter reported numbers or rendered content. Check behavior at the cheapest relevant layers before committing.
allowed-tools: Bash, Read, Grep, Glob
---

# Verify unchanged behavior

Run the inexpensive checks for every cleanup and add deeper comparisons for the affected layer.

1. Run `make test` and retain the complete command output. Do not pipe through a command that can hide the test exit status.
2. Rerun `uv run python model/cv/model_comparison.py`; when its prerequisites exist, also run `uv run python model/cv/generalization_primary.py`. Confirm tracked output JSON remains unchanged.
3. For loader or table changes, load every affected family under HEAD and the working tree and compare arrays with `np.array_equal`, including given-magnitude scalars and masks.
4. For figure changes, regenerate affected outputs and compare PDFs after removing `/CreationDate` and `/ModDate`. Restore metadata-only churn with `git restore`; render and inspect real content differences.
5. For likelihood, table-routing, optimizer, or fit-loop changes, rerun at least one representative affected study or CV fold and compare parameters, losses, predictions, and manifests field by field. Expand to all affected families when one study cannot exercise every path.
6. Run `make freshen-outputs`, then confirm `make all` is a no-op apart from sub-make status lines.

Report exactly which checks ran, which were not applicable, and every observed difference. Do not encode verification only in a commit message, and do not commit unless asked.
