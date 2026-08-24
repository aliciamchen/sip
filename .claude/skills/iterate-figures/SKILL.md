---
name: iterate-figures
description: Use when creating, revising, rendering, verifying, or committing generated manuscript figures, SI figures, schematic panels, or Illustrator components.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
---

# Iterate on figures

## Workflow

1. Identify the source script and narrowest Make target with `make help`. Main routes are `figures-panels`, `figures-nonfood-domains`, `figures-schematic`, `figures-lm-si`, and the `figures-si-*` targets.
2. Read `figures/scripts/plot_style.py` before changing shared styling. Read per-study facts from `study_registry.py`; do not introduce local metadata tables.
3. Regenerate only the affected family. If Make unexpectedly does nothing, inspect the target prerequisites before running scripts ad hoc.
4. Render and inspect every changed PDF or its PNG preview. Never claim a visual result from code or file existence alone.
5. For ambiguous or new compositions, make one scratch prototype first. Keep rejected mockups outside tracked figure scripts and fold only the accepted design into the implementation.
6. For manuscript-bound output, run `make sync-journal-figures`, rebuild with the manuscript skill, inspect the rendered page, and verify captions and in-text panel references.

Quick model-vs-human diagnostics are scratch outputs, not manuscript figures. Reuse the model-comparison cell definitions and do not commit them.

## Durable style rules

- Use sentence case and the terminology defined by the manuscript and `study_registry.py`.
- Keep fonts, palettes, colormaps, output roots, and reusable helpers in `plot_style.py`.
- Avoid white point edges and rotated tick labels. Show effort splits only for actions where effort is manipulated.
- Keep multi-panel axes, panel letters, and page origins aligned. Use `tight=False` for Illustrator components that must stack by page origin.
- Illustrator components are vector PDFs with editable text; PNGs are gitignored previews. SI files are complete figures and should match the main-text visual language.
- Treat faceting, legends, point layers, and panel selection as plot-specific choices unless an established helper encodes the convention.

## Before a requested commit

PDF regeneration can change metadata without changing content. Compare against HEAD with `/CreationDate` and `/ModDate` removed; restore metadata-only churn with `git restore`. Rasterize and inspect files whose normalized bytes differ. Stage an explicit file list, remove superseded outputs, and verify every manuscript `\includegraphics` target still resolves.
