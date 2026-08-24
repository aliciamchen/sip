---
name: regen-experiments
description: Use when experiments/scenarios.py, scenarios_nonfood.py, model/lm/prompts.py, or experiments/_lib code has been edited — including when the user says "i made some edits, can you regenerate the artifacts". Also use before committing any experiments/ change.
allowed-tools: Bash, Read, Grep, Glob
---

# Regenerate experiment artifacts after source edits

The scenario CSVs, per-experiment `stimuli.json`, counterbalancing JSON, entry files, and the SI LaTeX tables are all **generated**; their sources of truth are the `.py` files. The user frequently edits sources herself between or during sessions, so the chain below gets re-run often — and a partial run leaves stale artifacts that the deploy guard will catch later.

## The chain

1. `git diff --stat` first — discover her manual edits before assuming file state.
2. `make experiments` — regenerates scenario CSVs → `stimuli.json` → counterbalancing → entry files, in order.
3. `make check-experiments` — verifies nothing is still drifted (this is also what `bin/deploy-experiment` runs before every push).
4. SI LaTeX exports (no make target; skip if `SIP_journal/` is absent):
   - scenarios changed → `uv run python experiments/export_scenarios_latex.py`
   - `prompts.py` changed → `uv run python model/lm/export_prompts_latex.py`
   Never hand-edit or read the generated `si_*.tex` as a source — they mirror the `.py` files.
5. **Flag LM staleness.** A scenario or prompt change makes every `outputs/lm/<slug>/` table stale. Say so explicitly, but do not re-elicit — that is a paid run the user triggers herself (see the rerun-lm-elicitation skill).
6. If the preview server is running, remind her to hard-reload the browser (stale-looking stimuli after a regen are almost always browser cache).

## Committing

Commit the edited source and its regenerated artifacts together, staging an explicit file list (she often has unrelated work in flight from another tab — never `git add -A` or stage by directory). Lowercase verb-first message.
