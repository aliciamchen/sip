---
name: regen-experiments
description: Use when scenario sources, prompt sources, or shared experiment code changed, when generated experiment artifacts need refreshing, or before committing changes under experiments.
allowed-tools: Bash, Read, Grep, Glob
---

# Regenerate experiment artifacts

1. Inspect `git diff --stat` and the changed files so manual edits and the affected scope are clear.
2. Run `make experiments` to regenerate scenario CSVs, per-study stimuli, counterbalancing, and entry files.
3. Run `make check-experiments` to verify tracked assets match their sources.
4. If scenario sources changed and `SIP_journal/` is present, run `uv run python experiments/export_scenarios_latex.py`.
5. If `model/lm/prompts.py` changed, run `uv run python model/lm/export_prompts_latex.py` when the exporter allows the current vintage.
6. Flag only the LM tables affected by the changed scenario set, prompt surface, or elicitation stage. Shared browser code does not make LM tables stale. Do not launch a paid elicitation unless the user requests it; use the rerun-lm-elicitation skill then.
7. If a preview is open, hard-reload it after regeneration.

Never edit generated CSV, JSON, or SI LaTeX files as sources. If a commit is requested, stage explicit source and artifact files together and use a conventional commit message.
