# CLAUDE.md

Guidance for Claude Code sessions working in this repository. Project overview, the experiment roster, the utility model, and run instructions live in [README.md](../README.md). This file holds Claude-specific context that isn't in the public docs.

## Naming and structure conventions

- The stable identifier for each experiment is its directory slug in `data/<slug>/` and `experiments/<slug>/`. Paper-level experiment numbers shift as the writeup evolves; slugs don't. Slugs are all-underscore (no hyphens), so the per-experiment fit/predict/cv scripts can also be imported as modules if needed.
- Per-experiment scripts (e.g. `model/inverse/fit_food_inv_intimacy_3act.py`) are thin wrappers that import shared logic from `_dispatcher.py` (cv/) or `_helpers.py` (inverse/) and call its main with their hardcoded slug. To trace what a script does, follow the import.
- The active experiment roster lives in `Makefile`'s `EXPERIMENTS_FORWARD` and `EXPERIMENTS_INVERSE` variables. The six pre-3-action inverse food experiments (`food_inv_*_alt`, `food_inv_*_noalt` on the 4-action and 2-action sets) were archived in May 2026 — their data is under `data/legacy/`, their model scripts and qmds remain runnable, and their slugs are kept in the Makefile's `LEGACY_INVERSE` / `LEGACY_ANALYSIS_QMDS` lists. `make all` only touches active experiments.

## Terminology

The paper uses "desire" but internal variable names use "reward" (e.g. `p_high_reward`, `reward_condition`) or "motivation" — same concept, terminology changed mid-project for clarity. In `model/utility.py`, `V` is signed valence in [-1, +1]; in the effort experiment V is stipulated to 1 (reward held fixed at high) and `w_v` is non-identified, so it appears in fit-result tables for parallelism but stays near initialization.

## Submission status

The current journal version is in `SIP_journal/` (gitignored; its own git repo synced to Overleaf).

The camera-ready CogSci 2026 fork is in `cogsci-cr/` (gitignored; will be spun out into its own repo). It's a self-contained subset — slimmed copies of `model/`, the 3 cogsci experiments under `data/`, 4 analysis qmds, and the conference-paper LaTeX under `cogsci-cr/cogsci-2026/` — using the journal-version utility shape (no appeal term) but with hand-stipulated, action-only V/access/effort values from `cogsci-cr/model/stipulated_tables.py` instead of LM-elicited per-scenario tables. No `model/lm/` pipeline. The 3 experiments it covers: `food_forw_intimacy_desire`, `food_inv_intimacy_desire_alt`, `food_inv_desire_intimacy_alt`. `cogsci-cr/cogsci-2026/` is a nested git repo synced to Overleaf; reviews of the conference submission are in `cogsci-cr/cogsci-2026/cogsci-2026-reviews.md`. Don't edit HEAD code on the user's behalf to make `cogsci-cr/` work — keep the change isolated to that subfolder.

## Workflow

```
jsPsych experiments (experiments/) → JSON → json_to_csv.py → CSV (data/)
                                                              ↓
                                  model fits (model/) → predictions
                                                              ↓
                                  R/Quarto analysis (analysis/) → figures
```

## Common commands

The `Makefile` wraps everything; `make help` lists targets. Stage-specific details are in `.claude/rules/{analysis,data,experiments,model}.md`, which load on demand when Claude reads files in those directories.

## Environment setup

```bash
uv sync                  # Python deps; creates .venv
uv run python script.py  # run scripts
```

Key Python deps: JAX, memo-lang (probabilistic modeling DSL), pandas, numpy, optax. R deps are managed with `renv` (`renv.lock` at the repo root).

## Project instructions

- Always use Context7 when needing library/API documentation, code generation, setup, or configuration steps — without me having to explicitly ask.
- When changing CLAUDE.md or rules files, also update README.md if relevant. README.md is what reviewers and the public read.

## Utility helpers

- `utils.py` — `get_project_root()` for constructing paths relative to project root.
- `analysis/utils.R` — shared R helpers: `setup_analysis()`, `boot_cor()`, `calculate_belief_update()`.
