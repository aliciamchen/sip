# CLAUDE.md

Guidance for Claude Code sessions working in this repository. Project overview, the experiment roster, the utility model, and run instructions live in [README.md](../README.md). This file holds Claude-specific context that isn't in the public docs.

## Naming and structure conventions

- The stable identifier for each experiment is its directory slug in `data/<slug>/` and `experiments/<slug>/`. Paper-level experiment numbers shift as the writeup evolves; slugs don't. Slugs are all-underscore (no hyphens), so the per-experiment fit/cv scripts can also be imported as modules if needed.
- The active roster is four inverse-planning studies, all on the 3-action set: `food_inv_desire` (Study 1a), `food_inv_joint_de` (Study 1b), `food_inv_intimacy` (Study 2a), `food_inv_joint_ie` (Study 2b).
- Per-experiment scripts (e.g. `model/inverse/fit_food_inv_intimacy.py`) are thin wrappers that import shared logic from `_inverse_dispatcher.py` (cv/) or `_helpers.py` (inverse/) and call its main with their hardcoded slug. To trace what a script does, follow the import.
- The active experiment roster lives in `Makefile`'s `EXPERIMENTS_INVERSE` variable; `make all` runs fit → cv → analysis across all four (CV produces the out-of-sample predictions; there is no separate predict stage).

## Legacy data

Archived participant data from earlier (removed) experiments is under `data/legacy/` — there is no legacy *code*. A little active code reads `data/legacy/` (the `utils.R` demographics fallback, a guarded pilot overlay in `food-inv-desire-analysis.qmd`); those archived CSVs use older column names than the active ones.

## Terminology

The codebase uses **desire** and **risk** throughout. Two points worth knowing:

- The reward term is `w_v · desire · g`: `g` is the LM-elicited, desire-free goal-satisfaction; `desire` (`d ∈ [0, 1]`) is the inferred latent in 1a/1b and an observed `desire_condition` in 2a/2b. **risk** is the per-action discomfort feature, weight `w_d`.
- The fitted reward-term weight is `w_v` (column `param_w_v`), *not* `w_d` — keep it named `w_v`; don't "fix" it to `w_d`.

## Source of truth for project intent

The manuscript (`SIP_journal/main.tex`) is generally the most up-to-date description of the project: what the studies are, what design they have, what the model is, how it's fit, and what's being claimed. Treat it as the authoritative plan when the two are out of sync. The code often lags — a script, prompt, or utility shape can be a step or two behind what the manuscript now describes — and bringing the code in line with the manuscript is usually the right move.

The reverse can also happen: the user sometimes develops code first (a new prompt, a new fitting procedure, a new pipeline stage) before writing it up. In that case the code is ahead of the manuscript, and the manuscript needs to catch up rather than the code being rolled back. So discrepancies don't have a single default direction — they usually mean either the code needs an update or the manuscript needs one.

When you spot a discrepancy, don't silently reconcile it. Surface it: name the divergence, say which side looks newer based on context (recent edits, conversation, git log), and ask which direction to update before changing either one.

## Submission status

The current journal version is in `SIP_journal/` (gitignored; its own git repo synced to Overleaf).

The camera-ready CogSci 2026 fork is in `cogsci-cr/` (gitignored; its own repo, with the LaTeX in the nested `cogsci-cr/cogsci-2026/` synced to Overleaf). It's a self-contained subset with its own slimmed `model/`, data, analysis, and paper, using hand-stipulated utility tables instead of the LM pipeline. Keep any changes isolated to that subfolder — don't edit HEAD code to make `cogsci-cr/` work. Conference reviews: `cogsci-cr/cogsci-2026/cogsci-2026-reviews.md`.

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
