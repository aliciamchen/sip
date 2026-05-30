# CLAUDE.md

Guidance for Claude Code sessions working in this repository. Project overview, the experiment roster, the utility model, and run instructions live in [README.md](../README.md). This file holds Claude-specific context that isn't in the public docs.

## Naming and structure conventions

- The stable identifier for each experiment is its directory slug in `data/<slug>/` and `experiments/<slug>/`. Paper-level experiment numbers shift as the writeup evolves; slugs don't. Slugs are all-underscore (no hyphens), so the per-experiment fit/predict/cv scripts can also be imported as modules if needed.
- The active roster is four inverse-planning studies, all on the 3-action set: `food_inv_desire` (Study 1a), `food_inv_joint_de` (Study 1b), `food_inv_intimacy` (Study 2a), `food_inv_joint_ie` (Study 2b). The May 2026 refactor dropped the `_3act` suffix from active slugs — it only marked a transitional period when the 4-action and 2-action sets still had active studies; now those sets are entirely legacy, so the qualifier is gone.
- Per-experiment scripts (e.g. `model/inverse/fit_food_inv_intimacy.py`) are thin wrappers that import shared logic from `_inverse_dispatcher.py` (cv/) or `_helpers.py` (inverse/) and call its main with their hardcoded slug. To trace what a script does, follow the import.
- The active experiment roster lives in `Makefile`'s `EXPERIMENTS_INVERSE` variable. Legacy experiments are in `LEGACY_FORWARD` (the three forward-planning experiments — real data, archived to `data/legacy/` in May 2026 when the manuscript reframed around four inverse studies) and `LEGACY_INVERSE` (the two surviving 4-action `_noalt` inverse experiments). Both keep runnable per-slug `fit-/predict-/cv-` targets and legacy analysis qmds, but `make all` only touches active experiments.

## Terminology

The paper uses "desire." The experiment code and the data it saves now use "desire" too (`desire_low`/`desire_high` scenario paragraphs, the `desire_condition` factor), but the model's internal variables still use "reward" (e.g. `p_high_reward`, the memo `reward_condition`) and the processed-CSV column is "motivation" — same concept, the model-side names predate the experiment-side rename. The active LM scripts that read the scenario paragraphs were updated to the new column names; legacy code still reads `reward_*`. Planned for later: a larger change will rename "reward"/"motivation" to "desire" everywhere else too — the model internals (`p_high_reward`, the memo `reward_condition`, `param_w_v` naming) and the processed-CSV `motivation` column — to make the whole codebase consistent. That's deferred for now because it touches the fitting/CV/table code broadly. In `model/utility.py`, `V` is signed valence in [-1, +1]; in the effort experiment V is stipulated to 1 (reward held fixed at high) and `w_v` is non-identified, so it appears in fit-result tables for parallelism but stays near initialization.

## Source of truth for project intent

The manuscript (`SIP_journal/main.tex`) is generally the most up-to-date description of the project: what the studies are, what design they have, what the model is, how it's fit, and what's being claimed. Treat it as the authoritative plan when the two are out of sync. The code often lags — a script, prompt, or utility shape can be a step or two behind what the manuscript now describes — and bringing the code in line with the manuscript is usually the right move.

The reverse can also happen: the user sometimes develops code first (a new prompt, a new fitting procedure, a new pipeline stage) before writing it up. In that case the code is ahead of the manuscript, and the manuscript needs to catch up rather than the code being rolled back. So discrepancies don't have a single default direction — they usually mean either the code needs an update or the manuscript needs one.

When you spot a discrepancy, don't silently reconcile it. Surface it: name the divergence, say which side looks newer based on context (recent edits, conversation, git log), and ask which direction to update before changing either one.

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
