# CLAUDE.md

Guidance for Claude Code sessions working in this repository. Project overview, the experiment roster, the utility model, and run instructions live in [README.md](../README.md). This file holds Claude-specific context that isn't in the public docs.

## Naming and structure conventions

- The stable identifier for each experiment is its directory slug in `data/<slug>/` and `experiments/<slug>/`. Paper-level experiment numbers shift as the writeup evolves; slugs don't. Slugs are all-underscore (no hyphens), so the per-experiment fit/cv scripts can also be imported as modules if needed.
- The active roster is six inverse-planning studies, all on the 3-action structure: `food_inv_desire` (Study 1a), `food_inv_joint_de` (Study 1b), `food_inv_intimacy` (Study 2a), `food_inv_joint_ie` (Study 2b) on the food scenario set, plus `nonfood_inv_joint_de` (Study 3a, mirroring 1b) and `nonfood_inv_joint_ie` (Study 3b, mirroring 2b) on the nonfood set. The nonfood pair reuses the food joint studies' observers, fit helpers, and CV dispatcher — only the stimulus set, scenario labels, and LM tables differ (`domain="nonfood"` in the table-kwargs builders; `STUDY_SCENARIO_LABELS` in `model/tables.py`).
- Per-experiment scripts (e.g. `model/inverse/fit_food_inv_intimacy.py`) are thin wrappers that import shared logic from `_inverse_dispatcher.py` (cv/) or `_helpers.py` (inverse/) and call its main with their hardcoded slug. To trace what a script does, follow the import.
- The experiment roster lives in the `Makefile`: `EXPERIMENTS_INVERSE` holds the food studies (the data-dependent aggregates `make data`/`fit`/`cv`/`analysis` run over these), `EXPERIMENTS_NONFOOD` the two nonfood studies (no participant data yet — the Makefile's roster comment lists everything to change when Study 3 data lands). `make all` runs fit → cv → model-comparison → analysis over `EXPERIMENTS_INVERSE` as sequential sub-makes, so the stages stay ordered even under `make -j` (CV produces the out-of-sample predictions; there is no separate predict stage); per-study targets (`lm-/fit-/cv-/data-<slug>`) cover all six.

## Legacy data

Archived participant data from earlier (removed) experiments is under `data/legacy/` — there is no legacy *code*. A little active code reads `data/legacy/` (the `utils.R` demographics fallback); those archived CSVs use older column names than the active ones.

## Terminology

The codebase uses **desire** and **risk** throughout. Two points worth knowing:

- The reward term is `w_v · desire · g`: `g` is the LM-elicited, desire-free goal-satisfaction; `desire` (`d ∈ [0, 1]`) is the inferred latent in 1a/1b and an observed `desire_condition` in 2a/2b. **risk** is the per-action discomfort feature, weight `w_d`.
- The fitted reward-term weight is `w_v` (column `param_w_v`), *not* `w_d` — keep it named `w_v`; don't "fix" it to `w_d`.

## Source of truth for project intent

The manuscript (`SIP_journal/main.tex`) is generally the most up-to-date description of the project: what the studies are, what design they have, what the model is, how it's fit, and what's being claimed. Treat it as the authoritative plan when the two are out of sync. The code often lags — a script, prompt, or utility shape can be a step or two behind what the manuscript now describes — and bringing the code in line with the manuscript is usually the right move.

The reverse can also happen: the user sometimes develops code first (a new prompt, a new fitting procedure, a new pipeline stage) before writing it up. In that case the code is ahead of the manuscript, and the manuscript needs to catch up rather than the code being rolled back. So discrepancies don't have a single default direction — they usually mean either the code needs an update or the manuscript needs one.

When you spot a discrepancy, don't silently reconcile it. Surface it: name the divergence, say which side looks newer based on context (recent edits, conversation, git log), and ask which direction to update before changing either one.

## Decisions log

`notes/decisions.md` (local-only, like the rest of `notes/`) records design and methods decisions reached in conversation, so they don't get re-derived from scratch. Two obligations:

- **Before answering a design/methods question**, check the log. If a decision exists, start from its recorded rationale rather than re-litigating — the user may still overturn it, but deliberately.
- **At the end of a consulting or design discussion that reaches a conclusion**, propose an entry: date, the question, the decision, the why, and whether it's firm or provisional. Don't append without showing the proposed entry first.

## Submission status

The current journal version is in `SIP_journal/` (gitignored; its own git repo synced to Overleaf).

The camera-ready CogSci 2026 fork is in `cogsci-cr/` (gitignored; its own repo, with the LaTeX in the nested `cogsci-cr/cogsci-2026/` synced to Overleaf). It's a self-contained subset with its own slimmed `model/`, data, analysis, and paper, using hand-stipulated utility tables instead of the LM pipeline. Keep any changes isolated to that subfolder — don't edit HEAD code to make `cogsci-cr/` work. Conference reviews: `cogsci-cr/cogsci-2026/cogsci-2026-reviews.md`.

## Generated SI artifacts (don't hand-edit)

The Supplementary Material `\input`s LaTeX files that are generated from the code. Don't read or edit these `.tex` files to inspect or change a prompt or scenario — read the source and regenerate. Each carries an "AUTO-GENERATED — do not edit by hand" header.

- `SIP_journal/si_prompts.tex` ← `model/lm/export_prompts_latex.py`, rendered from `model/lm/prompts.py`. To read or change a prompt, go to `prompts.py` and re-run the script.
- `SIP_journal/si_scenarios_food.tex`, `si_scenarios_nonfood.tex` ← `experiments/export_scenarios_latex.py`, rendered from `experiments/scenarios.csv` / `scenarios_nonfood.csv`. Those CSVs are themselves generated from `experiments/scenarios.py` / `scenarios_nonfood.py` (see the experiments rules), so the scenarios' source of truth is the `.py` files — to change a scenario, edit the `.py`, regenerate the CSV, then re-run the table export.

## Workflow

```
jsPsych experiments (experiments/) → JSON → json_to_csv.py → CSV (data/)
                                                              ↓
                                  model fits + LOSO CV (model/) → out-of-sample predictions
                                                              ↓
                     paper figures: Python scripts (styled by plot_style.py) → figures/
                                                              ↓
                     make sync-journal-figures → SIP_journal/figures/ (Overleaf)
```

The R/Quarto qmds in `analysis/` are working/visualization documents, not the paper's figure source — the paper's figures come from the Python scripts above and its model-comparison statistics from `model/cv/model_comparison.py` (see `.claude/rules/analysis.md`).

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
- Before committing a nontrivial change under `model/` or `analysis/` (fitting/likelihood logic, data loaders, CV, new pipeline stages — not figure styling or prose), run `/code-review` on the diff and apply or surface the findings. Do this on your own initiative; the user won't ask.
- For anything involving Together AI (the LM pipeline's inference provider — chat/completions, batch, embeddings, fine-tuning, etc.), use the installed `togetherai-skills:*` skills and the `TogetherAIDocs` MCP server to fetch current docs rather than relying on training data.
- When changing CLAUDE.md or rules files, also update README.md if relevant. README.md is what reviewers and the public read.

## Utility helpers

- `utils.py` — `get_project_root()` for constructing paths relative to project root.
- `analysis/utils.R` — shared R helpers: `setup_analysis()`, `boot_cor()`, `calculate_belief_update()`.
- `plot_style.py` — shared style for **all** Python-generated figures (the `figures/schematic_panels/figure_schematic_plots.py` panels and the LM-elicitation SI figures in `model/lm/plot_si_validation.py` + `plot_alternatives.py`): `apply_style("si"|"schematic")`, `savefig()` → vector PDF + PNG preview into `figures/`, plus every palette and colormap (matched to `analysis/utils.R`). Change figure colors, fonts, or colormaps here, not inline in the plotting scripts. `make figures-lm-si` regenerates the LM SI figure set.
