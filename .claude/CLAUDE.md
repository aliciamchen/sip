# CLAUDE.md

Guidance for Claude Code sessions working in this repository. Project overview, the experiment roster, the utility model, and run instructions live in [README.md](../README.md). This file holds Claude-specific context that isn't in the public docs.

## Naming and structure conventions

- The stable identifier for each experiment is its directory slug in `data/<slug>/` and `experiments/<slug>/`. Paper-level experiment numbers shift as the writeup evolves; slugs don't. Slugs are all-underscore (no hyphens), so the per-experiment fit/cv scripts can also be imported as modules if needed.
- The active roster is six inverse-planning studies, all on the 3-action structure: `food_inv_desire` (Study 1a), `food_inv_joint_de` (Study 1b), `food_inv_intimacy` (Study 2a), `food_inv_joint_ie` (Study 2b) on the food scenario set, plus `nonfood_inv_joint_de` (Study 3a, mirroring 1b) and `nonfood_inv_joint_ie` (Study 3b, mirroring 2b) on the nonfood set. The nonfood pair reuses the food joint studies' observers, fit helpers, and CV dispatcher — only the stimulus set, scenario labels, and LM tables differ (`domain="nonfood"` in the table-kwargs builders; `STUDY_SCENARIO_LABELS` in `model/tables.py`).
- Per-experiment scripts are genuinely thin wrappers (~25 lines) that call a shared dispatcher with their hardcoded slug: `model/inverse/fit_<slug>.py` → `_fit_dispatcher.main(slug)`, `model/cv/cv_<slug>.py` → `_inverse_dispatcher.main_<family>(slug)`. The whole fit and CV protocol lives in those two dispatchers, keyed by a `_FAMILIES` registry; a wrapper carries nothing but its slug and docstring, and `model/test_fit_protocol.py` enforces that. To trace what a script does, follow the import.
- The experiment roster lives in the `Makefile`: `EXPERIMENTS_INVERSE` holds all six studies (the Study 3 nonfood pair was folded in on 2026-07-21, once their data, LM tables, fits/CV, and analysis qmds all existed), and the data-dependent aggregates `make data`/`fit`/`cv`/`model-comparison` run over it. `EXPERIMENTS_NONFOOD` is kept but empty (the roster-sync test reads both lists; it's the holding list for any future not-yet-data-complete study family). `make all` runs fit → cv → model-comparison → figures over `EXPERIMENTS_INVERSE` as sequential sub-makes, so the stages stay ordered even under `make -j` (CV produces the out-of-sample predictions; there is no separate predict stage); per-study targets (`lm-/fit-/cv-/data-<slug>`) cover the roster too.

## Legacy data

Archived participant data from earlier, superseded experiments sits in the local-only, gitignored `data/legacy/` — there is no legacy *code*, and nothing in the active pipeline reads it. Those archived CSVs use older column names than the active ones.

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
jsPsych experiments (experiments/) → JSON → data_prep/json_to_csv.py → CSV (data/)
                                                              ↓
                                  model fits + LOSO CV (model/) → out-of-sample predictions
                                                              ↓
     paper figures: Python scripts in figures/scripts/ (styled by plot_style.py)
                    → figures/panels/ (Illustrator components) + figures/si/ (finished)
                                                              ↓
                     make sync-journal-figures → SIP_journal/figures/ (Overleaf)
```

`data_prep/` holds only the raw-data conversion (`json_to_csv.py` + its test). Every figure comes from the Python scripts in `figures/scripts/` (the Illustrator results components, run with `make figures-panels`; the SI LM figures, run with `make figures-lm-si`), the model-comparison statistics from `model/cv/model_comparison.py`, and the manuscript's demographics from `model/export_results_latex.py`. There is no R or Quarto anywhere in the pipeline (the qmds and renv setup were removed for the public release; local R exploration files are gitignored).

## Common commands

The `Makefile` wraps everything; `make help` lists targets. Stage-specific details are in `.claude/rules/{data_prep,data,experiments,model}.md`, which load on demand when Claude reads files in those directories.

## Environment setup

```bash
uv sync                  # Python deps; creates .venv
uv run python script.py  # run scripts
```

Key Python deps: JAX, memo-lang (probabilistic modeling DSL), pandas, numpy, optax.

## Project instructions

- Always use Context7 when needing library/API documentation, code generation, setup, or configuration steps — without me having to explicitly ask.
- Before committing a nontrivial change under `model/` or `data_prep/` (fitting/likelihood logic, data loaders, CV, new pipeline stages — not figure styling or prose), run a code review on the diff (in Claude Code, the `/code-review` skill) and apply or surface the findings. Do this on your own initiative; the user won't ask. A pre-commit hook independently runs the full test suite (`make test`) whenever a staged file is under `model/`.
- For anything involving Together AI (the LM pipeline's inference provider — chat/completions, batch, embeddings, fine-tuning, etc.), use the installed `togetherai-skills:*` skills and the `TogetherAIDocs` MCP server to fetch current docs rather than relying on training data.
- When changing CLAUDE.md or rules files, also update README.md if relevant. README.md is what reviewers and the public read.

## Utility helpers

- `utils.py` — `get_project_root()` for constructing paths relative to project root.
- `study_registry.py` — the single source of truth for per-study metadata (given conditions, inferred latents with their `<rating>_update` / `delta_<latent>` column pairs, paper label, stimulus domain), plus `reported_base(slug)` for which variant the paper's "Base" column means. Imported by `model/cv/model_comparison.py`, `model/export_results_latex.py`, and the figure scripts, and listed in the Makefile's `FIG_SHARED`. Read a per-study fact from here rather than hardcoding it in a consumer. Details in `.claude/rules/model.md`.
- `figures/scripts/plot_style.py` — shared style for **all** Python-generated figures (every script in `figures/scripts/`: the main results figures, the `figure_schematic_plots.py` panels, and the LM-elicitation SI figures `figure_si_lm_validation.py` + `figure_si_consolidated.py`): `apply_style("si"|"schematic")`, `savefig()` → vector PDF + a gitignored PNG preview, into whichever output root the caller names (default `figures/si/`), plus every palette and colormap. It is the visual source of truth — change figure colors, fonts, or colormaps here, not inline in the plotting scripts. `make figures-panels` regenerates the Illustrator components; `make figures-lm-si` the LM SI set.
- `figures/scripts/` — output is split by consumer, with the roots named in `plot_style.py` (`PANELS_RESULTS`, `PANELS_LEGENDS`, `PANELS_SCHEMATIC`, `SI_DIR`): `figures/panels/` for Illustrator components and `figures/si/` for finished figures. The paper's results figures are assembled by hand, so `figure_paper_panels.py` writes components, not finished figures — the assembled per-study scripts were removed on 2026-08-02. Shared data prep is in `_data.py` (reusing `model/cv/model_comparison.py`'s cell specs and loaders), the points design in `_points.py`, and the pooled model-vs-humans panel in `_agg.py` (a helper module, not a script). Each renders the panels whose inputs exist, skips the rest with a printed note, and warns when CV outputs are stale relative to the data CSV.
