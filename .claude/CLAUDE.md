# Agent guide

This file contains durable, agent-facing context for this repository. The public project overview and run instructions are in `README.md`; the `Makefile` is the command and active-roster source of truth. `AGENTS.md` is a symlink to this file.

## Sources of truth

- Use directory slugs as stable study identifiers. Paper numbers can change, but the slugs in `study_registry.py` and the `Makefile` do not.
- Read per-study metadata from `study_registry.py`, including paper labels, domains, given conditions, inferred variables, and which variant the paper reports as "Base." Do not duplicate that metadata in consumers.
- Treat `SIP_journal/main.tex` as the usual source of truth for project intent and claims. Code can occasionally be newer. If manuscript and code disagree, identify the divergence, use recent edits and Git history to assess which is newer, and ask which direction to reconcile before changing either.
- Before answering a design or methods question, check the local-only `notes/decisions.md`. If the discussion reaches a new conclusion, propose a dated entry with the question, decision, rationale, and firm or provisional status; do not append it without showing the user first.

## Naming and model terminology

- Study slugs use underscores so per-study scripts remain importable. Fit and CV scripts are thin slug wrappers over registries in `model/inverse/_fit_dispatcher.py` and `model/cv/_inverse_dispatcher.py`; follow the dispatcher to understand behavior.
- The repository installs as an editable package with `uv sync`. Use package-qualified imports and do not add `sys.path.insert`. The only intentional exceptions are the build-directory inserts in `experiments/scenarios*.py`; `experiments/` is not a package. Figure scripts can import sibling helpers directly because their directory is on `sys.path` when run as scripts.
- The reward term is `w_v · desire · g`. Here `g` is desire-free goal satisfaction, desire is inferred in Studies 1a, 1b, and 3a and given in Studies 2a, 2b, and 3b, and risk is the per-action discomfort feature weighted by `w_d`.
- Keep the fitted reward weight named `w_v` and its output column `param_w_v`; do not rename it to the risk weight `w_d`.

## Repository boundaries

- `data/legacy/` contains local-only, gitignored participant data from superseded studies. No active code reads it.
- `SIP_journal/` is the current journal manuscript and a separate, gitignored Git repository synced to Overleaf.
- `cogsci-cr/` is the self-contained CogSci 2026 camera-ready fork. Keep fixes for that fork inside its directory; do not change the main pipeline to make the fork work.
- Generated SI files must not be edited by hand. `model/lm/prompts.py` is the source for `SIP_journal/si_prompts.tex`, rendered by `model/lm/export_prompts_latex.py`; `experiments/scenarios.py` and `scenarios_nonfood.py` generate the scenario CSVs, which `experiments/export_scenarios_latex.py` renders into `SIP_journal/si_scenarios_food.tex` and `si_scenarios_nonfood.tex`.

## Workflow and commands

```text
experiments/ -> raw JSON -> data_prep/json_to_csv.py -> data CSVs
                                                    -> model fits and LOSO CV
                                                    -> statistics and figures
```

The model's cross-validation outputs are the sole prediction source. Python scripts in `figures/scripts/` generate plotted manuscript figures and components; `figures/model-eqs/` also contains authored equation graphics. Use `make help` for current commands.

Set up and run Python through uv:

```bash
uv sync
uv run python path/to/script.py
```

Before changing files in `data/`, `data_prep/`, `experiments/`, or `model/`, read the corresponding `.claude/rules/<area>.md` unless the harness loaded it automatically.

## Project instructions

- Use Context7 whenever library or API documentation, code generation, setup, or configuration guidance is needed.
- Before committing a nontrivial change to fitting, likelihood, data-loading, CV, or pipeline logic under `model/` or `data_prep/`, run an available code-review skill on the diff and apply or surface its findings. The pre-commit hook also runs `make test` when staged files are under `model/`.
- For Together AI work, use the relevant `togetherai-skills:*` skill and current Together AI documentation rather than model recall.
- When changing this guide or a scoped rule, update `README.md` only when the public documentation is affected.
- `figures/scripts/plot_style.py` is the style and output-routing source of truth for Python-generated figures. Change shared colors, fonts, colormaps, and output roots there.
