# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a cognitive science research project investigating inverse planning and social inference in food-sharing scenarios involving saliva transfer. The project explores how people make decisions about sharing food based on relationship closeness (intimacy) and motivation (reward), and how observers infer relationship closeness or desire from observed actions.

The project comprises several experimental variants. Paper-level experiment numbering shifts as the writeup evolves, so the stable identifier for each variant is its data directory in `data/`.

- **Forward planning** (`data/food_forw_intimacy_desire/`) — actors choose actions based on intimacy and motivation.
- **Intimacy inference, alternatives shown** (`data/food_inv-intimacy_desire_alt/`) — observers see the scenario plus all four candidate actions, then infer relationship closeness from the one the actor took.
- **Desire inference, alternatives shown** (`data/food_inv-desire_intimacy_alt/`) — observers see all four candidate actions, then infer the actor's desire from the one they took.
- **Intimacy inference, no alternatives shown** (`data/food_inv-intimacy_desire_noalt/`) — observers see only the single action the actor took and infer relationship closeness; on the model side, counterfactual alternatives are LM-generated.
- **Desire inference, no alternatives shown** (`data/food_inv-desire_intimacy_noalt/`) — same noalt structure as above but flips the inference target: observers see only the chosen action and infer the actor's desire (slider endpoints are the scenario's `reward_low` / `reward_high` paragraphs).

A parallel pair of experiments manipulates **relative effort** instead of reward, using a different stimulus set (`experiments/scenarios_effort.csv`) in which each scenario has two actions and reward is held fixed at high. Data has been collected for both, and the full pipeline (LM scenario params, model fits, LOSO CV, analysis qmds) parallels the canonical pipeline.

- **Forward planning, effort** (`data/food_forw_intimacy_effort/`) — actors choose between two actions given intimacy (4 levels) × relative effort (2 levels); 2-slider linked-probability response.
- **Inverse planning, effort** (`data/food_inv-intimacy_effort_alt/`) — observers infer intimacy from the observed action (2 levels) × relative effort (2 levels); both candidate actions shown, prior/posterior intimacy sliders.
- **Inverse planning, effort inferred** (`data/food_inv-effort_intimacy_alt/`) — flips the inference direction of the previous experiment: observers see observed action (2 levels) × intimacy (4 levels) and infer the effort context from a slider whose endpoints are the scenario's `effort_low` / `effort_high` paragraphs. Effort is the latent (no effort paragraph in the vignette). Prior/posterior structure.

A second parallel pipeline tests **generalization beyond food sharing** using `experiments/scenarios_nonfood.csv` — 16 scenarios covering substance sharing (chapstick, towel, hat, hairbrush, harmonica, sunscreen), shared physical space (blanket, sleeping-bag, bed, locker-room, sauna), and informational/situational privacy (breakup, payment, gossip, home, navigation). The schema matches the canonical `scenarios.csv` with one extra `scenario_type` column. The five jsPsych experiments parallel the canonical food set one-to-one (no data collected yet):

- **`experiments/nonfood_forw_intimacy_desire/`** — non-food forward planning.
- **`experiments/nonfood_inv-intimacy_desire_alt/`** — non-food intimacy inference, alternatives shown.
- **`experiments/nonfood_inv-desire_intimacy_alt/`** — non-food desire inference, alternatives shown.
- **`experiments/nonfood_inv-intimacy_desire_noalt/`** — non-food intimacy inference, no alternatives shown.
- **`experiments/nonfood_inv-desire_intimacy_noalt/`** — non-food desire inference, no alternatives shown.

## LM prompts

Both the food and non-food pipelines use a single prompt set defined in `model/lm/prompts.py`. The access rubric covers three channel types — bodily-substance transfer, direct physical contact, and informational/private-resource disclosure — so the same prompt works for food sharing, shared objects, shared physical space, and privacy/information-disclosure scenarios. The earlier food-only prompts were retired after a side-by-side comparison showed the unified prompts produced equal or slightly better fits on the food data; they live in git history under commit `eb13d0e` if anyone needs to reproduce a pre-unification fit.

`lm/score_canonical_features.py` and `lm/generate_alternatives_motivation.py` accept a `--domain food|nonfood` flag that selects which scenario CSV to score and which output filename to write to.

## Intermediate conference submission

We are currently deciding how to extend this project for submission to a journal. An early version of this project has been submitted to a conference; the paper is in the `cogsci-2026` folder. The reviews for this submission are in `cogsci-2026/cogsci-2026-reviews.md`. The model in this submission is an old version that we are not using for the journal version. 

The journal version of the paper is in the `SIP_journal` folder. This is the most up-to-date version of the writeup. 

Both of these folders are gitignored but there are git repos within these folders that sync to Overleaf. 

## README.md

`README.md` in the project root is the public-facing documentation — reviewers and the public read it to understand the repo and how to run the code. When prompted to change CLAUDE.md or rules files, update the README too with relevant information. 

## Project instructions

Always use Context7 when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

## Environment setup

```bash
uv sync
```

This creates a `.venv` virtual environment with all dependencies. To run Python scripts:
```bash
uv run python script.py
```

Or activate the environment directly:
```bash
source .venv/bin/activate
```

Key dependencies: JAX, memo-lang (probabilistic modeling DSL), pandas, numpy, optax.

R dependencies are managed with `renv` (`renv.lock` at the repo root).

## Common commands

A `Makefile` at the repo root wraps the most common commands. Run `make help` for a list of targets. Underlying script invocations are documented in the rules files that load on demand: `.claude/rules/analysis.md` (data pipeline + quarto render) and `.claude/rules/model.md` (LM scenario params, fits, predictions, LOSO CV, tests).

## Workflow

```
jsPsych experiments (experiments/) → JSON → json_to_csv.py → CSV (data/)
                                                              ↓
                                                    R analysis (analysis/)
                                                              ↓
                                            Compare with model predictions (model/outputs/)
```

## Repository layout

Topic-specific details load on demand via `.claude/rules/` when Claude reads files in these areas:

- `analysis/` — R/Quarto analysis scripts (see `.claude/rules/analysis.md`)
- `data/` — processed experiment data (see `.claude/rules/data.md`)
- `experiments/` — jsPsych experiment code (see `.claude/rules/experiments.md`)
- `model/` — computational models (see `.claude/rules/model.md`)

Other top-level directories (kept here because they're cross-cutting or small):

- `cogsci-2026/` - CogSci 2026 paper source (LaTeX: `main.tex`, `cogsci.sty`, bibliography, figures). This is gitignored
- `submissions/` - Compiled paper PDFs for submission
- `figures/` - Rendered figures used in the paper
- `LM_evals/` - Language-model evaluation code (`experiments/`, `providers/`)
- `_quarto.yml` / `_output/` - Quarto project configuration and rendered output

## Utility functions

- `utils.py` - Provides `get_project_root()` for constructing paths relative to project root
- `analysis/utils.R` - Shared R functions: `setup_analysis()`, `boot_cor()`, `calculate_belief_update()`
