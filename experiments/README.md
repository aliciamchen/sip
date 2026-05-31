# Experiments

## Terminology note

The experiment code and the data it saves use "desire" (e.g., the `desire_low`/`desire_high` scenario paragraphs and the `desire_condition` factor). The model's internal variables and the processed-data column still use "reward"/"motivation" (e.g., `p_high_reward`, and the `motivation` column that `json_to_csv.py` writes) — the same concept under an older name that predates the switch to "desire" on the experiment side. A later, larger change will rename "reward"/"motivation" to "desire" on the model side too (the fitting, CV, and table code); it's deferred for now.

## Stimulus sources

The active stimulus set is one scenario CSV, generated from a Python source of truth. Edit the `.py` file and regenerate with `uv run python experiments/scenarios.py` — never edit the CSV directly, since the next regeneration will overwrite the edits. After regenerating, run `uv run python experiments/build/csv_to_json.py` to propagate the changes into each experiment's `json/stimuli.json`. The earlier 4-action / 2-action / non-food sets are archived under `experiments/legacy/` (see below).

### `scenarios.csv` — 3-action canonical (active Studies 1a, 1b, 2a, 2b)

The stimulus set for all four active inverse-planning experiments. Carries the effort paragraphs alongside the desire and intimacy framing so all three latent variables (desire, effort, intimacy) can be manipulated alongside the observed action.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier used in data files |
| `name_0`, `name_1` | Character names in the vignette |
| `vignette` | Base scenario description |
| `desire_phrase` | Scenario-specific completion of the desire question "How much do {name_0} and {name_1} both want to ___?" (e.g. `eat the hot dog`), so the question names the actual food rather than "the food". Used in Studies 1a and 1b |
| `desire_low`, `desire_high` | Two desire-state paragraphs |
| `low_risk_share_effort_low`, `low_risk_share_effort_high` | Two effort-state paragraphs — the low version makes the resource that `low_risk_share` relies on easy to obtain, the high version makes it costly |
| `no_share` | No sharing |
| `low_risk_share` | Low-risk (non-saliva) sharing — the effort cost lives in the `low_risk_share_effort_*` paragraph, not the action text |
| `high_risk_share` | High-risk (saliva) sharing |

### Legacy stimulus sets (`experiments/legacy/`)

Archived; their experiments are no longer active. Each is still generated from its `.py` source and regenerate-able via `build/csv_to_json.py` (legacy routing):

- `legacy/scenarios.csv` — 4-action canonical (no-share / no-risk / moderate-risk / high-risk; `action_0`…`action_3`), the original forward Study 1a + archived inverse experiments.
- `legacy/scenarios_effort.csv` — 2-action effort set (non-saliva `action_1` vs saliva `action_2`, with `effort_low`/`effort_high` paragraphs; reward fixed high), the original forward Study 1b.
- `legacy/scenarios_nonfood.csv` — non-food parallel of the 4-action set across substance / shared-space / privacy categories (adds a `scenario_type` column); a basis for the planned Study 3.

## Build scripts

The generators that turn source files into the artifacts each experiment loads are in [`experiments/build/`](build/) (never deployed — the deploy only pushes `_lib/` and the per-experiment dirs). The authored scenario data stays at `experiments/scenarios.py` (and `legacy/`).

- `csv_to_json.py` — scenario CSV → each experiment's `json/stimuli.json` (routing in its `SOURCES` table).
- `counterbalancing.py` — per-study `json/full_counterbalancing.json` (designs in its `STUDY_CONFIGS` registry; `--study <slug>` for one).
- `sync_entry_files.py` — the byte-identical `index.html` + `experiment.js` written into every active experiment.
- `_scenario_io.py` — shared `write_scenarios_csv()` that the `scenarios*.py` source files call.

The `Makefile` wraps these: `make stimuli`, `make counterbalancing`, `make entry-files`, or `make experiments` for all three.

## Active experiments

### Inverse planning (3-action set)

All four use `scenarios.csv` and follow the noalt-style presentation: the participant sees a single observed action per trial, with prior and posterior slider responses. The DV scales are desire as a continuous 0–100 rating with a scenario-specific question that names the characters and the food ("how much do Carissa and Josh both want to eat the hot dog?", not at all → moderately → extremely), effort as a continuous 0–100 rating between two states, and intimacy on a 0–100 numeric scale. Each experiment dir's `README.md` documents the design spec.

- [food_inv_desire](food_inv_desire/README.md) — **Study 1a**: infer desire under known effort + intimacy. Design 2 × 4 × 3.
- [food_inv_joint_de](food_inv_joint_de/README.md) — **Study 1b**: joint inference over desire and effort given intimacy. Design 4 × 3, two sliders per trial.
- [food_inv_intimacy](food_inv_intimacy/README.md) — **Study 2a**: infer intimacy under known desire + effort. Design 2 × 2 × 3.
- [food_inv_joint_ie](food_inv_joint_ie/README.md) — **Study 2b**: joint inference over intimacy and effort given desire. Design 2 × 3, two sliders per trial (intimacy on the 0–100 scale, effort with paragraph endpoints).

## Legacy experiment dirs

Legacy experiment code lives under [`experiments/legacy/`](legacy/); their model scripts and analysis qmds still run against the archived data under `data/legacy/` (outputs land in `model/outputs/legacy/`), via the Makefile's per-slug `LEGACY_FORWARD` / `LEGACY_INVERSE` targets.

- **Forward planning** (real data): `food_forw_intimacy_desire`, `food_forw_intimacy_effort`, `nonfood_forw_intimacy_desire` — the manuscript's earlier Studies 1a/1b plus a non-food forward, demoted to legacy in the May 2026 roster refactor.
- **Pre-3-action inverse**: `food_inv_intimacy_desire_alt`, `food_inv_desire_intimacy_alt`, `food_inv_intimacy_desire_noalt`, `food_inv_desire_intimacy_noalt`, `food_inv_intimacy_effort_alt`, `food_inv_effort_intimacy_alt`. The two `_noalt` dirs retain runnable model code; the `_alt` dirs are data-only (their model code was removed earlier).

The four non-food inverse stubs (`nonfood_inv_*`), scaffolded against the obsolete 4-action design and never run, were deleted in the May 2026 cleanup.

## Counterbalancing

A single registry-driven generator, [`build/counterbalancing.py`](build/counterbalancing.py), produces every experiment's `json/full_counterbalancing.json` (run it with no arguments for all four studies, or `--study <slug>` for one; via the Makefile, `make counterbalancing`). Each file is an array of N "sequences," each a 16-trial assignment of factor cells to the 16 scenarios. `experiment.js` reads a per-participant `condition_assignment` from jsPsychPipe and selects `counterbalancing[sequence_index]` (the index wrapped modulo the sequence count). The generator builds `n_rounds` rounds per study and rotates which scenario gets which cell within a round, for `n_rounds × 16` sequences total — 192 for Studies 1a/1b/2a (12 rounds) and 96 for Study 2b (6 rounds); each study's cell set, round count, and seed are in the script's `STUDY_CONFIGS` registry. Cells are distributed across the 16 slots by a balanced design so that every factor cell ends up in the same number of trial slots overall: when the cell space exceeds 16 (Study 1a has 2 × 4 × 3 = 24 cells) each round carries a balanced 16-cell subset (every cell in 8 of 12 rounds → 128 slots each); when it is at most 16 (Studies 1b/2a have 12 cells, 2b has 6) each round carries every cell once or twice plus a balanced set of extra slots (256 slots each). Rounds are interleaved by rotation index so sequential `condition_assignment` values spread early participants across all rounds rather than clustering them on one round's cell choices.

## Shared experiment code

Most of each experiment lives in shared modules under [`_lib/`](_lib/); each `trials.js` is reduced to its study-specific stimulus trials. The shared modules are:

- `bootstrap.js` — `runExperiment({ config, makeStimulusTrials, instructionsPages, consentTemplate })`: fetches assets, initializes jsPsych, assigns the counterbalancing condition, and **assembles the whole timeline** (consent → instructions → stimulus trials → exit survey → save → thank-you). The per-study `makeStimulusTrials` is slotted in; the rest is identical for every experiment, so timeline-wide changes happen here.
- `config.js` — the `DATAPIPE_IDS` map plus the settings shared by every experiment (attention-check index/tolerance, inter-trial durations, Prolific completion URL). Each `trials.js` builds its `CONFIG` with `makeConfig("<slug>")` (pass overrides as a second argument to depart from a default).
- `instructions.js` — `STUDY_INSTRUCTIONS`, all four studies' instruction pages in one place (shared notes + per-study pages), easy to compare.
- `scenario.js` — the per-trial building blocks: condition-paragraph getters (`getDesireText`/`getEffortText`), the intimacy descriptor, slider labels, the "press any key" page, the prior/posterior pause, and `scenarioStimulus(...)` which renders the vignette block + observed action + lead-in.
- `two-slider.js` — `makeTwoSliderForm(...)` renders two sliders on one page (Studies 1b/2b) via `survey-html-form`.
- `timeline.js`, `attention-check.js`, `memory-checks.js`, `style.css`, and the consent templates in `consent/` round out the boilerplate.

So a `trials.js` only defines that study's `makeStimulusTrials` (composing the `_lib` pieces) and exports `CONFIG`, `INSTRUCTIONS_PAGES`, and `makeStimulusTrials`. The per-experiment `index.html` and `experiment.js` are byte-identical across studies and are generated from a single source by `build/sync_entry_files.py` (run it after changing the entry template, e.g. a jsPsych version bump).

This shared layout means the experiments are not standalone folders anymore: each one references `../_lib/` via relative paths. Deploys (see below) need to push `_lib/` to the server alongside the experiment.

## Deploying experiments

Active experiments are hosted on the MIT athena Locker at `https://web.mit.edu/aliciach/www/sip/experiments/<slug>/`. Pushes go through [`bin/deploy-experiment`](../bin/deploy-experiment), which rsyncs the experiment directory plus the shared `_lib/` to `aliciach@athena.dialup.mit.edu:~/www/sip/experiments/`:

```bash
bin/deploy-experiment food_inv_desire           # push _lib/ + the experiment
bin/deploy-experiment food_inv_desire --dry-run # preview what would change
bin/deploy-experiment --lib-only                # push only _lib/ (after editing it)
bin/deploy-experiment --list                    # list the active slugs
```

The script rejects slugs that aren't in the active roster (the four experiments listed under "Active experiments" above), excludes `python/`, `README.md`, `.DS_Store`, and `*.bak` from the push, and runs rsync with `--delete` so stale files from earlier deploys get cleaned up. The server destination is overridable per-invocation with `RSYNC_DEST=user@host:/path`.

Before an experiment is launched to participants, its real DataPipe experiment ID needs to be filled into the `DATAPIPE_IDS` map in [`_lib/config.js`](_lib/config.js) (keyed by slug); a `TODO_FILL_IN_DATAPIPE_ID` placeholder means data won't save until a real ID is set. The Prolific completion URL is shared across all experiments and already set in the same file.
