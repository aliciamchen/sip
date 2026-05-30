# Experiments

## Terminology note

The experiment code and the data it saves use "desire" (e.g., the `desire_low`/`desire_high` scenario paragraphs and the `desire_condition` factor). The model's internal variables and the processed-data column still use "reward"/"motivation" (e.g., `p_high_reward`, and the `motivation` column that `json_to_csv.py` writes) — the same concept under an older name that predates the switch to "desire" on the experiment side. A later, larger change will rename "reward"/"motivation" to "desire" on the model side too (the fitting, CV, and table code); it's deferred for now.

## Stimulus sources

The active stimulus set is one scenario CSV, generated from a Python source of truth. Edit the `.py` file and regenerate with `uv run python experiments/scenarios.py` — never edit the CSV directly, since the next regeneration will overwrite the edits. After regenerating, run `uv run python experiments/csv_to_json.py` to propagate the changes into each experiment's `json/stimuli.json`. The earlier 4-action / 2-action / non-food sets are archived under `experiments/legacy/` (see below).

### `scenarios.csv` — 3-action canonical (active Studies 1a, 1b, 2a, 2b)

The stimulus set for all four active inverse-planning experiments. Carries the effort paragraphs alongside the desire and intimacy framing so all three latent variables (desire, effort, intimacy) can be manipulated alongside the observed action.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier used in data files |
| `name_0`, `name_1` | Character names in the vignette |
| `vignette` | Base scenario description |
| `desire_low`, `desire_high` | Two desire-state paragraphs |
| `effort_low`, `effort_high` | Two effort-state paragraphs — the low version makes the resource that `action_1` relies on easy to obtain, the high version makes it costly |
| `action_0` | No sharing |
| `action_1` | Low-risk (non-saliva) sharing — the effort cost lives in the `effort_*` paragraph, not the action text |
| `action_2` | High-risk (saliva) sharing |

### Legacy stimulus sets (`experiments/legacy/`)

Archived; their experiments are no longer active. Each is still generated from its `.py` source and regenerate-able via `csv_to_json.py` (legacy routing):

- `legacy/scenarios.csv` — 4-action canonical (no-share / no-risk / moderate-risk / high-risk; `action_0`…`action_3`), the original forward Study 1a + archived inverse experiments.
- `legacy/scenarios_effort.csv` — 2-action effort set (non-saliva `action_1` vs saliva `action_2`, with `effort_low`/`effort_high` paragraphs; reward fixed high), the original forward Study 1b.
- `legacy/scenarios_nonfood.csv` — non-food parallel of the 4-action set across substance / shared-space / privacy categories (adds a `scenario_type` column); a basis for the planned Study 3.

## Active experiments

### Inverse planning (3-action set)

All four use `scenarios.csv` and follow the noalt-style presentation: the participant sees a single observed action per trial, with prior and posterior slider responses. The DV scales are desire as a continuous 0–100 rating ("how much do they want to eat the food?", not-at-all → extremely), effort as a continuous 0–100 rating between two states, and intimacy on a 0–100 numeric scale. Each experiment dir's `README.md` documents the design spec.

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

Each experiment dir has `python/generate_counterbalancing.py` which produces `json/full_counterbalancing.json` — an array of N "sequences," each a 16-trial assignment of factor cells to the 16 scenarios. `experiment.js` reads a per-participant `condition_assignment` from jsPsychPipe and selects `counterbalancing[condition_assignment]`. Cells are distributed across the 16 slots as evenly as possible; when the cell space exceeds 16 (Study 1a has 2 × 4 × 3 = 24 cells), each participant samples a 16-cell subset and the cells balance across participants. The number of sequences is `n_cells × 16` rotations (e.g. Study 2b's 6 cells → 96 sequences).

## Shared experiment code

The active experiments share a single copy of the jsPsych boilerplate — consent + instructions screens, attention check, memory checks, exit survey, save, thank-you, and the merged stylesheet — in [`_lib/`](_lib/). Each per-experiment `experiment.js` is a thin call to `runExperiment()` from `_lib/bootstrap.js`, and each `trials.js` holds only the experiment-specific instruction text and prior/posterior trial rendering. The settings repeated across every experiment — the DataPipe ID map, the attention-check index and tolerance, the inter-trial durations, and the shared Prolific completion URL — are collected in [`_lib/config.js`](_lib/config.js); each `trials.js` builds its `CONFIG` with a single `makeConfig("<slug>")` call (passing overrides as a second argument if it needs to depart from a shared default). The four active experiments share a single consent form (`food-inverse`), which is in `_lib/consent/`.

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
