# Experiments

## Terminology note

Internal variable names use "reward" (e.g., `p_high_reward`, `reward_condition`) rather than "desire" — the public manuscript uses "desire" but the data column names were fixed before that rename.

## Stimulus sources

Four scenario CSVs, all generated from Python sources of truth. Edit the `.py` file and regenerate with `uv run python experiments/<name>.py` — never edit the CSV directly, since the next regeneration will overwrite the edits. After regenerating, run `uv run python experiments/csv_to_json.py` to propagate the changes into each experiment's `json/stimuli.json`.

### `scenarios.csv` — 4-action canonical (Study 1a, plus archived inverse experiments)

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier used in data files |
| `name_0`, `name_1` | Character names in the vignette |
| `vignette` | Base scenario description |
| `reward_low`, `reward_high` | Two motivation-state paragraphs |
| `action_0` … `action_3` | Four actions, ordered by saliva-transfer risk |

Action ordering:
- **Action 0**: No sharing
- **Action 1**: Sharing with no saliva risk (separate utensils, divide into portions)
- **Action 2**: Sharing with moderate saliva risk (opposite ends of a shared item)
- **Action 3**: Sharing with high saliva risk (same end, same utensil)

### `scenarios_effort.csv` — 2-action effort (Study 1b)

Same 16 scenarios as `scenarios.csv`, but collapsed to 2 actions and supplemented with separable effort paragraphs. Reward is held fixed at high and integrated into the shared vignette.

| Column | Description |
|--------|-------------|
| `scenario_label`, `name_0`, `name_1` | Same as canonical |
| `vignette` | Shared scenario narrative (same across both effort conditions) |
| `effort_low`, `effort_high` | Trailing paragraphs — the low version makes the resource that `action_1` relies on easy to obtain, the high version makes it costly |
| `action_1` | Non-saliva-sharing action |
| `action_2` | Saliva-sharing action (collapsed from `action_2`/`action_3` of `scenarios.csv`) |

### `scenarios_3act.csv` — 3-action canonical (Studies 2, 3a, 3b, 4a, 4b)

The new stimulus set used by all five inverse-planning experiments. Merges the effort paragraphs from `scenarios_effort.csv` into the canonical scenarios so all three latent variables (desire, effort, intimacy) can be manipulated alongside the observed action.

| Column | Description |
|--------|-------------|
| `scenario_label`, `name_0`, `name_1`, `vignette` | Same as canonical |
| `reward_low`, `reward_high` | Same as canonical |
| `effort_low`, `effort_high` | Same as effort set |
| `action_0` | No sharing (same as canonical action_0) |
| `action_1` | Low-risk sharing (separable from effort context — same wording as effort set's `action_1`) |
| `action_2` | High-risk sharing — the more intuitively plausible of the canonical `action_2`/`action_3` per scenario |

### `scenarios_nonfood.csv` — Non-food

A parallel set of 16 scenarios covering substance sharing (chapstick, hairbrush), shared physical space (blanket, sauna), and informational/situational privacy (breakup conversation, phone passcode). Schema matches `scenarios.csv` with one additional column, `scenario_type`, taking one of `substance`, `space`, or `privacy`. Currently only the forward variant has data.

## Active experiments

### Forward planning

- [food_forw_intimacy_desire](food_forw_intimacy_desire/README.md) — **Study 1a**: 4-action canonical with desire × intimacy crossing.
- [food_forw_intimacy_effort](food_forw_intimacy_effort/README.md) — **Study 1b**: 2-action effort experiment with effort × intimacy crossing; reward held at high.
- [nonfood_forw_intimacy_desire](nonfood_forw_intimacy_desire/README.md) — Non-food forward (parallels Study 1a on `scenarios_nonfood.csv`).

### Inverse planning (3-action set)

All five use `scenarios_3act.csv` and follow the noalt-style presentation: the participant sees a single observed action per trial, with prior and posterior slider responses. Each experiment dir's `README.md` documents the design spec.

- [food_inv_intimacy_3act](food_inv_intimacy_3act/README.md) — **Study 2**: infer intimacy under known desire + effort. Design 2 × 2 × 3.
- [food_inv_effort_3act](food_inv_effort_3act/README.md) — **Study 3a**: infer effort under known desire + intimacy. Design 2 × 4 × 3. Slider endpoints are the two effort paragraphs.
- [food_inv_desire_3act](food_inv_desire_3act/README.md) — **Study 3b**: infer desire under known effort + intimacy. Design 2 × 4 × 3. Slider endpoints are the two reward paragraphs.
- [food_inv_joint_de_3act](food_inv_joint_de_3act/README.md) — **Study 4a**: joint inference over desire and effort given intimacy. Design 4 × 3, two sliders per trial.
- [food_inv_joint_di_3act](food_inv_joint_di_3act/README.md) — **Study 4b**: joint inference over desire and intimacy given effort. Design 2 × 3, two sliders per trial (desire with paragraph endpoints, intimacy on the 0–100 maximally-formal-to-maximally-intimate scale).

## Legacy experiment dirs (slated for removal)

Six older inverse-planning experiment dirs from the previous 4-action / 2-action design remain on disk so their model scripts and analysis qmds still run against the archived data under `data/legacy/`:

- `food_inv_intimacy_desire_alt`, `food_inv_desire_intimacy_alt`, `food_inv_intimacy_desire_noalt`, `food_inv_desire_intimacy_noalt`, `food_inv_intimacy_effort_alt`, `food_inv_effort_intimacy_alt`.

Four non-food inverse stubs (`nonfood_inv_*`) were scaffolded against the obsolete 4-action design and were never run; they'll be retired along with the legacy food inverse dirs.

## Counterbalancing

Each experiment dir has `python/generate_counterbalancing.py` which produces `json/full_counterbalancing.json` — an array of N "sequences," each a 16-trial assignment of factor cells to the 16 scenarios. `experiment.js` reads a per-participant `condition_assignment` from jsPsychPipe and selects `counterbalancing[condition_assignment]`. For the new 3-action experiments, cells are distributed across the 16 slots as evenly as possible (with subsets of the cell space sampled per participant for Studies 3a/3b, which have 24 cells > 16 slots). Each experiment uses 192 sequences (12 rounds × 16 rotations).

## Shared experiment code

The active experiments share a single copy of the jsPsych boilerplate — consent + instructions screens, attention check, memory checks, exit survey, save, thank-you, and the merged stylesheet — in [`_lib/`](_lib/). Each per-experiment `experiment.js` is a thin call to `runExperiment()` from `_lib/bootstrap.js`, and each `trials.js` holds only the experiment-specific `CONFIG`, instruction text, and prior/posterior trial rendering. The three consent variants (`food-forward`, `food-inverse`, `nonfood-forward`) live in `_lib/consent/`.

This shared layout means the experiments are not standalone folders anymore: each one references `../_lib/` via relative paths. Deploys (see below) need to push `_lib/` to the server alongside the experiment.

## Deploying experiments

Active experiments are hosted on the MIT athena Locker at `https://web.mit.edu/aliciach/www/sip/experiments/<slug>/`. Pushes go through [`bin/deploy-experiment`](../bin/deploy-experiment), which rsyncs the experiment directory plus the shared `_lib/` to `aliciach@athena.dialup.mit.edu:~/www/sip/experiments/`:

```bash
bin/deploy-experiment food_inv_desire_3act           # push _lib/ + the experiment
bin/deploy-experiment food_inv_desire_3act --dry-run # preview what would change
bin/deploy-experiment --lib-only                     # push only _lib/ (after editing it)
bin/deploy-experiment --list                         # list the active slugs
```

The script rejects slugs that aren't in the active roster (the eight experiments listed under "Active experiments" above), excludes `python/`, `README.md`, `.DS_Store`, and `*.bak` from the push, and runs rsync with `--delete` so stale files from earlier deploys get cleaned up. The server destination is overridable per-invocation with `RSYNC_DEST=user@host:/path`.

Each experiment's `trials.js` still needs `PIPE_EXPERIMENT_ID` and `PROLIFIC_COMPLETION_URL` filled in with the real DataPipe and Prolific identifiers before it is launched to participants.
