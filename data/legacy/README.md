# Legacy data

This directory archives participant data from experiments that are no longer part of the active pipeline, which now consists of the four inverse-planning studies documented in [data/README.md](../README.md). The model and analysis code for these older experiments was removed in the June 2026 cleanup and is recoverable from git history; only the data is kept here.

Not everything in this directory is tracked in git. The processed CSVs for the three forward-planning experiments and the original Study 1a pilot are committed. The six superseded 4-action/2-action inverse experiments, the early design pilots (`pilots/`), and the `planning_comm/` side project were deliberately removed from git history when they were archived (May 2026) and exist only as local copies, as do all `raw_data/` directories.

## Column naming

The archived CSVs predate the project-wide rename of "reward"/"motivation" to **desire** and "access" to **risk**, so they keep the older column names: the forward-planning `main_trials.csv` files use `reward_condition` (and their long format uses `motivation`), and the archived inverse experiments use `reward_condition` and reward-paragraph columns. The active studies' CSVs use the current `desire` naming throughout.

## Forward-planning experiments (tracked)

Three experiments in which participants allocated probabilities across all available actions, collected before the manuscript was reorganized around inverse planning.

| Slug | Stimulus set | Manipulation |
|---|---|---|
| `food_forw_intimacy_desire` | 4 actions, food | desire (then "reward") × intimacy |
| `food_forw_intimacy_effort` | 2 actions, food | effort × intimacy |
| `nonfood_forw_intimacy_desire` | 4 actions, non-food (substance/space/privacy) | desire × intimacy |

`main_trials.csv` is wide format, one row per trial: `subject_id`, `scenario_label`, `intimacy_condition` (0, 50, 75, or 100), `reward_condition` or `effort_condition` ("low"/"high"), and one `action_<i>` probability column per action (4-action set: `action_0` = no sharing through `action_3` = high saliva risk; 2-action set: `action_1` = non-saliva-sharing, `action_2` = saliva-sharing).

`main_trials_long.csv` is one row per action: `subject_id`, `scenario_label`, `intimacy`, `motivation` or `effort`, `action` (index), `p_action`.

## Study 1a pilots (tracked)

`food_inv_desire_pilot/` holds the original pilot for manuscript Study 1a, with the same design as the active `food_inv_desire` (2 effort × 4 intimacy × 3 action) and the same column schema. The difference is the desire DV: the pilot collected a 0–100 "probability of two motivational states" slider, whereas Study 1a now uses a direct 0–100 desire rating.

`food_inv_desire_pilot_jun2026/` holds a second, later pilot (N = 20, collected 2026-06-17) that already used the current direct 0–100 desire rating, so it is closer to the final Study 1a design than the original pilot. It is archived here rather than retained in the active directory because it predates the 2026-06-19 intimacy-label rename: its `intimacy_condition` column still uses the old `neither` slug in place of `somewhat_formal`. Pooling it with the full sample would therefore split one intimacy level across two label strings, so it was moved aside before the full-sample run. It keeps the same column schema as the active study.

## Superseded inverse-planning experiments (local-only)

Six food-domain inverse-planning experiments collected against the original 4-action canonical set and 2-action effort set, superseded when the manuscript was reorganized around the single 3-action set and the current Studies 1a/1b/2a/2b.

| Slug | Inferred | Conditioning | Action set |
|---|---|---|---|
| `food_inv_intimacy_desire_alt` | intimacy | desire | 4 actions, alternatives shown |
| `food_inv_desire_intimacy_alt` | desire | intimacy | 4 actions, alternatives shown |
| `food_inv_intimacy_desire_noalt` | intimacy | desire | 4 actions, single action shown |
| `food_inv_desire_intimacy_noalt` | desire | intimacy | 4 actions, single action shown |
| `food_inv_intimacy_effort_alt` | intimacy | effort | 2 actions, alternatives shown |
| `food_inv_effort_intimacy_alt` | effort | intimacy | 2 actions, alternatives shown |

## Earlier pilots and side projects (local-only)

- `pilots/` — early pilots collected while iterating on the experimental design (discomfort, effort, forw_plan, planning_priors, planning-1, risk).
- `planning_comm/` — a parallel side project on planning and communication.
