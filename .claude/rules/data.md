---
paths:
  - "data/**/*"
---

# Data structure

Each experiment folder is named after its slug and contains:

- `raw_data/` — JSON files from jsPsych (gitignored).
- `main_trials.csv` — wide-format processed trials (all participants).
- `main_trials_long.csv` — long format with excluded participants removed; this is what the analysis qmds load.
- `exit_survey.csv` — demographics and attention/memory checks.

The active roster is four inverse experiments under `data/<slug>/` — `food_inv_desire` (1a), `food_inv_joint_de` (1b), `food_inv_intimacy` (2a), `food_inv_joint_ie` (2b) — none with collected data yet.

`data/legacy/` holds earlier experiments not part of the current pipeline: the original `planning_comm/` and `pilots/` side projects; the six pre-3-action inverse food experiments archived in May 2026; the three forward-planning experiments (`food_forw_intimacy_desire`, `food_forw_intimacy_effort`, `nonfood_forw_intimacy_desire`) moved to legacy in the May 2026 roster refactor (their forward model scripts read from `data/legacy/<slug>/` and stay runnable under `LEGACY_FORWARD`); and `food_inv_desire_pilot`, the original Study 1a pilot (collected on a 0–100 desire slider before the manuscript switched to a 1–7 Likert). Two of the pre-3-action experiments — `food_inv_intimacy_desire_noalt` and `food_inv_desire_intimacy_noalt` — still have runnable per-slug model targets under `LEGACY_INVERSE`. The four `_alt` siblings (`food_inv_intimacy_desire_alt`, `food_inv_desire_intimacy_alt`, `food_inv_intimacy_effort_alt`, `food_inv_effort_intimacy_alt`) used a pre-specified alternatives-shown paradigm and their model + analysis code has been removed; only the data remains for reproducibility. The archived inverse experiments are documented in [data/legacy/README.md](../../data/legacy/README.md); the directory is covered by the `legacy` gitignore rule.

`data/legacy/food_inv_desire_intimacy_alt/raw_data/` was originally collected under the URL slug `inv-plan-reward-final` (back when the experiment was called "reward inference"). The raw JSONs use Prolific PIDs, which `analysis/json_to_csv.py` deterministically anonymizes to UUIDs in the processed CSVs.

## Participant exclusion criteria

Participants are excluded if either is true:
- Failed attention check (`attention_passed != True`)
- Got 0 correct on memory check (`memory_correct_count == 0`)

`main_trials_long.csv` reflects exclusions; `main_trials.csv` does not.

## Anonymization

`analysis/json_to_csv.py` maps each Prolific PID to a deterministic UUID5 (namespace `6ba7b810-9dad-11d1-80b4-00c04fd430c8`); the mapping is regenerated from `raw_data/` on each run and never persisted to disk. Tracked CSVs only ever contain the anonymized UUIDs. The `raw_data/` directories are gitignored repo-wide.

## Where else to look

- The full experiment roster lives in [README.md](../../README.md).
- Per-CSV column documentation is in [data/README.md](../../data/README.md).
