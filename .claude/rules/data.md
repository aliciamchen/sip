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

`data/legacy/` holds earlier experiments not part of the current pipeline (`planning_comm/`, `pilots/`); the directory is covered by the `legacy` gitignore rule.

`data/food_inv_desire_intimacy_alt/raw_data/` was originally collected under the URL slug `inv-plan-reward-final` (back when the experiment was called "reward inference"). The raw JSONs use Prolific PIDs, which `analysis/json_to_csv.py` deterministically anonymizes to UUIDs in the processed CSVs.

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
