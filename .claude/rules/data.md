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

`data/legacy/` holds archived data from earlier experiments (forward-planning, pre-3-action inverse, side projects, and the original Study 1a pilot) that are no longer part of the pipeline — their model and analysis code was removed; only the data is kept for reproducibility. It's documented in [data/legacy/README.md](../../data/legacy/README.md). The forward-planning and Study 1a pilot CSVs are tracked; the superseded inverse experiments, `pilots/`, and `planning_comm/` are local-only (scoped `data/legacy/*` gitignore rules).

## Participant exclusion criteria

Participants are excluded only if **both** are true (i.e. they failed every check):
- Failed attention check (`attention_passed != True`)
- Got 0 correct on memory check (`memory_correct_count == 0`)

So the exclusion filter is `(attention_passed != True) & (memory_correct_count == 0)`; anyone who passes attention or answers at least one memory check correctly is retained.

`main_trials_long.csv` reflects exclusions; `main_trials.csv` does not. There is no comprehension-check exclusion: participants who fail the comprehension check (3 attempts) are ended via `jsPsych.abortExperiment` before the DataPipe save, so they never appear in `raw_data/`. The `exit_survey` rows carry `comprehension_attempt` (1–3) as a quality signal for the participants who did pass.

## Anonymization

`analysis/json_to_csv.py` maps each Prolific PID to a deterministic UUID5 (namespace `6ba7b810-9dad-11d1-80b4-00c04fd430c8`); the mapping is regenerated from `raw_data/` on each run and never persisted to disk. Tracked CSVs only ever contain the anonymized UUIDs. The `raw_data/` directories are gitignored repo-wide.

## Where else to look

- The full experiment roster lives in [README.md](../../README.md).
- Per-CSV column documentation is in [data/README.md](../../data/README.md).
