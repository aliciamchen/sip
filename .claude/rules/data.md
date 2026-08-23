---
paths:
  - "data/**/*"
---

# Data structure

Each experiment folder is named after its slug and contains:

- `raw_data/` — JSON files from jsPsych (gitignored).
- `main_trials.csv` — wide-format processed trials (all participants).
- `main_trials_long.csv` — long format with excluded participants removed; this is what the model and figure code load.
- `exit_survey.csv` — demographics and attention/memory checks.

The active roster is six inverse experiments under `data/<slug>/` — `food_inv_desire` (1a), `food_inv_joint_de` (1b), `food_inv_intimacy` (2a), `food_inv_joint_ie` (2b), `nonfood_inv_joint_de` (3a), `nonfood_inv_joint_ie` (3b). All six now have full samples (retained after exclusions: 1a ~480; 1b, 2a, 3a, 3b ~240 each; 2b ~120); all are in the Makefile's `EXPERIMENTS_INVERSE` as of the 2026-07-21 roster move.

`data/legacy/` holds archived data from earlier, superseded experiments (forward-planning, pre-3-action inverse, side projects, and the original Study 1a pilot). It is local-only and gitignored — nothing in the active pipeline reads it, and its model and analysis code was removed long ago.

## Participant exclusion criteria

The rule is per-study (the config's `exclusion_rule` in `data_prep/json_to_csv.py`); `memory_correct_count` counts questions (0–3), not checks:

- **1a (lax, preregistered)**: excluded only if `(attention_passed != True) & (memory_correct_count == 0)` — anyone who passes attention OR answers ≥1 memory question is retained.
- **1b/2a/2b/3a/3b (strict)**: retained only if `attention_passed == True & memory_correct_count > 0` — 1a's rule excluded 0 participants, so it was tightened for the later studies (matching the manuscript and their preregs). The nonfood studies' memory checks are the sleeping-bag (2 questions) + salary (1 question) scenarios in `NONFOOD_MEMORY_CHECKS`.

`main_trials_long.csv` reflects exclusions; `main_trials.csv` does not. There is no comprehension-check exclusion: participants who fail the comprehension check (3 attempts) are ended via `jsPsych.abortExperiment` before the DataPipe save, so they never appear in `raw_data/`. The `exit_survey` rows carry `comprehension_attempt` (1–3) as a quality signal for the participants who did pass.

## Anonymization

`data_prep/json_to_csv.py` maps each Prolific PID to a deterministic UUID5 (namespace `6ba7b810-9dad-11d1-80b4-00c04fd430c8`); the mapping is regenerated from `raw_data/` on each run and never persisted to disk. Tracked CSVs only ever contain the anonymized UUIDs. The `raw_data/` directories are gitignored repo-wide.

## Where else to look

- The full experiment roster lives in [README.md](../../README.md).
- Per-CSV column documentation is in [data/README.md](../../data/README.md).
