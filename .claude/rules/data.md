---
paths:
  - "data/**/*"
---

# Data structure

Each active study is in `data/<slug>/` and contains:

- `raw_data/`: gitignored jsPsych JSON containing participant identifiers.
- `main_trials.csv`: wide processed trials for all parsed participants.
- `main_trials_long.csv`: long processed trials after participant exclusions; model and figure code load this file.
- `exit_survey.csv`: demographics, attention checks, memory checks, and completion metadata.

The active roster and paper labels are defined in `study_registry.py` and the `Makefile`. `data/legacy/` is local-only archival data and is not an input to the active pipeline.

## Exclusions

`data_prep/json_to_csv.py` is the exclusion-rule source of truth.

- Study 1a uses its preregistered lax rule: exclude only participants who fail attention and answer no memory question correctly.
- Studies 1b, 2a, 2b, 3a, and 3b require both a passed attention check and at least one correct memory question.

`memory_correct_count` counts questions, not check screens. `main_trials_long.csv` reflects exclusions; `main_trials.csv` does not. Participants who fail the experiment's three-attempt comprehension gate are aborted before DataPipe saves, so they do not enter `raw_data/`.

## Anonymization

The converter replaces each Prolific PID with a deterministic UUID5 and never persists the mapping. Tracked CSVs contain only anonymized IDs; raw JSON remains gitignored.

See `data/README.md` for column definitions and `.claude/rules/data_prep.md` for conversion behavior.
