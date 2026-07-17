# Data codebook

The active roster is six inverse-planning studies on the 3-action set: four on the food scenario set and two on the non-food set. Each experiment folder contains `main_trials.csv` (all participants), `main_trials_long.csv` (exclusions applied; this is what the model and analysis code load), and `exit_survey.csv`, all produced from the gitignored `raw_data/` JSON by `analysis/json_to_csv.py`. Study 3a currently has pilot data only (`nonfood_inv_joint_de/`); its full sample and Study 3b's data have not been collected yet (3b's folder appears once collection starts — its conversion config already exists in `json_to_csv.py`, mirroring 2b's).

| Folder | Study | Inferred | Given |
|---|---|---|---|
| `food_inv_desire/` | 1a | desire | effort, intimacy |
| `food_inv_joint_de/` | 1b | desire + effort (jointly) | intimacy |
| `food_inv_intimacy/` | 2a | intimacy | desire, effort |
| `food_inv_joint_ie/` | 2b | intimacy + effort (jointly) | desire |
| `nonfood_inv_joint_de/` | 3a | desire + effort (jointly) | intimacy |
| `nonfood_inv_joint_ie/` | 3b | intimacy + effort (jointly) | desire |

Data from earlier experiments (forward-planning, the superseded 4-action/2-action inverse studies, early pilots, and side projects) is archived under [`legacy/`](legacy/README.md), which documents what is tracked versus kept local-only.

## Terminology note

The active CSVs use the current "desire" naming (e.g., `desire_condition`); the archived CSVs under `legacy/` keep older column names (see [legacy/README.md](legacy/README.md)).

## Exit survey (all experiments)

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `gender` | Self-reported gender |
| `age` | Self-reported age |
| `understood` | Whether participant understood the task ("yes"/"no") |
| `comments` | Free-text comments |
| `attention_passed` | Whether participant passed attention check (True/False) |
| `memory_correct_count` | Number of correct responses on memory check (0-3) |
| `comprehension_attempt` | Which attempt (1–3) the participant passed the comprehension check on. Everyone in the data passed (those who failed all three attempts are ended before data is saved and never appear in `raw_data/`), so this is a quality signal, not an exclusion field. |

## Main trials

Each row of `main_trials.csv` is one rating page (one prior or posterior elicitation for one scenario). All ratings are collected on 0–100 sliders and stored on the 0–1 scale. Columns shared by all studies:

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier (e.g., "apples") |
| `stimulus_index` | 0-based order in which the participant encountered this scenario (0–15; identical on the scenario's prior and posterior rows), copied from the jsPsych trial data. Supports the first-half-of-trials (repeated-exposure) robustness analyses. |
| `action_condition` | Observed action: "no_share", "low_risk_share", or "high_risk_share" |
| `stage` | "prior" or "posterior" (before vs. after seeing the action) |

In `main_trials_long.csv` the given-condition columns are renamed to bare factor names (`effort_condition` → `effort`, `intimacy_condition` → `intimacy`, `desire_condition` → `desire`) and excluded participants are removed.

### Study 1a — desire inference (`food_inv_desire/`)

| Column | Description |
|--------|-------------|
| `effort_condition` | "low" or "high" |
| `intimacy_condition` | "max_formal", "somewhat_formal", "somewhat_intimate", or "max_intimate" |
| `response` | Desire rating ("how much do they want the food?"), stored 0–1 |

### Study 1b — joint desire + effort (`food_inv_joint_de/`)

Two sliders per page (a `survey-html-form` trial), so each row carries both ratings.

| Column | Description |
|--------|-------------|
| `intimacy_condition` | "max_formal", "somewhat_formal", "somewhat_intimate", or "max_intimate" |
| `desire_rating` | Desire rating, stored 0–1 |
| `effort_rating` | Which effort situation is more likely, stored 0–1 (0 = effort-low paragraph, 1 = effort-high paragraph) |

### Study 2a — intimacy inference (`food_inv_intimacy/`)

| Column | Description |
|--------|-------------|
| `desire_condition` | "low" or "high" |
| `effort_condition` | "low" or "high" |
| `intimacy_rating` | Intimacy rating, stored 0–1 |

### Study 2b — joint intimacy + effort (`food_inv_joint_ie/`)

Two sliders per page, so each row carries both ratings.

| Column | Description |
|--------|-------------|
| `desire_condition` | "low" or "high" |
| `intimacy_rating` | Intimacy rating, stored 0–1 |
| `effort_rating` | Which effort situation is more likely, stored 0–1 (0 = effort-low paragraph, 1 = effort-high paragraph) |

### Studies 3a and 3b — non-food (`nonfood_inv_joint_de/`, `nonfood_inv_joint_ie/`)

The non-food studies repeat 1b's and 2b's designs on the non-food scenario set, so their CSVs will use exactly the same columns: Study 3a matches the Study 1b table above and Study 3b matches the Study 2b table. Study 3a currently has pilot data only; Study 3b's data has not been collected yet.

## Exclusion criteria

The exclusion rule is per-study (each study's `exclusion_rule` in `analysis/json_to_csv.py`), and `memory_correct_count` counts questions (three across the two memory checks):

- **Study 1a** (`food_inv_desire`) uses its preregistered lax rule: participants are excluded only if they failed the attention check **and** answered 0 memory questions correctly, i.e. `(attention_passed != True) & (memory_correct_count == 0)`.
- **Studies 1b, 2a, 2b, 3a, and 3b** use a stricter rule (1a's rule excluded no one, so it was tightened for the later studies): participants are retained only if they passed the attention check **and** answered at least one memory question correctly, i.e. excluded if `(attention_passed != True) | (memory_correct_count == 0)`. The non-food studies' memory checks come from their own scenario set (the sleeping-bag and salary scenarios, three questions total, in `experiments/_lib/memory-checks.js`), but the counting and the rule are the same.

(There is no comprehension-check exclusion: participants who fail the comprehension check after three attempts are ended before any data is saved, so they never appear in `raw_data/`.)

`main_trials_long.csv` reflects exclusions; `main_trials.csv` does not. `analysis/utils.R`'s `report_demographics()` reports both the total recruited and the count surviving exclusions.
