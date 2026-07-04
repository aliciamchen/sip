# Data codebook

The active roster is four inverse-planning studies on the 3-action set. Each experiment folder contains `main_trials.csv` (all participants), `main_trials_long.csv` (exclusions applied; this is what the model and analysis code load), and `exit_survey.csv`, all produced from the gitignored `raw_data/` JSON by `analysis/json_to_csv.py`.

| Folder | Study | Inferred | Given |
|---|---|---|---|
| `food_inv_desire/` | 1a | desire | effort, intimacy |
| `food_inv_joint_de/` | 1b | desire + effort (jointly) | intimacy |
| `food_inv_intimacy/` | 2a | intimacy | desire, effort |
| `food_inv_joint_ie/` | 2b | intimacy + effort (jointly) | desire |

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

Each row of `main_trials.csv` is one rating page (one prior or posterior elicitation for one scenario). All ratings are collected on 0–100 sliders and stored on the 0–1 scale. Columns shared by all four studies:

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier (e.g., "apples") |
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

## Exclusion criteria

The exclusion rule is per-study (each study's `exclusion_rule` in `analysis/json_to_csv.py`), and `memory_correct_count` counts questions (three across the two memory checks):

- **Study 1a** (`food_inv_desire`) uses its preregistered lax rule: participants are excluded only if they failed the attention check **and** answered 0 memory questions correctly, i.e. `(attention_passed != True) & (memory_correct_count == 0)`.
- **Studies 1b, 2a, and 2b** use a stricter rule (1a's rule excluded no one, so it was tightened for the later studies): participants are retained only if they passed the attention check **and** answered at least one memory question correctly, i.e. excluded if `(attention_passed != True) | (memory_correct_count == 0)`.

(There is no comprehension-check exclusion: participants who fail the comprehension check after three attempts are ended before any data is saved, so they never appear in `raw_data/`.)

`main_trials_long.csv` reflects exclusions; `main_trials.csv` does not. `analysis/utils.R`'s `report_demographics()` reports both the total recruited and the count surviving exclusions.
