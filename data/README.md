# Data codebook

The `data/` directory contains the processed, de-identified data from all six
studies. Each study directory contains three CSV files:

- `main_trials.csv` contains all prior and posterior ratings from participants
  who completed the study.
- `main_trials_long.csv` excludes participants who did not meet that study's
  inclusion criteria. The model and figure code use this file.
- `exit_survey.csv` contains demographics, study feedback, and the responses
  used to determine inclusion.

The raw jsPsych JSON files may contain identifying information and are not
included. `data_prep/json_to_csv.py` converts those files into the CSV files
described here.

| Directory | Study | Ratings participants provide | Information they are given |
|---|---|---|---|
| `food_inv_desire/` | 1a | desire | effort and intimacy |
| `food_inv_joint_de/` | 1b | desire and effort | intimacy |
| `food_inv_intimacy/` | 2a | intimacy | desire and effort |
| `food_inv_joint_ie/` | 2b | intimacy and effort | desire |
| `nonfood_inv_joint_de/` | 3a | desire and effort | intimacy |
| `nonfood_inv_joint_ie/` | 3b | intimacy and effort | desire |

## Trial data

Each row of `main_trials.csv` records one rating page for one scenario. Ratings
were collected on sliders from 0 to 100 and are stored as values from 0 to 1.
The following columns appear in every study:

| Column | Description |
|---|---|
| `subject_id` | An anonymized participant ID. |
| `scenario_label` | The scenario name, such as `apples`. |
| `stimulus_index` | The order in which the participant saw the scenario, from 0 to 15. |
| `action_condition` | The observed action: `no_share`, `low_risk_share`, or `high_risk_share`. |
| `stage` | Whether the rating was made before (`prior`) or after (`posterior`) the observed action. |

`main_trials_long.csv` contains the same rows after exclusions. In that file,
the columns for information given to participants use shorter names:
`effort_condition` becomes `effort`, `intimacy_condition` becomes `intimacy`,
and `desire_condition` becomes `desire`.

### Study 1a -- desire

| Column | Description |
|---|---|
| `effort_condition` | Whether obtaining another serving requires low or high effort. |
| `intimacy_condition` | The relationship description, from `max_formal` to `max_intimate`. |
| `response` | The participant's desire rating. |

### Study 1b -- desire and effort

| Column | Description |
|---|---|
| `intimacy_condition` | The relationship description, from `max_formal` to `max_intimate`. |
| `desire_rating` | The participant's desire rating. |
| `effort_rating` | The participant's judgment of which effort situation is more likely. A value of 0 corresponds to the low-effort description and 1 to the high-effort description. |

### Study 2a -- intimacy

| Column | Description |
|---|---|
| `desire_condition` | Whether the characters have low or high desire. |
| `effort_condition` | Whether obtaining another serving requires low or high effort. |
| `intimacy_rating` | The participant's intimacy rating, from maximally formal to maximally intimate. |

### Study 2b -- intimacy and effort

| Column | Description |
|---|---|
| `desire_condition` | Whether the characters have low or high desire. |
| `intimacy_rating` | The participant's intimacy rating, from maximally formal to maximally intimate. |
| `effort_rating` | The participant's judgment of which effort situation is more likely. A value of 0 corresponds to the low-effort description and 1 to the high-effort description. |

Studies 3a and 3b use non-food scenarios. Their columns match Studies 1b and
2b, respectively.

## Exit survey

| Column | Description |
|---|---|
| `subject_id` | An anonymized participant ID. |
| `gender` | The participant's self-reported gender. |
| `age` | The participant's self-reported age. |
| `understood` | Whether the participant reported understanding the task. |
| `comments` | Free-text feedback. |
| `attention_passed` | Whether the participant passed the attention check. |
| `memory_correct_count` | The number of correct answers across the three memory questions. |
| `comprehension_attempt` | The attempt on which the participant passed the comprehension check, from 1 to 3. |

## Exclusions

Study 1a excludes participants only when they both fail the attention check and
answer none of the memory questions correctly. The other five studies require
participants to pass the attention check and answer at least one memory
question correctly.

Participants must pass the comprehension check before beginning the study.
Those who do not pass within three attempts do not produce a saved data file,
so no additional comprehension-check exclusion is needed.
