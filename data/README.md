# Data Codebook


Each experiment folder contains `main_trials.csv`, `main_trials_long.csv`, and `exit_survey.csv`.

## Terminology note

In the inverse-planning experiments, internal variable names use "reward" (e.g., `p_high_reward`, `reward_condition`) or "motivation" rather than "desire" — we changed the terminology to "desire" after running the experiments, for clarity.

## Exit Survey (all experiments)

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `gender` | Self-reported gender |
| `age` | Self-reported age |
| `understood` | Whether participant understood the task ("yes"/"no") |
| `comments` | Free-text comments |
| `attention_passed` | Whether participant passed attention check (True/False) |
| `memory_correct_count` | Number of correct responses on memory check (0-3) |

## Forward planning (`forw_plan/`)

**main_trials.csv** (wide format - one row per trial):

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier (e.g., "hike", "wedding", "basketball") |
| `intimacy_condition` | Relationship closeness level (0, 50, 75, or 100) |
| `reward_condition` | Motivation condition ("low" or "high") |
| `action_0` | Probability allocated to action 0 (no sharing) |
| `action_1` | Probability allocated to action 1 (sharing with no saliva risk) |
| `action_2` | Probability allocated to action 2 (sharing with moderate saliva risk) |
| `action_3` | Probability allocated to action 3 (sharing with high saliva risk) |

**main_trials_long.csv** (long format - one row per action):

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier |
| `intimacy` | Relationship closeness level (0, 50, 75, or 100) |
| `motivation` | Motivation condition ("low" or "high") |
| `action` | Action index (0-3) |
| `p_action` | Probability allocated to this action |

## Intimacy inference, alternatives shown (`inv_plan_intimacy_alt/`)

**main_trials.csv** and **main_trials_long.csv**:

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier |
| `action_condition` | Observed action ("action_0" through "action_3") |
| `reward_condition` | Motivation condition shown ("low" or "high") |
| `stage` | Measurement timing ("prior" = before seeing action, "posterior" = after) |
| `intimacy_rating` | Participant's intimacy estimate (0-100 scale) |

Note: `main_trials_long.csv` uses `motivation` instead of `reward_condition`.

## Intimacy inference, no alternatives shown (`inv_plan_intimacy_noalt/`)

Same columns as `inv_plan_intimacy_alt/` above. The experimental procedure differs only in that participants see a single observed action at the posterior stage rather than the full four-action list; on the model side, counterfactual alternatives are supplied by a language model.

## Desire inference, alternatives shown (`inv_plan_desire_alt/`)

**main_trials.csv** and **main_trials_long.csv**:

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier |
| `action_condition` | Observed action ("action_0" through "action_3") |
| `intimacy_condition` / `intimacy` | Relationship closeness level shown (0, 50, 75, or 100) |
| `stage` | Measurement timing ("prior" or "posterior") |
| `response` | Participant's desire/motivation estimate (0-100 scale) |

## Exclusion Criteria

Participants are excluded from analysis if:
- `attention_passed != True`
- `memory_correct_count == 0`
