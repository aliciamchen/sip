# Data Codebook

Each experiment folder contains `main_trials.csv`, `main_trials_long.csv`, and `exit_survey.csv`.

The repository has eight active experiments. The six older inverse-planning experiments that the new design replaces are archived under [`legacy/`](legacy/README.md) (gitignored); their column schemas are still documented for the archived CSVs but the active codebook below covers only the current roster.

## Terminology note

Internal variable names use "reward" (e.g., `p_high_reward`, `reward_condition`) or "motivation" rather than "desire" — the public manuscript uses "desire" but the data column names were fixed before that rename.

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

## Forward planning

### Study 1a — Desire manipulation (`food_forw_intimacy_desire/`)

4-action canonical set (`legacy/scenarios.csv`). Design: 16 scenarios × 4 actions × intimacy (4 levels) × desire (2 levels).

**main_trials.csv** (wide format — one row per trial):

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

**main_trials_long.csv** (long format — one row per action):

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier |
| `intimacy` | Relationship closeness level (0, 50, 75, or 100) |
| `motivation` | Motivation condition ("low" or "high") |
| `action` | Action index (0–3) |
| `p_action` | Probability allocated to this action |

### Study 1b — Effort manipulation (`food_forw_intimacy_effort/`)

2-action effort set (`legacy/scenarios_effort.csv`). Design: 16 scenarios × 2 actions × intimacy (4 levels) × effort (2 levels), with reward held fixed at high.

**main_trials.csv** (wide format):

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier |
| `intimacy_condition` | Relationship closeness level |
| `effort_condition` | Effort condition ("low" or "high") |
| `action_1` | Probability allocated to action 1 (non-saliva-sharing) |
| `action_2` | Probability allocated to action 2 (saliva-sharing) |

**main_trials_long.csv** (long format):

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier |
| `intimacy` | Relationship closeness level |
| `effort` | Effort condition ("low" or "high") |
| `action` | Action index (1 = non-saliva, 2 = saliva) |
| `p_action` | Probability allocated to this action |

### Non-food forward (`nonfood_forw_intimacy_desire/`)

Same schema but on `legacy/scenarios_nonfood.csv`. Covers substance/space/privacy stimulus categories.

## Inverse planning — 3-action set

All four active inverse experiments use `scenarios.csv` and follow the noalt-style presentation: the participant sees one observed action per trial. Slider counts and known vs. inferred variables differ per study (see [the experiments README](../experiments/README.md) for design specs).

Each `main_trials_long.csv` has one row per slider response (so 1 or 2 rows per trial × prior/posterior). Common columns:

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier |
| `action_condition` | Observed action ("action_0", "action_1", or "action_2") |
| `stage` | "prior" or "posterior" |

Plus per-study factor columns and slider-response columns:

### Study 2 — Inverse intimacy (`food_inv_intimacy_3act/`)

Design: 2 (reward) × 2 (effort) × 3 (action). Observer infers intimacy.

| Column | Description |
|--------|-------------|
| `reward_condition` | "low" or "high" |
| `effort_condition` | "low" or "high" |
| `intimacy_rating` | Participant's intimacy estimate (0–100) |

### Study 3a — Effort inference (`food_inv_effort_3act/`)

Design: 2 (reward) × 4 (intimacy) × 3 (action). Observer infers effort. Slider endpoints are the two `effort_low` / `effort_high` paragraphs (also stored on each trial for downstream recovery).

| Column | Description |
|--------|-------------|
| `reward_condition` | "low" or "high" |
| `intimacy_condition` | 0, 50, 75, or 100 |
| `response` | Slider value 0–100, where 0 = `effort_low` endpoint and 100 = `effort_high` endpoint |
| `effort_low`, `effort_high` | The two effort-paragraph texts that anchor the slider on this trial |

### Study 3b — Desire inference (`food_inv_desire_3act/`)

Design: 2 (effort) × 4 (intimacy) × 3 (action). Observer infers desire. Slider endpoints are the two reward paragraphs.

| Column | Description |
|--------|-------------|
| `effort_condition` | "low" or "high" |
| `intimacy_condition` | 0, 50, 75, or 100 |
| `response` | Slider value 0–100, where 0 = `reward_low` endpoint and 100 = `reward_high` endpoint |
| `reward_low`, `reward_high` | The two reward-paragraph texts that anchor the slider on this trial |

### Study 4a — Joint desire + effort (`food_inv_joint_de_3act/`)

Design: 4 (intimacy) × 3 (action). Observer jointly infers desire and effort; two sliders per trial.

| Column | Description |
|--------|-------------|
| `intimacy_condition` | 0, 50, 75, or 100 |
| `response_target` | "reward" or "effort" (which slider this row records) |
| `response` | Slider value 0–100, with `reward_low`/`reward_high` or `effort_low`/`effort_high` endpoints depending on `response_target` |
| `reward_low`, `reward_high` | On reward rows |
| `effort_low`, `effort_high` | On effort rows |

### Study 4b — Joint desire + intimacy (`food_inv_joint_di_3act/`)

Design: 2 (effort) × 3 (action). Observer jointly infers desire and intimacy; two sliders per trial (desire slider with reward-paragraph endpoints, intimacy slider on the 0–100 scale).

| Column | Description |
|--------|-------------|
| `effort_condition` | "low" or "high" |
| `response_target` | "reward" or "intimacy" |
| `response` | Slider value 0–100. On reward rows: 0 = `reward_low`, 100 = `reward_high`. On intimacy rows: 0 = maximally formal, 100 = maximally intimate |
| `reward_low`, `reward_high` | On reward rows |

## Exclusion Criteria

Participants are excluded from analysis if:
- `attention_passed != True`
- `memory_correct_count == 0`

`main_trials_long.csv` reflects exclusions; `main_trials.csv` does not. `analysis/utils.R`'s `report_demographics()` reports both the total recruited and the count surviving exclusions.
