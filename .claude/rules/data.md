---
paths:
  - "data/**/*"
---

# Data structure

Main experiments are at the top level:

Each directory name reflects what the experiment measures, not a paper experiment number:

```
data/
  food_forw_intimacy_desire/        # Forward planning (actors choose actions)
  food_inv-intimacy_desire_alt/     # Intimacy inference, alternatives shown to participants
  food_inv-intimacy_desire_noalt/   # Intimacy inference, no alternatives shown (LM-generated counterfactuals on the model side)
  food_inv-desire_intimacy_alt/     # Desire inference, alternatives shown
  food_inv-desire_intimacy_noalt/   # Desire inference, no alternatives shown (relationship-keyed action space)
  food_forw_intimacy_effort/        # Forward planning, effort manipulation (2 actions, intimacy × effort, reward fixed at high)
  food_inv-intimacy_effort_alt/     # Inverse planning, effort manipulation (2 candidate actions × effort, intimacy inference)
  food_inv-effort_intimacy_alt/     # Inverse planning, effort inference (2 candidate actions × intimacy, effort inference)
  nonfood_forw_intimacy_desire/     # Non-food forward planning (parallels food_forw_intimacy_desire on scenarios_nonfood.csv)
  inv_plan_reward/                  # Earlier "reward inference" predecessor of food_inv-desire_intimacy_alt (raw only)
  planning_comm/                    # Communication experiment
  pilots/                           # Pilot experiments
```

Each experiment folder contains:
- `raw_data/` - JSON files from experiment
- `main_trials.csv` - Processed trial data (all participants)
- `main_trials_long.csv` - Long format with excluded participants removed
- `exit_survey.csv` - Demographic and attention check data

Participant exclusion criteria:
- Failed attention check (`attention_passed != True`)
- Got 0 correct on memory check (`memory_correct_count == 0`)
