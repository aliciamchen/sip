---
paths:
  - "data/**/*"
---

# Data structure

Main experiments are at the top level:

Each directory name reflects what the experiment measures, not a paper experiment number:

```
data/
  forw_plan/               # Forward planning (actors choose actions)
  inv_plan_intimacy_alt/   # Intimacy inference, alternatives shown to participants
  inv_plan_intimacy_noalt/ # Intimacy inference, no alternatives shown (LM-generated counterfactuals on the model side)
  inv_plan_desire_alt/     # Desire inference, alternatives shown
  forw_plan_effort/        # Forward planning, effort manipulation (2 actions, intimacy × effort, reward fixed at high)
  inv_plan_effort/         # Inverse planning, effort manipulation (2 candidate actions × effort, intimacy inference)
  inv_plan_effort_inferred/ # Inverse planning, effort inference (2 candidate actions × intimacy, effort inference)
  inv_plan_reward/         # Earlier "reward inference" predecessor of inv_plan_desire_alt (raw only)
  planning_comm/           # Communication experiment
  pilots/                  # Pilot experiments
```

Each experiment folder contains:
- `raw_data/` - JSON files from experiment
- `main_trials.csv` - Processed trial data (all participants)
- `main_trials_long.csv` - Long format with excluded participants removed
- `exit_survey.csv` - Demographic and attention check data

Participant exclusion criteria:
- Failed attention check (`attention_passed != True`)
- Got 0 correct on memory check (`memory_correct_count == 0`)
