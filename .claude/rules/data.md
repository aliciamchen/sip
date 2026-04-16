---
paths:
  - "data/**/*"
---

# Data structure

Main experiments are at the top level:

```
data/
  forw_plan/             # Experiment 1 (forward planning)
  inv_plan_intimacy/     # Experiment 2a (intimacy inference)
  inv_plan_desire/       # Experiment 2b (desire inference) — current
  inv_plan_reward/       # Earlier "reward inference" version of Exp 2b (raw only)
  planning_comm/         # Communication experiment
  pilots/                # Pilot experiments
```

Each experiment folder contains:
- `raw_data/` - JSON files from experiment
- `main_trials.csv` - Processed trial data (all participants)
- `main_trials_long.csv` - Long format with excluded participants removed
- `exit_survey.csv` - Demographic and attention check data

Participant exclusion criteria:
- Failed attention check (`attention_passed != True`)
- Got 0 correct on memory check (`memory_correct_count == 0`)
