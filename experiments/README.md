# Experiments

## Terminology note

In Experiment 2, internal variable names use "reward" (e.g., `p_high_reward`, `reward_condition`) rather than "desire" — we changed the terminology to "desire" after we ran the experiments, for clarity 

## Scenarios (`scenarios.csv`)

Spreadsheet of scenarios used to generate stimuli for the experiments.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier used in data files |
| `name_0`, `name_1` | Character names in the vignette |
| `vignette` | Base scenario description |
| `reward_low` | Text describing low motivation condition |
| `reward_high` | Text describing high motivation condition |
| `action_0` | Description of action 0 (no sharing) |
| `action_1` | Description of action 1 (minimal sharing) |
| `action_2` | Description of action 2 (moderate sharing) |
| `action_3` | Description of action 3 (maximum sharing) |

### Action Scale

Actions are ordered by degree of saliva-sharing risk:
- **Action 0**: No sharing
- **Action 1**: Sharing with no saliva risk (e.g., cutting food in half, using separate utensils)
- **Action 2**: Sharing with moderate saliva risk (e.g., eating from opposite ends)
- **Action 3**: Sharing with high saliva risk (e.g., same utensil, same bite location) 
## Main experiments

- [inv_plan_intimacy](inv_plan_intimacy/README.md) - Inverse planning: infer intimacy from observed actions
- [inv_plan_desire](inv_plan_desire/README.md) - Inverse planning: infer desire from observed actions
- [forw_plan](forw_plan/README.md) - Planning with better-defined rewards and reward manipulations
- [planning_comm](planning_comm/README.md) - Planning with communciative goals 

## Pilot experiments (pilots/)

- [planning_1](pilots/planning_1/README.md) - Initial planning task
- [planning_priors](pilots/planning_priors/README.md) - Prior beliefs about actions
- [risk](pilots/risk/README.md) - Risk/saliva transfer perception
- [effort](pilots/effort/README.md) - Effort cost perception
- [discomfort](pilots/discomfort/README.md) - Discomfort/saliva concern perception
