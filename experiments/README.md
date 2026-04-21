# Experiments

## Terminology note

In the inverse-planning experiments, internal variable names use "reward" (e.g., `p_high_reward`, `reward_condition`) rather than "desire" — we changed the terminology to "desire" after running the experiments, for clarity.

## Scenarios (`scenarios.csv`)

Spreadsheet of scenarios used to generate stimuli for the experiments.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier used in data files |
| `name_0`, `name_1` | Character names in the vignette |
| `vignette` | Base scenario description |
| `reward_low` | Text describing low motivation condition |
| `reward_high` | Text describing high motivation condition |
| `action_0` | Description of action 0 |
| `action_1` | Description of action 1 |
| `action_2` | Description of action 2 |
| `action_3` | Description of action 3 |

### Action Scale

Actions are ordered by degree of saliva-sharing risk:
- **Action 0**: No sharing
- **Action 1**: Sharing with no saliva risk (e.g., cutting food in half, using separate utensils)
- **Action 2**: Sharing with moderate saliva risk (e.g., eating from opposite ends)
- **Action 3**: Sharing with high saliva risk (e.g., same utensil, same bite location) 
## Main experiments

- [forw_plan](forw_plan/README.md) — Forward planning: actors choose actions given intimacy and motivation
- [inv_plan_intimacy_alt](inv_plan_intimacy_alt/README.md) — Inverse planning: infer intimacy from the observed action (four candidate actions shown to participants)
- [inv_plan_intimacy_noalt](inv_plan_intimacy_noalt/README.md) — Same inference as `inv_plan_intimacy_alt` but with the candidate actions hidden from participants; counterfactual alternatives are supplied by a language model on the model side
- [inv_plan_desire_alt](inv_plan_desire_alt/README.md) — Inverse planning: infer desire from the observed action (four candidate actions shown)
