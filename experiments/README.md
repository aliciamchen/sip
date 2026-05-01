# Experiments

## Terminology note

In the inverse-planning experiments, internal variable names use "reward" (e.g., `p_high_reward`, `reward_condition`) rather than "desire" — we changed the terminology to "desire" after running the experiments, for clarity.

## Scenarios (`scenarios.csv`)

Spreadsheet of scenarios used to generate stimuli for the experiments. The CSV is a generated artifact — the source of truth is `scenarios.py`, which writes the CSV when run. To edit scenarios, modify `scenarios.py` and regenerate:

```bash
uv run python experiments/scenarios.py
```

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

## Non-food scenarios (`scenarios_nonfood.csv`)

A parallel set of 16 scenarios covering non-food sharing situations, grouped by what is being shared: substance (e.g., chapstick, sunscreen, hairbrush), space (e.g., blanket, bed, sauna), and privacy (e.g., breakup conversation, payment, phone passcode). Like `scenarios.csv`, the CSV is generated from a Python source of truth — edit `scenarios_nonfood.py` and regenerate:

```bash
uv run python experiments/scenarios_nonfood.py
```

The schema matches `scenarios.csv` with one additional column, `scenario_type`, which takes one of `substance`, `space`, or `privacy`.

## Effort-manipulation scenarios (`scenarios_effort.csv`)

A parallel set of 16 food-sharing scenarios for an experiment that manipulates the relative effort of avoiding saliva sharing, rather than reward. Each scenario uses the same `scenario_label` and character names as `scenarios.csv`, but differs in three ways:

- Reward is held fixed at high and integrated into the vignette narrative (no separate `reward_low`/`reward_high` columns).
- The action space is collapsed to two actions: `action_1` is a non-saliva-sharing action (e.g., using an extra utensil, cutting a portion, using a second cup), and `action_2` is a plausible saliva-sharing action for the scenario.
- Each scenario has one shared `vignette` followed by one of two effort-manipulation paragraphs, `effort_low` or `effort_high`. The low paragraph makes the resource that `action_1` relies on (knife, extra plate, second cup, etc.) easy to obtain; the high paragraph makes it costly. The action text is identical across conditions. In the experiments, the effort paragraph is rendered as a separate paragraph immediately after the shared vignette.

Like the other two scenario CSVs, this one is generated from a Python source of truth — edit `scenarios_effort.py` and regenerate:

```bash
uv run python experiments/scenarios_effort.py
```

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier (matches `scenarios.csv`) |
| `name_0`, `name_1` | Character names in the vignette |
| `vignette` | Shared scenario narrative (same across both effort conditions) |
| `effort_low` | Trailing paragraph in which avoiding saliva sharing is easy |
| `effort_high` | Trailing paragraph in which avoiding saliva sharing is costly |
| `action_1` | Non-saliva-sharing action |
| `action_2` | Saliva-sharing action |

## Main experiments

- [food_forw_intimacy_desire](food_forw_intimacy_desire/README.md) — Forward planning: actors choose actions given intimacy and motivation
- [food_inv-intimacy_desire_alt](food_inv-intimacy_desire_alt/README.md) — Inverse planning: infer intimacy from the observed action (four candidate actions shown to participants)
- [food_inv-intimacy_desire_noalt](food_inv-intimacy_desire_noalt/README.md) — Same inference as `food_inv-intimacy_desire_alt` but with the candidate actions hidden from participants; counterfactual alternatives are supplied by a language model on the model side
- [food_inv-desire_intimacy_alt](food_inv-desire_intimacy_alt/README.md) — Inverse planning: infer desire from the observed action (four candidate actions shown)
- [food_inv-desire_intimacy_noalt](food_inv-desire_intimacy_noalt/README.md) — Same inference as `food_inv-desire_intimacy_alt` but with the candidate actions hidden; LM-generated counterfactual alternatives on the model side (no data collected yet)

## Effort-manipulation experiments

These use the `scenarios_effort.csv` stimulus set (two actions per scenario, reward held fixed at high) and vary relative effort as a second manipulation:

- [food_forw_intimacy_effort](food_forw_intimacy_effort/README.md) — Forward planning: actors choose between two actions given intimacy (4 levels) × relative effort (2 levels)
- [food_inv-intimacy_effort_alt](food_inv-intimacy_effort_alt/README.md) — Inverse planning: infer intimacy from the observed action (2 candidate actions shown) × relative effort (2 levels)
- [food_inv-effort_intimacy_alt](food_inv-effort_intimacy_alt/README.md) — Inverse planning: infer effort from the observed action (2 candidate actions shown) × intimacy (4 levels), with the two effort paragraphs as slider endpoints

## Non-food experiments

A second parallel pipeline that uses the `scenarios_nonfood.csv` stimulus set (substance sharing, shared space, and privacy) instead of food sharing. The five experiments mirror the canonical food set one-to-one in structure and counterbalancing, with the participant-facing copy generalized away from food. No data has been collected yet for any of them.

- [nonfood_forw_intimacy_desire](nonfood_forw_intimacy_desire/README.md)
- [nonfood_inv-intimacy_desire_alt](nonfood_inv-intimacy_desire_alt/README.md)
- [nonfood_inv-desire_intimacy_alt](nonfood_inv-desire_intimacy_alt/README.md)
- [nonfood_inv-intimacy_desire_noalt](nonfood_inv-intimacy_desire_noalt/README.md)
- [nonfood_inv-desire_intimacy_noalt](nonfood_inv-desire_intimacy_noalt/README.md)
