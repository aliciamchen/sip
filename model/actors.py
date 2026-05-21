"""
Actor memo models — both forward (`actor_forw_*`) and inverse (the
`actor_discrete_*` and `actor_continuous_*` families used inside observer
`thinks[...]` blocks).

Three model variants per shape: `_full`, `_discomfort_only`, `_base`. Padded
variants (`_padded`, `_padded_rel`) for the no-alternatives-shown observers
also live here. Effort-experiment 2-action actors (`_effort_*`) follow at the
bottom.

Dependency layer 2: imports from `tables.py` (enums, axes) and `utility.py`
(get_utility_*). `observers.py` imports from here.
"""

from memo import memo

from tables import (
    EffortConditions,
    IntimacyLevels,
    ObservedActions,
    PaddedActionSlots,
    RelationshipConditions,
    RewardConditions,
    Scenarios,
    actions,
    actions_3act,
    actions_effort,
)
from utility import (
    get_prior_padded,
    get_prior_padded_rel,
    get_utility_3act_base,
    get_utility_3act_base_disc,
    get_utility_3act_discomfort_only,
    get_utility_3act_discomfort_only_disc,
    get_utility_3act_full,
    get_utility_3act_full_disc,
    get_utility_base,
    get_utility_base_disc,
    get_utility_base_padded,
    get_utility_base_padded_rel,
    get_utility_discomfort_only,
    get_utility_discomfort_only_disc,
    get_utility_discomfort_only_padded,
    get_utility_discomfort_only_padded_rel,
    get_utility_effort_base,
    get_utility_effort_discomfort_only,
    get_utility_effort_full,
    get_utility_full,
    get_utility_full_disc,
    get_utility_full_padded,
    get_utility_full_padded_rel,
)


# ==============================================================================
# Forward-planning actors (canonical 4-action, continuous intimacy)
# ==============================================================================


@memo
def actor_forw_full[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full(
                action,
                scenario_idx,
                intimacy,
                reward_condition,
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_discomfort_only(
                action,
                scenario_idx,
                intimacy,
                reward_condition,
                alpha,
                w_d,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_base[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_base(
                action,
                scenario_idx,
                intimacy,
                reward_condition,
                alpha,
                w_v,
                w_e,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Inverse-planning actors — discrete relationship (observer infers reward)
# ==============================================================================


@memo
def actor_discrete_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full_disc(
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_discomfort_only_disc(
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                alpha,
                w_d,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_base[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_base_disc(
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                alpha,
                w_v,
                w_e,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Inverse-planning actors — continuous intimacy (observer infers intimacy)
# ==============================================================================


@memo
def actor_continuous_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full(
                action,
                scenario_idx,
                relationship,
                reward_condition,
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_discomfort_only(
                action,
                scenario_idx,
                relationship,
                reward_condition,
                alpha,
                w_d,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_base[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_base(
                action,
                scenario_idx,
                relationship,
                reward_condition,
                alpha,
                w_v,
                w_e,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Padded actors (no-alternatives-shown, motivation-keyed action space)
# ==============================================================================


@memo
def actor_continuous_full_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    access_table: ...,
    effort_table: ...,
    v_padded_table: ...,
    prior_table: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded(
            padded_slot, scenario_idx, observed_action, reward_condition, prior_table
        )
        * exp(
            get_utility_full_padded(
                padded_slot,
                scenario_idx,
                observed_action,
                relationship,
                reward_condition,
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
                v_padded_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_continuous_discomfort_only_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded(
            padded_slot, scenario_idx, observed_action, reward_condition, prior_table
        )
        * exp(
            get_utility_discomfort_only_padded(
                padded_slot,
                scenario_idx,
                observed_action,
                relationship,
                reward_condition,
                alpha,
                w_d,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_continuous_base_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    alpha,
    w_v,
    w_e,
    access_table: ...,
    effort_table: ...,
    v_padded_table: ...,
    prior_table: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded(
            padded_slot, scenario_idx, observed_action, reward_condition, prior_table
        )
        * exp(
            get_utility_base_padded(
                padded_slot,
                scenario_idx,
                observed_action,
                relationship,
                reward_condition,
                alpha,
                w_v,
                w_e,
                access_table,
                effort_table,
                v_padded_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


# ==============================================================================
# Padded actors — relationship-keyed (desire-noalt observer)
# ==============================================================================


@memo
def actor_continuous_full_padded_rel[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    access_table: ...,
    effort_table: ...,
    v_padded_table: ...,
    prior_table: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_rel(
            padded_slot,
            scenario_idx,
            observed_action,
            relationship_condition,
            prior_table,
        )
        * exp(
            get_utility_full_padded_rel(
                padded_slot,
                scenario_idx,
                observed_action,
                relationship_condition,
                reward_condition,
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
                v_padded_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_continuous_discomfort_only_padded_rel[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_rel(
            padded_slot,
            scenario_idx,
            observed_action,
            relationship_condition,
            prior_table,
        )
        * exp(
            get_utility_discomfort_only_padded_rel(
                padded_slot,
                scenario_idx,
                observed_action,
                relationship_condition,
                reward_condition,
                alpha,
                w_d,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_continuous_base_padded_rel[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha,
    w_v,
    w_e,
    access_table: ...,
    effort_table: ...,
    v_padded_table: ...,
    prior_table: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_rel(
            padded_slot,
            scenario_idx,
            observed_action,
            relationship_condition,
            prior_table,
        )
        * exp(
            get_utility_base_padded_rel(
                padded_slot,
                scenario_idx,
                observed_action,
                relationship_condition,
                reward_condition,
                alpha,
                w_v,
                w_e,
                access_table,
                effort_table,
                v_padded_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


# ==============================================================================
# Effort-experiment actors (2-action space)
# ==============================================================================


@memo
def actor_forw_effort_full[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_full(
                action,
                scenario_idx,
                intimacy,
                effort_condition,
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_effort_discomfort_only[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_discomfort_only(
                action,
                scenario_idx,
                intimacy,
                effort_condition,
                alpha,
                w_d,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_effort_base[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_base(
                action,
                scenario_idx,
                intimacy,
                effort_condition,
                alpha,
                w_v,
                w_e,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


# Continuous-intimacy effort actors (used inside the effort observers'
# `thinks[...]` blocks; mathematically identical to actor_forw_effort_* but
# bound to `relationship` instead of `intimacy` for parallelism with the
# canonical alt-shown observer memos).


@memo
def actor_continuous_effort_full[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_full(
                action,
                scenario_idx,
                relationship,
                effort_condition,
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_effort_discomfort_only[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_discomfort_only(
                action,
                scenario_idx,
                relationship,
                effort_condition,
                alpha,
                w_d,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_effort_base[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_base(
                action,
                scenario_idx,
                relationship,
                effort_condition,
                alpha,
                w_v,
                w_e,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# 3-action actors (Studies 2, 3a, 3b, 4a, 4b)
# ==============================================================================
# Three families of actors over the 3-action set, paralleling the canonical
# 4-action actors:
#   - `actor_forw_3act_*`: continuous-intimacy forward actor with all three
#     known variables (intimacy, reward, effort). Used inside the inverse
#     observers' `thinks[...]` blocks when intimacy is the inferred latent.
#   - `actor_discrete_3act_*`: discrete-relationship variant for observers
#     that infer reward or effort (intimacy is known at one of the four
#     RelationshipConditions levels).
#   - `actor_continuous_3act_*`: continuous-intimacy variant used by
#     observers that infer intimacy.


@memo
def actor_forw_3act_full[
    action: actions_3act,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_3act,
        wpp=exp(
            get_utility_3act_full(
                action,
                scenario_idx,
                intimacy,
                reward_condition,
                effort_condition,
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_3act_discomfort_only[
    action: actions_3act,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_3act,
        wpp=exp(
            get_utility_3act_discomfort_only(
                action,
                scenario_idx,
                intimacy,
                reward_condition,
                effort_condition,
                alpha,
                w_d,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_3act_base[
    action: actions_3act,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_3act,
        wpp=exp(
            get_utility_3act_base(
                action,
                scenario_idx,
                intimacy,
                reward_condition,
                effort_condition,
                alpha,
                w_v,
                w_e,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]


# Continuous-intimacy 3-action actor — used inside `observer_intimacy_3act_*`
# (Study 2) and inside the joint observers that infer intimacy (Study 4b).


@memo
def actor_continuous_3act_full[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_3act,
        wpp=exp(
            get_utility_3act_full(
                action,
                scenario_idx,
                relationship,
                reward_condition,
                effort_condition,
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_3act_discomfort_only[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_3act,
        wpp=exp(
            get_utility_3act_discomfort_only(
                action,
                scenario_idx,
                relationship,
                reward_condition,
                effort_condition,
                alpha,
                w_d,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_3act_base[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_3act,
        wpp=exp(
            get_utility_3act_base(
                action,
                scenario_idx,
                relationship,
                reward_condition,
                effort_condition,
                alpha,
                w_v,
                w_e,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]


# Discrete-relationship 3-action actor — used inside `observer_effort_3act_*`
# (Study 3a) and `observer_reward_3act_*` (Study 3b), where intimacy is fixed
# at one of the four RelationshipConditions levels.


@memo
def actor_discrete_3act_full[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_3act,
        wpp=exp(
            get_utility_3act_full_disc(
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                effort_condition,
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_3act_discomfort_only[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_3act,
        wpp=exp(
            get_utility_3act_discomfort_only_disc(
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                effort_condition,
                alpha,
                w_d,
                gamma,
                access_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_3act_base[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_3act,
        wpp=exp(
            get_utility_3act_base_disc(
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                effort_condition,
                alpha,
                w_v,
                w_e,
                access_table,
                effort_table,
                v_table,
            )
        ),
    )
    return Pr[actor.action == action]
