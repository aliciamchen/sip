"""
Observer memo models — the inverse-planning Bayesian inference layer.

Two target structures:
  - `observer_intimacy_*` — infers the actor's relationship/intimacy from an
    observed action (padded no-alt motivation-keyed; 3-action variants).
  - `observer_reward_*` — infers the actor's reward/motivation from an
    observed action (padded no-alt relationship-keyed; 3-action variants).

Three model variants per observer: `_full`, `_discomfort_only`, `_base`.

Dependency layer 3: imports from `tables.py`, `utility.py`, and `actors.py`.
"""

from memo import memo

from actors import (
    actor_continuous_3act_base,
    actor_continuous_3act_discomfort_only,
    actor_continuous_3act_full,
    actor_continuous_base_padded,
    actor_continuous_base_padded_rel,
    actor_continuous_discomfort_only_padded,
    actor_continuous_discomfort_only_padded_rel,
    actor_continuous_full_padded,
    actor_continuous_full_padded_rel,
    actor_discrete_3act_base,
    actor_discrete_3act_base_padded_desire,
    actor_discrete_3act_discomfort_only,
    actor_discrete_3act_discomfort_only_padded_desire,
    actor_discrete_3act_full,
    actor_discrete_3act_full_padded_desire,
    actor_forw_effort_base,
    actor_forw_effort_discomfort_only,
    actor_forw_effort_full,
)
from tables import (
    EffortConditions,
    IntimacyLevels,
    ObservedActions,
    ObservedActions3Act,
    PaddedActionSlots,
    PaddedActionSlots3Act,
    RelationshipConditions,
    RewardConditions,
    Scenarios,
    actions_3act,
    actions_effort,
)


# ==============================================================================
# Observer inferring intimacy — padded action space (no-alternatives-shown)
# ==============================================================================
# The padded observer infers intimacy from a single observed action, using a
# trial-specific action space that is the union of the observed action (slot 0)
# and the LM-generated counterfactual alternatives for that cell (slots 1..k).
# Remaining slots are null-padded; their tiny prior makes them contribute
# negligible mass to the softmax. Callers evaluate at padded_slot=0 since
# slot 0 always holds the observed canonical action.


@memo
def observer_intimacy_full_padded[
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
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    v_padded_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_full_padded[
                padded_slot,
                scenario_idx,
                observed_action,
                relationship,
                reward_condition,
            ](
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
                v_padded_table,
                prior_table,
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_discomfort_only_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    alpha,
    w_d,
    gamma,
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_discomfort_only_padded[
                padded_slot,
                scenario_idx,
                observed_action,
                relationship,
                reward_condition,
            ](alpha, w_d, gamma, access_table, effort_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_base_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    alpha,
    w_v,
    w_e,
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    v_padded_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_base_padded[
                padded_slot,
                scenario_idx,
                observed_action,
                relationship,
                reward_condition,
            ](alpha, w_v, w_e, access_table, effort_table, v_padded_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# ==============================================================================
# Observer inferring reward — padded action space, relationship-keyed
# ==============================================================================
# Used by `food_inv_desire_intimacy_noalt`. The observer knows scenario,
# observed_action, and relationship_condition; the latent is reward_condition.
# Action space is conditioned on (scenario, observed_action, relationship_condition)
# — i.e. the LM alternatives are elicited per relationship level so the
# counterfactual action set matches what the observer can see.


@memo
def observer_reward_full_padded_rel[
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
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    v_padded_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_full_padded_rel[
                padded_slot,
                scenario_idx,
                observed_action,
                relationship_condition,
                reward_condition,
            ](
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
                v_padded_table,
                prior_table,
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_discomfort_only_padded_rel[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha,
    w_d,
    gamma,
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_discomfort_only_padded_rel[
                padded_slot,
                scenario_idx,
                observed_action,
                relationship_condition,
                reward_condition,
            ](alpha, w_d, gamma, access_table, effort_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_base_padded_rel[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha,
    w_v,
    w_e,
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    v_padded_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_base_padded_rel[
                padded_slot,
                scenario_idx,
                observed_action,
                relationship_condition,
                reward_condition,
            ](alpha, w_v, w_e, access_table, effort_table, v_padded_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


# ==============================================================================
# 3-action single-target observers (Studies 2, 3a, 3b)
# ==============================================================================
# These are the alt-shown observers for the new inverse-planning experiments.
# Padded variants (open-world action space with LM-generated counterfactuals
# in slots 1..k) come in a later step once `lm_alternatives_*_3act.csv` exists.
#
# Study 2  — observer knows (reward, effort), infers intimacy.
# Study 3a — observer knows (reward, intimacy), infers effort.
# Study 3b — observer knows (effort, intimacy), infers reward.


# --- Study 2: infer intimacy -------------------------------------------------


@memo
def observer_intimacy_3act_full[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    v_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions_3act,
            wpp=actor_continuous_3act_full[
                action, scenario_idx, relationship, reward_condition, effort_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_3act_discomfort_only[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions_3act,
            wpp=actor_continuous_3act_discomfort_only[
                action, scenario_idx, relationship, reward_condition, effort_condition
            ](alpha, w_d, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_3act_base[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions_3act,
            wpp=actor_continuous_3act_base[
                action, scenario_idx, relationship, reward_condition, effort_condition
            ](alpha, w_v, w_e, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# --- Study 3a: infer effort -------------------------------------------------


@memo
def observer_effort_3act_full[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    v_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : knows(reward_condition),
        actor : chooses(effort_condition in EffortConditions, wpp=1),
        actor : chooses(
            action in actions_3act,
            wpp=actor_discrete_3act_full[
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                effort_condition,
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        effort_condition in EffortConditions,
        wpp=E[actor.effort_condition == effort_condition] ** alpha_observer,
    )
    return Pr[observer.effort_condition == effort_condition]


@memo
def observer_effort_3act_discomfort_only[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : knows(reward_condition),
        actor : chooses(effort_condition in EffortConditions, wpp=1),
        actor : chooses(
            action in actions_3act,
            wpp=actor_discrete_3act_discomfort_only[
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                effort_condition,
            ](alpha, w_d, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        effort_condition in EffortConditions,
        wpp=E[actor.effort_condition == effort_condition] ** alpha_observer,
    )
    return Pr[observer.effort_condition == effort_condition]


@memo
def observer_effort_3act_base[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : knows(reward_condition),
        actor : chooses(effort_condition in EffortConditions, wpp=1),
        actor : chooses(
            action in actions_3act,
            wpp=actor_discrete_3act_base[
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                effort_condition,
            ](alpha, w_v, w_e, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        effort_condition in EffortConditions,
        wpp=E[actor.effort_condition == effort_condition] ** alpha_observer,
    )
    return Pr[observer.effort_condition == effort_condition]


# --- Study 3b: infer reward (desire), LM-generated alternatives -------------
# Observer knows scenario, observed_action, effort_condition,
# relationship_condition; latent is reward_condition. Action space is per
# (scenario, observed_action, effort_condition, relationship_condition) — the
# LM generates plausible alternatives per cell, padded to MAX_ACTIONS_3ACT with
# the observed canonical action in slot 0.


@memo
def observer_reward_3act_full[
    padded_slot: PaddedActionSlots3Act,
    scenario_idx: Scenarios,
    observed_action: ObservedActions3Act,
    effort_condition: EffortConditions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    v_padded_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(effort_condition)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(effort_condition),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots3Act,
            wpp=actor_discrete_3act_full_padded_desire[
                padded_slot,
                scenario_idx,
                observed_action,
                effort_condition,
                relationship_condition,
                reward_condition,
            ](
                alpha,
                w_v,
                w_d,
                w_e,
                gamma,
                access_table,
                effort_table,
                v_padded_table,
                prior_table,
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_3act_discomfort_only[
    padded_slot: PaddedActionSlots3Act,
    scenario_idx: Scenarios,
    observed_action: ObservedActions3Act,
    effort_condition: EffortConditions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha,
    w_d,
    gamma,
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(effort_condition)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(effort_condition),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots3Act,
            wpp=actor_discrete_3act_discomfort_only_padded_desire[
                padded_slot,
                scenario_idx,
                observed_action,
                effort_condition,
                relationship_condition,
                reward_condition,
            ](alpha, w_d, gamma, access_table, effort_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_3act_base[
    padded_slot: PaddedActionSlots3Act,
    scenario_idx: Scenarios,
    observed_action: ObservedActions3Act,
    effort_condition: EffortConditions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha,
    w_v,
    w_e,
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    v_padded_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(effort_condition)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(effort_condition),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots3Act,
            wpp=actor_discrete_3act_base_padded_desire[
                padded_slot,
                scenario_idx,
                observed_action,
                effort_condition,
                relationship_condition,
                reward_condition,
            ](
                alpha,
                w_v,
                w_e,
                access_table,
                effort_table,
                v_padded_table,
                prior_table,
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


# ==============================================================================
# 3-action joint-target observers (Studies 4a, 4b)
# ==============================================================================
# Compute a joint posterior over two latent variables simultaneously, using
# memo's multi-choice syntax: `chooses(x in X, y in Y, wpp=...)` for the
# joint draw, and `Pr[..., ...]` for the joint return.
#
# Study 4a — observer knows intimacy, jointly infers (reward, effort).
# Study 4b — observer knows effort, jointly infers (reward, intimacy).
#
# Downstream code marginalizes the returned joint over each axis to produce
# the per-slider predictions matching the two ratings participants give.


# --- Study 4a: joint over (reward, effort) given intimacy --------------------


@memo
def observer_joint_de_3act_full[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    v_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(
            reward_condition in RewardConditions,
            effort_condition in EffortConditions,
            wpp=1,
        ),
        actor : chooses(
            action in actions_3act,
            wpp=actor_discrete_3act_full[
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                effort_condition,
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        effort_condition in EffortConditions,
        wpp=E[
            (actor.reward_condition == reward_condition)
            * (actor.effort_condition == effort_condition)
        ]
        ** alpha_observer,
    )
    return Pr[
        observer.reward_condition == reward_condition,
        observer.effort_condition == effort_condition,
    ]


@memo
def observer_joint_de_3act_discomfort_only[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(
            reward_condition in RewardConditions,
            effort_condition in EffortConditions,
            wpp=1,
        ),
        actor : chooses(
            action in actions_3act,
            wpp=actor_discrete_3act_discomfort_only[
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                effort_condition,
            ](alpha, w_d, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        effort_condition in EffortConditions,
        wpp=E[
            (actor.reward_condition == reward_condition)
            * (actor.effort_condition == effort_condition)
        ]
        ** alpha_observer,
    )
    return Pr[
        observer.reward_condition == reward_condition,
        observer.effort_condition == effort_condition,
    ]


@memo
def observer_joint_de_3act_base[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(
            reward_condition in RewardConditions,
            effort_condition in EffortConditions,
            wpp=1,
        ),
        actor : chooses(
            action in actions_3act,
            wpp=actor_discrete_3act_base[
                action,
                scenario_idx,
                relationship_condition,
                reward_condition,
                effort_condition,
            ](alpha, w_v, w_e, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        effort_condition in EffortConditions,
        wpp=E[
            (actor.reward_condition == reward_condition)
            * (actor.effort_condition == effort_condition)
        ]
        ** alpha_observer,
    )
    return Pr[
        observer.reward_condition == reward_condition,
        observer.effort_condition == effort_condition,
    ]


# --- Study 4b: joint over (reward, intimacy) given effort -------------------


@memo
def observer_joint_di_3act_full[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    access_table: ...,
    effort_table: ...,
    v_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(effort_condition),
        actor : chooses(
            reward_condition in RewardConditions,
            relationship in IntimacyLevels,
            wpp=1,
        ),
        actor : chooses(
            action in actions_3act,
            wpp=actor_continuous_3act_full[
                action, scenario_idx, relationship, reward_condition, effort_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        relationship in IntimacyLevels,
        wpp=E[
            (actor.reward_condition == reward_condition)
            * (actor.relationship == relationship)
        ]
        ** alpha_observer,
    )
    return Pr[
        observer.reward_condition == reward_condition,
        observer.relationship == relationship,
    ]


@memo
def observer_joint_di_3act_discomfort_only[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(effort_condition),
        actor : chooses(
            reward_condition in RewardConditions,
            relationship in IntimacyLevels,
            wpp=1,
        ),
        actor : chooses(
            action in actions_3act,
            wpp=actor_continuous_3act_discomfort_only[
                action, scenario_idx, relationship, reward_condition, effort_condition
            ](alpha, w_d, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        relationship in IntimacyLevels,
        wpp=E[
            (actor.reward_condition == reward_condition)
            * (actor.relationship == relationship)
        ]
        ** alpha_observer,
    )
    return Pr[
        observer.reward_condition == reward_condition,
        observer.relationship == relationship,
    ]


@memo
def observer_joint_di_3act_base[
    action: actions_3act,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(effort_condition),
        actor : chooses(
            reward_condition in RewardConditions,
            relationship in IntimacyLevels,
            wpp=1,
        ),
        actor : chooses(
            action in actions_3act,
            wpp=actor_continuous_3act_base[
                action, scenario_idx, relationship, reward_condition, effort_condition
            ](alpha, w_v, w_e, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        relationship in IntimacyLevels,
        wpp=E[
            (actor.reward_condition == reward_condition)
            * (actor.relationship == relationship)
        ]
        ** alpha_observer,
    )
    return Pr[
        observer.reward_condition == reward_condition,
        observer.relationship == relationship,
    ]
