"""
Observer memo models — the inverse-planning Bayesian inference layer.

Three target structures:
  - `observer_intimacy_*` — infers the actor's relationship/intimacy from an
    observed action (alt-shown 4-action; padded no-alt motivation-keyed).
  - `observer_reward_*` — infers the actor's reward/motivation from an
    observed action (alt-shown 4-action; padded no-alt relationship-keyed).
  - `observer_effort_intimacy_*` — infers the effort_condition (latent) from
    an observed action under a known intimacy (effort experiment, 2-action).

Three model variants per observer: `_full`, `_discomfort_only`, `_base`.

Dependency layer 3: imports from `tables.py`, `utility.py`, and `actors.py`.
"""

from memo import memo

from actors import (
    actor_continuous_base,
    actor_continuous_base_padded,
    actor_continuous_base_padded_rel,
    actor_continuous_discomfort_only,
    actor_continuous_discomfort_only_padded,
    actor_continuous_discomfort_only_padded_rel,
    actor_continuous_effort_base,
    actor_continuous_effort_discomfort_only,
    actor_continuous_effort_full,
    actor_continuous_full,
    actor_continuous_full_padded,
    actor_continuous_full_padded_rel,
    actor_discrete_base,
    actor_discrete_discomfort_only,
    actor_discrete_full,
    actor_forw_effort_base,
    actor_forw_effort_discomfort_only,
    actor_forw_effort_full,
)
from tables import (
    EffortConditions,
    IntimacyLevels,
    ObservedActions,
    PaddedActionSlots,
    RelationshipConditions,
    RewardConditions,
    Scenarios,
    actions,
    actions_effort,
)


# ==============================================================================
# Observer inferring intimacy (alt-shown action space)
# ==============================================================================


@memo
def observer_intimacy_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_full[
                action, scenario_idx, relationship, reward_condition
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
def observer_intimacy_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_discomfort_only[
                action, scenario_idx, relationship, reward_condition
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
def observer_intimacy_base[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_base[
                action, scenario_idx, relationship, reward_condition
            ](alpha, w_v, w_e, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# ==============================================================================
# Observer inferring reward (alt-shown action space)
# ==============================================================================


@memo
def observer_reward_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_full[
                action, scenario_idx, relationship_condition, reward_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_discomfort_only[
                action, scenario_idx, relationship_condition, reward_condition
            ](alpha, w_d, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_base[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_base[
                action, scenario_idx, relationship_condition, reward_condition
            ](alpha, w_v, w_e, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


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
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
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
                padded_slot, scenario_idx, observed_action, relationship, reward_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_padded_table, prior_table),
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
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ..., prior_table: ...):
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
                padded_slot, scenario_idx, observed_action, relationship, reward_condition
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
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
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
                padded_slot, scenario_idx, observed_action, relationship, reward_condition
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
# Used by `food_inv-desire_intimacy_noalt`. The observer knows scenario,
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
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
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
                padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_padded_table, prior_table),
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
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ..., prior_table: ...):
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
                padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition
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
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
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
                padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition
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
# Observer inferring intimacy — effort experiment (2-action space)
# ==============================================================================


@memo
def observer_intimacy_effort_full[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_continuous_effort_full[
                action, scenario_idx, relationship, effort_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_effort_discomfort_only[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_continuous_effort_discomfort_only[
                action, scenario_idx, relationship, effort_condition
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
def observer_intimacy_effort_base[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_continuous_effort_base[
                action, scenario_idx, relationship, effort_condition
            ](alpha, w_v, w_e, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# ==============================================================================
# Observer inferring effort condition (effort experiment, 2-action space)
# ==============================================================================
# Observed: (action, intimacy, scenario). Latent: effort_condition (low/high).
# Uniform prior over the two effort conditions; α_observer applies the usual
# inverse-planning softmax sharpness to the implied posterior.


@memo
def observer_effort_intimacy_full[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(intimacy)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(intimacy),
        actor : chooses(effort_condition in EffortConditions, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_forw_effort_full[
                action, scenario_idx, intimacy, effort_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        effort_condition in EffortConditions,
        wpp=E[actor.effort_condition == effort_condition] ** alpha_observer,
    )
    return Pr[observer.effort_condition == effort_condition]


@memo
def observer_effort_intimacy_discomfort_only[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(intimacy)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(intimacy),
        actor : chooses(effort_condition in EffortConditions, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_forw_effort_discomfort_only[
                action, scenario_idx, intimacy, effort_condition
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
def observer_effort_intimacy_base[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(intimacy)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(intimacy),
        actor : chooses(effort_condition in EffortConditions, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_forw_effort_base[
                action, scenario_idx, intimacy, effort_condition
            ](alpha, w_v, w_e, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        effort_condition in EffortConditions,
        wpp=E[actor.effort_condition == effort_condition] ** alpha_observer,
    )
    return Pr[observer.effort_condition == effort_condition]
