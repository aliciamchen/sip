from enum import IntEnum

import jax
import jax.numpy as jnp
from memo import memo

# ==============================================================================
# Constants
# ==============================================================================

actions = jnp.array([0, 1, 2, 3])
IntimacyLevels = jnp.arange(0, 1.01, 0.01)


class RewardConditions(IntEnum):
    LOW = 0
    HIGH = 1


class RelationshipConditions(IntEnum):
    ZERO = 0
    FIFTY = 1
    SEVENTY_FIVE = 2
    ONE_HUNDRED = 3


# ==============================================================================
# Scenario Labels
# ==============================================================================
# Scenarios indexed alphabetically (0-15)
SCENARIO_LABELS = [
    "apples",
    "basketball",
    "birthday",
    "brunch",
    "cooking",
    "dip",
    "drinks",
    "driving",
    "fair",
    "gala",
    "hike",
    "oysters",
    "social",
    "soup",
    "takeout",
    "wedding",
]
SCENARIO_TO_IDX = {label: idx for idx, label in enumerate(SCENARIO_LABELS)}


# ==============================================================================
# Basic Utility Functions
# ==============================================================================


@jax.jit
def get_intimacy(relationship_condition):
    """For the cases where the relationship condition is known, get the intimacy level from the relationship condition"""
    return jnp.array([0, 0.5, 0.75, 1])[relationship_condition]


@jax.jit
def get_risk(action):
    """Get risk level for each action (actions 0 and 1 have no saliva transfer)"""
    return jnp.array([0, 0, 1, 2])[action]


@jax.jit
def get_effort(action):
    """Get effort level for each action (food sharing is more effortful than no food sharing)"""
    return jnp.array([0, 1, 1, 1])[action]


@jax.jit
def get_access(action):
    """Graded bodily/spatial exposure of each action to the other person.

    access(a) = [0, 0.3, 1, 2][a]: action 0 opens nothing, action 1 (sharing with
    no saliva) opens a small amount of personal/spatial access, actions 2-3
    (double-dip, same-item) open progressively more bodily access via saliva
    transfer. The gap between 1 and 2 is larger than between 0 and 1, since
    saliva transfer is a qualitatively bigger step in exposure than simply
    eating together.
    """
    return jnp.array([0.0, 0.3, 1.0, 2.0])[action]


@jax.jit
def get_reward_base(action, reward_condition):
    """Get reward for action given reward condition.

    Aligned with forward planning reward (get_reward_forw) per preregistration:
    - Low reward: 0 for all actions (characters don't want to eat together)
    - High reward: 0 for action 0, 1 for actions 1-3 (base reward r_0=1)
    """
    low_reward = jnp.array([0, 0, 0, 0])
    high_reward = jnp.array([0, 1, 1, 1])
    which_reward = jnp.where(
        reward_condition == RewardConditions.LOW, low_reward, high_reward
    )
    return which_reward[action]


@jax.jit
def get_reward_from_intimacy(action, reward_condition, intimacy):
    """Scale reward based on intimacy.

    Aligned with get_reward_forw() per preregistration (r_0=1):
    - LOW motivation: 0 for all actions
    - HIGH motivation, action 0: 0
    - HIGH motivation, actions 1-3: 1 * (1 + intimacy)

    Higher intimacy increases the reward of sharing food together.
    """
    intimacy_multiplier = jnp.array([1, 1, 1, 1]) + intimacy * jnp.array([0, 1, 1, 1])
    base_reward = get_reward_base(action, reward_condition)
    return base_reward * intimacy_multiplier[action]


@jax.jit
def get_reward_from_relationship_condition(
    action, reward_condition, relationship_condition
):
    "scale reward based on relationship condition"
    intimacy = get_intimacy(relationship_condition)
    return get_reward_from_intimacy(action, reward_condition, intimacy)


@jax.jit
def get_discomfort_from_intimacy(action, intimacy):
    """Get discomfort for an action given an intimacy level in [0, 1]."""
    formality = 1 - intimacy
    risk = get_risk(action)
    return formality * risk


@jax.jit
def get_discomfort_from_relationship_condition(action, relationship_condition):
    """Get discomfort for an action given a discrete relationship condition."""
    intimacy = get_intimacy(relationship_condition)
    formality = 1 - intimacy
    risk = get_risk(action)
    return formality * risk


# ==============================================================================
# Forward Planning Utility Functions
# ==============================================================================


@jax.jit
def get_sharing_cost(action):
    """Get sharing/coordination cost for each action.
    Action 0 (not eating) has no cost; actions 1-3 (sharing) have cost 1.
    """
    return jnp.array([0, 1, 1, 1])[action]


@jax.jit
def get_reward_forw(action, reward_condition, intimacy):
    """Get reward for forward planning model.

    Action 0 always has 0 reward.
    Actions 1-3: base reward (1 for high motivation, 0 for low) scaled by (1 + intimacy).
    This captures that eating together is more rewarding when:
    - motivation is high (they want to eat the food)
    - intimacy is high (closer relationship makes sharing more rewarding)
    """
    # Base reward: 1 for high motivation, 0 for low
    base = jnp.where(reward_condition == RewardConditions.HIGH, 1.0, 0.0)
    # Only actions 1-3 get reward (action 0 = not eating = no reward)
    action_has_reward = jnp.array([0, 1, 1, 1])[action]
    # Scale by intimacy: higher intimacy -> higher reward of sharing
    return base * action_has_reward * (1 + intimacy)


@jax.jit
def get_reward_forw_vanilla(action, reward_condition):
    """Get reward for vanilla model (no intimacy scaling).

    Action 0 always has 0 reward.
    Actions 1-3: base reward (1 for high motivation, 0 for low).
    """
    base = jnp.where(reward_condition == RewardConditions.HIGH, 1.0, 0.0)
    action_has_reward = jnp.array([0, 1, 1, 1])[action]
    return base * action_has_reward


@jax.jit
def get_utility_forw_full(action, intimacy, reward_condition, alpha, w_r, w_d, w_c):
    """Full forward planning utility with intimacy-scaled reward and discomfort.

    U = w_r * r(a|s,I) - w_d * d(a|I) - w_c * c(a)
    where:
    - r(a|s,I) = reward scaled by motivation s and intimacy I
    - d(a|I) = (1-I) * risk(a) = discomfort from saliva transfer
    - c(a) = sharing cost (NOT scaled by intimacy per pre-registration)
    """
    reward = get_reward_forw(action, reward_condition, intimacy)
    discomfort = get_discomfort_from_intimacy(action, intimacy)
    sharing_cost = get_sharing_cost(action)
    return alpha * (w_r * reward - w_d * discomfort - w_c * sharing_cost)


@jax.jit
def get_utility_forw_vanilla(action, intimacy, reward_condition, alpha, w_r, w_d, w_c):
    """Vanilla forward planning utility (no intimacy scaling).

    U = w_r * r(a|s) - w_d * risk(a) - w_c * c(a)
    Reward and discomfort are NOT scaled by intimacy.
    """
    reward = get_reward_forw_vanilla(action, reward_condition)
    risk = get_risk(action)  # raw risk, not scaled by intimacy
    sharing_cost = get_sharing_cost(action)
    return alpha * (w_r * reward - w_d * risk - w_c * sharing_cost)


@jax.jit
def get_utility_forw_discomfort_only(action, intimacy, alpha, w_d):
    """Discomfort-only forward planning utility.

    U = -w_d * d(a|I)
    Only considers how intimacy mitigates discomfort from risky actions.
    """
    discomfort = get_discomfort_from_intimacy(action, intimacy)
    return alpha * (-w_d * discomfort)


# ==============================================================================
# Inverse Planning Utility Functions
# ==============================================================================


@jax.jit
def get_utility_discomfort_only_discrete(
    action,
    relationship_condition,
    reward_condition,
    alpha,
    w_r,
    w_d,
    w_c,
):
    return (
        alpha
        * -1
        * get_discomfort_from_relationship_condition(action, relationship_condition)
    )


@jax.jit
def get_utility_discomfort_only_continuous(
    action,
    intimacy,
    reward_condition,
    alpha,
    w_r,
    w_d,
    w_c,
):
    return alpha * -1 * get_discomfort_from_intimacy(action, intimacy)


@jax.jit
def get_utility_vanilla_inv_plan(
    action,
    relationship,
    reward_condition,
    alpha,
    w_r,
    w_d,
    w_c,
):
    return alpha * (
        w_r * get_reward_base(action, reward_condition)
        - w_d * get_risk(action)
        - w_c * get_effort(action)
    )


@jax.jit
def get_utility_full_model_discrete(
    action,
    relationship_condition,
    reward_condition,
    alpha,
    w_r,
    w_d,
    w_c,
):
    """Pre-registered full model: effort NOT scaled by intimacy."""
    return alpha * (
        w_r
        * get_reward_from_relationship_condition(
            action, reward_condition, relationship_condition
        )
        - w_d
        * get_discomfort_from_relationship_condition(action, relationship_condition)
        - w_c * get_effort(action)
    )


@jax.jit
def get_utility_full_model_discrete_modified(
    action,
    relationship_condition,
    reward_condition,
    alpha,
    w_r,
    w_d,
    w_c,
    beta,
):
    """Modified full model: reward scaled by (1 + beta * intimacy).

    beta controls how much observers think actors' reward scales with intimacy.
    beta=0: no intimacy scaling on reward (vanilla-like)
    beta=1: pre-registered model (full intimacy scaling)
    """
    intimacy = get_intimacy(relationship_condition)
    risk = get_risk(action)

    # Reward with discounted intimacy scaling
    base = jnp.where(reward_condition == RewardConditions.HIGH, 1.0, 0.0)
    action_has_reward = jnp.array([0, 1, 1, 1])[action]
    reward = base * action_has_reward * (1 + beta * intimacy)

    # Discomfort unchanged (standard model)
    discomfort = (1 - intimacy) * risk

    return alpha * (w_r * reward - w_d * discomfort - w_c * get_effort(action))


@jax.jit
def get_utility_full_model_continuous(
    action,
    intimacy,
    reward_condition,
    alpha,
    w_r,
    w_d,
    w_c,
):
    """Pre-registered full model: effort NOT scaled by intimacy."""
    return alpha * (
        w_r * get_reward_from_intimacy(action, reward_condition, intimacy)
        - w_d * get_discomfort_from_intimacy(action, intimacy)
        - w_c * get_effort(action)
    )


@jax.jit
def get_utility_full_model_continuous_modified(
    action,
    intimacy,
    reward_condition,
    alpha,
    w_r,
    w_d,
    w_c,
    beta,
):
    """Modified full model: reward scaled by (1 + beta * intimacy).

    beta controls how much observers think actors' reward scales with intimacy.
    beta=0: no intimacy scaling on reward (vanilla-like)
    beta=1: pre-registered model (full intimacy scaling)
    """
    risk = get_risk(action)

    # Reward with discounted intimacy scaling
    base = jnp.where(reward_condition == RewardConditions.HIGH, 1.0, 0.0)
    action_has_reward = jnp.array([0, 1, 1, 1])[action]
    reward = base * action_has_reward * (1 + beta * intimacy)

    # Discomfort unchanged (standard model)
    discomfort = (1 - intimacy) * risk

    return alpha * (w_r * reward - w_d * discomfort - w_c * get_effort(action))


# ==============================================================================
# Access-Based Utility Functions
# ==============================================================================
# Canonical reformulation (access_full):
#   U(a|s, I) = w_v * V(a|s)
#             + w_r * access(a) * I
#             - w_d * access(a) * (1 - I)
#             - w_e * effort(a)
# where V(a|s) is the food reward (not scaled by intimacy) and access(a) is
# graded bodily/spatial exposure to the other person.
#
# Three variants are fit to data:
#   - access_full  : full utility above (the main model)
#   - access_only  : only the two access terms (no food reward, no effort cost)
#   - no_access    : only food reward and effort cost (base model / baseline)
#
# Each variant has one continuous-intimacy utility (shared by forward-planning
# and continuous-observer memo models) and one discrete wrapper keyed on
# RelationshipConditions (used by observer_reward_* memo models).


@jax.jit
def get_utility_access_full(
    action, intimacy, reward_condition, alpha, w_v, w_r, w_d, w_e
):
    V = get_reward_base(action, reward_condition)
    access = get_access(action)
    effort = get_effort(action)
    return alpha * (
        w_v * V
        + w_r * access * intimacy
        - w_d * access * (1 - intimacy)
        - w_e * effort
    )


@jax.jit
def get_utility_access_full_disc(
    action, relationship_condition, reward_condition, alpha, w_v, w_r, w_d, w_e
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_access_full(
        action, intimacy, reward_condition, alpha, w_v, w_r, w_d, w_e
    )


@jax.jit
def get_utility_access_only(
    action, intimacy, reward_condition, alpha, w_r, w_d
):
    """Access-only utility: both positive and negative access terms, no food reward, no effort.

    U = alpha * (w_r * access(a) * I  -  w_d * access(a) * (1 - I))

    Tests whether the access terms alone — stripped of the food-reward motive
    and the physical-effort cost — can account for behavior.
    """
    access = get_access(action)
    return alpha * (w_r * access * intimacy - w_d * access * (1 - intimacy))


@jax.jit
def get_utility_access_only_disc(
    action, relationship_condition, reward_condition, alpha, w_r, w_d
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_access_only(
        action, intimacy, reward_condition, alpha, w_r, w_d
    )


@jax.jit
def get_utility_no_access(
    action, intimacy, reward_condition, alpha, w_v, w_e
):
    V = get_reward_base(action, reward_condition)
    effort = get_effort(action)
    return alpha * (w_v * V - w_e * effort)


@jax.jit
def get_utility_no_access_disc(
    action, relationship_condition, reward_condition, alpha, w_v, w_e
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_no_access(
        action, intimacy, reward_condition, alpha, w_v, w_e
    )


# ==============================================================================
# Inverse Planning Actor Models (Discrete)
# ==============================================================================


@memo
def actor_discrete_discomfort_only[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c):
    cast: [actor]
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_discomfort_only_discrete(
                action,
                relationship_condition,
                reward_condition,
                alpha,
                w_r,
                w_d,
                w_c,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_vanilla_inv_plan[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c):
    cast: [actor]
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_vanilla_inv_plan(
                action,
                relationship_condition,
                reward_condition,
                alpha,
                w_r,
                w_d,
                w_c,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_full_model[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c):
    cast: [actor]
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full_model_discrete(
                action,
                relationship_condition,
                reward_condition,
                alpha,
                w_r,
                w_d,
                w_c,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_full_model_modified[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c, beta):
    cast: [actor]
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full_model_discrete_modified(
                action,
                relationship_condition,
                reward_condition,
                alpha,
                w_r,
                w_d,
                w_c,
                beta,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Inverse Planning Actor Models (Continuous)
# ==============================================================================


@memo
def actor_continuous_discomfort_only[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c):
    cast: [actor]
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_discomfort_only_continuous(
                action,
                relationship,
                reward_condition,
                alpha,
                w_r,
                w_d,
                w_c,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_vanilla_inv_plan[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c):
    cast: [actor]
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_vanilla_inv_plan(
                action,
                relationship,
                reward_condition,
                alpha,
                w_r,
                w_d,
                w_c,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_full_model[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c):
    cast: [actor]
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full_model_continuous(
                action,
                relationship,
                reward_condition,
                alpha,
                w_r,
                w_d,
                w_c,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_full_model_modified[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c, beta):
    cast: [actor]
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full_model_continuous_modified(
                action,
                relationship,
                reward_condition,
                alpha,
                w_r,
                w_d,
                w_c,
                beta,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Forward Planning Actor Models
# ==============================================================================
@memo
def actor_forw_full[
    action: actions,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c):
    cast: [actor]
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_forw_full(
                action,
                intimacy,
                reward_condition,
                alpha,
                w_r,
                w_d,
                w_c,
            )
        ),
    )
    return Pr[actor.action == action]


# Vanilla model: no intimacy scaling
@memo
def actor_forw_vanilla[
    action: actions,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c):
    cast: [actor]
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_forw_vanilla(
                action,
                intimacy,
                reward_condition,
                alpha,
                w_r,
                w_d,
                w_c,
            )
        ),
    )
    return Pr[actor.action == action]


# Discomfort-only model: only considers discomfort
@memo
def actor_forw_discomfort_only[
    action: actions,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d):
    cast: [actor]
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_forw_discomfort_only(
                action,
                intimacy,
                alpha,
                w_d,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Observer Models - Inferring Intimacy
# ==============================================================================


@memo
def observer_intimacy_discomfort_only[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c, alpha_observer):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_discomfort_only[
                action, relationship, reward_condition
            ](alpha, w_r, w_d, w_c),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_vanilla_inv_plan[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c, alpha_observer):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_vanilla_inv_plan[
                action, relationship, reward_condition
            ](alpha, w_r, w_d, w_c),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_full_model[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c, alpha_observer):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_full_model[action, relationship, reward_condition](
                alpha, w_r, w_d, w_c
            ),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_full_model_modified[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c, alpha_observer, beta):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_full_model_modified[
                action, relationship, reward_condition
            ](alpha, w_r, w_d, w_c, beta),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# ==============================================================================
# Observer Models - Inferring Reward
# ==============================================================================


@memo
def observer_reward_discomfort_only[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c, alpha_observer):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_discomfort_only[
                action, relationship_condition, reward_condition
            ](alpha, w_r, w_d, w_c),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_vanilla_inv_plan[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c, alpha_observer):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_vanilla_inv_plan[
                action, relationship_condition, reward_condition
            ](alpha, w_r, w_d, w_c),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_full_model[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c, alpha_observer):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_full_model[
                action, relationship_condition, reward_condition
            ](alpha, w_r, w_d, w_c),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_full_model_modified[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, w_c, alpha_observer, beta):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_full_model_modified[
                action, relationship_condition, reward_condition
            ](alpha, w_r, w_d, w_c, beta),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


# ==============================================================================
# Access-Based Models — Forward Planning (Actor, Exp 1)
# ==============================================================================


@memo
def actor_forw_access_full[
    action: actions,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_r, w_d, w_e):
    cast: [actor]
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_full(
                action, intimacy, reward_condition, alpha, w_v, w_r, w_d, w_e
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_access_only[
    action: actions,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d):
    cast: [actor]
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_only(
                action, intimacy, reward_condition, alpha, w_r, w_d
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_no_access[
    action: actions,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e):
    cast: [actor]
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_no_access(
                action, intimacy, reward_condition, alpha, w_v, w_e
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Access-Based Models — Inverse Planning Actor (Discrete)
# ==============================================================================


@memo
def actor_discrete_access_full[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_r, w_d, w_e):
    cast: [actor]
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_full_disc(
                action,
                relationship_condition,
                reward_condition,
                alpha,
                w_v,
                w_r,
                w_d,
                w_e,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_access_only[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d):
    cast: [actor]
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_only_disc(
                action,
                relationship_condition,
                reward_condition,
                alpha,
                w_r,
                w_d,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_no_access[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_e):
    cast: [actor]
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_no_access_disc(
                action,
                relationship_condition,
                reward_condition,
                alpha,
                w_v,
                w_e,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Access-Based Models — Inverse Planning Actor (Continuous)
# ==============================================================================


@memo
def actor_continuous_access_full[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_r, w_d, w_e):
    cast: [actor]
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_full(
                action, relationship, reward_condition, alpha, w_v, w_r, w_d, w_e
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_access_only[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d):
    cast: [actor]
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_only(
                action, relationship, reward_condition, alpha, w_r, w_d
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_no_access[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e):
    cast: [actor]
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_no_access(
                action, relationship, reward_condition, alpha, w_v, w_e
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Access-Based Models — Observer Inferring Intimacy (Exp 2a)
# ==============================================================================


@memo
def observer_intimacy_access_full[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_r, w_d, w_e, alpha_observer):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_access_full[
                action, relationship, reward_condition
            ](alpha, w_v, w_r, w_d, w_e),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_access_only[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, alpha_observer):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_access_only[
                action, relationship, reward_condition
            ](alpha, w_r, w_d),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_no_access[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, alpha_observer):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_no_access[
                action, relationship, reward_condition
            ](alpha, w_v, w_e),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# ==============================================================================
# Access-Based Models — Observer Inferring Reward (Exp 2b)
# ==============================================================================


@memo
def observer_reward_access_full[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_r, w_d, w_e, alpha_observer):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_access_full[
                action, relationship_condition, reward_condition
            ](alpha, w_v, w_r, w_d, w_e),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_access_only[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, alpha_observer):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_access_only[
                action, relationship_condition, reward_condition
            ](alpha, w_r, w_d),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_no_access[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, alpha_observer):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_no_access[
                action, relationship_condition, reward_condition
            ](alpha, w_v, w_e),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]
