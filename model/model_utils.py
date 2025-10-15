from memo import memo
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from enum import IntEnum


# Constants

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


# Utility functions


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
def get_reward(action, reward_condition):
    """Get reward for action given reward condition
    Low reward: the characters don't particularly want to eat the food together. So regardless of whether they eat or not eat the food, they get the same reward.
    High reward: the characters want to eat the food together. So the reward is higher if they eat the food together.
    """
    low_reward = jnp.array([1, 1, 1, 1])
    high_reward = jnp.array([0, 2, 2, 2])
    which_reward = jnp.where(
        reward_condition == RewardConditions.LOW, low_reward, high_reward
    )
    return which_reward[action]


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


# Models


class ModelLabels(IntEnum): # will i use these?
    DISCOMFORT_ONLY = 0
    VANILLA_INV_PLAN = 1
    FULL_MODEL = 2


@jax.jit
def get_utility_discomfort_only_discrete(
    action,
    relationship_condition,
    reward_condition,
    alpha,
    w_r,
    w_c,
    w_e,
):
    return alpha * -1 * get_discomfort_from_relationship_condition(action, relationship_condition)


@jax.jit
def get_utility_discomfort_only_continuous(
    action,
    intimacy,
    reward_condition,
    alpha,
    w_r,
    w_c,
    w_e,
):
    return alpha * -1 * get_discomfort_from_intimacy(action, intimacy)


@jax.jit
def get_utility_vanilla_inv_plan(
    action,
    relationship,
    reward_condition,
    alpha,
    w_r,
    w_c,
    w_e,
):
    return alpha * (
        w_r * get_reward(action, reward_condition)
        - w_c * get_risk(action)
        - w_e * get_effort(action)
    )


@jax.jit
def get_utility_full_model_discrete(
    action,
    relationship_condition,
    reward_condition,
    alpha,
    w_r,
    w_c,
    w_e,
):
    return alpha * (
        w_r * get_reward(action, reward_condition)
        - w_c * get_discomfort_from_relationship_condition(action, relationship_condition)
        - w_e * get_effort(action)
    )


@jax.jit
def get_utility_full_model_continuous(
    action,
    intimacy,
    reward_condition,
    alpha,
    w_r,
    w_c,
    w_e,
):
    return alpha * (
        w_r * get_reward(action, reward_condition)
        - w_c * get_discomfort_from_intimacy(action, intimacy)
        - w_e * get_effort(action)
    )


# memo functions


## Actor models picking actions given discrete relationships (4 options)
@memo
def actor_discrete_discomfort_only[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_c, w_e
):
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
                w_c,
                w_e,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_vanilla_inv_plan[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_c, w_e
):
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
                w_c,
                w_e,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_full_model[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_c, w_e
):
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
                w_c,
                w_e,
            )
        ),
    )
    return Pr[actor.action == action]


## Continuous actor models
@memo
def actor_continuous_discomfort_only[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_c, w_e
):
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
                w_c,
                w_e,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_vanilla_inv_plan[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_c, w_e
):
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
                w_c,
                w_e,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_full_model[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_c, w_e
):
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
                w_c,
                w_e,
            )
        ),
    )
    return Pr[actor.action == action]


# Observers that are inferring the relationship given an observed action and reward


@memo
def observer_intimacy_discomfort_only[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_c, w_e
):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_discomfort_only[
                action, relationship, reward_condition
            ](alpha, w_r, w_c, w_e),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels, wpp=E[actor.relationship == relationship]
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_vanilla_inv_plan[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_c, w_e
):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_vanilla_inv_plan[
                action, relationship, reward_condition
            ](alpha, w_r, w_c, w_e),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels, wpp=E[actor.relationship == relationship]
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_full_model[
    action: actions, relationship: IntimacyLevels, reward_condition: RewardConditions
](
    alpha, w_r, w_c, w_e
):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_full_model[action, relationship, reward_condition](
                alpha, w_r, w_c, w_e
            ),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels, wpp=E[actor.relationship == relationship]
    )
    return Pr[observer.relationship == relationship]


# Observers that are inferring the reward given an observed action and relationship


@memo
def observer_reward_discomfort_only[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_c, w_e
):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_discomfort_only[
                action, relationship_condition, reward_condition
            ](alpha, w_r, w_c, w_e),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition],
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_vanilla_inv_plan[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_c, w_e
):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_vanilla_inv_plan[
                action, relationship_condition, reward_condition
            ](alpha, w_r, w_c, w_e),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition],
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_full_model[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_c, w_e
):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_full_model[
                action, relationship_condition, reward_condition
            ](alpha, w_r, w_c, w_e),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition],
    )

    return Pr[observer.reward_condition == reward_condition]
