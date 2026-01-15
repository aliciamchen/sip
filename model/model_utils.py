from memo import memo
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from enum import IntEnum
from pathlib import Path


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
# LLM-Derived Scenario-Specific Parameters (loaded from CSV)
# ==============================================================================
# Scenarios indexed alphabetically (0-15)
SCENARIO_LABELS = [
    'apples', 'basketball', 'birthday', 'brunch', 'cooking', 'dip',
    'drinks', 'driving', 'fair', 'gala', 'hike', 'oysters',
    'social', 'soup', 'takeout', 'wedding'
]
SCENARIO_TO_IDX = {label: idx for idx, label in enumerate(SCENARIO_LABELS)}
ScenarioIndices = jnp.arange(16)


def _load_lm_params():
    """Load LLM-derived parameters from CSV file.

    Returns:
        LLM_RISK: jnp.array of shape (16, 4) - risk values per scenario/action
        LLM_EFFORT: jnp.array of shape (16, 4) - effort values per scenario/action
        LLM_REWARD: jnp.array of shape (16,) - reward values per scenario
    """
    csv_path = Path(__file__).parent / "lm_scenario_params.csv"
    df = pd.read_csv(csv_path)

    # Pivot to get [scenario, action] arrays, sorted alphabetically by scenario
    risk_pivot = df.pivot(index='scenario_label', columns='action', values='risk').sort_index()
    effort_pivot = df.pivot(index='scenario_label', columns='action', values='effort').sort_index()

    # Reward is per-scenario (same for all actions), take first row per scenario
    reward_df = df.groupby('scenario_label')['reward'].first().sort_index()

    return (
        jnp.array(risk_pivot.values),
        jnp.array(effort_pivot.values),
        jnp.array(reward_df.values)
    )


# Load LLM parameters at module initialization
LLM_RISK, LLM_EFFORT, LLM_REWARD = _load_lm_params()

# Scenario-aware getters for LLM parameters
@jax.jit
def get_risk_lm(action, scenario_idx):
    return LLM_RISK[scenario_idx, action]


@jax.jit
def get_effort_lm(action, scenario_idx):
    return LLM_EFFORT[scenario_idx, action]


@jax.jit
def get_reward_base_lm(scenario_idx):
    return LLM_REWARD[scenario_idx]


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
def get_reward_base(action, reward_condition):
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
def get_reward_from_intimacy(action, reward_condition, intimacy):
    "scale reward based on intimacy"
    intimacy_multiplier = jnp.array([1, 1, 1, 1]) + intimacy * jnp.array([0, 1, 1, 1]) # higher intimacy -> higher reward of sharing the food together in the first place
    base_reward = get_reward_base(action, reward_condition)
    return base_reward * intimacy_multiplier[action]

@jax.jit
def get_reward_from_relationship_condition(action, reward_condition, relationship_condition):
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
    - c(a) = sharing cost
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


@jax.jit
def get_utility_forw_full_lm(action, intimacy, reward_condition, scenario_idx, alpha, w_r, w_d, w_c):
    """Full forward planning utility using LLM-derived scenario-specific parameters.

    Same structure as get_utility_forw_full but uses LLM values for risk, effort, reward.
    """
    # Reward: LLM base reward scaled by (1 + intimacy) for high motivation
    base_reward = jnp.where(reward_condition == RewardConditions.HIGH, get_reward_base_lm(scenario_idx), 0.0)
    action_has_reward = jnp.array([0, 1, 1, 1])[action]
    reward = base_reward * action_has_reward * (1 + intimacy)

    # Discomfort: (1 - intimacy) * LLM risk
    formality = 1 - intimacy
    risk = get_risk_lm(action, scenario_idx)
    discomfort = formality * risk

    # Effort: LLM effort
    effort = get_effort_lm(action, scenario_idx)

    return alpha * (w_r * reward - w_d * discomfort - w_c * effort)


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
    return alpha * -1 * get_discomfort_from_relationship_condition(action, relationship_condition)


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
    return alpha * (
        w_r * get_reward_from_relationship_condition(action, reward_condition, relationship_condition)
        - w_d * get_discomfort_from_relationship_condition(action, relationship_condition)
        - w_c * get_effort(action)
    )


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
    return alpha * (
        w_r * get_reward_from_intimacy(action, reward_condition, intimacy)
        - w_d * get_discomfort_from_intimacy(action, intimacy)
        - w_c * get_effort(action)
    )


# ==============================================================================
# Inverse Planning Actor Models (Discrete)
# ==============================================================================


@memo
def actor_discrete_discomfort_only[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_d, w_c
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
](
    alpha, w_r, w_d, w_c
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
](
    alpha, w_r, w_d, w_c
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
                w_d,
                w_c,
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
](
    alpha, w_r, w_d, w_c
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
](
    alpha, w_r, w_d, w_c
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
](
    alpha, w_r, w_d, w_c
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
                w_d,
                w_c,
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
](
    alpha, w_r, w_d, w_c
):
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
](
    alpha, w_r, w_d, w_c
):
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
](
    alpha, w_d
):
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


# Full model with LLM-derived scenario-specific parameters
@memo
def actor_forw_full_lm[
    action: actions,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](
    scenario_idx, alpha, w_r, w_d, w_c
):
    cast: [actor]
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_forw_full_lm(
                action,
                intimacy,
                reward_condition,
                scenario_idx,
                alpha,
                w_r,
                w_d,
                w_c,
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
](
    alpha, w_r, w_d, w_c
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
            ](alpha, w_r, w_d, w_c),
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
    alpha, w_r, w_d, w_c
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
            ](alpha, w_r, w_d, w_c),
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
    alpha, w_r, w_d, w_c
):
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
        relationship in IntimacyLevels, wpp=E[actor.relationship == relationship]
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
](
    alpha, w_r, w_d, w_c
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
            ](alpha, w_r, w_d, w_c),
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
    alpha, w_r, w_d, w_c
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
            ](alpha, w_r, w_d, w_c),
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
    alpha, w_r, w_d, w_c
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
            ](alpha, w_r, w_d, w_c),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition],
    )
    return Pr[observer.reward_condition == reward_condition]
