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


# ==============================================================================
# Empirical Priors (loaded from CSV)
# ==============================================================================

def _load_empirical_priors():
    """Load empirical priors from CSV files as JAX arrays.

    Returns:
        EMPIRICAL_PRIOR_INTIMACY: jnp.array of shape (2, 101) - [motivation, intimacy_idx]
            motivation: 0=low, 1=high
            intimacy_idx: 0-100 corresponding to intimacy values 0.00-1.00
        EMPIRICAL_PRIOR_REWARD_BINARY: jnp.array of shape (4, 2) - [relationship_condition, reward_condition]
            relationship_condition: 0-3 for 0%, 50%, 75%, 100% intimacy
            reward_condition: 0=low, 1=high
    """
    # Load intimacy priors
    intimacy_csv_path = Path(__file__).parent / "empirical_priors_intimacy.csv"
    if intimacy_csv_path.exists():
        df_int = pd.read_csv(intimacy_csv_path)
        # Reshape to (2, 101): [motivation, intimacy_idx]
        # motivation: low=0, high=1
        low_prior = df_int[df_int['motivation'] == 'low'].sort_values('intimacy_100')['density'].values
        high_prior = df_int[df_int['motivation'] == 'high'].sort_values('intimacy_100')['density'].values
        empirical_prior_intimacy = jnp.array([low_prior, high_prior])
    else:
        # Fallback to uniform if file doesn't exist
        empirical_prior_intimacy = jnp.ones((2, 101)) / 101

    # Load reward priors and convert to binary
    reward_csv_path = Path(__file__).parent / "empirical_priors_reward.csv"
    if reward_csv_path.exists():
        df_rew = pd.read_csv(reward_csv_path)
        # For each intimacy condition, compute P(high) = mean / 100
        # intimacy_condition: 0, 50, 75, 100 -> relationship_condition: 0, 1, 2, 3
        intimacy_to_rel = {0: 0, 50: 1, 75: 2, 100: 3}
        reward_grid = np.arange(101)
        binary_priors = np.zeros((4, 2))
        for int_cond in [0, 50, 75, 100]:
            rel_cond = intimacy_to_rel[int_cond]
            subset = df_rew[df_rew['intimacy_condition'] == int_cond].sort_values('reward_value')
            density = subset['density'].values
            prior_mean = (reward_grid * density).sum()
            p_high = prior_mean / 100.0
            p_low = 1 - p_high
            binary_priors[rel_cond, 0] = p_low
            binary_priors[rel_cond, 1] = p_high
        empirical_prior_reward_binary = jnp.array(binary_priors)
    else:
        # Fallback to uniform if file doesn't exist
        empirical_prior_reward_binary = jnp.ones((4, 2)) / 2

    return empirical_prior_intimacy, empirical_prior_reward_binary


# Load empirical priors at module initialization
EMPIRICAL_PRIOR_INTIMACY, EMPIRICAL_PRIOR_REWARD_BINARY = _load_empirical_priors()


@jax.jit
def get_empirical_prior_intimacy(relationship_value, reward_condition):
    """Get empirical prior density for intimacy given motivation (reward_condition).

    Args:
        relationship_value: float in [0, 1] representing intimacy level
        reward_condition: 0 (low motivation) or 1 (high motivation)
    Returns:
        Prior density at this intimacy level for the given motivation condition
    """
    idx = jnp.round(relationship_value * 100).astype(jnp.int32)
    idx = jnp.clip(idx, 0, 100)  # Ensure index is in bounds
    return EMPIRICAL_PRIOR_INTIMACY[reward_condition, idx]


@jax.jit
def get_empirical_prior_reward(reward_condition, relationship_condition):
    """Get empirical prior probability for reward given intimacy condition.

    Args:
        reward_condition: 0 (low) or 1 (high)
        relationship_condition: 0, 1, 2, 3 (for 0%, 50%, 75%, 100% intimacy)
    Returns:
        Prior probability for this reward level given the intimacy condition
    """
    return EMPIRICAL_PRIOR_REWARD_BINARY[relationship_condition, reward_condition]


# Mixed prior getters (blend between empirical and uniform priors)
UNIFORM_PRIOR_INTIMACY = 1.0 / 101.0  # Uniform over 101 intimacy levels


@jax.jit
def get_mixed_prior_intimacy(relationship_value, reward_condition, prior_weight):
    """Get mixed prior for intimacy: blend of empirical and uniform.

    prior = prior_weight * empirical + (1 - prior_weight) * uniform

    Args:
        relationship_value: float in [0, 1] representing intimacy level
        reward_condition: 0 (low motivation) or 1 (high motivation)
        prior_weight: weight on empirical prior (0 = uniform, 1 = pure empirical)
    Returns:
        Mixed prior density at this intimacy level
    """
    empirical = get_empirical_prior_intimacy(relationship_value, reward_condition)
    return prior_weight * empirical + (1 - prior_weight) * UNIFORM_PRIOR_INTIMACY


@jax.jit
def get_mixed_prior_reward(reward_condition, relationship_condition, prior_weight):
    """Get mixed prior for reward: blend of empirical and uniform.

    prior = prior_weight * empirical + (1 - prior_weight) * uniform

    Args:
        reward_condition: 0 (low) or 1 (high)
        relationship_condition: 0, 1, 2, 3 (for 0%, 50%, 75%, 100% intimacy)
        prior_weight: weight on empirical prior (0 = uniform, 1 = pure empirical)
    Returns:
        Mixed prior probability for this reward level
    """
    empirical = get_empirical_prior_reward(reward_condition, relationship_condition)
    uniform = 0.5  # Uniform over 2 reward levels
    return prior_weight * empirical + (1 - prior_weight) * uniform


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


@jax.jit
def get_utility_forw_vanilla_lm(action, intimacy, reward_condition, scenario_idx, alpha, w_r, w_d, w_c):
    """Vanilla forward planning utility using LLM-derived scenario-specific parameters.

    No intimacy scaling - reward and risk are independent of relationship.
    Uses LLM values for risk, effort, and reward.
    """
    # Reward: LLM base reward (no intimacy scaling)
    base_reward = jnp.where(reward_condition == RewardConditions.HIGH, get_reward_base_lm(scenario_idx), 0.0)
    action_has_reward = jnp.array([0, 1, 1, 1])[action]
    reward = base_reward * action_has_reward

    # Risk: LLM risk (no intimacy scaling)
    risk = get_risk_lm(action, scenario_idx)

    # Effort: LLM effort
    effort = get_effort_lm(action, scenario_idx)

    return alpha * (w_r * reward - w_d * risk - w_c * effort)


@jax.jit
def get_utility_forw_discomfort_only_lm(action, intimacy, scenario_idx, alpha, w_d):
    """Discomfort-only forward planning utility using LLM-derived risk.

    Only considers how intimacy mitigates discomfort from risky actions.
    Uses LLM risk values instead of stipulated values.
    """
    formality = 1 - intimacy
    risk = get_risk_lm(action, scenario_idx)
    discomfort = formality * risk
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
# Inverse Planning Utility Functions (LM-based)
# ==============================================================================


@jax.jit
def get_utility_inv_plan_full_lm_continuous(
    action,
    intimacy,
    reward_condition,
    scenario_idx,
    alpha,
    w_r,
    w_d,
    w_c,
):
    """Full inverse planning utility with LLM-derived scenario params (continuous intimacy)."""
    base_reward = jnp.where(reward_condition == RewardConditions.HIGH, get_reward_base_lm(scenario_idx), 0.0)
    action_has_reward = jnp.array([0, 1, 1, 1])[action]
    reward = base_reward * action_has_reward * (1 + intimacy)

    formality = 1 - intimacy
    risk = get_risk_lm(action, scenario_idx)
    discomfort = formality * risk

    effort = get_effort_lm(action, scenario_idx)

    return alpha * (w_r * reward - w_d * discomfort - w_c * effort)


@jax.jit
def get_utility_inv_plan_full_lm_discrete(
    action,
    relationship_condition,
    reward_condition,
    scenario_idx,
    alpha,
    w_r,
    w_d,
    w_c,
):
    """Full inverse planning utility with LLM-derived scenario params (discrete intimacy)."""
    intimacy = get_intimacy(relationship_condition)
    return get_utility_inv_plan_full_lm_continuous(
        action, intimacy, reward_condition, scenario_idx, alpha, w_r, w_d, w_c
    )


@jax.jit
def get_utility_inv_plan_vanilla_lm(
    action,
    relationship,
    reward_condition,
    scenario_idx,
    alpha,
    w_r,
    w_d,
    w_c,
):
    """Vanilla inverse planning utility with LLM-derived scenario params (no intimacy scaling)."""
    base_reward = jnp.where(reward_condition == RewardConditions.HIGH, get_reward_base_lm(scenario_idx), 0.0)
    action_has_reward = jnp.array([0, 1, 1, 1])[action]
    reward = base_reward * action_has_reward

    risk = get_risk_lm(action, scenario_idx)
    effort = get_effort_lm(action, scenario_idx)

    return alpha * (w_r * reward - w_d * risk - w_c * effort)


@jax.jit
def get_utility_inv_plan_discomfort_only_lm_continuous(
    action,
    intimacy,
    reward_condition,
    scenario_idx,
    alpha,
    w_r,
    w_d,
    w_c,
):
    """Discomfort-only inverse planning utility with LLM risk (continuous intimacy)."""
    formality = 1 - intimacy
    risk = get_risk_lm(action, scenario_idx)
    discomfort = formality * risk
    return alpha * (-w_d * discomfort)


@jax.jit
def get_utility_inv_plan_discomfort_only_lm_discrete(
    action,
    relationship_condition,
    reward_condition,
    scenario_idx,
    alpha,
    w_r,
    w_d,
    w_c,
):
    """Discomfort-only inverse planning utility with LLM risk (discrete intimacy)."""
    intimacy = get_intimacy(relationship_condition)
    return get_utility_inv_plan_discomfort_only_lm_continuous(
        action, intimacy, reward_condition, scenario_idx, alpha, w_r, w_d, w_c
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
# Inverse Planning Actor Models (Discrete, LM-based)
# ==============================================================================


@memo
def actor_discrete_full_model_lm[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    scenario_idx, alpha, w_r, w_d, w_c
):
    cast: [actor]
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_inv_plan_full_lm_discrete(
                action,
                relationship_condition,
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


@memo
def actor_discrete_vanilla_lm[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    scenario_idx, alpha, w_r, w_d, w_c
):
    cast: [actor]
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_inv_plan_vanilla_lm(
                action,
                relationship_condition,
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


@memo
def actor_discrete_discomfort_only_lm[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    scenario_idx, alpha, w_r, w_d, w_c
):
    cast: [actor]
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_inv_plan_discomfort_only_lm_discrete(
                action,
                relationship_condition,
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
# Inverse Planning Actor Models (Continuous, LM-based)
# ==============================================================================


@memo
def actor_continuous_full_model_lm[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    scenario_idx, alpha, w_r, w_d, w_c
):
    cast: [actor]
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_inv_plan_full_lm_continuous(
                action,
                relationship,
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


@memo
def actor_continuous_vanilla_lm[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    scenario_idx, alpha, w_r, w_d, w_c
):
    cast: [actor]
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_inv_plan_vanilla_lm(
                action,
                relationship,
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


@memo
def actor_continuous_discomfort_only_lm[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    scenario_idx, alpha, w_r, w_d, w_c
):
    cast: [actor]
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_inv_plan_discomfort_only_lm_continuous(
                action,
                relationship,
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


# Vanilla model with LLM-derived scenario-specific parameters
@memo
def actor_forw_vanilla_lm[
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
            get_utility_forw_vanilla_lm(
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


# Discomfort-only model with LLM-derived scenario-specific parameters
@memo
def actor_forw_discomfort_only_lm[
    action: actions,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](
    scenario_idx, alpha, w_d
):
    cast: [actor]
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_forw_discomfort_only_lm(
                action,
                intimacy,
                scenario_idx,
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
](
    alpha, w_r, w_d, w_c, alpha_observer
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
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_vanilla_inv_plan[
    action: actions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_d, w_c, alpha_observer
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
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_full_model[
    action: actions, relationship: IntimacyLevels, reward_condition: RewardConditions
](
    alpha, w_r, w_d, w_c, alpha_observer
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
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# ==============================================================================
# Observer Models - Inferring Intimacy (LM-based)
# ==============================================================================


@memo
def observer_intimacy_full_model_lm[
    action: actions, relationship: IntimacyLevels, reward_condition: RewardConditions
](
    scenario_idx, alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_full_model_lm[action, relationship, reward_condition](
                scenario_idx, alpha, w_r, w_d, w_c
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
def observer_intimacy_vanilla_lm[
    action: actions, relationship: IntimacyLevels, reward_condition: RewardConditions
](
    scenario_idx, alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_vanilla_lm[action, relationship, reward_condition](
                scenario_idx, alpha, w_r, w_d, w_c
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
def observer_intimacy_discomfort_only_lm[
    action: actions, relationship: IntimacyLevels, reward_condition: RewardConditions
](
    scenario_idx, alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_discomfort_only_lm[action, relationship, reward_condition](
                scenario_idx, alpha, w_r, w_d, w_c
            ),
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
](
    alpha, w_r, w_d, w_c, alpha_observer
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
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_vanilla_inv_plan[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_d, w_c, alpha_observer
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
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_full_model[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_d, w_c, alpha_observer
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
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


# ==============================================================================
# Observer Models - Inferring Reward (LM-based)
# ==============================================================================


@memo
def observer_reward_full_model_lm[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    scenario_idx, alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_full_model_lm[
                action, relationship_condition, reward_condition
            ](scenario_idx, alpha, w_r, w_d, w_c),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_vanilla_lm[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    scenario_idx, alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_vanilla_lm[
                action, relationship_condition, reward_condition
            ](scenario_idx, alpha, w_r, w_d, w_c),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_discomfort_only_lm[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    scenario_idx, alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_discomfort_only_lm[
                action, relationship_condition, reward_condition
            ](scenario_idx, alpha, w_r, w_d, w_c),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


# ==============================================================================
# Observer Models - Inferring Intimacy (Empirical Prior)
# ==============================================================================


@memo
def observer_intimacy_full_model_empirical_prior[
    action: actions, relationship: IntimacyLevels, reward_condition: RewardConditions
](
    alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(
            relationship in IntimacyLevels,
            wpp=get_empirical_prior_intimacy(relationship, reward_condition)
        ),
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
def observer_intimacy_vanilla_empirical_prior[
    action: actions, relationship: IntimacyLevels, reward_condition: RewardConditions
](
    alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(
            relationship in IntimacyLevels,
            wpp=get_empirical_prior_intimacy(relationship, reward_condition)
        ),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_vanilla_inv_plan[action, relationship, reward_condition](
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
def observer_intimacy_discomfort_only_empirical_prior[
    action: actions, relationship: IntimacyLevels, reward_condition: RewardConditions
](
    alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(reward_condition),
        actor : chooses(
            relationship in IntimacyLevels,
            wpp=get_empirical_prior_intimacy(relationship, reward_condition)
        ),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_discomfort_only[action, relationship, reward_condition](
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


# ==============================================================================
# Observer Models - Inferring Reward (Empirical Prior)
# ==============================================================================


@memo
def observer_reward_full_model_empirical_prior[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(
            reward_condition in RewardConditions,
            wpp=get_empirical_prior_reward(reward_condition, relationship_condition)
        ),
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
def observer_reward_vanilla_empirical_prior[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(
            reward_condition in RewardConditions,
            wpp=get_empirical_prior_reward(reward_condition, relationship_condition)
        ),
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
def observer_reward_discomfort_only_empirical_prior[
    action: actions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](
    alpha, w_r, w_d, w_c, alpha_observer
):
    cast: [actor, observer]
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(relationship_condition),
        actor : chooses(
            reward_condition in RewardConditions,
            wpp=get_empirical_prior_reward(reward_condition, relationship_condition)
        ),
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