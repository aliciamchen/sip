from enum import IntEnum
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
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


class Scenarios(IntEnum):
    """Memo-friendly enum of the 16 scenarios (alphabetical order)."""
    APPLES = 0
    BASKETBALL = 1
    BIRTHDAY = 2
    BRUNCH = 3
    COOKING = 4
    DIP = 5
    DRINKS = 6
    DRIVING = 7
    FAIR = 8
    GALA = 9
    HIKE = 10
    OYSTERS = 11
    SOCIAL = 12
    SOUP = 13
    TAKEOUT = 14
    WEDDING = 15


# ==============================================================================
# LLM-derived scenario-specific parameters (access / effort / reward)
# ==============================================================================


def load_lm_scenario_params(filepath=None):
    """Load scenario-specific LLM parameter tables for access and effort.

    Returns a dict with:
      - "access": jnp.array of shape (16, 4)  — per (scenario, action)
      - "effort": jnp.array of shape (16, 4)  — per (scenario, action)

    Reward is NOT loaded here; it's stipulated in the utility functions as a
    binary goal-satisfaction gate (V=1 iff the action satisfies the active
    goal: sharing under HIGH motivation, or not-sharing under LOW motivation).

    Rows are ordered by SCENARIO_LABELS (alphabetical). Raises FileNotFoundError
    if the CSV is missing — run `uv run python model/lm_scenario_params.py` first.
    """
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent / "outputs" / "lm_scenario_params.csv"
        )
    df = pd.read_csv(filepath)
    access = np.zeros((len(SCENARIO_LABELS), 4), dtype=np.float32)
    effort = np.zeros((len(SCENARIO_LABELS), 4), dtype=np.float32)
    for _, row in df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        a = int(row["action"])
        access[s_idx, a] = row["access"]
        effort[s_idx, a] = row["effort"]
    return {
        "access": jnp.array(access),
        "effort": jnp.array(effort),
    }


LLM_TABLES = load_lm_scenario_params()


# ==============================================================================
# Basic utility helpers
# ==============================================================================


@jax.jit
def get_intimacy(relationship_condition):
    """Map a discrete relationship condition to a continuous intimacy level in [0, 1]."""
    return jnp.array([0, 0.5, 0.75, 1])[relationship_condition]


# ==============================================================================
# Access-model utility functions
# ==============================================================================
# Canonical utility:
#   U(a|s, I, scen) =  w_v * V(a|s)
#                    + w_r * access[scen, a] * I
#                    - w_d * access[scen, a] * (1 - I)
#                    - w_e * effort[scen, a]
#
# Reward V(a|s) is stipulated as a binary goal-satisfaction gate: V=1 iff the
# action satisfies the active goal. Under HIGH motivation the goal is to eat/
# share, so V=1 for sharing actions (action != 0); under LOW motivation the
# goal is to not eat, so V=1 for action 0. V=0 otherwise.
#
# Access and effort are LLM-derived per-scenario (access_table, effort_table)
# and passed as memo parameters so memo can JIT-compile without baking them
# into the compiled graph.
#
# Three ablations:
#   - access_full : full utility above (main model)
#   - access_only : only the two access terms (drop w_v*V and w_e*effort)
#   - no_access   : base model (w_v*V - w_e*effort)


@jax.jit
def get_stipulated_reward(action, reward_condition):
    """Stipulated binary reward encoding goal satisfaction.

    V=1 iff the action satisfies the active goal:
      - HIGH motivation (goal = eat/share): V=1 for sharing actions (1-3)
      - LOW  motivation (goal = not eat):  V=1 for action 0 (no sharing)
    """
    motivation = jnp.where(reward_condition == RewardConditions.HIGH, 1.0, 0.0)
    action_is_share = jnp.array([0.0, 1.0, 1.0, 1.0])[action]
    return motivation * action_is_share + (1.0 - motivation) * (1.0 - action_is_share)


@jax.jit
def get_utility_access_full(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_v, w_r, w_d, w_e,
    access_table, effort_table,
):
    access = access_table[scenario_idx, action]
    effort = effort_table[scenario_idx, action]
    V = get_stipulated_reward(action, reward_condition)
    return alpha * (
        w_v * V
        + w_r * access * intimacy
        - w_d * access * (1 - intimacy)
        - w_e * effort
    )


@jax.jit
def get_utility_access_full_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_v, w_r, w_d, w_e,
    access_table, effort_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_access_full(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_v, w_r, w_d, w_e,
        access_table, effort_table,
    )


@jax.jit
def get_utility_access_only(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_r, w_d,
    access_table, effort_table,
):
    access = access_table[scenario_idx, action]
    return alpha * (w_r * access * intimacy - w_d * access * (1 - intimacy))


@jax.jit
def get_utility_access_only_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_r, w_d,
    access_table, effort_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_access_only(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_r, w_d,
        access_table, effort_table,
    )


@jax.jit
def get_utility_no_access(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table,
):
    effort = effort_table[scenario_idx, action]
    V = get_stipulated_reward(action, reward_condition)
    return alpha * (w_v * V - w_e * effort)


@jax.jit
def get_utility_no_access_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_no_access(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_v, w_e,
        access_table, effort_table,
    )


# ==============================================================================
# Forward-planning actor models (Exp 1)
# ==============================================================================


@memo
def actor_forw_access_full[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_r, w_d, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_full(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_v, w_r, w_d, w_e,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_access_only[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_only(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_r, w_d,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_no_access[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_no_access(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Inverse-planning actor models (discrete relationship)
# ==============================================================================


@memo
def actor_discrete_access_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_r, w_d, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_full_disc(
                action, scenario_idx, relationship_condition, reward_condition,
                alpha, w_v, w_r, w_d, w_e,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_access_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_only_disc(
                action, scenario_idx, relationship_condition, reward_condition,
                alpha, w_r, w_d,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_no_access[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_no_access_disc(
                action, scenario_idx, relationship_condition, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Inverse-planning actor models (continuous intimacy)
# ==============================================================================


@memo
def actor_continuous_access_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_r, w_d, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_full(
                action, scenario_idx, relationship, reward_condition,
                alpha, w_v, w_r, w_d, w_e,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_access_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_only(
                action, scenario_idx, relationship, reward_condition,
                alpha, w_r, w_d,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_no_access[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_no_access(
                action, scenario_idx, relationship, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Observer inferring intimacy (Exp 2a)
# ==============================================================================


@memo
def observer_intimacy_access_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_r, w_d, w_e, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_access_full[
                action, scenario_idx, relationship, reward_condition
            ](alpha, w_v, w_r, w_d, w_e, access_table, effort_table),
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
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_access_only[
                action, scenario_idx, relationship, reward_condition
            ](alpha, w_r, w_d, access_table, effort_table),
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
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_no_access[
                action, scenario_idx, relationship, reward_condition
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
# Observer inferring reward (Exp 2b)
# ==============================================================================


@memo
def observer_reward_access_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_r, w_d, w_e, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_access_full[
                action, scenario_idx, relationship_condition, reward_condition
            ](alpha, w_v, w_r, w_d, w_e, access_table, effort_table),
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
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_r, w_d, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_access_only[
                action, scenario_idx, relationship_condition, reward_condition
            ](alpha, w_r, w_d, access_table, effort_table),
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
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_no_access[
                action, scenario_idx, relationship_condition, reward_condition
            ](alpha, w_v, w_e, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]
