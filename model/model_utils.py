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


# Padded-action constants for the no-alternatives-shown variant (Exp 2a no-alt).
# See load_padded_lm_tables() below for how per-cell action tables are built.
MAX_ACTIONS = 16
padded_slots = jnp.arange(MAX_ACTIONS)


class PaddedActionSlots(IntEnum):
    """Memo-friendly enum of padded action slot indices."""
    S0 = 0
    S1 = 1
    S2 = 2
    S3 = 3
    S4 = 4
    S5 = 5
    S6 = 6
    S7 = 7
    S8 = 8
    S9 = 9
    S10 = 10
    S11 = 11
    S12 = 12
    S13 = 13
    S14 = 14
    S15 = 15


class ObservedActions(IntEnum):
    """The 4 canonical observed actions (the experimental stimulus dimension)."""
    A0 = 0
    A1 = 1
    A2 = 2
    A3 = 3


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


# Canonical is_share mapping for the 4 experimenter-authored actions.
# Slot 0 of the padded tables always holds the observed canonical action, so
# its is_share is determined by this vector. Matches get_stipulated_reward.
CANONICAL_IS_SHARE = jnp.array([0.0, 1.0, 1.0, 1.0])


def load_padded_lm_tables(
    canonical_path=None,
    alternatives_path=None,
    alternatives_features_path=None,
):
    """Build padded (16, 4, 2, MAX_ACTIONS) tables for access, effort, is_share,
    and prior.

    Slot 0 of every (scenario, observed_action, motivation) cell holds the
    observed canonical action's features (from lm_scenario_params.csv).
    Slots 1..k hold the LM-generated alternatives for that cell (from
    lm_alternatives.csv + lm_alternatives_features.csv). Remaining slots are
    null-padded with access=0, effort=0, is_share=0. The prior is uniform over
    valid (non-null) slots; null slots get a tiny epsilon (1e-8) rather than
    exactly 0 to keep `E[...] ** alpha_observer` differentiable in the observer
    memo, while still contributing <1e-6 of softmax mass.

    Returns a dict {access, effort, is_share, prior} (each jnp.array of shape
    (16, 4, 2, MAX_ACTIONS)), or None if the alternative CSVs are missing.
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    canonical_path = canonical_path or outputs_dir / "lm_scenario_params.csv"
    alternatives_path = alternatives_path or outputs_dir / "lm_alternatives.csv"
    alternatives_features_path = (
        alternatives_features_path or outputs_dir / "lm_alternatives_features.csv"
    )

    if not alternatives_path.exists() or not alternatives_features_path.exists():
        return None

    canonical_df = pd.read_csv(canonical_path)
    alts_df = pd.read_csv(alternatives_path)
    feats_df = pd.read_csv(alternatives_features_path)

    n_scenarios = len(SCENARIO_LABELS)
    n_observed = 4
    n_motivations = 2
    shape = (n_scenarios, n_observed, n_motivations, MAX_ACTIONS)
    access = np.zeros(shape, dtype=np.float32)
    effort = np.zeros(shape, dtype=np.float32)
    is_share = np.zeros(shape, dtype=np.float32)
    # valid_mask[s, o, m, slot] = True if the slot holds a real action (not null pad)
    valid_mask = np.zeros(shape, dtype=bool)

    # Canonical (slot 0) per cell
    canon_lookup = {}
    for _, row in canonical_df.iterrows():
        canon_lookup[(row["scenario_label"], int(row["action"]))] = (
            float(row["access"]),
            float(row["effort"]),
        )
    for scenario in SCENARIO_LABELS:
        s_idx = SCENARIO_TO_IDX[scenario]
        for observed in range(n_observed):
            a_access, a_effort = canon_lookup[(scenario, observed)]
            for motivation in range(n_motivations):
                access[s_idx, observed, motivation, 0] = a_access
                effort[s_idx, observed, motivation, 0] = a_effort
                is_share[s_idx, observed, motivation, 0] = float(
                    CANONICAL_IS_SHARE[observed]
                )
                valid_mask[s_idx, observed, motivation, 0] = True

    # LM-generated alternatives (slots 1..k) per cell
    is_share_lookup = {
        (r["scenario_label"], r["observed_action"], r["motivation"], int(r["alt_idx"])): int(
            r["is_share"]
        )
        for _, r in alts_df.iterrows()
    }
    motivation_to_idx = {"low": int(RewardConditions.LOW), "high": int(RewardConditions.HIGH)}
    observed_str_to_idx = {f"action_{i}": i for i in range(n_observed)}

    for _, row in feats_df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        o_idx = observed_str_to_idx[row["observed_action"]]
        m_idx = motivation_to_idx[row["motivation"]]
        alt_idx = int(row["alt_idx"])
        slot = alt_idx + 1  # +1 because slot 0 is the observed canonical action
        if slot >= MAX_ACTIONS:
            continue  # truncate silently; warn below
        access[s_idx, o_idx, m_idx, slot] = float(row["access"])
        effort[s_idx, o_idx, m_idx, slot] = float(row["effort"])
        is_share[s_idx, o_idx, m_idx, slot] = float(
            is_share_lookup.get(
                (row["scenario_label"], row["observed_action"], row["motivation"], alt_idx),
                0,
            )
        )
        valid_mask[s_idx, o_idx, m_idx, slot] = True

    NULL_EPSILON = 1e-8
    n_valid = valid_mask.sum(axis=-1, keepdims=True)  # (16, 4, 2, 1)
    prior_table = np.where(
        valid_mask, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON
    ).astype(np.float32)

    max_alt_count = (
        feats_df.groupby(["scenario_label", "observed_action", "motivation"])
        .size()
        .max()
    )
    if max_alt_count + 1 > MAX_ACTIONS:
        print(
            f"WARNING: largest cell has {max_alt_count} LM-generated alternatives + "
            f"1 observed = {max_alt_count + 1} actions, exceeding MAX_ACTIONS={MAX_ACTIONS}. "
            "Extra alternatives were truncated."
        )

    return {
        "access": jnp.array(access),
        "effort": jnp.array(effort),
        "is_share": jnp.array(is_share),
        "prior": jnp.array(prior_table),
    }


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
#   - access_only : only the access-discomfort term (-w_d * access * (1-I))
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
    alpha, w_v, w_d, w_e,
    access_table, effort_table,
):
    access = access_table[scenario_idx, action]
    effort = effort_table[scenario_idx, action]
    V = get_stipulated_reward(action, reward_condition)
    return alpha * (
        w_v * V
        - w_d * access * (1 - intimacy)
        - w_e * effort
    )


@jax.jit
def get_utility_access_full_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_v, w_d, w_e,
    access_table, effort_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_access_full(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_v, w_d, w_e,
        access_table, effort_table,
    )


@jax.jit
def get_utility_access_only(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_d,
    access_table, effort_table,
):
    access = access_table[scenario_idx, action]
    return alpha * (-w_d * access * (1 - intimacy))


@jax.jit
def get_utility_access_only_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_d,
    access_table, effort_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_access_only(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_d,
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
](alpha, w_v, w_d, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_full(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_v, w_d, w_e,
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
](alpha, w_d, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_only(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_d,
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
](alpha, w_v, w_d, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_full_disc(
                action, scenario_idx, relationship_condition, reward_condition,
                alpha, w_v, w_d, w_e,
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
](alpha, w_d, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_only_disc(
                action, scenario_idx, relationship_condition, reward_condition,
                alpha, w_d,
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
](alpha, w_v, w_d, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_full(
                action, scenario_idx, relationship, reward_condition,
                alpha, w_v, w_d, w_e,
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
](alpha, w_d, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_access_only(
                action, scenario_idx, relationship, reward_condition,
                alpha, w_d,
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
](alpha, w_v, w_d, w_e, alpha_observer, access_table: ..., effort_table: ...):
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
            ](alpha, w_v, w_d, w_e, access_table, effort_table),
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
](alpha, w_d, alpha_observer, access_table: ..., effort_table: ...):
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
            ](alpha, w_d, access_table, effort_table),
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
](alpha, w_v, w_d, w_e, alpha_observer, access_table: ..., effort_table: ...):
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
            ](alpha, w_v, w_d, w_e, access_table, effort_table),
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
](alpha, w_d, alpha_observer, access_table: ..., effort_table: ...):
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
            ](alpha, w_d, access_table, effort_table),
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


# ==============================================================================
# Padded-action utility and observer (Exp 2a no-alt variant)
# ==============================================================================
# The padded observer infers intimacy from a single observed action, using a
# trial-specific action space that is the union of the observed action (slot 0)
# and the LM-generated counterfactual alternatives for that cell (slots 1..k).
# Remaining slots are null-padded (access=effort=is_share=0, utility=0).


@jax.jit
def get_prior_padded(
    padded_slot, scenario_idx, observed_action, reward_condition, prior_table,
):
    """Look up the actor-prior weight for this slot. Null-padded slots have 0."""
    return prior_table[scenario_idx, observed_action, reward_condition, padded_slot]


@jax.jit
def get_stipulated_reward_padded(
    padded_slot, scenario_idx, observed_action, reward_condition, is_share_table,
):
    """Binary goal-satisfaction gate using the is_share tag for arbitrary actions.

    is_share_table has shape (16, 4, 2, MAX_ACTIONS) — indexed by
    (scenario, observed_action, motivation, padded_slot). Under HIGH motivation
    the goal is to eat: V=1 iff is_share==1. Under LOW: V=1 iff is_share==0.
    """
    motivation = jnp.where(reward_condition == RewardConditions.HIGH, 1.0, 0.0)
    action_is_share = is_share_table[scenario_idx, observed_action, reward_condition, padded_slot]
    return motivation * action_is_share + (1.0 - motivation) * (1.0 - action_is_share)


@jax.jit
def get_utility_access_full_padded(
    padded_slot, scenario_idx, observed_action, intimacy, reward_condition,
    alpha, w_v, w_d, w_e,
    access_table, effort_table, is_share_table,
):
    access = access_table[scenario_idx, observed_action, reward_condition, padded_slot]
    effort = effort_table[scenario_idx, observed_action, reward_condition, padded_slot]
    V = get_stipulated_reward_padded(
        padded_slot, scenario_idx, observed_action, reward_condition, is_share_table,
    )
    return alpha * (
        w_v * V
        - w_d * access * (1 - intimacy)
        - w_e * effort
    )


@memo
def actor_continuous_access_full_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, access_table: ..., effort_table: ..., is_share_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded(padded_slot, scenario_idx, observed_action, reward_condition, prior_table) * exp(
            get_utility_access_full_padded(
                padded_slot, scenario_idx, observed_action, relationship, reward_condition,
                alpha, w_v, w_d, w_e,
                access_table, effort_table, is_share_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def observer_intimacy_access_full_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, alpha_observer, access_table: ..., effort_table: ..., is_share_table: ..., prior_table: ...):
    # The memo returns a posterior on relationship indexed by padded_slot as
    # the observation. Callers should evaluate at padded_slot=0, since slot 0
    # always holds the observed canonical action (see load_padded_lm_tables).
    # The actor prior_table multiplies into the softmax: wpp = prior * exp(U).
    # Null-padded slots have prior=0 so they contribute no mass.
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
            wpp=actor_continuous_access_full_padded[
                padded_slot, scenario_idx, observed_action, relationship, reward_condition
            ](alpha, w_v, w_d, w_e, access_table, effort_table, is_share_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# --- access_only padded variant: drops w_v*V and w_e*effort ---


@jax.jit
def get_utility_access_only_padded(
    padded_slot, scenario_idx, observed_action, intimacy, reward_condition,
    alpha, w_d,
    access_table, effort_table, is_share_table,
):
    access = access_table[scenario_idx, observed_action, reward_condition, padded_slot]
    return alpha * (-w_d * access * (1 - intimacy))


@memo
def actor_continuous_access_only_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, access_table: ..., effort_table: ..., is_share_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded(padded_slot, scenario_idx, observed_action, reward_condition, prior_table) * exp(
            get_utility_access_only_padded(
                padded_slot, scenario_idx, observed_action, relationship, reward_condition,
                alpha, w_d,
                access_table, effort_table, is_share_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def observer_intimacy_access_only_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, alpha_observer, access_table: ..., effort_table: ..., is_share_table: ..., prior_table: ...):
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
            wpp=actor_continuous_access_only_padded[
                padded_slot, scenario_idx, observed_action, relationship, reward_condition
            ](alpha, w_d, access_table, effort_table, is_share_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# --- no_access padded variant: drops w_d*access ---


@jax.jit
def get_utility_no_access_padded(
    padded_slot, scenario_idx, observed_action, intimacy, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table, is_share_table,
):
    effort = effort_table[scenario_idx, observed_action, reward_condition, padded_slot]
    V = get_stipulated_reward_padded(
        padded_slot, scenario_idx, observed_action, reward_condition, is_share_table,
    )
    return alpha * (w_v * V - w_e * effort)


@memo
def actor_continuous_no_access_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., is_share_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded(padded_slot, scenario_idx, observed_action, reward_condition, prior_table) * exp(
            get_utility_no_access_padded(
                padded_slot, scenario_idx, observed_action, relationship, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table, is_share_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def observer_intimacy_no_access_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., is_share_table: ..., prior_table: ...):
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
            wpp=actor_continuous_no_access_padded[
                padded_slot, scenario_idx, observed_action, relationship, reward_condition
            ](alpha, w_v, w_e, access_table, effort_table, is_share_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]
