"""
Pure utility functions (jax-jit-compiled, dimension-agnostic) used by all
actor and observer memo models.

Three model variants — Full, Discomfort-only, Base — each have:
  - a canonical form: `get_utility_<variant>(action, scenario_idx, intimacy, ...)`
  - a discrete-relationship form (`_disc`): wraps the canonical form, mapping
    a discrete relationship_condition to a continuous intimacy via get_intimacy
  - a padded form (`_padded`): for the no-alternatives-shown observer; indexes
    into per-cell padded action tables of shape (16, 4, 2, MAX_ACTIONS)
  - a relationship-keyed padded form (`_padded_rel`): for the desire-noalt
    observer; padded tables keyed on (scenario, observed, relationship)

Effort-experiment counterparts (`get_utility_effort_*`) use a 2-action space
with a stipulated V=1 (reward fixed HIGH).

Canonical utility:
  U(a|s, I, scen, m) =  w_v * V(a|s, m)
                      - w_d * access[scen, a] * (1 - I)^gamma
                      - w_e * effort[scen, a]

V is signed in [-1, +1]: positive when an action serves the active state,
negative when it actively works against it. access and effort use the
canonical (16, 4) tables; V uses a (16, 4, 2) table indexed by motivation.

Dependency layer 1: imports from `tables.py` only. `actors.py` and
`observers.py` import from here.
"""

import jax
import jax.numpy as jnp

from tables import RELATIONSHIP_LEVEL_VALUES


# ==============================================================================
# Basic helpers
# ==============================================================================


@jax.jit
def get_intimacy(relationship_condition):
    """Map a discrete relationship condition to a continuous intimacy level in [0, 1]."""
    return jnp.array([0, 0.5, 0.75, 1])[relationship_condition]


@jax.jit
def get_lm_v(action, scenario_idx, reward_condition, v_table):
    """LM-elicited signed valence: v_table[scenario_idx, action, reward_condition].

    v_table has shape (16, 4, 2). Values in [-1, +1]. Positive = action serves
    the active state; negative = action actively counterproductive.
    """
    return v_table[scenario_idx, action, reward_condition]


# ==============================================================================
# Canonical (4-action) utility — Full / Discomfort-only / Base
# ==============================================================================


@jax.jit
def get_utility_full(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_table,
):
    access = access_table[scenario_idx, action]
    effort = effort_table[scenario_idx, action]
    V = get_lm_v(action, scenario_idx, reward_condition, v_table)
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * V
        - w_d * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@jax.jit
def get_utility_full_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_full(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_v, w_d, w_e, gamma,
        access_table, effort_table, v_table,
    )


@jax.jit
def get_utility_discomfort_only(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    access = access_table[scenario_idx, action]
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (-w_d * access * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_discomfort_only_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_discomfort_only(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_d, gamma,
        access_table, effort_table,
    )


@jax.jit
def get_utility_base(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table, v_table,
):
    effort = effort_table[scenario_idx, action]
    V = get_lm_v(action, scenario_idx, reward_condition, v_table)
    return alpha * (w_v * V - w_e * effort)


@jax.jit
def get_utility_base_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table, v_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_base(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_v, w_e,
        access_table, effort_table, v_table,
    )


# ==============================================================================
# Padded (no-alternatives-shown) utility helpers — motivation-keyed action space
# ==============================================================================


@jax.jit
def get_prior_padded(
    padded_slot, scenario_idx, observed_action, reward_condition, prior_table,
):
    """Look up the actor-prior weight for this slot. Null-padded slots have 0."""
    return prior_table[scenario_idx, observed_action, reward_condition, padded_slot]


@jax.jit
def get_lm_v_padded(
    padded_slot, scenario_idx, observed_action, reward_condition, v_padded_table,
):
    """LM-elicited signed valence for an arbitrary action in the padded action
    space.

    v_padded_table has shape (16, 4, 2, MAX_ACTIONS) — indexed by
    (scenario, observed_action, motivation, padded_slot). Slot 0 is the
    canonical action (V from lm_scenario_v.csv); slots 1..k are LM-generated
    alternatives (V from lm_alternatives_v.csv); remaining slots are
    null-padded with V=0 (no contribution after multiplying by zero prior).
    """
    return v_padded_table[scenario_idx, observed_action, reward_condition, padded_slot]


@jax.jit
def get_utility_full_padded(
    padded_slot, scenario_idx, observed_action, intimacy, reward_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_padded_table,
):
    access = access_table[scenario_idx, observed_action, reward_condition, padded_slot]
    effort = effort_table[scenario_idx, observed_action, reward_condition, padded_slot]
    V = get_lm_v_padded(
        padded_slot, scenario_idx, observed_action, reward_condition, v_padded_table,
    )
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * V
        - w_d * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@jax.jit
def get_utility_discomfort_only_padded(
    padded_slot, scenario_idx, observed_action, intimacy, reward_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    access = access_table[scenario_idx, observed_action, reward_condition, padded_slot]
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (-w_d * access * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_base_padded(
    padded_slot, scenario_idx, observed_action, intimacy, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table, v_padded_table,
):
    effort = effort_table[scenario_idx, observed_action, reward_condition, padded_slot]
    V = get_lm_v_padded(
        padded_slot, scenario_idx, observed_action, reward_condition, v_padded_table,
    )
    return alpha * (w_v * V - w_e * effort)


# ==============================================================================
# Relationship-keyed padded utility (desire-noalt observer)
# ==============================================================================


@jax.jit
def get_prior_padded_rel(
    padded_slot, scenario_idx, observed_action, relationship_condition, prior_table,
):
    """Look up the actor-prior weight for this slot under the relationship-keyed
    action space. Null-padded slots have ~0 (1e-8 epsilon)."""
    return prior_table[scenario_idx, observed_action, relationship_condition, padded_slot]


@jax.jit
def get_lm_v_padded_rel(
    padded_slot, scenario_idx, observed_action, relationship_condition,
    reward_condition, v_padded_table,
):
    """LM-elicited signed valence for an arbitrary action in the relationship-
    keyed padded action space. v_padded_table has shape
    (16, 4, 4, MAX_ACTIONS, 2) — indexed by
    (scenario, observed_action, relationship, padded_slot, motivation)."""
    return v_padded_table[
        scenario_idx, observed_action, relationship_condition, padded_slot, reward_condition,
    ]


@jax.jit
def get_utility_full_padded_rel(
    padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_padded_table,
):
    intimacy = RELATIONSHIP_LEVEL_VALUES[relationship_condition]
    access = access_table[scenario_idx, observed_action, relationship_condition, padded_slot]
    effort = effort_table[scenario_idx, observed_action, relationship_condition, padded_slot]
    V = get_lm_v_padded_rel(
        padded_slot, scenario_idx, observed_action, relationship_condition,
        reward_condition, v_padded_table,
    )
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * V
        - w_d * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@jax.jit
def get_utility_discomfort_only_padded_rel(
    padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    intimacy = RELATIONSHIP_LEVEL_VALUES[relationship_condition]
    access = access_table[scenario_idx, observed_action, relationship_condition, padded_slot]
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (-w_d * access * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_base_padded_rel(
    padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table, v_padded_table,
):
    effort = effort_table[scenario_idx, observed_action, relationship_condition, padded_slot]
    V = get_lm_v_padded_rel(
        padded_slot, scenario_idx, observed_action, relationship_condition,
        reward_condition, v_padded_table,
    )
    return alpha * (w_v * V - w_e * effort)


# ==============================================================================
# Effort-experiment utility (2-action space, V stipulated to 1)
# ==============================================================================


@jax.jit
def get_stipulated_reward_effort(action):
    """Constant V = 1 for both actions (reward fixed HIGH, both end in eating).
    w_v is therefore non-identified in the softmax but kept as a fitted
    parameter for parallelism with the canonical 4-action pipeline."""
    return jnp.array([1.0, 1.0])[action]


@jax.jit
def get_utility_effort_full(
    action, scenario_idx, intimacy, effort_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table,
):
    access = access_table[scenario_idx, effort_condition, action]
    effort = effort_table[scenario_idx, effort_condition, action]
    V = get_stipulated_reward_effort(action)
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * V
        - w_d * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@jax.jit
def get_utility_effort_discomfort_only(
    action, scenario_idx, intimacy, effort_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    access = access_table[scenario_idx, effort_condition, action]
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (-w_d * access * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_effort_base(
    action, scenario_idx, intimacy, effort_condition,
    alpha, w_v, w_e,
    access_table, effort_table,
):
    effort = effort_table[scenario_idx, effort_condition, action]
    V = get_stipulated_reward_effort(action)
    return alpha * (w_v * V - w_e * effort)
