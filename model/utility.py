"""
Pure utility functions (jax-jit-compiled, dimension-agnostic) used by the padded
inverse actor and observer memo models.

Each study has a padded utility family `get_utility_<variant>_padded_<study>` in
three variants — `full`, `discomfort_only`, `base` — plus `get_prior_padded_*`
and `get_lm_g_padded_*` helpers, indexing per-cell padded action tables.

Utility:
  U(a|s, I, d) =  w_v * d * g(a)
               -  w_d * risk[scen, a] * (1 - I)^gamma
               -  w_e * effort[scen, a]

The reward term `w_v * d * g` multiplies desire d ∈ [0, 1] by the desire-free
goal-satisfaction g(a) ∈ [0, 1]. Intimacy I ∈ [0, 1] modulates the risk term.

Dependency layer 1: imports from `tables.py` only. `actors.py` and
`observers.py` import from here.
"""

import jax
import jax.numpy as jnp

from tables import RELATIONSHIP_LEVEL_VALUES


# ==============================================================================
# Padded utilities for Study 1a — desire inference with LM-generated alternatives
# ==============================================================================
# `food_inv_desire`. Observer knows scenario, observed_action, effort_condition,
# relationship_condition; the latent is desire (continuous, inferred over the
# 101-bin DesireLevels grid). The LM enumerates plausible alternatives per
# (scenario, observed_action, effort_condition, relationship_condition) cell and
# the observer's actor softmaxes over `{observed_action} ∪ alternatives`, padded
# to `MAX_ACTIONS`. The reward term is `w_v · desire · g`, with g the
# desire-free goal-satisfaction.


@jax.jit
def get_prior_padded_desire(
    padded_slot,
    scenario_idx,
    observed_action,
    effort_condition,
    relationship_condition,
    prior_table,
):
    """Actor-prior weight for this slot in Study 1a's padded action space.

    Null-padded slots have ~0 mass (1e-8 epsilon), so the actor's softmax
    effectively skips them while keeping `E[...] ** alpha_observer`
    differentiable."""
    return prior_table[
        scenario_idx,
        observed_action,
        effort_condition,
        relationship_condition,
        padded_slot,
    ]


@jax.jit
def get_lm_g_padded_desire(
    padded_slot,
    scenario_idx,
    observed_action,
    effort_condition,
    relationship_condition,
    g_padded_table,
):
    """LM-elicited goal-satisfaction g for an arbitrary slot in Study 1a's padded
    action space. g_padded_table has shape (16, 3, 2, 4, MAX_ACTIONS) —
    indexed by (scenario, observed_action, effort_condition, relationship, slot).
    g is desire-free; desire enters as the multiplier in w_v · desire · g."""
    return g_padded_table[
        scenario_idx,
        observed_action,
        effort_condition,
        relationship_condition,
        padded_slot,
    ]


@jax.jit
def get_utility_full_padded_desire(
    padded_slot,
    scenario_idx,
    observed_action,
    effort_condition,
    relationship_condition,
    desire,
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    risk_table,
    effort_table,
    g_padded_table,
):
    intimacy = RELATIONSHIP_LEVEL_VALUES[relationship_condition]
    risk = risk_table[
        scenario_idx,
        observed_action,
        effort_condition,
        relationship_condition,
        padded_slot,
    ]
    effort = effort_table[
        scenario_idx,
        observed_action,
        effort_condition,
        relationship_condition,
        padded_slot,
    ]
    g = get_lm_g_padded_desire(
        padded_slot,
        scenario_idx,
        observed_action,
        effort_condition,
        relationship_condition,
        g_padded_table,
    )
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * desire * g - w_d * risk * jnp.power(one_minus_I, gamma) - w_e * effort
    )


@jax.jit
def get_utility_discomfort_only_padded_desire(
    padded_slot,
    scenario_idx,
    observed_action,
    effort_condition,
    relationship_condition,
    desire,
    alpha,
    w_d,
    gamma,
    risk_table,
    effort_table,
):
    intimacy = RELATIONSHIP_LEVEL_VALUES[relationship_condition]
    risk = risk_table[
        scenario_idx,
        observed_action,
        effort_condition,
        relationship_condition,
        padded_slot,
    ]
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (-w_d * risk * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_base_padded_desire(
    padded_slot,
    scenario_idx,
    observed_action,
    effort_condition,
    relationship_condition,
    desire,
    alpha,
    w_v,
    w_e,
    risk_table,
    effort_table,
    g_padded_table,
):
    effort = effort_table[
        scenario_idx,
        observed_action,
        effort_condition,
        relationship_condition,
        padded_slot,
    ]
    g = get_lm_g_padded_desire(
        padded_slot,
        scenario_idx,
        observed_action,
        effort_condition,
        relationship_condition,
        g_padded_table,
    )
    return alpha * (w_v * desire * g - w_e * effort)


# =============================================================================
# Padded utilities for the migrated studies (1b, 2a, 2b)
# =============================================================================
# Each study's LM-generated alternative set is indexed by the cell grid =
# (scenario, observed_action, <variables the observer/participant sees>). A
# feature gets an extra axis when the variable it depends on is *inferred* (so
# the alt set is shared across that variable's hypotheses but the feature value
# differs): effort gains an effort_condition axis when effort is inferred. g
# (goal-satisfaction) is desire-free, so it carries no desire axis. risk is
# intimacy- and effort-independent by construction, so it is only indexed by the
# cell grid + slot.


# --- Study 1b (joint_de): observer knows intimacy, infers (desire, effort) ---
# Cell grid: (scenario, observed_action, relationship_condition). effort is
# inferred -> effort table carries an effort_condition feature axis. desire is
# inferred, but g is desire-free (no desire axis).
#   risk:   (16, 3, 4, S)            [scenario, obs, relationship, slot]
#   effort: (16, 3, 4, 2, S)         [scenario, obs, relationship, effort_condition, slot]
#   g:      (16, 3, 4, S)            [scenario, obs, relationship, slot]
#   prior:  (16, 3, 4, S)            [scenario, obs, relationship, slot]


@jax.jit
def get_prior_padded_joint_de(
    padded_slot, scenario_idx, observed_action, relationship_condition, prior_table
):
    return prior_table[
        scenario_idx, observed_action, relationship_condition, padded_slot
    ]


@jax.jit
def get_lm_g_padded_joint_de(
    padded_slot,
    scenario_idx,
    observed_action,
    relationship_condition,
    g_padded_table,
):
    """Desire-free goal-satisfaction g for Study 1b's padded action space.
    g_padded_table has shape (16, 3, 4, MAX_ACTIONS) — indexed by
    (scenario, observed_action, relationship, slot)."""
    return g_padded_table[
        scenario_idx,
        observed_action,
        relationship_condition,
        padded_slot,
    ]


@jax.jit
def get_utility_full_padded_joint_de(
    padded_slot,
    scenario_idx,
    observed_action,
    relationship_condition,
    desire,
    effort_condition,
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    risk_table,
    effort_table,
    g_padded_table,
):
    intimacy = RELATIONSHIP_LEVEL_VALUES[relationship_condition]
    risk = risk_table[
        scenario_idx, observed_action, relationship_condition, padded_slot
    ]
    effort = effort_table[
        scenario_idx,
        observed_action,
        relationship_condition,
        effort_condition,
        padded_slot,
    ]
    g = get_lm_g_padded_joint_de(
        padded_slot,
        scenario_idx,
        observed_action,
        relationship_condition,
        g_padded_table,
    )
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * desire * g - w_d * risk * jnp.power(one_minus_I, gamma) - w_e * effort
    )


@jax.jit
def get_utility_discomfort_only_padded_joint_de(
    padded_slot,
    scenario_idx,
    observed_action,
    relationship_condition,
    desire,
    effort_condition,
    alpha,
    w_d,
    gamma,
    risk_table,
    effort_table,
):
    intimacy = RELATIONSHIP_LEVEL_VALUES[relationship_condition]
    risk = risk_table[
        scenario_idx, observed_action, relationship_condition, padded_slot
    ]
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (-w_d * risk * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_base_padded_joint_de(
    padded_slot,
    scenario_idx,
    observed_action,
    relationship_condition,
    desire,
    effort_condition,
    alpha,
    w_v,
    w_e,
    risk_table,
    effort_table,
    g_padded_table,
):
    effort = effort_table[
        scenario_idx,
        observed_action,
        relationship_condition,
        effort_condition,
        padded_slot,
    ]
    g = get_lm_g_padded_joint_de(
        padded_slot,
        scenario_idx,
        observed_action,
        relationship_condition,
        g_padded_table,
    )
    return alpha * (w_v * desire * g - w_e * effort)


# --- Study 2a (intimacy): observer knows (desire, effort), infers intimacy ----
# Cell grid: (scenario, observed_action, desire_condition, effort_condition).
# intimacy is inferred (continuous) -> risk is modulated by (1-I)^gamma in the
# utility but the table has no intimacy axis. desire + effort are observed, so
# both index the cell grid; effort feature is taken at the observed effort.
#   risk:   (16, 3, 2, 2, S)         [scenario, obs, desire, effort, slot]
#   effort: (16, 3, 2, 2, S)         [scenario, obs, desire, effort, slot]
#   g:      (16, 3, 2, 2, S)         [scenario, obs, desire, effort, slot]
#   prior:  (16, 3, 2, 2, S)         [scenario, obs, desire, effort, slot]
# `intimacy_level` is the continuous value the observer's actor hypothesizes.


@jax.jit
def get_prior_padded_intimacy(
    padded_slot,
    scenario_idx,
    observed_action,
    desire_condition,
    effort_condition,
    prior_table,
):
    return prior_table[
        scenario_idx, observed_action, desire_condition, effort_condition, padded_slot
    ]


@jax.jit
def get_lm_g_padded_intimacy(
    padded_slot,
    scenario_idx,
    observed_action,
    desire_condition,
    effort_condition,
    g_padded_table,
):
    """Desire-free goal-satisfaction g for Study 2a's padded action space.
    g_padded_table has shape (16, 3, 2, 2, MAX_ACTIONS) — indexed by
    (scenario, observed_action, desire, effort, slot)."""
    return g_padded_table[
        scenario_idx,
        observed_action,
        desire_condition,
        effort_condition,
        padded_slot,
    ]


@jax.jit
def get_utility_full_padded_intimacy(
    padded_slot,
    scenario_idx,
    observed_action,
    desire_condition,
    effort_condition,
    intimacy_level,
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    risk_table,
    effort_table,
    g_padded_table,
    desire_table,
):
    risk = risk_table[
        scenario_idx, observed_action, desire_condition, effort_condition, padded_slot
    ]
    effort = effort_table[
        scenario_idx, observed_action, desire_condition, effort_condition, padded_slot
    ]
    g = get_lm_g_padded_intimacy(
        padded_slot,
        scenario_idx,
        observed_action,
        desire_condition,
        effort_condition,
        g_padded_table,
    )
    # Desire is given (observer-visible): look up the LM-rated scalar for this
    # scenario × desire condition and use it as the desire multiplier.
    desire = desire_table[scenario_idx, desire_condition]
    one_minus_I = jnp.maximum(1.0 - intimacy_level, 1e-8)
    return alpha * (
        w_v * desire * g - w_d * risk * jnp.power(one_minus_I, gamma) - w_e * effort
    )


@jax.jit
def get_utility_discomfort_only_padded_intimacy(
    padded_slot,
    scenario_idx,
    observed_action,
    desire_condition,
    effort_condition,
    intimacy_level,
    alpha,
    w_d,
    gamma,
    risk_table,
    effort_table,
):
    risk = risk_table[
        scenario_idx, observed_action, desire_condition, effort_condition, padded_slot
    ]
    one_minus_I = jnp.maximum(1.0 - intimacy_level, 1e-8)
    return alpha * (-w_d * risk * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_base_padded_intimacy(
    padded_slot,
    scenario_idx,
    observed_action,
    desire_condition,
    effort_condition,
    intimacy_level,
    alpha,
    w_v,
    w_e,
    risk_table,
    effort_table,
    g_padded_table,
    desire_table,
):
    effort = effort_table[
        scenario_idx, observed_action, desire_condition, effort_condition, padded_slot
    ]
    g = get_lm_g_padded_intimacy(
        padded_slot,
        scenario_idx,
        observed_action,
        desire_condition,
        effort_condition,
        g_padded_table,
    )
    desire = desire_table[scenario_idx, desire_condition]
    return alpha * (w_v * desire * g - w_e * effort)


# --- Study 2b (joint_ie): observer knows desire, infers (intimacy, effort) ----
# Cell grid: (scenario, observed_action, desire_condition). intimacy inferred
# (continuous, no table axis). effort inferred -> effort gains an
# effort_condition feature axis. desire observed -> indexes cell grid + g.
#   risk:   (16, 3, 2, S)            [scenario, obs, desire, slot]
#   effort: (16, 3, 2, 2, S)         [scenario, obs, desire, effort_condition, slot]
#   g:      (16, 3, 2, S)            [scenario, obs, desire, slot]
#   prior:  (16, 3, 2, S)            [scenario, obs, desire, slot]


@jax.jit
def get_prior_padded_joint_ie(
    padded_slot, scenario_idx, observed_action, desire_condition, prior_table
):
    return prior_table[scenario_idx, observed_action, desire_condition, padded_slot]


@jax.jit
def get_lm_g_padded_joint_ie(
    padded_slot, scenario_idx, observed_action, desire_condition, g_padded_table
):
    """Desire-free goal-satisfaction g for Study 2b's padded action space.
    g_padded_table has shape (16, 3, 2, MAX_ACTIONS) — indexed by
    (scenario, observed_action, desire, slot)."""
    return g_padded_table[scenario_idx, observed_action, desire_condition, padded_slot]


@jax.jit
def get_utility_full_padded_joint_ie(
    padded_slot,
    scenario_idx,
    observed_action,
    desire_condition,
    effort_condition,
    intimacy_level,
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    risk_table,
    effort_table,
    g_padded_table,
    desire_table,
):
    risk = risk_table[scenario_idx, observed_action, desire_condition, padded_slot]
    effort = effort_table[
        scenario_idx, observed_action, desire_condition, effort_condition, padded_slot
    ]
    g = get_lm_g_padded_joint_ie(
        padded_slot, scenario_idx, observed_action, desire_condition, g_padded_table
    )
    # Desire is given (observer-visible): use the LM-rated scalar for this
    # scenario × desire condition as the desire multiplier.
    desire = desire_table[scenario_idx, desire_condition]
    one_minus_I = jnp.maximum(1.0 - intimacy_level, 1e-8)
    return alpha * (
        w_v * desire * g - w_d * risk * jnp.power(one_minus_I, gamma) - w_e * effort
    )


@jax.jit
def get_utility_discomfort_only_padded_joint_ie(
    padded_slot,
    scenario_idx,
    observed_action,
    desire_condition,
    effort_condition,
    intimacy_level,
    alpha,
    w_d,
    gamma,
    risk_table,
    effort_table,
):
    risk = risk_table[scenario_idx, observed_action, desire_condition, padded_slot]
    one_minus_I = jnp.maximum(1.0 - intimacy_level, 1e-8)
    return alpha * (-w_d * risk * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_base_padded_joint_ie(
    padded_slot,
    scenario_idx,
    observed_action,
    desire_condition,
    effort_condition,
    intimacy_level,
    alpha,
    w_v,
    w_e,
    risk_table,
    effort_table,
    g_padded_table,
    desire_table,
):
    effort = effort_table[
        scenario_idx, observed_action, desire_condition, effort_condition, padded_slot
    ]
    g = get_lm_g_padded_joint_ie(
        padded_slot, scenario_idx, observed_action, desire_condition, g_padded_table
    )
    desire = desire_table[scenario_idx, desire_condition]
    return alpha * (w_v * desire * g - w_e * effort)
