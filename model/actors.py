"""
Actor policies — plain-JAX softmax policies over the padded LM-alternatives
action space, one per (observer family, ablation variant).

Each function mirrors its memo original in `memo_spec.py`: same name, same
signature, same output axis order (padded_slot first, then the cell axes, then
the latent grid axes). The policy is

    pi(slot | cell, latents)  ∝  prior[cell, slot] · exp(U(slot, cell, latents))

normalized over the slot axis. Null-padded slots carry a tiny epsilon prior
(`tables.NULL_EPSILON` = 1e-8), so they effectively drop out of the softmax
while keeping the observer's sharpening differentiable.

The latent axes are evaluated by broadcasting: every axis of the output table
gets a reserved dimension (`_axis`), the utility functions from `utility.py`
are called once with the broadcast index/value arrays — the same arrays the
memo compiler would enumerate — and the result is expanded to the family's
full table shape (an ablation whose utility ignores a latent is constant along
that axis).

Vintage note: this formulation compiles to a different XLA graph than the memo
actors, so results differ from theirs by ~1 float32 ulp (fusion/FMA choices),
which compounds into small fitted-parameter drift. All committed outputs are a
single vintage regenerated under this code (bin/regenerate-vintage.sh);
`test_model_compliance.py` enforces semantic equivalence with the memo actors
(values + gradients, all twelve variants) at test tolerance.

Three model variants per family: `_full`, `_discomfort_only`, `_base`.

Dependency layer 2: imports from `tables.py` (enums, axes) and `utility.py`
(get_utility_*_padded_*). `observers.py` (the Bayesian inversions) and
`inverse/_reweighting.py` (the surprise term) call these; `memo_spec.py` holds
the memo originals.
"""

import jax
import jax.numpy as jnp

from model.tables import (
    DesireLevels,
    EffortConditions,
    IntimacyLevels,
    ObservedActions,
    PaddedActionSlots,
    RelationshipConditions,
    DesireConditions,
    Scenarios,
)
from model.utility import (
    get_prior_padded_desire,
    get_prior_padded_intimacy,
    get_prior_padded_joint_de,
    get_prior_padded_joint_ie,
    get_utility_base_padded_desire,
    get_utility_base_padded_intimacy,
    get_utility_base_padded_joint_de,
    get_utility_base_padded_joint_ie,
    get_utility_discomfort_only_padded_desire,
    get_utility_discomfort_only_padded_intimacy,
    get_utility_discomfort_only_padded_joint_de,
    get_utility_discomfort_only_padded_joint_ie,
    get_utility_full_padded_desire,
    get_utility_full_padded_intimacy,
    get_utility_full_padded_joint_de,
    get_utility_full_padded_joint_ie,
)


def _axis(values, ndim, axis):
    """A 1-D axis grid reshaped to broadcast along its reserved output
    dimension: all-1 shape except `axis`. Index axes pass an arange over the
    enum; value axes (the continuous latent grids) pass the grid itself."""
    arr = jnp.asarray(values)
    shape = [1] * ndim
    shape[axis] = arr.shape[0]
    return arr.reshape(shape)


def _slot_policy(prior, utility, shape):
    """Softmax of `prior · exp(utility)` over the slot axis (axis 0),
    broadcast to the family's full table shape (`prior` and `utility` may be
    constant — size 1 — along latent axes their variant ignores).

    `nan_to_num` mirrors the memo spec's guard: a cell whose every slot
    underflows (or overflows) in float32 at extreme parameters yields an
    all-zero policy rather than NaN. The log-space observers mask zeros and
    NaN alike, but `_reweighting.action_surprise` consumes the raw policy —
    without the guard a pathological cold-start draw would poison that
    restart's loss with NaN instead of a large-but-finite surprise."""
    w = prior * jnp.exp(utility)
    return jnp.broadcast_to(jnp.nan_to_num(w / w.sum(axis=0, keepdims=True)), shape)


# Enum-axis index grids (value grids are DesireLevels / IntimacyLevels).
_SLOTS = jnp.arange(len(PaddedActionSlots))
_SCENARIOS = jnp.arange(len(Scenarios))
_OBSERVED = jnp.arange(len(ObservedActions))
_EFFORTS = jnp.arange(len(EffortConditions))
_RELATIONSHIPS = jnp.arange(len(RelationshipConditions))
_DESIRE_CONDS = jnp.arange(len(DesireConditions))


# ==============================================================================
# Study 1a (desire): axes (slot, scenario, observed_action, effort,
# relationship, desire)
# ==============================================================================

_D_SLOT = _axis(_SLOTS, 6, 0)
_D_SCEN = _axis(_SCENARIOS, 6, 1)
_D_OBS = _axis(_OBSERVED, 6, 2)
_D_EFF = _axis(_EFFORTS, 6, 3)
_D_REL = _axis(_RELATIONSHIPS, 6, 4)
_D_DESIRE = _axis(DesireLevels, 6, 5)
# Full table shape, derived from the reserved-dimension axis constants so the
# two can never drift apart (each axis occupies its own dimension, so the
# broadcast of their shapes IS the table shape).
_DESIRE_SHAPE = jnp.broadcast_shapes(
    *(a.shape for a in (_D_SLOT, _D_SCEN, _D_OBS, _D_EFF, _D_REL, _D_DESIRE))
)


@jax.jit
def actor_discrete_full_padded_desire(
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    relationship_values,
):
    """Study 1a full-utility actor policy; mirrors
    `memo_spec.actor_discrete_full_padded_desire`."""
    prior = get_prior_padded_desire(
        _D_SLOT, _D_SCEN, _D_OBS, _D_EFF, _D_REL, prior_table
    )
    utility = get_utility_full_padded_desire(
        _D_SLOT,
        _D_SCEN,
        _D_OBS,
        _D_EFF,
        _D_REL,
        _D_DESIRE,
        alpha,
        w_v,
        w_d,
        w_e,
        gamma,
        risk_table,
        effort_table,
        g_padded_table,
        relationship_values,
    )
    return _slot_policy(prior, utility, _DESIRE_SHAPE)


@jax.jit
def actor_discrete_discomfort_only_padded_desire(
    alpha,
    w_d,
    gamma,
    risk_table,
    effort_table,
    prior_table,
    relationship_values,
):
    """Study 1a discomfort-only actor policy; mirrors
    `memo_spec.actor_discrete_discomfort_only_padded_desire`."""
    prior = get_prior_padded_desire(
        _D_SLOT, _D_SCEN, _D_OBS, _D_EFF, _D_REL, prior_table
    )
    utility = get_utility_discomfort_only_padded_desire(
        _D_SLOT,
        _D_SCEN,
        _D_OBS,
        _D_EFF,
        _D_REL,
        _D_DESIRE,
        alpha,
        w_d,
        gamma,
        risk_table,
        effort_table,
        relationship_values,
    )
    return _slot_policy(prior, utility, _DESIRE_SHAPE)


@jax.jit
def actor_discrete_base_padded_desire(
    alpha,
    w_v,
    w_e,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
):
    """Study 1a base actor policy; mirrors
    `memo_spec.actor_discrete_base_padded_desire`."""
    prior = get_prior_padded_desire(
        _D_SLOT, _D_SCEN, _D_OBS, _D_EFF, _D_REL, prior_table
    )
    utility = get_utility_base_padded_desire(
        _D_SLOT,
        _D_SCEN,
        _D_OBS,
        _D_EFF,
        _D_REL,
        _D_DESIRE,
        alpha,
        w_v,
        w_e,
        risk_table,
        effort_table,
        g_padded_table,
    )
    return _slot_policy(prior, utility, _DESIRE_SHAPE)


# ==============================================================================
# Studies 1b/3a (joint_de): axes (slot, scenario, observed_action,
# relationship, desire, effort)
# ==============================================================================

_JDE_SLOT = _axis(_SLOTS, 6, 0)
_JDE_SCEN = _axis(_SCENARIOS, 6, 1)
_JDE_OBS = _axis(_OBSERVED, 6, 2)
_JDE_REL = _axis(_RELATIONSHIPS, 6, 3)
_JDE_DESIRE = _axis(DesireLevels, 6, 4)
_JDE_EFF = _axis(_EFFORTS, 6, 5)
_JOINT_DE_SHAPE = jnp.broadcast_shapes(
    *(
        a.shape
        for a in (_JDE_SLOT, _JDE_SCEN, _JDE_OBS, _JDE_REL, _JDE_DESIRE, _JDE_EFF)
    )
)


@jax.jit
def actor_discrete_full_padded_joint_de(
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    relationship_values,
):
    """Study 1b/3a full-utility actor policy; mirrors
    `memo_spec.actor_discrete_full_padded_joint_de`."""
    prior = get_prior_padded_joint_de(
        _JDE_SLOT, _JDE_SCEN, _JDE_OBS, _JDE_REL, prior_table
    )
    utility = get_utility_full_padded_joint_de(
        _JDE_SLOT,
        _JDE_SCEN,
        _JDE_OBS,
        _JDE_REL,
        _JDE_DESIRE,
        _JDE_EFF,
        alpha,
        w_v,
        w_d,
        w_e,
        gamma,
        risk_table,
        effort_table,
        g_padded_table,
        relationship_values,
    )
    return _slot_policy(prior, utility, _JOINT_DE_SHAPE)


@jax.jit
def actor_discrete_discomfort_only_padded_joint_de(
    alpha,
    w_d,
    gamma,
    risk_table,
    effort_table,
    prior_table,
    relationship_values,
):
    """Study 1b/3a discomfort-only actor policy; mirrors
    `memo_spec.actor_discrete_discomfort_only_padded_joint_de`."""
    prior = get_prior_padded_joint_de(
        _JDE_SLOT, _JDE_SCEN, _JDE_OBS, _JDE_REL, prior_table
    )
    utility = get_utility_discomfort_only_padded_joint_de(
        _JDE_SLOT,
        _JDE_SCEN,
        _JDE_OBS,
        _JDE_REL,
        _JDE_DESIRE,
        _JDE_EFF,
        alpha,
        w_d,
        gamma,
        risk_table,
        effort_table,
        relationship_values,
    )
    return _slot_policy(prior, utility, _JOINT_DE_SHAPE)


@jax.jit
def actor_discrete_base_padded_joint_de(
    alpha,
    w_v,
    w_e,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
):
    """Study 1b/3a base actor policy; mirrors
    `memo_spec.actor_discrete_base_padded_joint_de`."""
    prior = get_prior_padded_joint_de(
        _JDE_SLOT, _JDE_SCEN, _JDE_OBS, _JDE_REL, prior_table
    )
    utility = get_utility_base_padded_joint_de(
        _JDE_SLOT,
        _JDE_SCEN,
        _JDE_OBS,
        _JDE_REL,
        _JDE_DESIRE,
        _JDE_EFF,
        alpha,
        w_v,
        w_e,
        risk_table,
        effort_table,
        g_padded_table,
    )
    return _slot_policy(prior, utility, _JOINT_DE_SHAPE)


# ==============================================================================
# Study 2a (intimacy): axes (slot, scenario, observed_action, desire_condition,
# effort, relationship)
# ==============================================================================

_I_SLOT = _axis(_SLOTS, 6, 0)
_I_SCEN = _axis(_SCENARIOS, 6, 1)
_I_OBS = _axis(_OBSERVED, 6, 2)
_I_DES = _axis(_DESIRE_CONDS, 6, 3)
_I_EFF = _axis(_EFFORTS, 6, 4)
_I_REL = _axis(IntimacyLevels, 6, 5)
_INTIMACY_SHAPE = jnp.broadcast_shapes(
    *(a.shape for a in (_I_SLOT, _I_SCEN, _I_OBS, _I_DES, _I_EFF, _I_REL))
)


@jax.jit
def actor_continuous_full_padded_intimacy(
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    desire_table,
):
    """Study 2a full-utility actor policy; mirrors
    `memo_spec.actor_continuous_full_padded_intimacy`."""
    prior = get_prior_padded_intimacy(
        _I_SLOT, _I_SCEN, _I_OBS, _I_DES, _I_EFF, prior_table
    )
    utility = get_utility_full_padded_intimacy(
        _I_SLOT,
        _I_SCEN,
        _I_OBS,
        _I_DES,
        _I_EFF,
        _I_REL,
        alpha,
        w_v,
        w_d,
        w_e,
        gamma,
        risk_table,
        effort_table,
        g_padded_table,
        desire_table,
    )
    return _slot_policy(prior, utility, _INTIMACY_SHAPE)


@jax.jit
def actor_continuous_discomfort_only_padded_intimacy(
    alpha, w_d, gamma, risk_table, effort_table, prior_table
):
    """Study 2a discomfort-only actor policy; mirrors
    `memo_spec.actor_continuous_discomfort_only_padded_intimacy`."""
    prior = get_prior_padded_intimacy(
        _I_SLOT, _I_SCEN, _I_OBS, _I_DES, _I_EFF, prior_table
    )
    utility = get_utility_discomfort_only_padded_intimacy(
        _I_SLOT,
        _I_SCEN,
        _I_OBS,
        _I_DES,
        _I_EFF,
        _I_REL,
        alpha,
        w_d,
        gamma,
        risk_table,
        effort_table,
    )
    return _slot_policy(prior, utility, _INTIMACY_SHAPE)


@jax.jit
def actor_continuous_base_padded_intimacy(
    alpha,
    w_v,
    w_e,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    desire_table,
):
    """Study 2a base actor policy; mirrors
    `memo_spec.actor_continuous_base_padded_intimacy`."""
    prior = get_prior_padded_intimacy(
        _I_SLOT, _I_SCEN, _I_OBS, _I_DES, _I_EFF, prior_table
    )
    utility = get_utility_base_padded_intimacy(
        _I_SLOT,
        _I_SCEN,
        _I_OBS,
        _I_DES,
        _I_EFF,
        _I_REL,
        alpha,
        w_v,
        w_e,
        risk_table,
        effort_table,
        g_padded_table,
        desire_table,
    )
    return _slot_policy(prior, utility, _INTIMACY_SHAPE)


# ==============================================================================
# Studies 2b/3b (joint_ie): axes (slot, scenario, observed_action,
# desire_condition, relationship, effort)
# ==============================================================================
# NOTE the utility signatures take effort_condition BEFORE intimacy_level, while
# the declared axis order puts relationship before effort — so the reserved
# dimensions (from the axis order) and the call order (from the signature)
# differ here.

_JIE_SLOT = _axis(_SLOTS, 6, 0)
_JIE_SCEN = _axis(_SCENARIOS, 6, 1)
_JIE_OBS = _axis(_OBSERVED, 6, 2)
_JIE_DES = _axis(_DESIRE_CONDS, 6, 3)
_JIE_REL = _axis(IntimacyLevels, 6, 4)
_JIE_EFF = _axis(_EFFORTS, 6, 5)
_JOINT_IE_SHAPE = jnp.broadcast_shapes(
    *(a.shape for a in (_JIE_SLOT, _JIE_SCEN, _JIE_OBS, _JIE_DES, _JIE_REL, _JIE_EFF))
)


@jax.jit
def actor_continuous_full_padded_joint_ie(
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    desire_table,
):
    """Study 2b/3b full-utility actor policy; mirrors
    `memo_spec.actor_continuous_full_padded_joint_ie`."""
    prior = get_prior_padded_joint_ie(
        _JIE_SLOT, _JIE_SCEN, _JIE_OBS, _JIE_DES, prior_table
    )
    utility = get_utility_full_padded_joint_ie(
        _JIE_SLOT,
        _JIE_SCEN,
        _JIE_OBS,
        _JIE_DES,
        _JIE_EFF,
        _JIE_REL,
        alpha,
        w_v,
        w_d,
        w_e,
        gamma,
        risk_table,
        effort_table,
        g_padded_table,
        desire_table,
    )
    return _slot_policy(prior, utility, _JOINT_IE_SHAPE)


@jax.jit
def actor_continuous_discomfort_only_padded_joint_ie(
    alpha, w_d, gamma, risk_table, effort_table, prior_table
):
    """Study 2b/3b discomfort-only actor policy; mirrors
    `memo_spec.actor_continuous_discomfort_only_padded_joint_ie`."""
    prior = get_prior_padded_joint_ie(
        _JIE_SLOT, _JIE_SCEN, _JIE_OBS, _JIE_DES, prior_table
    )
    utility = get_utility_discomfort_only_padded_joint_ie(
        _JIE_SLOT,
        _JIE_SCEN,
        _JIE_OBS,
        _JIE_DES,
        _JIE_EFF,
        _JIE_REL,
        alpha,
        w_d,
        gamma,
        risk_table,
        effort_table,
    )
    return _slot_policy(prior, utility, _JOINT_IE_SHAPE)


@jax.jit
def actor_continuous_base_padded_joint_ie(
    alpha,
    w_v,
    w_e,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    desire_table,
):
    """Study 2b/3b base actor policy; mirrors
    `memo_spec.actor_continuous_base_padded_joint_ie`."""
    prior = get_prior_padded_joint_ie(
        _JIE_SLOT, _JIE_SCEN, _JIE_OBS, _JIE_DES, prior_table
    )
    utility = get_utility_base_padded_joint_ie(
        _JIE_SLOT,
        _JIE_SCEN,
        _JIE_OBS,
        _JIE_DES,
        _JIE_EFF,
        _JIE_REL,
        alpha,
        w_v,
        w_e,
        risk_table,
        effort_table,
        g_padded_table,
        desire_table,
    )
    return _slot_policy(prior, utility, _JOINT_IE_SHAPE)
