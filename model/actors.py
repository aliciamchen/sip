"""
Actor policies — plain-JAX softmax policies over the padded LM-alternatives
action space, one per (observer family, ablation variant).

Each function mirrors its memo original in `memo_spec.py` exactly: same name,
same signature, same output axis order (padded_slot first, then the cell axes,
then the latent grid axes). The policy is

    pi(slot | cell, latents)  ∝  prior[cell, slot] · exp(U(slot, cell, latents))

normalized over the slot axis. Null-padded slots carry a tiny epsilon prior
(`tables.NULL_EPSILON` = 1e-8), so they effectively drop out of the softmax
while keeping the observer's sharpening differentiable.

BIT-EXACTNESS CONSTRAINT. These functions deliberately transliterate the JAX
code the memo compiler generates (`_mesh_eval` = memo.lib's ffi/construct_vmap:
broadcast the axis meshes dense, flatten, vmap the scalar table function;
`_slot_policy` = the generated normalize-and-layout epilogue, including its
reversed internal axis order and final transpose). A mathematically identical
but differently-shaped formulation — e.g. batched advanced indexing on
reserved-dimension index arrays — produces ~1-ulp float32 drift through
different XLA fusion/FMA choices (verified 2026-08-23 on the fitted losses and
gradients of all six studies), which would force regenerating every committed
fit and CV output. The transliteration instead reproduces the memo actors
bitwise, so the committed output vintage carries over unchanged. Don't
"simplify" the mesh evaluation or the epilogue without re-verifying bitwise
agreement against `memo_spec` on the real tables at the fitted parameters.

Three model variants per family: `_full`, `_discomfort_only`, `_base`.

Dependency layer 2: imports from `tables.py` (enums, axes) and `utility.py`
(get_utility_*_padded_*). `observers.py` (the Bayesian inversions) and
`inverse/_reweighting.py` (the surprise term) call these; `memo_spec.py` holds
the memo originals and `test_model_compliance.py` enforces policy ≡ memo on
every variant.
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


def _mesh_eval(fn, mesh_args, static_args):
    """Evaluate a scalar-in-scalar-out table function over broadcast axis
    meshes, exactly the way the memo compiler does (memo.lib.ffi →
    construct_vmap): broadcast the mesh args to their common shape, flatten,
    vmap the scalar function over them with the parameters and tables closed
    over, and reshape back."""
    target = jnp.broadcast_shapes(*[m.shape for m in mesh_args])
    flat = [jnp.broadcast_to(m, target).reshape(-1) for m in mesh_args]
    out = jax.vmap(lambda *ms: fn(*ms, *static_args))(*flat)
    return out.reshape(target)


def _memo_axes(domains):
    """The memo compiler's internal mesh layout for one actor's declared axes.

    `domains[k]` is the domain of declared axis k (an IntEnum class for index
    axes, a grid array for the continuous latents); axis 0 is the padded slot.
    The compiler gives declared axis k the reshape (-1, 1×k) — so the declared
    order is REVERSED in array dims — and the slot CHOICE variable a leading
    seventh dim. Returns (choice_mesh, per-axis meshes for axes 1..n-1)."""
    n = len(domains)
    meshes = [
        jnp.array(list(d) if isinstance(d, type) else d).reshape((-1,) + (1,) * k)
        for k, d in enumerate(domains)
    ]
    choice = jnp.array(list(domains[0])).reshape((-1,) + (1,) * n)
    return choice, meshes[1:]


def _slot_policy(choice, prior, utility):
    """Normalize `prior · exp(utility)` over the slot-choice dim and return the
    policy in the declared axis order — a transliteration of the memo
    compiler's epilogue, kept op-for-op (the multiply by ones, nan_to_num, and
    the swap/squeeze/transpose layout dance included) so the compiled graph,
    and therefore every float32 rounding, matches the memo actor exactly.
    In this layout the slot choice is dim -7 and the declared axes sit
    reversed in dims -6..-1 (the size-1 dim -1 is the compiler's unused
    output-slot placeholder)."""
    op_mul = prior * jnp.exp(utility)
    ll = jnp.ones(jnp.broadcast_shapes(choice.shape), dtype=jnp.float32) * op_mul
    ll = jnp.nan_to_num(ll / ll.sum(axis=-7, keepdims=True))
    post = ll * 1.0
    post = jnp.swapaxes(post, -7, -1)
    return post.squeeze(axis=(-7,)).transpose()


# ==============================================================================
# Study 1a (desire): axes (slot, scenario, observed_action, effort,
# relationship, desire)
# ==============================================================================

_D_CHOICE, (_D_SCEN, _D_OBS, _D_EFF, _D_REL, _D_DESIRE) = _memo_axes(
    [
        PaddedActionSlots,
        Scenarios,
        ObservedActions,
        EffortConditions,
        RelationshipConditions,
        DesireLevels,
    ]
)
_D_PRIOR_MESH = (_D_CHOICE, _D_SCEN, _D_OBS, _D_EFF, _D_REL)
_D_MESH = (_D_CHOICE, _D_SCEN, _D_OBS, _D_EFF, _D_REL, _D_DESIRE)


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
    prior = _mesh_eval(get_prior_padded_desire, _D_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_full_padded_desire,
        _D_MESH,
        (
            alpha,
            w_v,
            w_d,
            w_e,
            gamma,
            risk_table,
            effort_table,
            g_padded_table,
            relationship_values,
        ),
    )
    return _slot_policy(_D_CHOICE, prior, utility)


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
    prior = _mesh_eval(get_prior_padded_desire, _D_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_discomfort_only_padded_desire,
        _D_MESH,
        (alpha, w_d, gamma, risk_table, effort_table, relationship_values),
    )
    return _slot_policy(_D_CHOICE, prior, utility)


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
    prior = _mesh_eval(get_prior_padded_desire, _D_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_base_padded_desire,
        _D_MESH,
        (alpha, w_v, w_e, risk_table, effort_table, g_padded_table),
    )
    return _slot_policy(_D_CHOICE, prior, utility)


# ==============================================================================
# Studies 1b/3a (joint_de): axes (slot, scenario, observed_action,
# relationship, desire, effort)
# ==============================================================================

_JDE_CHOICE, (_JDE_SCEN, _JDE_OBS, _JDE_REL, _JDE_DESIRE, _JDE_EFF) = _memo_axes(
    [
        PaddedActionSlots,
        Scenarios,
        ObservedActions,
        RelationshipConditions,
        DesireLevels,
        EffortConditions,
    ]
)
_JDE_PRIOR_MESH = (_JDE_CHOICE, _JDE_SCEN, _JDE_OBS, _JDE_REL)
_JDE_MESH = (_JDE_CHOICE, _JDE_SCEN, _JDE_OBS, _JDE_REL, _JDE_DESIRE, _JDE_EFF)


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
    prior = _mesh_eval(get_prior_padded_joint_de, _JDE_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_full_padded_joint_de,
        _JDE_MESH,
        (
            alpha,
            w_v,
            w_d,
            w_e,
            gamma,
            risk_table,
            effort_table,
            g_padded_table,
            relationship_values,
        ),
    )
    return _slot_policy(_JDE_CHOICE, prior, utility)


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
    prior = _mesh_eval(get_prior_padded_joint_de, _JDE_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_discomfort_only_padded_joint_de,
        _JDE_MESH,
        (alpha, w_d, gamma, risk_table, effort_table, relationship_values),
    )
    return _slot_policy(_JDE_CHOICE, prior, utility)


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
    prior = _mesh_eval(get_prior_padded_joint_de, _JDE_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_base_padded_joint_de,
        _JDE_MESH,
        (alpha, w_v, w_e, risk_table, effort_table, g_padded_table),
    )
    return _slot_policy(_JDE_CHOICE, prior, utility)


# ==============================================================================
# Study 2a (intimacy): axes (slot, scenario, observed_action, desire_condition,
# effort, relationship)
# ==============================================================================

_I_CHOICE, (_I_SCEN, _I_OBS, _I_DES, _I_EFF, _I_REL) = _memo_axes(
    [
        PaddedActionSlots,
        Scenarios,
        ObservedActions,
        DesireConditions,
        EffortConditions,
        IntimacyLevels,
    ]
)
_I_PRIOR_MESH = (_I_CHOICE, _I_SCEN, _I_OBS, _I_DES, _I_EFF)
_I_MESH = (_I_CHOICE, _I_SCEN, _I_OBS, _I_DES, _I_EFF, _I_REL)


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
    prior = _mesh_eval(get_prior_padded_intimacy, _I_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_full_padded_intimacy,
        _I_MESH,
        (
            alpha,
            w_v,
            w_d,
            w_e,
            gamma,
            risk_table,
            effort_table,
            g_padded_table,
            desire_table,
        ),
    )
    return _slot_policy(_I_CHOICE, prior, utility)


@jax.jit
def actor_continuous_discomfort_only_padded_intimacy(
    alpha, w_d, gamma, risk_table, effort_table, prior_table
):
    """Study 2a discomfort-only actor policy; mirrors
    `memo_spec.actor_continuous_discomfort_only_padded_intimacy`."""
    prior = _mesh_eval(get_prior_padded_intimacy, _I_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_discomfort_only_padded_intimacy,
        _I_MESH,
        (alpha, w_d, gamma, risk_table, effort_table),
    )
    return _slot_policy(_I_CHOICE, prior, utility)


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
    prior = _mesh_eval(get_prior_padded_intimacy, _I_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_base_padded_intimacy,
        _I_MESH,
        (alpha, w_v, w_e, risk_table, effort_table, g_padded_table, desire_table),
    )
    return _slot_policy(_I_CHOICE, prior, utility)


# ==============================================================================
# Studies 2b/3b (joint_ie): axes (slot, scenario, observed_action,
# desire_condition, relationship, effort)
# ==============================================================================
# NOTE the utility signatures take effort_condition BEFORE intimacy_level, while
# the declared axis order puts relationship before effort — so the mesh ranks
# (from the declared order) and the call order (from the signature) differ here.

_JIE_CHOICE, (_JIE_SCEN, _JIE_OBS, _JIE_DES, _JIE_REL, _JIE_EFF) = _memo_axes(
    [
        PaddedActionSlots,
        Scenarios,
        ObservedActions,
        DesireConditions,
        IntimacyLevels,
        EffortConditions,
    ]
)
_JIE_PRIOR_MESH = (_JIE_CHOICE, _JIE_SCEN, _JIE_OBS, _JIE_DES)
_JIE_MESH = (_JIE_CHOICE, _JIE_SCEN, _JIE_OBS, _JIE_DES, _JIE_EFF, _JIE_REL)


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
    prior = _mesh_eval(get_prior_padded_joint_ie, _JIE_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_full_padded_joint_ie,
        _JIE_MESH,
        (
            alpha,
            w_v,
            w_d,
            w_e,
            gamma,
            risk_table,
            effort_table,
            g_padded_table,
            desire_table,
        ),
    )
    return _slot_policy(_JIE_CHOICE, prior, utility)


@jax.jit
def actor_continuous_discomfort_only_padded_joint_ie(
    alpha, w_d, gamma, risk_table, effort_table, prior_table
):
    """Study 2b/3b discomfort-only actor policy; mirrors
    `memo_spec.actor_continuous_discomfort_only_padded_joint_ie`."""
    prior = _mesh_eval(get_prior_padded_joint_ie, _JIE_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_discomfort_only_padded_joint_ie,
        _JIE_MESH,
        (alpha, w_d, gamma, risk_table, effort_table),
    )
    return _slot_policy(_JIE_CHOICE, prior, utility)


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
    prior = _mesh_eval(get_prior_padded_joint_ie, _JIE_PRIOR_MESH, (prior_table,))
    utility = _mesh_eval(
        get_utility_base_padded_joint_ie,
        _JIE_MESH,
        (alpha, w_v, w_e, risk_table, effort_table, g_padded_table, desire_table),
    )
    return _slot_policy(_JIE_CHOICE, prior, utility)
