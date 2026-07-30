"""
Observer models — the inverse-planning Bayesian inference layer. One family
per active study, each over the padded LM-alternatives action space:
  - `observer_intimacy_*` (Study 2a) — knows (desire, effort), infers intimacy.
  - `observer_desire_*`   (Study 1a) — knows (effort, intimacy), infers desire.
  - `observer_joint_de_*` (Studies 1b/3a) — infers desire + effort.
  - `observer_joint_ie_*` (Studies 2b/3b) — infers intimacy + effort.

Three model variants per observer: `_full`, `_discomfort_only`, `_base`.

All twelve observers are plain-JAX Bayesian inversions of the actor memos,
computed in log space (`_sharpened_posterior_logspace`): the observer posterior
is a masked softmax of alpha_observer * log(actor policy) over the latent axes.

Two reasons the plain-JAX form is used rather than the memo-generated one, both
of which the generated code cannot be made to avoid from outside:

  - Memory. The memo joint indicator expectation compiles to a ~202 x 202
    latent cross-product per cell, ~7.5 GB of XLA temps per gradient step at
    K = 20.
  - Numerics. Memo's `wpp = E[latent == z] ** alpha_observer` raises a
    normalized row to a power, which underflows in float32: an entire diffuse
    latent row collapses to zero above alpha_observer ~ 15-20, silently fencing
    the optimizer out of that region. Log space subtracts the row max before
    exponentiating and is exact at any alpha.

The memo observers are kept as `_*_memo_reference` — the authoritative statement
of the model semantics; CHANGE MODEL SEMANTICS IN BOTH — and
`test_model_compliance.py` verifies the fast path against them on every variant,
in the parameter regime where the references are numerically healthy.

Dependency layer 3: imports from `tables.py` and `actors.py`.
"""

import jax.numpy as jnp
from memo import memo

from actors import (
    actor_continuous_base_padded_intimacy,
    actor_continuous_base_padded_joint_ie,
    actor_continuous_discomfort_only_padded_intimacy,
    actor_continuous_discomfort_only_padded_joint_ie,
    actor_continuous_full_padded_intimacy,
    actor_continuous_full_padded_joint_ie,
    actor_discrete_base_padded_desire,
    actor_discrete_base_padded_joint_de,
    actor_discrete_discomfort_only_padded_desire,
    actor_discrete_discomfort_only_padded_joint_de,
    actor_discrete_full_padded_desire,
    actor_discrete_full_padded_joint_de,
)
from tables import (
    DesireLevels,
    EffortConditions,
    IntimacyLevels,
    ObservedActions,
    PaddedActionSlots,
    RelationshipConditions,
    DesireConditions,
    Scenarios,
)


# ==============================================================================
# Single-target observers (Studies 1a, 2a)
# ==============================================================================
# These are the alt-shown observers for the new inverse-planning experiments.
# Padded variants (open-world action space with LM-generated counterfactuals
# in slots 1..k) come in a later step once `lm_alternatives_*.csv` exists.
#
# Study 2a — observer knows (desire, effort), infers intimacy.
# Study 1a — observer knows (effort, intimacy), infers desire.


# --- Study 2a: infer intimacy (LM-generated alternatives) --------------------
# Observer knows (desire, effort); infers intimacy (continuous, 101 bins). The
# actor reasons over the padded LM-alternatives action space; slot 0 is the
# observed action. Output dims:
#   (padded_slot, scenario, observed_action, desire, effort, relationship)
# and the fit/CV slice slot 0.


@memo
def _observer_intimacy_full_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    desire_condition: DesireConditions,
    effort_condition: EffortConditions,
    relationship: IntimacyLevels,
](
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    desire_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(desire_condition)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(desire_condition),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_full_padded_intimacy[
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                effort_condition,
                relationship,
            ](
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
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def _observer_intimacy_discomfort_only_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    desire_condition: DesireConditions,
    effort_condition: EffortConditions,
    relationship: IntimacyLevels,
](
    alpha,
    w_d,
    gamma,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(desire_condition)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(desire_condition),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_discomfort_only_padded_intimacy[
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                effort_condition,
                relationship,
            ](alpha, w_d, gamma, risk_table, effort_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def _observer_intimacy_base_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    desire_condition: DesireConditions,
    effort_condition: EffortConditions,
    relationship: IntimacyLevels,
](
    alpha,
    w_v,
    w_e,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    desire_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(desire_condition)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(desire_condition),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_base_padded_intimacy[
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                effort_condition,
                relationship,
            ](
                alpha,
                w_v,
                w_e,
                risk_table,
                effort_table,
                g_padded_table,
                prior_table,
                desire_table,
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# --- Study 1a: infer desire, LM-generated alternatives ----------------------
# Observer knows scenario, observed_action, effort_condition,
# relationship_condition; the latent is desire (continuous, over DesireLevels).
# The LM generates plausible alternatives per (scenario, observed_action,
# effort_condition, relationship_condition) cell, padded to MAX_ACTIONS with the
# observed action in slot 0.


@memo
def _observer_desire_full_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    effort_condition: EffortConditions,
    relationship_condition: RelationshipConditions,
    desire: DesireLevels,
](
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    relationship_values: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(effort_condition)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(effort_condition),
        actor : knows(relationship_condition),
        actor : chooses(desire in DesireLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_discrete_full_padded_desire[
                padded_slot,
                scenario_idx,
                observed_action,
                effort_condition,
                relationship_condition,
                desire,
            ](
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
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        desire in DesireLevels,
        wpp=E[actor.desire == desire] ** alpha_observer,
    )
    return Pr[observer.desire == desire]


@memo
def _observer_desire_discomfort_only_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    effort_condition: EffortConditions,
    relationship_condition: RelationshipConditions,
    desire: DesireLevels,
](
    alpha,
    w_d,
    gamma,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    prior_table: ...,
    relationship_values: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(effort_condition)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(effort_condition),
        actor : knows(relationship_condition),
        actor : chooses(desire in DesireLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_discrete_discomfort_only_padded_desire[
                padded_slot,
                scenario_idx,
                observed_action,
                effort_condition,
                relationship_condition,
                desire,
            ](
                alpha,
                w_d,
                gamma,
                risk_table,
                effort_table,
                prior_table,
                relationship_values,
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        desire in DesireLevels,
        wpp=E[actor.desire == desire] ** alpha_observer,
    )
    return Pr[observer.desire == desire]


@memo
def _observer_desire_base_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    effort_condition: EffortConditions,
    relationship_condition: RelationshipConditions,
    desire: DesireLevels,
](
    alpha,
    w_v,
    w_e,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(effort_condition)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(effort_condition),
        actor : knows(relationship_condition),
        actor : chooses(desire in DesireLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_discrete_base_padded_desire[
                padded_slot,
                scenario_idx,
                observed_action,
                effort_condition,
                relationship_condition,
                desire,
            ](
                alpha,
                w_v,
                w_e,
                risk_table,
                effort_table,
                g_padded_table,
                prior_table,
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        desire in DesireLevels,
        wpp=E[actor.desire == desire] ** alpha_observer,
    )
    return Pr[observer.desire == desire]


# ==============================================================================
# Joint-target observers (Studies 1b/3a, 2b/3b) — memo REFERENCE implementations
# ==============================================================================
# The `_*_memo_reference` functions below are the original memo statements of
# the joint observers: the actor draws its two latents uniformly, chooses a
# slot via its softmax policy, the observer conditions on the slot and chooses
# the latent pair with wpp = E[joint indicator]^alpha_observer. They are the
# authoritative semantics, but memo compiles that joint indicator expectation
# into an outer × inner latent cross-product — (101·2)² per (cell × slot ×
# run), ~7.5 GB of XLA temps per K=20 gradient step — which made the joint
# fits and CV memory-bound. They are NOT used by fits or CV; the fast
# equivalents below (`observer_joint_*`) are, and the compliance suite proves
# the two agree everywhere. Keep both in sync if the model semantics change.
#
# Study 1b/3a — observer knows intimacy, jointly infers (desire, effort).
# Study 2b/3b — observer knows desire, jointly infers (intimacy, effort).
#
# Downstream code marginalizes the returned joint over each axis to produce
# the per-slider predictions matching the two ratings participants give.


# --- Study 1b: joint over (desire, effort) given intimacy (LM alternatives) --
# Observer knows intimacy; infers (desire, effort). Padded LM-alternatives
# action space; slot 0 is the observed action. Output dims:
#   (padded_slot, scenario, observed_action, relationship, desire, effort)
# and the fit/CV slice slot 0, returning the joint posterior over (desire, effort).


@memo
def _observer_joint_de_full_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    desire: DesireLevels,
    effort_condition: EffortConditions,
](
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    relationship_values: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(relationship_condition),
        actor : chooses(
            desire in DesireLevels,
            effort_condition in EffortConditions,
            wpp=1,
        ),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_discrete_full_padded_joint_de[
                padded_slot,
                scenario_idx,
                observed_action,
                relationship_condition,
                desire,
                effort_condition,
            ](
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
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        desire in DesireLevels,
        effort_condition in EffortConditions,
        wpp=E[(actor.desire == desire) * (actor.effort_condition == effort_condition)]
        ** alpha_observer,
    )
    return Pr[
        observer.desire == desire,
        observer.effort_condition == effort_condition,
    ]


@memo
def _observer_joint_de_discomfort_only_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    desire: DesireLevels,
    effort_condition: EffortConditions,
](
    alpha,
    w_d,
    gamma,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    prior_table: ...,
    relationship_values: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(relationship_condition),
        actor : chooses(
            desire in DesireLevels,
            effort_condition in EffortConditions,
            wpp=1,
        ),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_discrete_discomfort_only_padded_joint_de[
                padded_slot,
                scenario_idx,
                observed_action,
                relationship_condition,
                desire,
                effort_condition,
            ](
                alpha,
                w_d,
                gamma,
                risk_table,
                effort_table,
                prior_table,
                relationship_values,
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        desire in DesireLevels,
        effort_condition in EffortConditions,
        wpp=E[(actor.desire == desire) * (actor.effort_condition == effort_condition)]
        ** alpha_observer,
    )
    return Pr[
        observer.desire == desire,
        observer.effort_condition == effort_condition,
    ]


@memo
def _observer_joint_de_base_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    desire: DesireLevels,
    effort_condition: EffortConditions,
](
    alpha,
    w_v,
    w_e,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(relationship_condition),
        actor : chooses(
            desire in DesireLevels,
            effort_condition in EffortConditions,
            wpp=1,
        ),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_discrete_base_padded_joint_de[
                padded_slot,
                scenario_idx,
                observed_action,
                relationship_condition,
                desire,
                effort_condition,
            ](alpha, w_v, w_e, risk_table, effort_table, g_padded_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        desire in DesireLevels,
        effort_condition in EffortConditions,
        wpp=E[(actor.desire == desire) * (actor.effort_condition == effort_condition)]
        ** alpha_observer,
    )
    return Pr[
        observer.desire == desire,
        observer.effort_condition == effort_condition,
    ]


# --- Study 2b: joint over (relationship, effort) given desire (LM alts) ------
# Observer knows desire; infers (intimacy, effort). Intimacy is continuous
# (IntimacyLevels, 101 bins). Padded LM-alternatives action space; slot 0 is the
# observed action. Output dims:
#   (padded_slot, scenario, observed_action, desire, relationship, effort)
# and the fit/CV slice slot 0, returning the joint posterior over
# (relationship, effort); downstream code marginalizes to the two sliders.


@memo
def _observer_joint_ie_full_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    desire_condition: DesireConditions,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    desire_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(desire_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(desire_condition),
        actor : chooses(
            relationship in IntimacyLevels,
            effort_condition in EffortConditions,
            wpp=1,
        ),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_full_padded_joint_ie[
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                relationship,
                effort_condition,
            ](
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
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        effort_condition in EffortConditions,
        wpp=E[
            (actor.relationship == relationship)
            * (actor.effort_condition == effort_condition)
        ]
        ** alpha_observer,
    )
    return Pr[
        observer.relationship == relationship,
        observer.effort_condition == effort_condition,
    ]


@memo
def _observer_joint_ie_discomfort_only_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    desire_condition: DesireConditions,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](
    alpha,
    w_d,
    gamma,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    prior_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(desire_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(desire_condition),
        actor : chooses(
            relationship in IntimacyLevels,
            effort_condition in EffortConditions,
            wpp=1,
        ),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_discomfort_only_padded_joint_ie[
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                relationship,
                effort_condition,
            ](alpha, w_d, gamma, risk_table, effort_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        effort_condition in EffortConditions,
        wpp=E[
            (actor.relationship == relationship)
            * (actor.effort_condition == effort_condition)
        ]
        ** alpha_observer,
    )
    return Pr[
        observer.relationship == relationship,
        observer.effort_condition == effort_condition,
    ]


@memo
def _observer_joint_ie_base_memo_reference[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    desire_condition: DesireConditions,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](
    alpha,
    w_v,
    w_e,
    alpha_observer,
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    desire_table: ...,
):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(desire_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(desire_condition),
        actor : chooses(
            relationship in IntimacyLevels,
            effort_condition in EffortConditions,
            wpp=1,
        ),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_base_padded_joint_ie[
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                relationship,
                effort_condition,
            ](
                alpha,
                w_v,
                w_e,
                risk_table,
                effort_table,
                g_padded_table,
                prior_table,
                desire_table,
            ),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        effort_condition in EffortConditions,
        wpp=E[
            (actor.relationship == relationship)
            * (actor.effort_condition == effort_condition)
        ]
        ** alpha_observer,
    )
    return Pr[
        observer.relationship == relationship,
        observer.effort_condition == effort_condition,
    ]


# ==============================================================================
# Joint-target observers — fast implementations (used by fits and CV)
# ==============================================================================
# Direct Bayesian inversion of the actor policy in plain JAX. In the memo
# reference above, the observer's belief about the actor's latent pair after
# observing the slot is, by Bayes with the actor's UNIFORM latent prior,
#
#   posterior(l1, l2 | slot, cell) ∝ policy(slot | l1, l2, cell),
#
# and the observer's choice wpp = posterior^alpha_observer renormalized over
# (l1, l2) cancels every normalization constant:
#
#   observer(l1, l2 | slot, cell) = policy^α / Σ_{l1',l2'} policy^α.
#
# The actor memos already produce the full policy table (a few MB), so the
# whole inversion is an elementwise power + a normalize over the two latent
# axes — no outer × inner cross-product. Each fast observer forwards its
# arguments (minus alpha_observer) verbatim to its actor memo, mirroring the
# reference's actor call, and returns a table with the same axis order as the
# reference: (padded_slot, scenario, observed_action, <given condition>,
# latent_1, latent_2). Equivalence to the reference — values and gradients,
# every variant — is enforced by test_model_compliance.py.


def _sharpened_posterior_logspace(policy, alpha_observer, n_latent_axes):
    """Invert an actor slot-choice policy into the observer's sharpened
    posterior over the trailing `n_latent_axes` latent axes, computed in log
    space: a masked softmax of alpha_observer * log(policy).

    Mathematically identical to a normalize -> power -> renormalize formulation
    (the normalization constant cancels inside the softmax), but immune to its
    float32 failure: powering a normalized row raises diffuse entries (~5e-3
    over a 202-bin row) to ~1e-51 at alpha_observer ~ 22 — below float32's
    ~1e-45 floor — so entire rows underflow to zero and renormalize to garbage,
    silently fencing the optimizer out of the high-alpha region. The same
    parameter vector scored 1573 in float64 and 16900 in float32. Subtracting
    the row max before exponentiating keeps the dominant entries representable
    at any alpha, so the computed object is the likelihood the equations define.

    Because the two formulations differ numerically in exactly the regime the
    optimizer visits, fits are not comparable across them: a set of fit and CV
    outputs must be generated entirely under one or the other, never mixed.

    Zero-probability entries stay exactly 0. Two sources of them: padded null
    slots (whose prior is `tables.NULL_EPSILON` = 1e-8, so their policy is a
    tiny constant rather than literally 0 — the all-null-row branch below is
    therefore rarely reached in production) and, more importantly, float32
    underflow of the actor's softmax at extreme per-latent utility, which can
    zero a subset of latent hypotheses inside an otherwise valid slot. A slot
    with no probability mass at all returns an all-zero posterior rather than
    NaN (the memo reference is NaN there; downstream code never consumes those
    slots).

    BOTH risky ops are guarded by the double-where pattern — the argument is
    sanitized BEFORE the op, not only after it. This matters for `exp` as much
    as for `log`: a masked entry's logit is 0, so `exp(0 - m)` with the row's
    shared `m` very negative (a diffuse row at alpha ~ 22 gives m ~ -117)
    overflows to +inf, and although the outer `where` zeroes the forward value,
    the backward pass then evaluates 0 * inf = NaN and — because `m` is shared
    across the row — poisons the gradient of every entry in it, valid ones
    included. This is not hypothetical: guarding only the output was a real bug
    here, and it reintroduced a high-alpha gradient cliff for mixed rows, the
    very failure this formulation exists to remove."""
    axes = tuple(range(policy.ndim - n_latent_axes, policy.ndim))
    positive = policy > 0.0
    # Sanitize before log: masked entries evaluate on 1.0 -> logit 0.0.
    log_pol = jnp.log(jnp.where(positive, policy, 1.0))
    logits = alpha_observer * log_pol
    masked = jnp.where(positive, logits, -jnp.inf)
    m = jnp.max(masked, axis=axes, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)  # all-null rows: shift by 0
    # Sanitize before exp too: masked entries exponentiate 0.0, not -m.
    shifted = jnp.where(positive, logits - m, 0.0)
    w = jnp.where(positive, jnp.exp(shifted), 0.0)
    denom = w.sum(axis=axes, keepdims=True)
    return w / jnp.where(denom > 0.0, denom, 1.0)


def _sharpened_joint_posterior(policy, alpha_observer):
    """Sharpened Bayes inversion over the trailing two (latent) axes — the
    joint families' entry point; see `_sharpened_posterior_logspace`."""
    return _sharpened_posterior_logspace(policy, alpha_observer, n_latent_axes=2)


def observer_joint_de_full(
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    relationship_values,
):
    """Study 1b/3a full-utility joint observer (fast path); equivalent to
    `_observer_joint_de_full_memo_reference`. Output dims:
    (padded_slot, scenario, observed_action, relationship, desire, effort)."""
    policy = actor_discrete_full_padded_joint_de(
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
    )
    return _sharpened_joint_posterior(policy, alpha_observer)


def observer_joint_de_discomfort_only(
    alpha,
    w_d,
    gamma,
    alpha_observer,
    risk_table,
    effort_table,
    prior_table,
    relationship_values,
):
    """Study 1b/3a discomfort-only joint observer (fast path); equivalent to
    `_observer_joint_de_discomfort_only_memo_reference`."""
    policy = actor_discrete_discomfort_only_padded_joint_de(
        alpha,
        w_d,
        gamma,
        risk_table,
        effort_table,
        prior_table,
        relationship_values,
    )
    return _sharpened_joint_posterior(policy, alpha_observer)


def observer_joint_de_base(
    alpha,
    w_v,
    w_e,
    alpha_observer,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
):
    """Study 1b/3a base joint observer (fast path); equivalent to
    `_observer_joint_de_base_memo_reference`."""
    policy = actor_discrete_base_padded_joint_de(
        alpha, w_v, w_e, risk_table, effort_table, g_padded_table, prior_table
    )
    return _sharpened_joint_posterior(policy, alpha_observer)


def observer_joint_ie_full(
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    desire_table,
):
    """Study 2b/3b full-utility joint observer (fast path); equivalent to
    `_observer_joint_ie_full_memo_reference`. Output dims:
    (padded_slot, scenario, observed_action, desire_condition, intimacy,
    effort)."""
    policy = actor_continuous_full_padded_joint_ie(
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
    )
    return _sharpened_joint_posterior(policy, alpha_observer)


def observer_joint_ie_discomfort_only(
    alpha,
    w_d,
    gamma,
    alpha_observer,
    risk_table,
    effort_table,
    prior_table,
):
    """Study 2b/3b discomfort-only joint observer (fast path); equivalent to
    `_observer_joint_ie_discomfort_only_memo_reference`."""
    policy = actor_continuous_discomfort_only_padded_joint_ie(
        alpha, w_d, gamma, risk_table, effort_table, prior_table
    )
    return _sharpened_joint_posterior(policy, alpha_observer)


def observer_joint_ie_base(
    alpha,
    w_v,
    w_e,
    alpha_observer,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    desire_table,
):
    """Study 2b/3b base joint observer (fast path); equivalent to
    `_observer_joint_ie_base_memo_reference`."""
    policy = actor_continuous_base_padded_joint_ie(
        alpha,
        w_v,
        w_e,
        risk_table,
        effort_table,
        g_padded_table,
        prior_table,
        desire_table,
    )
    return _sharpened_joint_posterior(policy, alpha_observer)


# ==============================================================================
# Variant registries — single source of truth for the three ablations
# ==============================================================================
# Each ablation fits the same utility-weight set across every observer family
# (only the observer function differs), so the param-name lists live in one
# place and each family's registry is built from them. The fit wrappers
# (model/inverse/fit_*.py) and the CV dispatcher both import these instead of
# re-declaring the mapping, so a variant added or a weight list changed here
# updates fit and CV together.

VARIANT_PARAM_NAMES = {
    "full": ["w_v", "w_d", "w_e", "gamma"],
    "discomfort_only": ["w_d", "gamma"],
    "base": ["w_v", "w_e"],
}


def _build_variants(full_fn, discomfort_only_fn, base_fn):
    """Map each ablation name to (observer_fn, utility_param_names) for one
    observer family, drawing the param names from VARIANT_PARAM_NAMES."""
    fns = {
        "full": full_fn,
        "discomfort_only": discomfort_only_fn,
        "base": base_fn,
    }
    return {
        name: (fns[name], VARIANT_PARAM_NAMES[name]) for name in VARIANT_PARAM_NAMES
    }


# ==============================================================================
# Fast single-latent observers (plain-JAX inversions of the actor memos)
# ==============================================================================
# Same construction as the joint fast paths: the actor memo supplies the slot
# policy over the latent grid, and the observer is its sharpened Bayes
# inversion over the single trailing latent axis (log-space; see
# `_sharpened_posterior_logspace`). The memo originals above are kept as
# `_observer_*_memo_reference` — the authoritative statement of the semantics;
# change model semantics in both — and test_model_compliance.py enforces
# fast ≡ reference on every variant. Converted alongside the
# log-space sharpening: the powering these memos apply inside `wpp`
# (E[latent == z] ** alpha_observer) has the same float32 underflow as the old
# joint path and cannot be stabilized from outside the generated code.


def observer_desire_full(
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    relationship_values,
):
    """Study 1a full observer (fast path); equivalent to
    `_observer_desire_full_memo_reference`. Output dims:
    (padded_slot, scenario, observed_action, effort, relationship, desire)."""
    policy = actor_discrete_full_padded_desire(
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
    )
    return _sharpened_posterior_logspace(policy, alpha_observer, n_latent_axes=1)


def observer_desire_discomfort_only(
    alpha,
    w_d,
    gamma,
    alpha_observer,
    risk_table,
    effort_table,
    prior_table,
    relationship_values,
):
    """Study 1a discomfort-only observer (fast path); equivalent to
    `_observer_desire_discomfort_only_memo_reference`."""
    policy = actor_discrete_discomfort_only_padded_desire(
        alpha, w_d, gamma, risk_table, effort_table, prior_table, relationship_values
    )
    return _sharpened_posterior_logspace(policy, alpha_observer, n_latent_axes=1)


def observer_desire_base(
    alpha,
    w_v,
    w_e,
    alpha_observer,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
):
    """Study 1a base observer (fast path); equivalent to
    `_observer_desire_base_memo_reference`."""
    policy = actor_discrete_base_padded_desire(
        alpha, w_v, w_e, risk_table, effort_table, g_padded_table, prior_table
    )
    return _sharpened_posterior_logspace(policy, alpha_observer, n_latent_axes=1)


def observer_intimacy_full(
    alpha,
    w_v,
    w_d,
    w_e,
    gamma,
    alpha_observer,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    desire_table,
):
    """Study 2a full observer (fast path); equivalent to
    `_observer_intimacy_full_memo_reference`. Output dims:
    (padded_slot, scenario, observed_action, desire_condition, effort,
    relationship)."""
    policy = actor_continuous_full_padded_intimacy(
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
    )
    return _sharpened_posterior_logspace(policy, alpha_observer, n_latent_axes=1)


def observer_intimacy_discomfort_only(
    alpha,
    w_d,
    gamma,
    alpha_observer,
    risk_table,
    effort_table,
    prior_table,
):
    """Study 2a discomfort-only observer (fast path); equivalent to
    `_observer_intimacy_discomfort_only_memo_reference`."""
    policy = actor_continuous_discomfort_only_padded_intimacy(
        alpha, w_d, gamma, risk_table, effort_table, prior_table
    )
    return _sharpened_posterior_logspace(policy, alpha_observer, n_latent_axes=1)


def observer_intimacy_base(
    alpha,
    w_v,
    w_e,
    alpha_observer,
    risk_table,
    effort_table,
    g_padded_table,
    prior_table,
    desire_table,
):
    """Study 2a base observer (fast path); equivalent to
    `_observer_intimacy_base_memo_reference`."""
    policy = actor_continuous_base_padded_intimacy(
        alpha,
        w_v,
        w_e,
        risk_table,
        effort_table,
        g_padded_table,
        prior_table,
        desire_table,
    )
    return _sharpened_posterior_logspace(policy, alpha_observer, n_latent_axes=1)


VARIANTS_DESIRE = _build_variants(
    observer_desire_full, observer_desire_discomfort_only, observer_desire_base
)
VARIANTS_JOINT_DE = _build_variants(
    observer_joint_de_full, observer_joint_de_discomfort_only, observer_joint_de_base
)
VARIANTS_INTIMACY = _build_variants(
    observer_intimacy_full, observer_intimacy_discomfort_only, observer_intimacy_base
)
VARIANTS_JOINT_IE = _build_variants(
    observer_joint_ie_full, observer_joint_ie_discomfort_only, observer_joint_ie_base
)
