"""
Observer models — the inverse-planning Bayesian inference layer. One family
per active study, each over the padded LM-alternatives action space:
  - `observer_intimacy_*` (Study 2a) — knows (desire, effort), infers intimacy.
  - `observer_desire_*`   (Study 1a) — knows (effort, intimacy), infers desire.
  - `observer_joint_de_*` (Studies 1b/3a) — infers desire + effort.
  - `observer_joint_ie_*` (Studies 2b/3b) — infers intimacy + effort.

Three model variants per observer: `_full`, `_discomfort_only`, `_base`.

The single-latent observers are memo models. The joint observers are plain-JAX
Bayesian inversions of the actor memos: the memo formulation of the joint
indicator expectation compiled to a ~202 outer × 202 inner latent cross-product
per (cell × slot × run) — ~7.5 GB of XLA temps per gradient step at K=20 — to
compute what is mathematically an elementwise power + normalize of the actor's
policy table (see `_sharpened_joint_posterior`). The original memo joint
observers are kept below as `_*_memo_reference` — the authoritative statement
of the model semantics — and `test_model_compliance.py` verifies the fast path
against them (values and gradients) on every variant.

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
def observer_intimacy_full[
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
def observer_intimacy_discomfort_only[
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
def observer_intimacy_base[
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
def observer_desire_full[
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
def observer_desire_discomfort_only[
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
def observer_desire_base[
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


def _sharpened_joint_posterior(policy, alpha_observer):
    """Invert an actor slot-choice policy into the observer's sharpened joint
    posterior over the two latent axes (the trailing two axes of `policy`).

    The posterior is normalized BEFORE the power is applied, mirroring the memo
    reference's op order (it exponentiates E[indicator], a normalized belief).
    This is not just cosmetic: the constants cancel mathematically either way,
    but at fitted alpha_observer ~10 a raw policy row of small values (~1e-3)
    raised first would underflow float32 row-wide (1e-3^10.3 ~ 1e-31 summed
    over a 202-bin row), where the normalized row's largest entry is O(1) and
    survives at any realistic alpha.

    Zero-probability entries (padded null slots have prior 0, so their policy
    is exactly 0 for every latent pair) are kept at 0 through the double-where
    pattern — `0 ** alpha` has a non-finite gradient for alpha < 1, and a NaN
    in any table entry would poison the fit's gradients even when the loss
    only reads slot 0. A slot with no probability mass at all (a null slot)
    normalizes to an all-zero posterior rather than NaN; the memo reference is
    NaN there, and downstream code never consumes those slots either way.

    Known shared fragility (present identically in the memo reference, so it
    is deliberately NOT changed here): at large alpha_observer (~10, the
    fitted scale) the gradient of the renormalization can go non-finite in
    float32 when no latent hypothesis dominates a row. The Adam loop's
    non-finite-loss abandon (see `_fit_with_adam`) contains it — a restart
    that walks into the regime is dropped in favor of the best finite iterate
    — and the compliance suite pins the fast path to the reference's exact
    NaN pattern there. Hardening the numerics (e.g. a log-space softmax)
    would change gradients relative to the reference vintage and is a
    deliberate modeling decision, not a refactor."""
    positive = policy > 0.0
    polsum = policy.sum(axis=(-2, -1), keepdims=True)
    posterior = policy / jnp.where(polsum > 0.0, polsum, 1.0)
    powered = jnp.where(
        positive, jnp.where(positive, posterior, 1.0) ** alpha_observer, 0.0
    )
    denom = powered.sum(axis=(-2, -1), keepdims=True)
    return powered / jnp.where(denom > 0.0, denom, 1.0)


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
