"""
Observer memo models — the inverse-planning Bayesian inference layer. One
family per active study, each over the padded LM-alternatives action space:
  - `observer_intimacy_*` (Study 2a) — knows (desire, effort), infers intimacy.
  - `observer_desire_*`   (Study 1a) — knows (effort, intimacy), infers desire.
  - `observer_joint_de_*` (Study 1b) — infers desire + effort.
  - `observer_joint_ie_*` (Study 2b) — infers intimacy + effort.

Three model variants per observer: `_full`, `_discomfort_only`, `_base`.

Dependency layer 3: imports from `tables.py` and `actors.py`.
"""

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
# observed canonical action. Output dims:
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
# observed canonical action in slot 0.


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
            ](alpha, w_d, gamma, risk_table, effort_table, prior_table),
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
# Joint-target observers (Studies 1b, 2b)
# ==============================================================================
# Compute a joint posterior over two latent variables simultaneously, using
# memo's multi-choice syntax: `chooses(x in X, y in Y, wpp=...)` for the
# joint draw, and `Pr[..., ...]` for the joint return.
#
# Study 1b — observer knows intimacy, jointly infers (desire, effort).
# Study 2b — observer knows effort, jointly infers (desire, intimacy).
#
# Downstream code marginalizes the returned joint over each axis to produce
# the per-slider predictions matching the two ratings participants give.


# --- Study 1b: joint over (desire, effort) given intimacy (LM alternatives) --
# Observer knows intimacy; infers (desire, effort). Padded LM-alternatives
# action space; slot 0 is the observed action. Output dims:
#   (padded_slot, scenario, observed_action, relationship, desire, effort)
# and the fit/CV slice slot 0, returning the joint posterior over (desire, effort).


@memo
def observer_joint_de_full[
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
def observer_joint_de_discomfort_only[
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
            ](alpha, w_d, gamma, risk_table, effort_table, prior_table),
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
def observer_joint_de_base[
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
def observer_joint_ie_full[
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
def observer_joint_ie_discomfort_only[
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
def observer_joint_ie_base[
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
