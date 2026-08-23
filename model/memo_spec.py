"""
Memo specification of the model — the executable statement of the actor and
observer semantics in the memo probabilistic-programming DSL, kept as the
authority the plain-JAX production code is tested against. NOT on the
production path: fits, CV, and the reweighting all run the plain-JAX
equivalents (the actor policies in `actors.py`, the observer inversions in
`observers.py`), and only `test_model_compliance.py` imports this module.

Layout mirrors the production modules:
  - Actor memos (`actor_*_padded_*`): softmax policies over the padded action
    slots, one per (family, variant) — the originals of the plain-JAX policies
    in `actors.py`, under the same names.
  - Observer memos (`_observer_*_memo_reference`): the actor nested inside the
    observer's `thinks[...]` block, conditioned on the observed slot, choosing
    the latent(s) with wpp = E[latent indicator] ** alpha_observer — the
    originals of the plain-JAX inversions in `observers.py`.

Two reasons production left memo (details in `observers.py`): the joint
indicator expectation compiles to a ~202 × 202 latent cross-product per cell
(~7.5 GB of XLA temps per K=20 gradient step), and `E[...] ** alpha_observer`
underflows float32 above alpha_observer ~ 15–20, silently fencing fits out of
the high-alpha region — neither fixable from outside the generated code.

CHANGE MODEL SEMANTICS HERE AND IN THE PRODUCTION MODULES TOGETHER;
`test_model_compliance.py` enforces production ≡ spec (values, and gradients
where the spec is numerically healthy) on all twelve variants.
"""

from memo import memo

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


# ==============================================================================
# Padded actors for Study 1a (desire inference, LM-generated alternatives)
# ==============================================================================
# Used inside `observer_desire_*` (Study 1a). The actor knows scenario,
# observed_action, effort_condition, relationship_condition, desire_condition;
# the latent (desire) is sampled by the observer's `thinks` block. The actor
# softmaxes over `MAX_ACTIONS` padded slots, with the observed
# action in slot 0 and LM-generated alternatives in slots 1..k. Null slots get
# a tiny epsilon prior (1e-8) so they effectively don't contribute to the
# softmax but still keep `E[...] ** alpha_observer` differentiable.


@memo
def actor_discrete_full_padded_desire[
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
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    relationship_values: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(effort_condition)
    actor: knows(relationship_condition)
    actor: knows(desire)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_desire(
            padded_slot,
            scenario_idx,
            observed_action,
            effort_condition,
            relationship_condition,
            prior_table,
        )
        * exp(
            get_utility_full_padded_desire(
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
                relationship_values,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_discrete_discomfort_only_padded_desire[
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
    risk_table: ...,
    effort_table: ...,
    prior_table: ...,
    relationship_values: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(effort_condition)
    actor: knows(relationship_condition)
    actor: knows(desire)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_desire(
            padded_slot,
            scenario_idx,
            observed_action,
            effort_condition,
            relationship_condition,
            prior_table,
        )
        * exp(
            get_utility_discomfort_only_padded_desire(
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
                relationship_values,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_discrete_base_padded_desire[
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
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(effort_condition)
    actor: knows(relationship_condition)
    actor: knows(desire)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_desire(
            padded_slot,
            scenario_idx,
            observed_action,
            effort_condition,
            relationship_condition,
            prior_table,
        )
        * exp(
            get_utility_base_padded_desire(
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
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


# =============================================================================
# Padded actors for the migrated studies (1b, 2a, 2b)
# =============================================================================
# Like actor_*_padded_desire, these live inside the observer's `thinks[...]`
# block: the actor knows the full latent state and chooses a padded slot. The
# observer marginalizes the slot choice over the latent(s) it infers.


# --- Study 1b (joint_de): intimacy observed (4-level), infers (desire, effort) ---


@memo
def actor_discrete_full_padded_joint_de[
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
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    relationship_values: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship_condition)
    actor: knows(desire)
    actor: knows(effort_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_joint_de(
            padded_slot,
            scenario_idx,
            observed_action,
            relationship_condition,
            prior_table,
        )
        * exp(
            get_utility_full_padded_joint_de(
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
                relationship_values,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_discrete_discomfort_only_padded_joint_de[
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
    risk_table: ...,
    effort_table: ...,
    prior_table: ...,
    relationship_values: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship_condition)
    actor: knows(desire)
    actor: knows(effort_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_joint_de(
            padded_slot,
            scenario_idx,
            observed_action,
            relationship_condition,
            prior_table,
        )
        * exp(
            get_utility_discomfort_only_padded_joint_de(
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
                relationship_values,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_discrete_base_padded_joint_de[
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
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship_condition)
    actor: knows(desire)
    actor: knows(effort_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_joint_de(
            padded_slot,
            scenario_idx,
            observed_action,
            relationship_condition,
            prior_table,
        )
        * exp(
            get_utility_base_padded_joint_de(
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
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


# --- Study 2a (intimacy): (desire, effort) observed, infers intimacy ----------


@memo
def actor_continuous_full_padded_intimacy[
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
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    desire_table: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(desire_condition)
    actor: knows(effort_condition)
    actor: knows(relationship)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_intimacy(
            padded_slot,
            scenario_idx,
            observed_action,
            desire_condition,
            effort_condition,
            prior_table,
        )
        * exp(
            get_utility_full_padded_intimacy(
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                effort_condition,
                relationship,
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
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_continuous_discomfort_only_padded_intimacy[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    desire_condition: DesireConditions,
    effort_condition: EffortConditions,
    relationship: IntimacyLevels,
](alpha, w_d, gamma, risk_table: ..., effort_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(desire_condition)
    actor: knows(effort_condition)
    actor: knows(relationship)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_intimacy(
            padded_slot,
            scenario_idx,
            observed_action,
            desire_condition,
            effort_condition,
            prior_table,
        )
        * exp(
            get_utility_discomfort_only_padded_intimacy(
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                effort_condition,
                relationship,
                alpha,
                w_d,
                gamma,
                risk_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_continuous_base_padded_intimacy[
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
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    desire_table: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(desire_condition)
    actor: knows(effort_condition)
    actor: knows(relationship)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_intimacy(
            padded_slot,
            scenario_idx,
            observed_action,
            desire_condition,
            effort_condition,
            prior_table,
        )
        * exp(
            get_utility_base_padded_intimacy(
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                effort_condition,
                relationship,
                alpha,
                w_v,
                w_e,
                risk_table,
                effort_table,
                g_padded_table,
                desire_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


# --- Study 2b (joint_ie): desire observed, infers (intimacy, effort) ----------


@memo
def actor_continuous_full_padded_joint_ie[
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
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    desire_table: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(desire_condition)
    actor: knows(relationship)
    actor: knows(effort_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_joint_ie(
            padded_slot, scenario_idx, observed_action, desire_condition, prior_table
        )
        * exp(
            get_utility_full_padded_joint_ie(
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                effort_condition,
                relationship,
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
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_continuous_discomfort_only_padded_joint_ie[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    desire_condition: DesireConditions,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, risk_table: ..., effort_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(desire_condition)
    actor: knows(relationship)
    actor: knows(effort_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_joint_ie(
            padded_slot, scenario_idx, observed_action, desire_condition, prior_table
        )
        * exp(
            get_utility_discomfort_only_padded_joint_ie(
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                effort_condition,
                relationship,
                alpha,
                w_d,
                gamma,
                risk_table,
                effort_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_continuous_base_padded_joint_ie[
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
    risk_table: ...,
    effort_table: ...,
    g_padded_table: ...,
    prior_table: ...,
    desire_table: ...,
):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(desire_condition)
    actor: knows(relationship)
    actor: knows(effort_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_joint_ie(
            padded_slot, scenario_idx, observed_action, desire_condition, prior_table
        )
        * exp(
            get_utility_base_padded_joint_ie(
                padded_slot,
                scenario_idx,
                observed_action,
                desire_condition,
                effort_condition,
                relationship,
                alpha,
                w_v,
                w_e,
                risk_table,
                effort_table,
                g_padded_table,
                desire_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


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
