"""
Actor memo models — the padded inverse actors (`actor_discrete_*_padded_*` for
discrete observed intimacy, `actor_continuous_*_padded_*` for inferred
intimacy) used inside the observers' `thinks[...]` blocks.

Three model variants per study: `_full`, `_discomfort_only`, `_base`.

Dependency layer 2: imports from `tables.py` (enums, axes) and `utility.py`
(get_utility_*_padded_*). `observers.py` imports from here.
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
