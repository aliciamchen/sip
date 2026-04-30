"""
Utility functions and memo actor models for the effort-manipulation forward-
planning experiment (forw_plan_effort).

Parallel to model/model_utils.py, but adapted for:
  - 2 actions per scenario (action 0 = non-saliva-share, action 1 = saliva-share).
  - An effort_condition covariate (LOW, HIGH) carried by the vignette text.
  - Reward held fixed at HIGH; V(a|s) = 1 stipulated for both actions. w_v is
    kept in the utility for consistency with the canonical 4-action pipeline
    but is non-identified under the softmax (V constant across actions) and
    will stay near its initialization during fitting.

Scenario labels are shared with the canonical pipeline (same 16 scenarios,
same alphabetical ordering) so Scenarios / SCENARIO_LABELS / SCENARIO_TO_IDX
are reused from model_utils.
"""

from enum import IntEnum
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from memo import memo

# Reuse the shared scenario enum / index map
from model_utils import (
    SCENARIO_LABELS,
    SCENARIO_TO_IDX,
    IntimacyLevels,
    Scenarios,
)


# ==============================================================================
# Constants
# ==============================================================================

actions_effort = jnp.array([0, 1])  # 0 = action_1 (non-share), 1 = action_2 (share)


class EffortConditions(IntEnum):
    LOW = 0
    HIGH = 1


EFFORT_CONDITION_TO_IDX = {"low": 0, "high": 1}
N_ACTIONS_EFFORT = 2
N_EFFORT_CONDITIONS = 2


# ==============================================================================
# LLM-derived scenario-specific parameter tables (effort pipeline)
# ==============================================================================


def load_lm_scenario_params_effort(filepath=None):
    """Load access and effort tables for the effort experiment.

    Returns a dict with:
      - "access": jnp.array of shape (16, 2, 2) — (scenario, effort_condition, action)
      - "effort": jnp.array of shape (16, 2, 2)

    Raises FileNotFoundError if the CSV is missing — run
    `uv run python model/lm_scenario_params_effort.py` first.
    """
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent
            / "outputs"
            / "lm_scenario_params_effort.csv"
        )
    df = pd.read_csv(filepath)
    shape = (len(SCENARIO_LABELS), N_EFFORT_CONDITIONS, N_ACTIONS_EFFORT)
    access = np.zeros(shape, dtype=np.float32)
    effort = np.zeros(shape, dtype=np.float32)
    for _, row in df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        e_idx = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        # CSV action 1 -> internal 0, CSV action 2 -> internal 1
        a_idx = int(row["action"]) - 1
        access[s_idx, e_idx, a_idx] = row["access"]
        effort[s_idx, e_idx, a_idx] = row["effort"]
    return {"access": jnp.array(access), "effort": jnp.array(effort)}


LLM_TABLES_EFFORT = load_lm_scenario_params_effort()


def load_lm_scenario_params_effort_marginal(filepath=None):
    """Load effort-marginal access ratings (vignette without effort paragraph).

    Used by the inv_plan_effort_inferred experiment, where the observer does
    not see the effort paragraph and so cannot perceive any effort-induced
    setting differences when reasoning about action access.

    Returns a jnp.array of shape (16, 2, 2) with the marginal access value
    broadcast across the effort_condition dimension — so it slots into the
    same indexing pattern as the conditional table without changing any
    downstream code. Returns None if the CSV is missing (the conditional
    table is then used everywhere as a fallback).
    """
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent
            / "outputs"
            / "lm_scenario_params_effort_marginal.csv"
        )
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    n_scen, n_act = len(SCENARIO_LABELS), N_ACTIONS_EFFORT
    flat = np.zeros((n_scen, n_act), dtype=np.float32)
    for _, row in df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        a_idx = int(row["action"]) - 1
        flat[s_idx, a_idx] = row["access"]
    # Broadcast across effort_condition so the actor utility's existing
    # indexing access_table[scenario, effort, action] returns the same
    # value for both effort conditions.
    broadcast = np.broadcast_to(flat[:, None, :], (n_scen, N_EFFORT_CONDITIONS, n_act))
    return jnp.array(broadcast)


_access_marg = load_lm_scenario_params_effort_marginal()
if _access_marg is not None:
    LLM_TABLES_EFFORT["access_marg"] = _access_marg


# ==============================================================================
# Utility functions
# ==============================================================================
# V(a|s) = 1 for both actions (reward fixed at HIGH, both end in eating). w_v
# is therefore non-identified in the softmax but kept as a fitted parameter for
# parallelism with the canonical 4-action pipeline.


@jax.jit
def get_stipulated_reward_effort(action):
    """Constant V = 1 for both actions."""
    return jnp.array([1.0, 1.0])[action]


@jax.jit
def get_utility_effort_access_full(
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
def get_utility_effort_access_only(
    action, scenario_idx, intimacy, effort_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    access = access_table[scenario_idx, effort_condition, action]
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (-w_d * access * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_effort_no_access(
    action, scenario_idx, intimacy, effort_condition,
    alpha, w_v, w_e,
    access_table, effort_table,
):
    effort = effort_table[scenario_idx, effort_condition, action]
    V = get_stipulated_reward_effort(action)
    return alpha * (w_v * V - w_e * effort)


# ==============================================================================
# Forward-planning actor models
# ==============================================================================


@memo
def actor_forw_effort_access_full[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_access_full(
                action, scenario_idx, intimacy, effort_condition,
                alpha, w_v, w_d, w_e, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_effort_access_only[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_access_only(
                action, scenario_idx, intimacy, effort_condition,
                alpha, w_d, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_effort_no_access[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_no_access(
                action, scenario_idx, intimacy, effort_condition,
                alpha, w_v, w_e,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Continuous-intimacy actor models (observer uses these inside `thinks[...]`)
# ==============================================================================
# These are mathematically identical to the actor_forw_effort_* models but use
# `relationship` as the index name to match the binding convention in the
# alt-shown observer memos.


@memo
def actor_continuous_effort_access_full[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_access_full(
                action, scenario_idx, relationship, effort_condition,
                alpha, w_v, w_d, w_e, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_effort_access_only[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_access_only(
                action, scenario_idx, relationship, effort_condition,
                alpha, w_d, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_effort_no_access[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_no_access(
                action, scenario_idx, relationship, effort_condition,
                alpha, w_v, w_e,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Observer inferring intimacy (effort experiment, 2-action space)
# ==============================================================================


@memo
def observer_intimacy_effort_access_full[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_continuous_effort_access_full[
                action, scenario_idx, relationship, effort_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_effort_access_only[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_continuous_effort_access_only[
                action, scenario_idx, relationship, effort_condition
            ](alpha, w_d, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_effort_no_access[
    action: actions_effort,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(effort_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(effort_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_continuous_effort_no_access[
                action, scenario_idx, relationship, effort_condition
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
# Observer inferring effort condition (effort experiment, 2-action space)
# ==============================================================================
# Observed: (action, intimacy, scenario). Latent: effort_condition (low/high).
# Uniform prior over the two effort conditions; α_observer applies the usual
# inverse-planning softmax sharpness to the implied posterior.


@memo
def observer_effort_inferred_access_full[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(intimacy)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(intimacy),
        actor : chooses(effort_condition in EffortConditions, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_forw_effort_access_full[
                action, scenario_idx, intimacy, effort_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        effort_condition in EffortConditions,
        wpp=E[actor.effort_condition == effort_condition] ** alpha_observer,
    )
    return Pr[observer.effort_condition == effort_condition]


@memo
def observer_effort_inferred_access_only[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(intimacy)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(intimacy),
        actor : chooses(effort_condition in EffortConditions, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_forw_effort_access_only[
                action, scenario_idx, intimacy, effort_condition
            ](alpha, w_d, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        effort_condition in EffortConditions,
        wpp=E[actor.effort_condition == effort_condition] ** alpha_observer,
    )
    return Pr[observer.effort_condition == effort_condition]


@memo
def observer_effort_inferred_no_access[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(intimacy)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(intimacy),
        actor : chooses(effort_condition in EffortConditions, wpp=1),
        actor : chooses(
            action in actions_effort,
            wpp=actor_forw_effort_no_access[
                action, scenario_idx, intimacy, effort_condition
            ](alpha, w_v, w_e, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        effort_condition in EffortConditions,
        wpp=E[actor.effort_condition == effort_condition] ** alpha_observer,
    )
    return Pr[observer.effort_condition == effort_condition]


