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


def load_lm_action_priors_effort(filepath=None):
    """Load π(a|s,e) from lm_action_priors_effort.csv.

    Returns a jnp.array of shape (16, 2, 2) with rows summing to 1 within each
    (scenario, effort_condition) pair, or None if the CSV is missing.
    """
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent
            / "outputs"
            / "lm_action_priors_effort.csv"
        )
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath)
    shape = (len(SCENARIO_LABELS), N_EFFORT_CONDITIONS, N_ACTIONS_EFFORT)
    priors = np.zeros(shape, dtype=np.float32)
    for _, row in df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        e_idx = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        a_idx = int(row["action"]) - 1
        priors[s_idx, e_idx, a_idx] = row["prior"]
    return jnp.array(priors)


ACTION_PRIOR_EFFORT = load_lm_action_priors_effort()
if ACTION_PRIOR_EFFORT is not None:
    LLM_TABLES_EFFORT["action_prior"] = ACTION_PRIOR_EFFORT


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
    alpha, w_v, w_d, w_e,
    access_table, effort_table,
):
    access = access_table[scenario_idx, effort_condition, action]
    effort = effort_table[scenario_idx, effort_condition, action]
    V = get_stipulated_reward_effort(action)
    return alpha * (
        w_v * V
        - w_d * access * (1 - intimacy)
        - w_e * effort
    )


@jax.jit
def get_utility_effort_access_only(
    action, scenario_idx, intimacy, effort_condition,
    alpha, w_d,
    access_table, effort_table,
):
    access = access_table[scenario_idx, effort_condition, action]
    return alpha * (-w_d * access * (1 - intimacy))


@jax.jit
def get_utility_effort_no_access(
    action, scenario_idx, intimacy, effort_condition,
    alpha, w_v, w_e,
    access_table, effort_table,
):
    effort = effort_table[scenario_idx, effort_condition, action]
    V = get_stipulated_reward_effort(action)
    return alpha * (w_v * V - w_e * effort)


@jax.jit
def get_action_prior_effort(scenario_idx, effort_condition, action, prior_table):
    return prior_table[scenario_idx, effort_condition, action]


# ==============================================================================
# Forward-planning actor models (6 variants)
# ==============================================================================


@memo
def actor_forw_effort_access_full[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_access_full(
                action, scenario_idx, intimacy, effort_condition,
                alpha, w_v, w_d, w_e,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_effort_access_full_prior[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_d, w_e, beta_prior, access_table: ..., effort_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=(
            get_action_prior_effort(scenario_idx, effort_condition, action, prior_table)
            ** beta_prior
        ) * exp(
            get_utility_effort_access_full(
                action, scenario_idx, intimacy, effort_condition,
                alpha, w_v, w_d, w_e,
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
](alpha, w_d, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=exp(
            get_utility_effort_access_only(
                action, scenario_idx, intimacy, effort_condition,
                alpha, w_d,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_effort_access_only_prior[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_d, beta_prior, access_table: ..., effort_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=(
            get_action_prior_effort(scenario_idx, effort_condition, action, prior_table)
            ** beta_prior
        ) * exp(
            get_utility_effort_access_only(
                action, scenario_idx, intimacy, effort_condition,
                alpha, w_d,
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


@memo
def actor_forw_effort_no_access_prior[
    action: actions_effort,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    effort_condition: EffortConditions,
](alpha, w_v, w_e, beta_prior, access_table: ..., effort_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(effort_condition)
    actor: chooses(
        action in actions_effort,
        wpp=(
            get_action_prior_effort(scenario_idx, effort_condition, action, prior_table)
            ** beta_prior
        ) * exp(
            get_utility_effort_no_access(
                action, scenario_idx, intimacy, effort_condition,
                alpha, w_v, w_e,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]
