"""
Forward-planning extensions to the canonical actor. Most variants here are
non-food-only; one (`actor_forw_gamma`) supports both domains.

NOTE (post γ-promotion): γ is now part of the CANONICAL utility in
`model_utils.py` (`get_utility_access_full` etc. take γ as a positional
arg). The `actor_forw_gamma` and `access_full_gamma` here are therefore
numerically identical to the canonical `actor_forw_access_full` /
`access_full` — kept as a redundant cross-check and to keep the
exploratory variants (typed_gamma, gamma_alpha, gamma_vpow) buildable.

Variants:

  access_full_gamma — replaces the linear (1 - I) intimacy modulator with
  a single power-law exponent gamma:

      U(a | s, I) = w_v * V - w_d * access * (1 - I)^gamma - w_e * effort

  Now equivalent to the canonical access_full. Kept for historical /
  cross-checking purposes.

  access_full_typed_gamma (NON-FOOD ONLY) — adds channel-specific access
  weights on top of gamma:

      U(a | s, I) = w_v * V
                  - w_d_type[s] * access * (1 - I)^gamma
                  - w_e * effort

  where type[s] ∈ {substance, space, privacy}. Free params:
      w_v, w_d_substance, w_d_space, w_d_privacy, w_e, gamma  (6 total).
  Tests whether the three channels carry different per-unit access costs
  even after the intimacy curve is allowed to be nonlinear.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
from memo import memo
from model_utils import (
    NONFOOD_SCENARIO_TO_IDX,
    IntimacyLevels,
    RewardConditions,
    Scenarios,
    actions,
    get_lm_v,
)


# ==============================================================================
# Non-food channel structure: maps each scenario_idx to its scenario_type.
#  0 = substance, 1 = space, 2 = privacy
# ==============================================================================

NONFOOD_TYPE_TO_IDX = {"substance": 0, "space": 1, "privacy": 2}
NONFOOD_TYPE_LABELS = ["substance", "space", "privacy"]


def _build_nonfood_scenario_type_idx_table():
    """Returns a jnp.array of shape (16,) mapping each non-food scenario_idx
    (alphabetical, matches NONFOOD_SCENARIO_TO_IDX) to a type index in
    {0=substance, 1=space, 2=privacy}, read from scenarios_nonfood.csv."""
    project_root = Path(__file__).resolve().parent.parent
    scenarios_csv = project_root / "experiments" / "scenarios_nonfood.csv"
    df = pd.read_csv(scenarios_csv)[["scenario_label", "scenario_type"]].drop_duplicates()
    arr = [None] * len(NONFOOD_SCENARIO_TO_IDX)
    for _, row in df.iterrows():
        idx = NONFOOD_SCENARIO_TO_IDX[row["scenario_label"]]
        arr[idx] = NONFOOD_TYPE_TO_IDX[row["scenario_type"]]
    if any(v is None for v in arr):
        missing = [lbl for lbl, i in NONFOOD_SCENARIO_TO_IDX.items() if arr[i] is None]
        raise ValueError(f"Missing scenario_type assignment for: {missing}")
    return jnp.array(arr, dtype=jnp.int32)


NONFOOD_SCENARIO_TYPE_IDX_TABLE = _build_nonfood_scenario_type_idx_table()


@jax.jit
def get_utility_gamma(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_table,
):
    access = access_table[scenario_idx, action]
    effort = effort_table[scenario_idx, action]
    V = get_lm_v(action, scenario_idx, reward_condition, v_table)
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * V
        - w_d * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@memo
def actor_forw_gamma[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_gamma(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_v, w_d, w_e, gamma,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@jax.jit
def get_utility_typed_gamma(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_v, w_e, gamma,
    w_d_per_scenario,
    access_table, effort_table, v_table,
):
    """w_d_per_scenario: jnp.array of shape (16,) with the channel-specific
    w_d already broadcast to each scenario_idx. The fit script builds it
    from a 3-vector (substance/space/privacy) and the scenario_type idx
    table at NONFOOD_SCENARIO_TYPE_IDX_TABLE."""
    access = access_table[scenario_idx, action]
    effort = effort_table[scenario_idx, action]
    V = get_lm_v(action, scenario_idx, reward_condition, v_table)
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    w_d_s = w_d_per_scenario[scenario_idx]
    return alpha * (
        w_v * V
        - w_d_s * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@memo
def actor_forw_typed_gamma[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, gamma,
  w_d_per_scenario: ...,
  access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_typed_gamma(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_v, w_e, gamma,
                w_d_per_scenario,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Decisiveness / dynamic-range variants
# ==============================================================================
# access_full_gamma_alpha: free α, w_v fixed = 1 (canonical-rescaling invariance
# test; should give identical NLL to access_full_gamma).
# access_full_gamma_vpow: V_eff = sign(V) * |V|^beta; β free; tests whether the
# LM-rated V scale is too compressed for human responses.


@jax.jit
def get_utility_gamma_alpha_free(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_d, w_e, gamma,
    access_table, effort_table, v_table,
):
    """Same as get_utility_gamma but with w_v fixed = 1.0; α is free."""
    access = access_table[scenario_idx, action]
    effort = effort_table[scenario_idx, action]
    V = get_lm_v(action, scenario_idx, reward_condition, v_table)
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        V
        - w_d * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@memo
def actor_forw_gamma_alpha_free[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_gamma_alpha_free(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_d, w_e, gamma,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@jax.jit
def get_utility_gamma_vpow(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_v, w_d, w_e, gamma, beta,
    access_table, effort_table, v_table,
):
    """V_eff = sign(V) * |V|^beta; β > 0. β=1 is identity (matches gamma).
    β < 1 stretches V toward ±1 (sharpening); β > 1 compresses toward 0."""
    access = access_table[scenario_idx, action]
    effort = effort_table[scenario_idx, action]
    V = get_lm_v(action, scenario_idx, reward_condition, v_table)
    V_eff = jnp.sign(V) * jnp.power(jnp.abs(V) + 1e-8, beta)
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * V_eff
        - w_d * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@memo
def actor_forw_gamma_vpow[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, beta, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_gamma_vpow(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_v, w_d, w_e, gamma, beta,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]
