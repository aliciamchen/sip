"""
Compliance tests for the active inverse-planning models.

Run standalone:  uv run python model/test_model_compliance.py
Or with pytest:  uv run python -m pytest model/test_model_compliance.py -v

Coverage uses the Study 1a (desire) padded utility + observer as a representative
of the active family (`observer_{desire,intimacy,joint_de,joint_ie}_*`), all of
which share the same `w_v · desire · g − w_d · risk · (1 − I)^γ − w_e · effort`
utility skeleton and the padded LM-alternatives action space:
  - utility ablation algebra (full collapses to base and to discomfort_only),
  - ablation invariances (discomfort_only is desire-free),
  - observer posterior normalization at the observed slot.
"""

import jax.numpy as jnp
import numpy as np

from observers import observer_desire_full
from tables import MAX_ACTIONS, N_ACTIONS, RELATIONSHIP_LEVEL_VALUES, SCENARIO_LABELS
from utility import (
    get_utility_base_padded_desire,
    get_utility_discomfort_only_padded_desire,
    get_utility_full_padded_desire,
)

N_S = len(SCENARIO_LABELS)  # 16
N_O = N_ACTIONS  # 3 canonical observed actions
N_E = 2  # effort conditions
N_R = 4  # relationship conditions
S = MAX_ACTIONS  # padded slots


def _synthetic_desire_tables():
    """Synthetic padded tables shaped (16, 3, 2, 4, S) for the desire study.
    Slots 0..2 hold the three canonical actions; remaining slots are null-padded
    (prior ≈ 0). Feature values are deterministic functions of the slot index."""
    shape = (N_S, N_O, N_E, N_R, S)
    risk = np.zeros(shape, dtype=np.float32)
    effort = np.zeros(shape, dtype=np.float32)
    g = np.zeros(shape, dtype=np.float32)
    prior = np.full(shape, 1e-8, dtype=np.float32)
    for s in range(N_S):
        for o in range(N_O):
            for e in range(N_E):
                for r in range(N_R):
                    for slot in range(N_O):  # 3 valid canonical slots
                        risk[s, o, e, r, slot] = 0.5 * (slot + 1)
                        effort[s, o, e, r, slot] = 0.3 * (slot + 1)
                        g[s, o, e, r, slot] = (slot + 1) / N_O
                        prior[s, o, e, r, slot] = 1.0 / N_O
    return jnp.array(risk), jnp.array(effort), jnp.array(g), jnp.array(prior)


# Representative cell + weights used by the algebra tests.
_CELL = (1, 0, 2, 1, 2)  # padded_slot, scenario, observed, effort, relationship
_DESIRE = 0.6
_ALPHA, _W_V, _W_D, _W_E, _GAMMA = 1.0, 1.2, 0.7, 0.5, 1.0
# LM-rated intimacy magnitude per relationship level (the full/discomfort_only
# desire utilities + observer now take this as a param; placeholder here).
_REL = RELATIONSHIP_LEVEL_VALUES


def test_utility_ablation_algebra():
    """full reduces to base (w_d=0) and to discomfort_only (w_v=w_e=0)."""
    risk, effort, g, _ = _synthetic_desire_tables()

    u_full_wd0 = float(
        get_utility_full_padded_desire(
            *_CELL, _DESIRE, _ALPHA, _W_V, 0.0, _W_E, _GAMMA, risk, effort, g, _REL
        )
    )
    u_base = float(
        get_utility_base_padded_desire(
            *_CELL, _DESIRE, _ALPHA, _W_V, _W_E, risk, effort, g
        )
    )
    assert abs(u_full_wd0 - u_base) < 1e-6, (
        f"full(w_d=0) should match base: {u_full_wd0} vs {u_base}"
    )

    u_full_only = float(
        get_utility_full_padded_desire(
            *_CELL, _DESIRE, _ALPHA, 0.0, _W_D, 0.0, _GAMMA, risk, effort, g, _REL
        )
    )
    u_disc = float(
        get_utility_discomfort_only_padded_desire(
            *_CELL, _DESIRE, _ALPHA, _W_D, _GAMMA, risk, effort, _REL
        )
    )
    assert abs(u_full_only - u_disc) < 1e-6, (
        f"full(w_v=w_e=0) should match discomfort_only: {u_full_only} vs {u_disc}"
    )
    print("✓ full collapses to base (w_d=0) and discomfort_only (w_v=w_e=0)")


def test_discomfort_only_invariant_to_desire():
    """discomfort_only drops the desire term, so it cannot depend on desire."""
    risk, effort, _, _ = _synthetic_desire_tables()
    u_low = float(
        get_utility_discomfort_only_padded_desire(
            *_CELL, 0.2, _ALPHA, _W_D, _GAMMA, risk, effort, _REL
        )
    )
    u_high = float(
        get_utility_discomfort_only_padded_desire(
            *_CELL, 0.9, _ALPHA, _W_D, _GAMMA, risk, effort, _REL
        )
    )
    assert abs(u_low - u_high) < 1e-9, (
        f"discomfort_only should be invariant to desire: {u_low} vs {u_high}"
    )
    print("✓ discomfort_only utility is invariant to desire")


def test_observer_desire_posterior_sums_to_one():
    """The desire observer's posterior over DesireLevels sums to 1 at slot 0
    (the observed canonical action), across a sample of cells."""
    risk, effort, g, prior = _synthetic_desire_tables()
    result = observer_desire_full(
        _ALPHA, _W_V, _W_D, _W_E, _GAMMA, 1.0, risk, effort, g, prior, _REL
    )
    # shape: (padded_slot, scenario, observed_action, effort, relationship, desire)
    for s in [0, 8, 15]:
        for o in range(N_O):
            for e in range(N_E):
                for r in range(N_R):
                    psum = float(result[0, s, o, e, r, :].sum())
                    assert np.isclose(psum, 1.0, atol=1e-4), (
                        f"desire posterior doesn't sum to 1: {psum} "
                        f"(slot=0, s={s}, o={o}, e={e}, r={r})"
                    )
    print("✓ desire observer posteriors (slot=0) sum to 1")


def run_all_tests():
    print("=" * 60)
    print("Active model compliance tests")
    print("=" * 60)
    test_utility_ablation_algebra()
    test_discomfort_only_invariant_to_desire()
    test_observer_desire_posterior_sums_to_one()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
