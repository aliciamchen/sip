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
  - observer posterior normalization at the observed slot (single and joint),
  - the mixture likelihoods against a plain-numpy reference,
  - null-padded slots absorbing negligible actor probability at fitted-scale
    weights,
  - the table loaders' fail-fast validation of NaN features and missing
    given-magnitude scalars.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "inverse"))

from actors import actor_discrete_full_padded_desire
from observers import observer_desire_full, observer_joint_de_full
from tables import (
    MAX_ACTIONS,
    N_ACTIONS,
    RELATIONSHIP_LEVEL_VALUES,
    SCENARIO_LABELS,
    _assert_no_missing_scalars,
    _validate_padded_tables,
)
from utility import (
    get_utility_base_padded_desire,
    get_utility_discomfort_only_padded_desire,
    get_utility_full_padded_desire,
)

N_S = len(SCENARIO_LABELS)  # 16
N_O = N_ACTIONS  # 3 observed actions
N_E = 2  # effort conditions
N_R = 4  # relationship conditions
S = MAX_ACTIONS  # padded slots


def _synthetic_desire_tables():
    """Synthetic padded tables shaped (16, 3, 2, 4, S) for the desire study.
    Slots 0..2 hold the three observed actions; remaining slots are null-padded
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
                    for slot in range(N_O):  # 3 valid observed-action slots
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
    (the observed action), across a sample of cells."""
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


def test_mixture_nll_1d_matches_reference():
    """mixture_nll_1d agrees with a plain-numpy evaluation of
    −log[(1/K) Σ_k N(u | δ_k, σ²)]."""
    from _helpers import mixture_nll_1d

    rng = np.random.default_rng(3)
    deltas = rng.normal(0.0, 0.2, size=20)
    sigma, u = 0.25, 0.13
    pdf = np.exp(-0.5 * ((u - deltas) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    expected = -np.log(pdf.mean())
    got = float(mixture_nll_1d(jnp.array(u), jnp.array(deltas), jnp.array(sigma)))
    assert abs(got - expected) < 1e-5, f"{got} vs {expected}"
    print("✓ mixture_nll_1d matches the numpy reference")


def test_mixture_nll_2d_matches_reference():
    """mixture_nll_2d agrees with a plain-numpy bivariate isotropic mixture."""
    from _helpers import mixture_nll_2d

    rng = np.random.default_rng(4)
    deltas = rng.normal(0.0, 0.2, size=(20, 2))
    sigma = 0.3
    u = np.array([0.1, -0.2])
    sq = ((u[None, :] - deltas) ** 2).sum(axis=1)
    pdf = np.exp(-0.5 * sq / sigma**2) / (2 * np.pi * sigma**2)
    expected = -np.log(pdf.mean())
    got = float(mixture_nll_2d(jnp.array(u), jnp.array(deltas), jnp.array(sigma)))
    assert abs(got - expected) < 1e-5, f"{got} vs {expected}"
    print("✓ mixture_nll_2d matches the numpy reference")


def test_observer_joint_de_posterior_sums_to_one():
    """The joint (desire, effort) observer's posterior sums to 1 at slot 0."""
    n_rel, n_eff, S = 4, 2, MAX_ACTIONS
    shape = (N_S, N_O, n_rel, S)
    risk = np.zeros(shape, dtype=np.float32)
    g = np.zeros(shape, dtype=np.float32)
    prior = np.full(shape, 1e-8, dtype=np.float32)
    effort = np.zeros((N_S, N_O, n_rel, n_eff, S), dtype=np.float32)
    for slot in range(N_O):
        risk[..., slot] = 0.5 * (slot + 1)
        g[..., slot] = (slot + 1) / N_O
        prior[..., slot] = 1.0 / N_O
        effort[..., 0, slot] = 0.2 * (slot + 1)
        effort[..., 1, slot] = 0.4 * (slot + 1)
    result = observer_joint_de_full(
        _ALPHA,
        _W_V,
        _W_D,
        _W_E,
        _GAMMA,
        1.3,
        jnp.array(risk),
        jnp.array(effort),
        jnp.array(g),
        jnp.array(prior),
        _REL,
    )
    # shape: (slot, scenario, observed_action, relationship, desire_101, effort_2)
    for s in [0, 15]:
        for o in range(N_O):
            for r in range(n_rel):
                psum = float(result[0, s, o, r, :, :].sum())
                assert np.isclose(psum, 1.0, atol=1e-4), (
                    f"joint posterior doesn't sum to 1: {psum} (s={s}, o={o}, r={r})"
                )
    print("✓ joint_de observer posteriors (slot=0) sum to 1")


def test_null_padding_mass_negligible():
    """Null-padded slots (prior 1e-8, features 0 → utility 0) must absorb a
    negligible share of the actor's choice probability at fitted-scale weights.
    Uses the largest weights fitted so far (Study 1a full: w_v≈12.4, w_d≈5.2,
    w_e≈8.9) at desire=0, where real actions' utilities are most negative. This
    holds because every real cell contains at least one low-cost action (a
    no-share-type action with risk ≈ effort ≈ 0); if all real actions had
    utilities below ≈ −14, the 1e-8 epsilon would start to compete."""
    risk, effort, g, prior = _synthetic_desire_tables()
    probs = actor_discrete_full_padded_desire(
        1.0, 12.4, 5.2, 8.9, 0.094, risk, effort, g, prior, _REL
    )
    # shape: (slot, scenario, observed_action, effort, relationship, desire)
    probs0 = np.asarray(probs[..., 0])  # desire = 0 (worst case for the reward term)
    pad_mass = probs0[N_O:].sum(axis=0)
    assert pad_mass.max() < 1e-3, f"padding absorbs {pad_mass.max():.2e} probability"
    print(f"✓ null-padding mass ≤ {pad_mass.max():.2e} at fitted-scale weights")


def test_loader_validation_rejects_nan_features():
    """_validate_padded_tables must reject NaN features at valid slots and
    accept NaN at null-padded (invalid) slots."""
    arr = np.zeros((2, 3), dtype=np.float32)
    valid = np.zeros((2, 3), dtype=bool)
    valid[0, 0] = True
    arr[1, 2] = np.nan  # invalid slot: fine
    _validate_padded_tables("test", {"risk": (arr, valid)})
    arr[0, 0] = np.nan  # valid slot: must raise
    try:
        _validate_padded_tables("test", {"risk": (arr, valid)})
    except ValueError:
        print("✓ loader validation rejects NaN features at valid slots")
    else:
        raise AssertionError("NaN at a valid slot was not rejected")


def test_loader_validation_rejects_missing_scalars():
    """_assert_no_missing_scalars must reject NaN given-magnitude entries."""
    ok = np.array([[0.1, 0.9]], dtype=np.float32)
    _assert_no_missing_scalars("test", "desire", ok)
    bad = np.array([[0.1, np.nan]], dtype=np.float32)
    try:
        _assert_no_missing_scalars("test", "desire", bad)
    except ValueError:
        print("✓ loader validation rejects missing given-magnitude scalars")
    else:
        raise AssertionError("missing scalar was not rejected")


def run_all_tests():
    print("=" * 60)
    print("Active model compliance tests")
    print("=" * 60)
    test_utility_ablation_algebra()
    test_discomfort_only_invariant_to_desire()
    test_observer_desire_posterior_sums_to_one()
    test_observer_joint_de_posterior_sums_to_one()
    test_mixture_nll_1d_matches_reference()
    test_mixture_nll_2d_matches_reference()
    test_null_padding_mass_negligible()
    test_loader_validation_rejects_nan_features()
    test_loader_validation_rejects_missing_scalars()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
