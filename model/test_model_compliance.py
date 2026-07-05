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
    given-magnitude scalars,
  - the data loader's fail-fast validation (unmapped condition labels,
    duplicate stage rows), the JSONL loader's duplicate-key conflict check,
    the multistart fit's all-NaN failure, and the fit provenance manifest's
    write/verify round-trip.
"""

import json
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

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


def test_data_loader_rejects_unmapped_label():
    """_map_condition must raise on a condition label with no model index
    (e.g. the pre-rename 'neither'), not silently produce a NaN index."""
    from _helpers import _map_condition

    from tables import INTIMACY_CONDITION_TO_IDX

    df = pd.DataFrame({"intimacy": ["max_formal", "neither", "max_intimate"]})
    try:
        _map_condition(df, "intimacy", INTIMACY_CONDITION_TO_IDX, "test")
    except ValueError as e:
        assert "neither" in str(e), f"offending label not named: {e}"
        print("✓ data loader rejects unmapped condition labels")
    else:
        raise AssertionError("unmapped label was not rejected")


def test_data_loader_rejects_duplicate_stage_rows():
    """_validate_long_raw must reject duplicate (subject, scenario, stage) rows
    (they would cross-join in the prior↔posterior merge) and out-of-[0, 1]
    ratings."""
    from _helpers import _validate_long_raw

    ok = pd.DataFrame(
        {
            "subject_id": ["s1", "s1"],
            "scenario_label": ["apples", "apples"],
            "stage": ["prior", "posterior"],
            "desire_rating": [0.5, 0.7],
        }
    )
    _validate_long_raw(ok, ["desire_rating"], "test")
    dup = pd.concat([ok, ok.iloc[[1]]], ignore_index=True)
    try:
        _validate_long_raw(dup, ["desire_rating"], "test")
    except ValueError:
        print("✓ data loader rejects duplicate (subject, scenario, stage) rows")
    else:
        raise AssertionError("duplicate stage rows were not rejected")
    out_of_range = ok.assign(desire_rating=[0.5, 1.7])
    try:
        _validate_long_raw(out_of_range, ["desire_rating"], "test")
    except ValueError:
        print("✓ data loader rejects ratings outside [0, 1]")
    else:
        raise AssertionError("out-of-range rating was not rejected")


def test_jsonl_loader_rejects_conflicting_duplicates():
    """_run_sources_jsonl must reject a repeated observed-action key with a
    different value (silent last-write-wins) while identical repeats pass."""
    from tables import _run_sources_jsonl

    def _record(risk):
        return {
            "run_id": 0,
            "scenario_label": "apples",
            "observed_action": "no_share",
            "effort_condition": "low",
            "actions": [
                {
                    "slot": 0,
                    "alt_idx": None,
                    "is_observed": True,
                    "action_text": "x",
                    "risk": risk,
                    "effort": 0.2,
                    "g": 0.9,
                }
            ],
        }

    def _load(records):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "lm_runs.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            return _run_sources_jsonl(path, ["effort_condition"])

    _load([_record(0.1), _record(0.1)])  # identical repeat: fine
    try:
        _load([_record(0.1), _record(0.5)])
    except ValueError:
        print("✓ JSONL loader rejects duplicate keys with conflicting values")
    else:
        raise AssertionError("conflicting duplicate was not rejected")


def test_fit_multistart_raises_on_all_nan():
    """_fit_multistart must raise (not return None params) when every restart's
    loss is NaN from the start."""
    from _helpers import _fit_multistart

    def loss_fn(params):
        return jnp.sum(params) * jnp.nan

    try:
        _fit_multistart(
            loss_fn,
            n_params=2,
            n_restarts=2,
            max_steps=5,
            verbose=False,
            seed_key="test|all_nan",
        )
    except RuntimeError:
        print("✓ multistart fit raises when every restart's NLL is NaN")
    else:
        raise AssertionError("all-NaN multistart did not raise")


def test_delta_helpers_match_reference():
    """delta_latent / delta_joint (the single source of the belief-update δ that
    both the JAX fit losses and the numpy CV scorer call) must match an
    independent reference, and give identical results for jnp and numpy inputs.
    Guards the shared δ definition: if it silently changed, fit and CV would
    still agree with each other (they call one function) but score the wrong
    quantity — so we pin it to a hand-written reference here."""
    from _helpers import EFFORT_PRIOR_MEAN, GRID, PRIOR_MEAN, delta_joint, delta_latent

    grid_np = np.asarray(GRID)
    pm, epm = float(PRIOR_MEAN), float(EFFORT_PRIOR_MEAN)
    rng = np.random.default_rng(0)

    def ref_latent(density):  # posterior mean − prior, independent formulation
        return np.array([float(np.sum(row * grid_np)) for row in density]) - pm

    def ref_joint(joint):  # marginalize effort by adding the two slabs explicitly
        marg = joint[:, :, 0] + joint[:, :, 1]  # (K, n_grid)
        latent = np.array([float(np.sum(row * grid_np)) for row in marg]) - pm
        p_high = joint[:, :, 1].sum(axis=1) - epm  # (K,)
        return latent, p_high

    # 1-D latent: (K, n_grid) normalized densities
    dens = rng.random((8, grid_np.shape[0]))
    dens /= dens.sum(axis=1, keepdims=True)
    ref = ref_latent(dens)
    d_np = delta_latent(dens, grid_np, pm)
    # The fit runs in JAX (float32), the CV in numpy (float64); the reference is
    # float64, so the numpy path must match tightly and the jnp path only to
    # float32 precision — the point is formula identity, not bit-level equality.
    d_jx = np.asarray(delta_latent(jnp.asarray(dens), GRID, PRIOR_MEAN))
    assert np.allclose(d_np, ref), "delta_latent (numpy) != reference"
    assert np.allclose(d_jx, ref, atol=1e-5), "delta_latent (jnp) != reference"
    assert np.allclose(d_np, d_jx, atol=1e-5), "delta_latent: numpy and jnp disagree"

    # 2-D (latent, effort) joint: (K, n_grid, 2) normalized over both last axes
    joint = rng.random((8, grid_np.shape[0], 2))
    joint /= joint.sum(axis=(1, 2), keepdims=True)
    r_lat, r_eff = ref_joint(joint)
    n_lat, n_eff = delta_joint(joint, grid_np, pm, epm)
    j_lat, j_eff = delta_joint(jnp.asarray(joint), GRID, PRIOR_MEAN, EFFORT_PRIOR_MEAN)
    assert np.allclose(n_lat, r_lat) and np.allclose(n_eff, r_eff), "delta_joint != ref"
    assert np.allclose(np.asarray(j_lat), r_lat, atol=1e-5) and np.allclose(
        np.asarray(j_eff), r_eff, atol=1e-5
    ), "delta_joint jnp != reference"
    print("✓ delta_latent/delta_joint match reference and agree across jnp/numpy")


def test_fit_manifest_round_trip():
    """write_fit_manifest / verify_fit_manifest: a fresh write verifies; a
    changed data CSV or a tampered fit output is refused; a *missing* manifest
    warns and proceeds (returns None) rather than blocking, so CV can still run
    on a fit produced before provenance tracking existed."""
    from _helpers import verify_fit_manifest, write_fit_manifest

    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        (out / "fit_results.json").write_text('[{"model": "full"}]')
        (out / "fit_restarts.jsonl").write_text("{}\n")
        data_csv = out / "main_trials_long.csv"
        data_csv.write_text("subject_id,response\ns1,0.5\n")

        # No manifest yet: must NOT raise (legacy fit stays usable).
        assert (
            verify_fit_manifest("test_slug", output_dir=out, data_csv=data_csv) is None
        ), "missing manifest should warn and return None, not raise"

        write_fit_manifest("test_slug", out, data_csv=data_csv)
        verify_fit_manifest("test_slug", output_dir=out, data_csv=data_csv)

        data_csv.write_text("subject_id,response\ns1,0.9\n")
        try:
            verify_fit_manifest("test_slug", output_dir=out, data_csv=data_csv)
        except RuntimeError:
            pass
        else:
            raise AssertionError("changed data CSV was not refused")
        data_csv.write_text("subject_id,response\ns1,0.5\n")

        (out / "fit_results.json").write_text('[{"model": "tampered"}]')
        try:
            verify_fit_manifest("test_slug", output_dir=out, data_csv=data_csv)
        except RuntimeError:
            print("✓ fit manifest verifies clean outputs and refuses stale ones")
        else:
            raise AssertionError("tampered fit_results.json was not refused")


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
    test_data_loader_rejects_unmapped_label()
    test_data_loader_rejects_duplicate_stage_rows()
    test_jsonl_loader_rejects_conflicting_duplicates()
    test_fit_multistart_raises_on_all_nan()
    test_fit_manifest_round_trip()
    test_delta_helpers_match_reference()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
