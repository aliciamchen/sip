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

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "inverse"))

from actors import actor_discrete_full_padded_desire
from observers import (
    VARIANT_PARAM_NAMES,
    _observer_joint_de_base_memo_reference,
    _observer_joint_de_discomfort_only_memo_reference,
    _observer_joint_de_full_memo_reference,
    _observer_joint_ie_base_memo_reference,
    _observer_joint_ie_discomfort_only_memo_reference,
    _observer_joint_ie_full_memo_reference,
    observer_desire_full,
    observer_joint_de_base,
    observer_joint_de_discomfort_only,
    observer_joint_de_full,
    observer_joint_ie_base,
    observer_joint_ie_discomfort_only,
    observer_joint_ie_full,
)
from tables import (
    INTIMACY_CONDITIONS,
    MAX_ACTIONS,
    N_ACTIONS,
    RELATIONSHIP_LEVEL_VALUES,
    SCENARIO_LABELS,
    DesireConditions,
    RelationshipConditions,
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


def _synthetic_joint_tables(family, seed=7):
    """Seeded synthetic tables for one joint family, shaped like one elicitation
    run's slice. Null-slot priors are 1e-8 (not exactly 0) so the memo reference
    is NaN-free on every slot and the fast path can be compared on the full
    table, matching the convention of the normalization tests above. Axis sizes
    come from the enums so the fixture tracks the production table shapes."""
    rng = np.random.default_rng(seed)
    n_cond = (
        len(RelationshipConditions) if family == "joint_de" else len(DesireConditions)
    )
    cell = (N_S, N_O, n_cond)
    risk = rng.uniform(0.0, 1.0, size=(*cell, S)).astype(np.float32)
    g = rng.uniform(0.0, 1.0, size=(*cell, S)).astype(np.float32)
    effort = rng.uniform(0.0, 1.0, size=(*cell, N_E, S)).astype(np.float32)
    prior = np.full((*cell, S), 1e-8, dtype=np.float32)
    prior[..., :N_O] = 1.0 / N_O
    tables = dict(
        risk_table=jnp.array(risk),
        effort_table=jnp.array(effort),
        g_padded_table=jnp.array(g),
        prior_table=jnp.array(prior),
    )
    if family == "joint_de":
        tables["relationship_values"] = _REL
    else:
        tables["desire_table"] = jnp.array(
            rng.uniform(0.1, 1.0, size=(N_S, len(DesireConditions))).astype(np.float32)
        )
    return tables


# The six (fast, memo-reference) joint observer pairs. Weight names per
# variant come from the production registry (VARIANT_PARAM_NAMES — the same
# list the fits use, so a renamed or reordered weight updates the test in
# lockstep); the table-name tuples are deliberately hand-written as an
# independent statement of each signature (alpha_observer sits between the
# weights and the tables).
_JOINT_OBSERVER_CASES = [
    (
        "joint_de/full",
        observer_joint_de_full,
        _observer_joint_de_full_memo_reference,
        "joint_de",
        VARIANT_PARAM_NAMES["full"],
        (
            "risk_table",
            "effort_table",
            "g_padded_table",
            "prior_table",
            "relationship_values",
        ),
    ),
    (
        "joint_de/discomfort_only",
        observer_joint_de_discomfort_only,
        _observer_joint_de_discomfort_only_memo_reference,
        "joint_de",
        VARIANT_PARAM_NAMES["discomfort_only"],
        ("risk_table", "effort_table", "prior_table", "relationship_values"),
    ),
    (
        "joint_de/base",
        observer_joint_de_base,
        _observer_joint_de_base_memo_reference,
        "joint_de",
        VARIANT_PARAM_NAMES["base"],
        ("risk_table", "effort_table", "g_padded_table", "prior_table"),
    ),
    (
        "joint_ie/full",
        observer_joint_ie_full,
        _observer_joint_ie_full_memo_reference,
        "joint_ie",
        VARIANT_PARAM_NAMES["full"],
        ("risk_table", "effort_table", "g_padded_table", "prior_table", "desire_table"),
    ),
    (
        "joint_ie/discomfort_only",
        observer_joint_ie_discomfort_only,
        _observer_joint_ie_discomfort_only_memo_reference,
        "joint_ie",
        VARIANT_PARAM_NAMES["discomfort_only"],
        ("risk_table", "effort_table", "prior_table"),
    ),
    (
        "joint_ie/base",
        observer_joint_ie_base,
        _observer_joint_ie_base_memo_reference,
        "joint_ie",
        VARIANT_PARAM_NAMES["base"],
        ("risk_table", "effort_table", "g_padded_table", "prior_table", "desire_table"),
    ),
]

_JOINT_WEIGHT_VALUES = {"w_v": _W_V, "w_d": _W_D, "w_e": _W_E, "gamma": _GAMMA}


def test_joint_observers_match_memo_reference():
    """The fast joint observers (direct Bayesian inversion of the actor policy
    in plain JAX) must reproduce the original memo observers — the ground truth
    for the model's semantics — on every slot, cell, and latent bin, across
    ablations and alpha_observer regimes (>1, >>1, <1). This is the guarantee
    that the memory optimization changed the computation, not the model."""
    for family in ("joint_de", "joint_ie"):
        tables = _synthetic_joint_tables(family)
        for label, fast_fn, ref_fn, fam, weights, table_names in _JOINT_OBSERVER_CASES:
            if fam != family:
                continue
            for alpha_obs in (1.3, 10.3, 0.7):
                args = (
                    [_ALPHA]
                    + [_JOINT_WEIGHT_VALUES[w] for w in weights]
                    + [alpha_obs]
                    + [tables[t] for t in table_names]
                )
                fast = np.asarray(fast_fn(*args))
                ref = np.asarray(ref_fn(*args))
                assert np.isfinite(fast).all(), f"{label}: fast path has non-finite"
                assert np.allclose(fast, ref, rtol=1e-4, atol=1e-6), (
                    f"{label} (alpha_obs={alpha_obs}): fast observer deviates from "
                    f"memo reference; max abs diff {np.abs(fast - ref).max():.3e}"
                )
    print("✓ fast joint observers match the memo reference (all variants)")


def test_joint_observer_gradients_match_memo_reference():
    """Fits differentiate through the observer, so the fast path's gradients
    must match the memo reference's too. Readout is restricted to slot 0 (the
    only slot the fit and CV consume). alpha_observer is probed both in the
    everyday regime (1.3) and at the fitted scale (10.3), where BOTH
    implementations share a known non-finite-gradient fragility (see
    `_sharpened_joint_posterior`) — `equal_nan=True` pins the fast path to the
    reference's exact NaN pattern there, so a change that makes one
    implementation diverge (going NaN where the other stays finite, or vice
    versa) fails loudly."""
    for family, fast_fn, ref_fn in [
        ("joint_de", observer_joint_de_full, _observer_joint_de_full_memo_reference),
        ("joint_ie", observer_joint_ie_full, _observer_joint_ie_full_memo_reference),
    ]:
        tables = _synthetic_joint_tables(family)
        table_names = next(
            t
            for lbl, f, r, fam, w, t in _JOINT_OBSERVER_CASES
            if fam == family and f is fast_fn
        )
        table_args = [tables[t] for t in table_names]

        def readout(fn, w_v, alpha_obs):
            out = fn(_ALPHA, w_v, _W_D, _W_E, _GAMMA, alpha_obs, *table_args)
            return jnp.log(out[0] + 1e-9).sum()  # slot 0 only

        for alpha_obs in (1.3, 10.3):
            for argnum in (0, 1):  # d/dw_v and d/dalpha_observer
                g_fast = jax.grad(
                    lambda wv, ao: readout(fast_fn, wv, ao), argnums=argnum
                )(jnp.float32(_W_V), jnp.float32(alpha_obs))
                g_ref = jax.grad(
                    lambda wv, ao: readout(ref_fn, wv, ao), argnums=argnum
                )(jnp.float32(_W_V), jnp.float32(alpha_obs))
                assert np.allclose(
                    float(g_fast), float(g_ref), rtol=1e-3, equal_nan=True
                ), (
                    f"{family} (alpha_obs={alpha_obs}): gradient (argnum {argnum}) "
                    f"fast={float(g_fast):.6g} vs reference={float(g_ref):.6g}"
                )
    print("✓ fast joint observer gradients match the memo reference")


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


def test_beta_prior_on_grid_normalizes_and_nests_uniform():
    from _priors import GRID, beta_prior_on_grid

    # exact uniform at (m=0.5, nu=2): Beta(1,1) has zero log-pdf everywhere
    w = beta_prior_on_grid(jnp.array(0.5), 2.0)
    assert w.shape == (101,)
    assert jnp.allclose(w, jnp.ones(101) / 101, atol=1e-7)
    # normalization + mean recovery for concentrated priors, batched input
    ms = jnp.array([0.2, 0.5, 0.8])
    for nu in (2.0, 8.0, 32.0):
        w = beta_prior_on_grid(ms, nu)  # (3, 101)
        assert jnp.allclose(w.sum(-1), 1.0, atol=1e-6)
        means = w @ GRID
        assert jnp.all(jnp.diff(means) > 0)  # monotone in m
        if nu >= 8.0:
            assert jnp.max(jnp.abs(means - ms)) < 0.03
    print("✓ beta_prior_on_grid normalizes, nests uniform, recovers the mean")


def test_reweight_grid_uniform_weights_is_identity():
    from _priors import GRID, beta_prior_on_grid, reweight_grid

    rng = np.random.default_rng(0)
    post = rng.dirichlet(np.ones(101), size=(4,))  # (4, 101)
    w = beta_prior_on_grid(jnp.full((4,), 0.5), 2.0)
    out = reweight_grid(jnp.asarray(post), w)
    assert jnp.allclose(out, post, atol=1e-6)
    # informative prior shifts the posterior mean toward the prior mean
    w_hi = beta_prior_on_grid(jnp.full((4,), 0.9), 8.0)
    out_hi = reweight_grid(jnp.asarray(post), w_hi)
    assert jnp.all(out_hi @ GRID > jnp.asarray(post) @ GRID)
    assert jnp.allclose(out_hi.sum(-1), 1.0, atol=1e-6)
    print("✓ reweight_grid is identity under uniform weights, shifts under prior")


def test_reweight_joint_matches_manual_and_nests_uniform():
    from _priors import beta_prior_on_grid, reweight_joint

    rng = np.random.default_rng(1)
    j = rng.dirichlet(np.ones(202), size=(3,)).reshape(3, 101, 2)
    j = jnp.asarray(j)
    assert reweight_joint(j) is j  # both None: no-op
    out_u = reweight_joint(
        j, beta_prior_on_grid(jnp.full((3,), 0.5), 2.0), jnp.full((3,), 0.5)
    )
    assert jnp.allclose(out_u, j, atol=1e-6)
    # manual reference for an informative case
    w = beta_prior_on_grid(jnp.full((3,), 0.3), 8.0)  # (3, 101)
    p = jnp.full((3,), 0.2)
    ref = j * w[:, :, None] * jnp.stack([1 - p, p], -1)[:, None, :]
    ref = ref / ref.sum((-2, -1), keepdims=True)
    assert jnp.allclose(reweight_joint(j, w, p), ref, atol=1e-6)
    print("✓ reweight_joint matches manual reference and nests uniform")


def _tiny_joint_ie_table_kwargs(K=2):
    """K-run joint_ie observer table kwargs for the nesting test: K independent
    seeded single-run slices from `_synthetic_joint_tables` stacked on a leading
    run axis (the shape `_build_observer_tables_runs` vmaps over). Gives
    risk (K,16,3,2,S), effort (K,16,3,2,2,S), g/prior (K,16,3,2,S), and
    desire_table (K,16,2) — the production 2b run-axis shapes."""
    runs = [_synthetic_joint_tables("joint_ie", seed=100 + k) for k in range(K)]
    return {key: jnp.stack([r[key] for r in runs], axis=0) for key in runs[0]}


def test_informative_prior_nests_uniform_fit_loss():
    """priors with m=0.5 everywhere and nu fixed at 2 must reproduce the
    uniform-path loss and gradient exactly (spec: uniform nested)."""
    from observers import VARIANTS_JOINT_IE

    obs_fn, utility_names = VARIANTS_JOINT_IE["full"]
    tk = _tiny_joint_ie_table_kwargs()  # the synthetic-table fixture (K=2)
    K = tk["risk_table"].shape[0]
    action = jnp.array([0, 1, 2, 1])
    scen = jnp.array([0, 1, 2, 3])
    des = jnp.array([0, 1, 0, 1])
    u_int = jnp.array([0.1, -0.2, 0.05, 0.3])
    u_eff = jnp.array([-0.1, 0.2, 0.0, -0.3])

    from _helpers import fit_joint_ie_observer_joint

    # Probe each loss at a fixed init via a 1-restart, max_steps=1 fit and
    # compare the recorded init NLLs (the loss at the same point).
    init_uniform = jnp.ones(len(utility_names) + 2)
    _, nll_u, rec_u = fit_joint_ie_observer_joint(
        obs_fn,
        utility_names,
        action,
        scen,
        des,
        u_int,
        u_eff,
        tk,
        n_restarts=1,
        init_params=init_uniform,
        max_steps=1,
        verbose=False,
    )
    n_scen = tk["risk_table"].shape[1]
    priors = {
        "m_latent": jnp.full((K, n_scen, 2), 0.5),
        "p_effort": jnp.full((K, n_scen, 2), 0.5),
    }
    init_inf = jnp.concatenate([init_uniform, jnp.array([2.0])])  # nu = 2
    _, nll_i, rec_i = fit_joint_ie_observer_joint(
        obs_fn,
        utility_names,
        action,
        scen,
        des,
        u_int,
        u_eff,
        tk,
        n_restarts=1,
        init_params=init_inf,
        max_steps=1,
        verbose=False,
        priors=priors,
    )
    assert abs(rec_u[0]["nll"] - rec_i[0]["nll"]) < 1e-4, (
        f"informative m=0.5/nu=2 must nest uniform: "
        f"{rec_u[0]['nll']} vs {rec_i[0]['nll']}"
    )
    print("✓ informative prior (m=0.5, nu=2) nests the uniform-path fit loss")


def _tmpdir():
    """A fresh temp directory as a Path (the file has no shared fixture helper)."""
    return Path(tempfile.mkdtemp())


def _write_priors_fixture(tmp_path, slug, rows):
    import json as _json

    d = tmp_path / "model" / "outputs" / "lm" / slug
    d.mkdir(parents=True)
    with open(d / "lm_priors.jsonl", "w") as f:
        for r in rows:
            f.write(_json.dumps(r) + "\n")
    return d / "lm_priors.jsonl"


def test_load_lm_priors_joint_de_shapes_and_values():
    from tables import SCENARIO_LABELS, load_lm_priors

    rows = [
        {
            "run_id": k,
            "scenario_label": s,
            "intimacy_condition": lvl,
            "prior_desire": 0.7,
            "prior_effort_high": 0.3,
        }
        for k in range(2)
        for s in SCENARIO_LABELS
        for lvl in INTIMACY_CONDITIONS
    ]
    path = _write_priors_fixture(_tmpdir(), "food_inv_joint_de", rows)
    out = load_lm_priors("food_inv_joint_de", filename=str(path))
    assert out["desire_m"].shape == (2, 16, 4)
    assert out["effort_p"].shape == (2, 16, 4)
    # Stored float32 (as every loader here), so compare with a float32 tolerance.
    assert abs(float(out["desire_m"][0, 0, 0]) - 0.7) < 1e-6
    print("✓ load_lm_priors joint_de shapes and values")


def test_load_lm_priors_missing_cell_raises():
    from tables import SCENARIO_LABELS, load_lm_priors

    rows = [
        {
            "run_id": 0,
            "scenario_label": s,
            "intimacy_condition": lvl,
            "prior_desire": 0.5,
            "prior_effort_high": 0.5,
        }
        for s in SCENARIO_LABELS
        for lvl in INTIMACY_CONDITIONS
    ][:-1]  # drop one cell
    path = _write_priors_fixture(_tmpdir(), "food_inv_joint_de", rows)
    try:
        load_lm_priors("food_inv_joint_de", filename=str(path))
        assert False, "expected ValueError on missing cell"
    except ValueError:
        print("✓ load_lm_priors raises on a missing cell")


def test_load_lm_priors_out_of_range_raises():
    from tables import SCENARIO_LABELS, load_lm_priors

    rows = [
        {
            "run_id": 0,
            "scenario_label": s,
            "intimacy_condition": lvl,
            "prior_desire": 1.7,  # out of [0, 1]
            "prior_effort_high": 0.5,
        }
        for s in SCENARIO_LABELS
        for lvl in INTIMACY_CONDITIONS
    ]
    path = _write_priors_fixture(_tmpdir(), "food_inv_joint_de", rows)
    try:
        load_lm_priors("food_inv_joint_de", filename=str(path))
        assert False, "expected ValueError on out-of-range scalar"
    except ValueError:
        print("✓ load_lm_priors raises on an out-of-range scalar")


def test_load_lm_priors_base_broadcasts_relationship():
    from tables import SCENARIO_LABELS, load_lm_priors

    rows = [
        {
            "run_id": 0,
            "scenario_label": s,
            "prior_desire": 0.6,
            "prior_effort_high": 0.4,
        }
        for s in SCENARIO_LABELS
    ]
    d = _tmpdir() / "model" / "outputs" / "lm" / "food_inv_joint_de"
    d.mkdir(parents=True)
    import json as _json

    with open(d / "lm_priors_base.jsonl", "w") as f:
        for r in rows:
            f.write(_json.dumps(r) + "\n")
    out = load_lm_priors(
        "food_inv_joint_de", base=True, filename=str(d / "lm_priors_base.jsonl")
    )
    assert out["desire_m"].shape == (1, 16, 4)
    assert jnp.allclose(out["desire_m"][0, :, 0], out["desire_m"][0, :, 3])
    print("✓ load_lm_priors base broadcasts across the relationship axis")


def _load_prompts_module():
    """Load model/lm/prompts.py in isolation (no package import machinery), so
    the prompt tests exercise the source file directly."""
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "prompts", Path(__file__).resolve().parent / "lm" / "prompts.py"
    )
    prompts = _ilu.module_from_spec(spec)
    spec.loader.exec_module(prompts)
    return prompts


def _load_generate_alternatives_module():
    """Load model/lm/generate_alternatives.py in isolation, for its importable,
    API-free helpers (e.g. `_output_path_for`)."""
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "generate_alternatives",
        Path(__file__).resolve().parent / "lm" / "generate_alternatives.py",
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_alternatives_prompt_arms():
    prompts = _load_prompts_module()

    base_sp = prompts.alternatives_system_prompt()
    assert base_sp == prompts.ALTERNATIVES_SYSTEM_PROMPT
    assert "nothing is shared at all" not in base_sp
    hint_sp = prompts.alternatives_system_prompt(refusal_hint=True)
    assert (
        "including options where nothing is shared at all (declining, "
        "keeping it to oneself, or forgoing it)" in hint_sp
    )
    # the clause replaces the differ-in sentence's period, once
    assert hint_sp.count("nothing is shared at all") == 1

    up = prompts.alternatives_user_prompt(
        "VIG.",
        "They shared.",
        intimacy_level="max_formal",
        effort_hypotheses=("LOW PARA.", "HIGH PARA."),
        unknown_desire_object="the hot dog",
    )
    assert "One of the following is true of the situation" in up
    assert up.index("LOW PARA.") < up.index("HIGH PARA.")
    assert "do not know how much the two people want the hot dog" in up
    # epistemic block sits after the condition paragraphs, before the action
    assert up.index("maximally formal") < up.index("One of the following")
    assert up.index("One of the following") < up.index("They shared.")

    up2 = prompts.alternatives_user_prompt("VIG.", "ACT.", unknown_intimacy=True)
    assert "do not know how close or formal" in up2

    # Output-path suffixing (the vintage side files each arm writes).
    ga = _load_generate_alternatives_module()
    assert (
        ga._output_path_for("food_inv_joint_de", False, "refusal_hint").name
        == "lm_alternatives_refusal_hint.jsonl"
    )
    assert (
        ga._output_path_for("food_inv_desire", True, "refusal_hint_hyp").name
        == "lm_alternatives_base_refusal_hint_hyp.jsonl"
    )
    print("✓ alternatives prompt arms + suffixed output paths")


def test_prior_prompts_compose_condition_texts():
    prompts = _load_prompts_module()

    up = prompts.prior_desire_user_prompt(
        "VIGNETTE.", "the hot dog", condition_texts=("REL.", "EFFORT.")
    )
    assert "VIGNETTE." in up and "REL." in up and "EFFORT." in up
    assert "the hot dog" in up and '"desire"' in up
    assert prompts.PRIOR_DESIRE_SYSTEM_PROMPT.startswith(
        "You are a participant in a human study"
    )

    ue = prompts.prior_effort_user_prompt("VIGNETTE.", "LOW TEXT.", "HIGH TEXT.")
    assert ue.index("LOW TEXT.") < ue.index("HIGH TEXT.")  # low = 0 endpoint
    assert '"effort"' in ue

    ui = prompts.prior_intimacy_user_prompt("VIGNETTE.", condition_texts=("DESIRE.",))
    assert "relationship" in ui and '"intimacy"' in ui
    assert "0" in prompts.PRIOR_INTIMACY_SYSTEM_PROMPT
    print("✓ prior prompts compose condition texts and expose the scalar keys")


def test_elicit_priors_cell_grids():
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "elicit_priors", Path(__file__).resolve().parent / "lm" / "elicit_priors.py"
    )
    ep = _ilu.module_from_spec(spec)
    spec.loader.exec_module(ep)

    cells_1a = ep._build_prior_cells("food_inv_desire", base=False)
    assert len(cells_1a) == 16 * 2 * 4  # scenario x effort x relationship
    assert cells_1a[0]["quantities"] == ("prior_desire",)
    assert len(ep._build_prior_cells("food_inv_desire", base=True)) == 16 * 2

    cells_1b = ep._build_prior_cells("food_inv_joint_de", base=False)
    assert len(cells_1b) == 16 * 4
    assert set(cells_1b[0]["quantities"]) == {"prior_desire", "prior_effort_high"}
    # relationship sentence present in condition_texts, effort paragraphs held
    # out of condition_texts (they are the effort question's endpoints)
    assert any("relationship" in t for t in cells_1b[0]["condition_texts"])

    cells_2a = ep._build_prior_cells("food_inv_intimacy", base=False)
    assert len(cells_2a) == 16 * 2 * 2
    assert cells_2a[0]["quantities"] == ("prior_intimacy",)
    assert len(cells_2a[0]["condition_texts"]) == 2  # desire + effort paragraphs

    cells_2b = ep._build_prior_cells("food_inv_joint_ie", base=False)
    assert len(cells_2b) == 16 * 2
    assert set(cells_2b[0]["quantities"]) == {"prior_intimacy", "prior_effort_high"}
    print("✓ elicit_priors cell grids have the expected shapes and quantities")


def run_all_tests():
    print("=" * 60)
    print("Active model compliance tests")
    print("=" * 60)
    test_utility_ablation_algebra()
    test_discomfort_only_invariant_to_desire()
    test_observer_desire_posterior_sums_to_one()
    test_observer_joint_de_posterior_sums_to_one()
    test_joint_observers_match_memo_reference()
    test_joint_observer_gradients_match_memo_reference()
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
    test_beta_prior_on_grid_normalizes_and_nests_uniform()
    test_reweight_grid_uniform_weights_is_identity()
    test_reweight_joint_matches_manual_and_nests_uniform()
    test_informative_prior_nests_uniform_fit_loss()
    test_load_lm_priors_joint_de_shapes_and_values()
    test_load_lm_priors_missing_cell_raises()
    test_load_lm_priors_out_of_range_raises()
    test_load_lm_priors_base_broadcasts_relationship()
    test_alternatives_prompt_arms()
    test_prior_prompts_compose_condition_texts()
    test_elicit_priors_cell_grids()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
