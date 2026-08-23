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
    _observer_desire_base_memo_reference,
    _observer_desire_discomfort_only_memo_reference,
    _observer_desire_full_memo_reference,
    _observer_intimacy_base_memo_reference,
    _observer_intimacy_discomfort_only_memo_reference,
    _observer_intimacy_full_memo_reference,
    _observer_joint_de_base_memo_reference,
    _observer_joint_de_discomfort_only_memo_reference,
    _observer_joint_de_full_memo_reference,
    _observer_joint_ie_base_memo_reference,
    _observer_joint_ie_discomfort_only_memo_reference,
    _observer_joint_ie_full_memo_reference,
    observer_desire_base,
    observer_desire_discomfort_only,
    observer_desire_full,
    observer_intimacy_base,
    observer_intimacy_discomfort_only,
    observer_intimacy_full,
    observer_joint_de_base,
    observer_joint_de_discomfort_only,
    observer_joint_de_full,
    observer_joint_ie_base,
    observer_joint_ie_discomfort_only,
    observer_joint_ie_full,
    _sharpened_joint_posterior,
)
from tables import (
    INTIMACY_CONDITIONS,
    MAX_ACTIONS,
    N_ACTIONS,
    RELATIONSHIP_LEVEL_VALUES,
    SCENARIO_LABELS,
    DesireConditions,
    EffortConditions,
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
    must match the memo reference's wherever the reference is numerically
    healthy, and — since the 2026-07-29 log-space sharpening — must be FINITE
    even where the reference's power-form gradient degrades. (Before that
    change both implementations shared the fragility and the test pinned the
    fast path to the reference's exact NaN pattern; the log-space path
    deliberately removes it, which is why fits must be regenerated rather than
    mixed across the change.) Readout is restricted to slot 0 (the only slot
    the fit and CV consume)."""
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

        # 1.3 is the everyday regime (reference healthy, so the value
        # comparison bites); 10.3 is the old fitted scale; 25.0 and 60.0 are
        # the region the log-space rewrite exists to open up, where only the
        # finiteness requirement applies. Without the last two this test
        # provides no evidence in the regime that matters.
        for alpha_obs in (1.3, 10.3, 25.0, 60.0):
            for argnum in (0, 1):  # d/dw_v and d/dalpha_observer
                g_fast = jax.grad(
                    lambda wv, ao: readout(fast_fn, wv, ao), argnums=argnum
                )(jnp.float32(_W_V), jnp.float32(alpha_obs))
                g_ref = jax.grad(
                    lambda wv, ao: readout(ref_fn, wv, ao), argnums=argnum
                )(jnp.float32(_W_V), jnp.float32(alpha_obs))
                assert np.isfinite(float(g_fast)), (
                    f"{family} (alpha_obs={alpha_obs}): fast gradient "
                    f"(argnum {argnum}) is non-finite — the log-space path "
                    "must stay differentiable at fitted-scale alpha"
                )
                if np.isfinite(float(g_ref)):
                    assert np.allclose(float(g_fast), float(g_ref), rtol=1e-3), (
                        f"{family} (alpha_obs={alpha_obs}): gradient "
                        f"(argnum {argnum}) fast={float(g_fast):.6g} vs "
                        f"reference={float(g_ref):.6g}"
                    )
    print("✓ fast joint observer gradients match the memo reference")


def _synthetic_single_tables(family, seed=11):
    """Seeded synthetic tables for one single-latent family, shaped like one
    elicitation run's slice (see .claude/rules/model.md, per-study shapes).
    Null-slot priors are 1e-8 so the memo reference is NaN-free (as in
    `_synthetic_joint_tables`)."""
    rng = np.random.default_rng(seed)
    if family == "desire":
        cell = (N_S, N_O, len(EffortConditions), len(RelationshipConditions))
    else:  # intimacy
        cell = (N_S, N_O, len(DesireConditions), len(EffortConditions))
    risk = rng.uniform(0.0, 1.0, size=(*cell, S)).astype(np.float32)
    g = rng.uniform(0.0, 1.0, size=(*cell, S)).astype(np.float32)
    effort = rng.uniform(0.0, 1.0, size=(*cell, S)).astype(np.float32)
    prior = np.full((*cell, S), 1e-8, dtype=np.float32)
    prior[..., :N_ACTIONS] = 1.0 / N_ACTIONS
    tables = dict(
        risk_table=jnp.array(risk),
        effort_table=jnp.array(effort),
        g_padded_table=jnp.array(g),
        prior_table=jnp.array(prior),
    )
    if family == "desire":
        tables["relationship_values"] = _REL
    else:
        tables["desire_table"] = jnp.array(
            rng.uniform(0.1, 1.0, size=(N_S, len(DesireConditions))).astype(np.float32)
        )
    return tables


_SINGLE_OBSERVER_CASES = [
    (
        "desire_full",
        observer_desire_full,
        _observer_desire_full_memo_reference,
        "desire",
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
        "desire_discomfort_only",
        observer_desire_discomfort_only,
        _observer_desire_discomfort_only_memo_reference,
        "desire",
        VARIANT_PARAM_NAMES["discomfort_only"],
        ("risk_table", "effort_table", "prior_table", "relationship_values"),
    ),
    (
        "desire_base",
        observer_desire_base,
        _observer_desire_base_memo_reference,
        "desire",
        VARIANT_PARAM_NAMES["base"],
        ("risk_table", "effort_table", "g_padded_table", "prior_table"),
    ),
    (
        "intimacy_full",
        observer_intimacy_full,
        _observer_intimacy_full_memo_reference,
        "intimacy",
        VARIANT_PARAM_NAMES["full"],
        ("risk_table", "effort_table", "g_padded_table", "prior_table", "desire_table"),
    ),
    (
        "intimacy_discomfort_only",
        observer_intimacy_discomfort_only,
        _observer_intimacy_discomfort_only_memo_reference,
        "intimacy",
        VARIANT_PARAM_NAMES["discomfort_only"],
        ("risk_table", "effort_table", "prior_table"),
    ),
    (
        "intimacy_base",
        observer_intimacy_base,
        _observer_intimacy_base_memo_reference,
        "intimacy",
        VARIANT_PARAM_NAMES["base"],
        ("risk_table", "effort_table", "g_padded_table", "prior_table", "desire_table"),
    ),
]


def test_single_latent_observers_match_memo_reference():
    """The fast single-latent observers (2026-07-29 conversion) must reproduce
    the original memo observers on every slot, cell, and latent bin, across
    ablations and alpha_observer regimes where the reference is numerically
    healthy — the same guarantee the joint conversion carries."""
    for family in ("desire", "intimacy"):
        tables = _synthetic_single_tables(family)
        for label, fast_fn, ref_fn, fam, weights, table_names in _SINGLE_OBSERVER_CASES:
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
                assert np.isfinite(fast).all(), f"{label}: fast path non-finite"
                assert np.allclose(fast, ref, rtol=1e-4, atol=1e-6), (
                    f"{label} (alpha_obs={alpha_obs}): fast observer deviates "
                    f"from memo reference; max abs diff "
                    f"{np.abs(fast - ref).max():.3e}"
                )
    print("✓ fast single-latent observers match the memo reference (all variants)")


def test_sharpening_survives_high_alpha():
    """The log-space sharpening must stay a valid probability distribution AND
    stay differentiable at alpha_observer far beyond the float32 power-form's
    underflow point (~15-20 for diffuse rows), where the old formulation
    collapsed entire rows to zero and renormalized to garbage — the failure that
    silently fenced fits out of the high-alpha region.

    Two row shapes, because they exercise different failure modes:
      - fully positive diffuse: the ORIGINAL power-form bug (forward values).
      - MIXED (diffuse positives beside exact zeros, as float32 underflow of the
        actor softmax produces for extreme latent hypotheses): the regime where
        guarding `exp`'s output but not its argument gave 0 * inf = NaN in the
        backward pass, poisoning the whole row through the shared row max. That
        was a real bug shipped and caught in review on 2026-07-29; this is its
        regression test, so it must check GRADIENTS, not just values.
    """
    rng = np.random.default_rng(3)

    def build(n_zero_latents, p_obs=1e-3):
        """A SURPRISING observed action: slot 0 carries probability ~p_obs under
        every latent hypothesis, with the remaining mass spread over the other
        active slots (the actor policy is normalized over slots, as produced by
        the actor memos). The magnitude is what matters — the row max enters the
        exponent as alpha * log(p_obs), so a slot that is merely one-of-three
        likely (~0.33) never overflows and would make this test vacuous, while a
        genuinely unlikely observed action (~1e-3, routine for a surprising
        action against 12 alternatives) drives alpha * log p to ~-150."""
        policy = np.zeros((S, 101, 2), dtype=np.float32)
        obs = p_obs * rng.uniform(0.9, 1.1, size=(101, 2))
        rest = rng.uniform(0.9, 1.1, size=(N_ACTIONS - 1, 101, 2))
        rest *= (1.0 - obs) / rest.sum(0, keepdims=True)
        policy[0] = obs.astype(np.float32)
        policy[1:N_ACTIONS] = rest.astype(np.float32)
        if n_zero_latents:  # exact zeros inside otherwise-valid slots
            policy[:N_ACTIONS, :n_zero_latents, :] = 0.0
        return jnp.array(policy)

    for label, n_zero in (("fully positive", 0), ("mixed with exact zeros", 5)):
        policy = build(n_zero)
        for alpha_obs in (22.0, 60.0):
            post = np.asarray(_sharpened_joint_posterior(policy, alpha_obs))
            assert np.isfinite(post).all(), (
                f"{label}: non-finite forward at alpha={alpha_obs}"
            )
            sums = post[:N_ACTIONS].sum(axis=(-2, -1))
            assert np.allclose(sums, 1.0, atol=1e-4), (
                f"{label}: active-slot posteriors do not normalize at "
                f"alpha={alpha_obs}: {sums} (the old power-form gave all-zero rows)"
            )
            assert np.allclose(post[N_ACTIONS:], 0.0), "null slots must stay zero"
            if n_zero:
                assert np.allclose(post[:N_ACTIONS, :n_zero, :], 0.0), (
                    f"{label}: zero-policy latents must stay zero"
                )

            def readout(a, policy=policy):
                return jnp.log(_sharpened_joint_posterior(policy, a)[0] + 1e-9).sum()

            g = float(jax.grad(readout)(jnp.float32(alpha_obs)))
            assert np.isfinite(g), (
                f"{label}: d/d_alpha_observer is non-finite at alpha={alpha_obs} "
                "— sanitize the exponent BEFORE exp (double-where), not only "
                "the result"
            )
    print("✓ log-space sharpening survives high alpha_observer (values + gradients)")


def test_single_latent_observer_gradients_are_finite():
    """The six fast single-latent observers must be differentiable across the
    alpha_observer range fits now explore, including past the old float32 cliff.
    No gradient test existed for this family before the 2026-07-29 conversion
    (only the joint families had one), which is part of why the exp-guard bug
    went unnoticed."""
    for family in ("desire", "intimacy"):
        tables = _synthetic_single_tables(family)
        for label, fast_fn, _ref, fam, weights, table_names in _SINGLE_OBSERVER_CASES:
            if fam != family:
                continue
            table_args = [tables[t] for t in table_names]
            w_first = weights[0]  # w_v for full/base, w_d for discomfort_only

            def readout(wv, ao, fn=fast_fn, weights=weights, table_args=table_args):
                vals = [
                    wv if w == weights[0] else _JOINT_WEIGHT_VALUES[w] for w in weights
                ]
                out = fn(_ALPHA, *vals, ao, *table_args)
                return jnp.log(out[0] + 1e-9).sum()  # slot 0 only

            for alpha_obs in (1.3, 10.3, 25.0):
                for argnum in (0, 1):
                    g = float(
                        jax.grad(readout, argnums=argnum)(
                            jnp.float32(_JOINT_WEIGHT_VALUES[w_first]),
                            jnp.float32(alpha_obs),
                        )
                    )
                    assert np.isfinite(g), (
                        f"{label} (alpha_obs={alpha_obs}): gradient "
                        f"(argnum {argnum}) is non-finite"
                    )
    print("✓ fast single-latent observer gradients are finite across alpha regimes")


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


def test_parse_run_config_args_defaults():
    """No flags → the default RunConfig (reweighting on), which routes outputs
    to outputs/<slug>/. This is the reported config, so a plain invocation of
    any fit or CV script must keep reproducing the paper's numbers in place."""
    from _helpers import parse_run_config_args

    from run_config import RunConfig

    cfg = parse_run_config_args([])
    assert cfg == RunConfig(), f"defaults not the reported config: {cfg}"
    assert cfg.is_default
    assert not cfg.no_reweighting
    print("✓ parse_run_config_args defaults to the reported config")


def test_parse_run_config_args_no_reweighting():
    """--no-reweighting must reach the scope rule and empty it for every
    (study, variant) — the switch is what makes the preregistered model
    runnable, and a config that parsed the flag but left the reweighting on
    would look right while fitting the reported model into the prereg's
    directory."""
    import _reweighting
    from _helpers import parse_run_config_args

    cfg = parse_run_config_args(["--no-reweighting"])
    assert cfg.no_reweighting
    assert not cfg.is_default
    # Every reweighted (study, variant) pair in the roster must go quiet, and
    # `None` is the fit helpers' preregistered path.
    reweighted = [
        (slug, "full")
        for slug, targets in _reweighting.STUDY_CONTRASTIVE.items()
        if targets
    ]
    assert reweighted, "no reweighted studies to check — STUDY_CONTRASTIVE is empty"
    for slug, variant in reweighted:
        names = ["w_v", "w_d", "w_e", "gamma"]
        assert _reweighting.uses_reweighting(slug, names), (
            f"{slug}/{variant} is not reweighted by default — fix the fixture"
        )
        assert not _reweighting.uses_reweighting(
            slug, names, enabled=not cfg.no_reweighting
        ), f"{slug}/{variant} still reweights under --no-reweighting"
        assert (
            _reweighting.config_for(
                slug, variant, names, enabled=not cfg.no_reweighting
            )
            is None
        ), f"{slug}/{variant} still built a reweighting config"
    print("✓ --no-reweighting disables the reweighting for every study")


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


def test_alternatives_prompt_latent_awareness():
    prompts = _load_prompts_module()

    # No refusal clause anywhere; single system prompt form.
    assert "nothing is shared at all" not in prompts.ALTERNATIVES_SYSTEM_PROMPT

    # Latent-awareness is always driven by the caller's per-study kwargs. 1b
    # (infers desire + effort): effort hypotheses + desire unknown, inserted
    # after the condition paragraphs and before the observed action.
    up = prompts.alternatives_user_prompt(
        "VIG.",
        "They shared.",
        intimacy_level="max_formal",
        effort_hypotheses=("LOW PARA.", "HIGH PARA."),
        unknown_desire_object="the hot dog",
    )
    assert "One of the following is true of the situation" in up
    assert up.index("LOW PARA.") < up.index("HIGH PARA.")
    # following the effort block (1b/3a), the desire line says "also"
    assert "You also do not know how much the two people would like the hot dog" in up
    assert up.index("maximally formal") < up.index("One of the following")
    assert up.index("One of the following") < up.index("They shared.")

    # 1a form: desire line is the sole epistemic statement — no dangling "also"
    up_1a = prompts.alternatives_user_prompt(
        "VIG.", "ACT.", unknown_desire_object="the hot dog"
    )
    assert "You do not know how much the two people would like the hot dog" in up_1a
    assert "also" not in up_1a

    # 2a/2b form: relationship unknown.
    up2 = prompts.alternatives_user_prompt("VIG.", "ACT.", unknown_intimacy=True)
    assert "do not know how formal or intimate" in up2

    # Output path is the single default vintage (no arm suffix).
    ga = _load_generate_alternatives_module()
    assert (
        ga._output_path_for("food_inv_joint_de", False).name == "lm_alternatives.jsonl"
    )
    assert (
        ga._output_path_for("food_inv_desire", True).name
        == "lm_alternatives_base.jsonl"
    )
    print("✓ alternatives prompt always latent-aware; single default vintage")


# ==============================================================================
# Comparison-set reweighting (model/inverse/_reweighting.py)
#
# This is the likelihood-layer module the reported fits depend on, and it was
# derived from a standalone prototype (model/diagnostics/tier3_surprise.py, a
# local development artifact that is not part of the repository). Two
# implementations of the paper's headline mechanism can drift apart silently, so
# the reweighting is pinned two ways:
#
#   1. against a self-contained numpy oracle written from the SI formula, which
#      needs no prototype to run;
#   2. against the prototype itself, when its local module is importable, so the
#      diagnostics and the reported pipeline are demonstrably the same math.
#
# Same discipline as the observer `_memo_reference` tests above.
# ==============================================================================


def _oracle_gated_prior(prior, surprise, v, eta, floor=0.0, active_eps=1e-6):
    """Independent numpy reference for p_eta(a) ∝ p(a) · exp(eta · S · v(a)),
    normalized over the active slots of each row.

    Written from the SI statement of the weighting rather than from the JAX
    code, so it is a real oracle: it computes the multiplicative form directly
    instead of reproducing the log-space masked-softmax the implementation uses
    for numerical stability. Agreement is therefore evidence the stable form is
    correct, not just self-consistent.
    """
    prior = np.asarray(prior, dtype=np.float64)
    active = prior > active_eps
    w = np.where(active, prior * np.exp(eta * np.asarray(surprise)[..., None] * v), 0.0)
    denom = w.sum(-1, keepdims=True)
    out = np.where(active, w / np.where(denom > 0.0, denom, 1.0), floor)
    if floor:
        out = out / out.sum(-1, keepdims=True)
    return out


def _reweighting_fixture(seed=0, n_slots=6):
    """A prior table with a realistic mix: most slots active, some exactly zero
    (null padding), and one row deliberately near-degenerate."""
    rng = np.random.default_rng(seed)
    prior = rng.random((2, 4, n_slots)) + 0.05
    prior[..., -2:] = 0.0  # null-padded slots, as the loaders produce
    prior[0, 0, 1:] = 0.0  # a row with a single active slot
    prior = prior / np.where(
        prior.sum(-1, keepdims=True) > 0, prior.sum(-1, keepdims=True), 1.0
    )
    surprise = rng.random((2, 4)) * 4.0  # nats
    v = rng.random((2, 4, n_slots))  # sensitivity scores on the [0,1] feature scale
    return prior, surprise, v


def test_reweighting_matches_numpy_oracle():
    """The production weighting equals the multiplicative form it implements in
    log space, on well-formed rows and across the fitted eta range."""
    import _reweighting

    prior, surprise, v = _reweighting_fixture()
    for eta in (0.0, 0.5, 1.01, 2.34, 3.01, 8.0):
        got = np.asarray(
            _reweighting.gated_prior(
                jnp.asarray(prior), jnp.asarray(surprise), jnp.asarray(v), eta
            ),
            dtype=np.float64,
        )
        want = _oracle_gated_prior(prior, surprise, v, eta)
        assert np.allclose(got, want, atol=1e-6), (
            f"eta={eta}: max |diff| = {np.abs(got - want).max():.2e}"
        )
        # Rows with any active slot must be proper distributions; inactive slots
        # must stay exactly zero at floor=0 (they are null padding, not actions).
        rows = got.reshape(-1, got.shape[-1])
        act = (prior > _reweighting.ACTIVE_EPS).reshape(-1, got.shape[-1])
        for r, a in zip(rows, act):
            if a.any():
                assert abs(r.sum() - 1.0) < 1e-6
                assert np.all(r[~a] == 0.0)
    print("✓ gated_prior matches the numpy oracle for the weighting formula")


def test_reweighting_nests_the_unreweighted_model_at_eta_zero():
    """eta = 0 must return the input prior exactly. The paper reports the
    reweighted model as nesting the preregistered one, and every nesting check
    and likelihood-ratio statement depends on this holding to float precision.
    """
    import _reweighting

    prior, surprise, v = _reweighting_fixture(seed=3)
    got = np.asarray(
        _reweighting.gated_prior(
            jnp.asarray(prior), jnp.asarray(surprise), jnp.asarray(v), 0.0
        ),
        dtype=np.float64,
    )
    assert np.allclose(got, prior, atol=1e-7), (
        f"not nested at eta=0: max |diff| = {np.abs(got - prior).max():.2e}"
    )
    print("✓ reweighting nests the unreweighted model exactly at eta = 0")


def test_reweighting_survives_an_all_inactive_row():
    """An all-zero prior row (every slot null-padded) must yield zeros, not NaN.

    This is a DELIBERATE difference from the diagnostics prototype, which has no
    such guard and NaNs here — production added the finite-max and zero-denom
    guards because a NaN in one cell poisons the whole fit through the shared
    row max. Pinned so nobody "simplifies" production back down to the
    prototype.
    """
    import _reweighting

    prior = np.zeros((1, 2, 4))
    prior[0, 1, :2] = 0.5  # one good row, one all-inactive row
    surprise = np.array([[1.0, 2.0]])
    v = np.ones((1, 2, 4))
    got = np.asarray(
        _reweighting.gated_prior(
            jnp.asarray(prior), jnp.asarray(surprise), jnp.asarray(v), 2.0
        ),
        dtype=np.float64,
    )
    assert np.all(np.isfinite(got)), f"non-finite output: {got}"
    assert np.all(got[0, 0] == 0.0), "all-inactive row should stay zero"
    assert abs(got[0, 1].sum() - 1.0) < 1e-9, "the good row must still normalize"
    # And the gradient must be finite, since this feeds a fitted objective.
    g = jax.grad(
        lambda e: jnp.sum(
            _reweighting.gated_prior(
                jnp.asarray(prior), jnp.asarray(surprise), jnp.asarray(v), e
            )
        )
    )(2.0)
    assert jnp.isfinite(g), f"non-finite gradient through an all-inactive row: {g}"
    print("✓ reweighting guards all-inactive rows (value and gradient finite)")


def test_reweighting_matches_the_diagnostics_prototype():
    """Production and the diagnostics prototype must be the same math.

    The prototype takes `active` and `log_prior` as arguments where production
    computes them internally; its callers all compute them identically
    (`prior > _ACTIVE_EPS`, `log(clip(prior, 1e-12))`), so feeding those makes
    the comparison exact. Skips with a printed note if model/diagnostics/ has
    been removed — the numpy-oracle test above is the durable guard.
    """
    import _reweighting

    sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnostics"))
    try:
        from tier3_surprise import _ACTIVE_EPS, _gated_prior
    except Exception as e:  # noqa: BLE001 - diagnostics is optional
        print(f"– skipped prototype cross-check (diagnostics unavailable: {e})")
        return

    assert _ACTIVE_EPS == _reweighting.ACTIVE_EPS, (
        f"active-slot threshold diverged: prototype {_ACTIVE_EPS} vs "
        f"production {_reweighting.ACTIVE_EPS}"
    )

    prior, surprise, v = _reweighting_fixture(seed=7)
    # Drop the single-active-slot row: it is fine in both, but the prototype has
    # no all-inactive guard, so keep the comparison to rows both define.
    pj = jnp.asarray(prior)
    active = pj > _ACTIVE_EPS
    log_prior = jnp.log(jnp.clip(pj, 1e-12, None))
    for eta in (0.0, 1.01, 2.34, 3.01):
        proto = np.asarray(
            _gated_prior(
                pj, active, log_prior, jnp.asarray(surprise), jnp.asarray(v), eta
            ),
            dtype=np.float64,
        )
        prod = np.asarray(
            _reweighting.gated_prior(pj, jnp.asarray(surprise), jnp.asarray(v), eta),
            dtype=np.float64,
        )
        assert np.allclose(proto, prod, atol=1e-6), (
            f"eta={eta}: production and prototype disagree, max |diff| = "
            f"{np.abs(proto - prod).max():.2e} — the diagnostics results in "
            "RESULTS.md no longer describe the reported model"
        )
    print("✓ reweighting matches the diagnostics prototype (same math)")


def test_reweighting_scope_rule_matches_the_reported_matrix():
    """The scope rule derives which (study, variant) pairs carry an eta. The
    paper reports 12 of the 18 pairs as reweighted: a target is dropped when the
    variant lacks the utility channel it acts through (base has no w_d, so the
    intimacy target cannot apply; every variant keeps w_e).
    """
    import _reweighting
    from observers import (
        VARIANTS_DESIRE,
        VARIANTS_INTIMACY,
        VARIANTS_JOINT_DE,
        VARIANTS_JOINT_IE,
    )

    registries = {
        "food_inv_desire": VARIANTS_DESIRE,
        "food_inv_intimacy": VARIANTS_INTIMACY,
        "food_inv_joint_de": VARIANTS_JOINT_DE,
        "food_inv_joint_ie": VARIANTS_JOINT_IE,
        "nonfood_inv_joint_de": VARIANTS_JOINT_DE,
        "nonfood_inv_joint_ie": VARIANTS_JOINT_IE,
    }
    n_pairs = n_reweighted = 0
    for slug, reg in registries.items():
        for variant, (_fn, param_names) in reg.items():
            n_pairs += 1
            targets = _reweighting.variant_targets(slug, param_names)
            used = _reweighting.uses_reweighting(slug, param_names)
            assert used == (len(targets) > 0)
            n_reweighted += bool(used)
            # No target may survive without its channel.
            for t in targets:
                assert _reweighting.TARGET_REQUIRES_WEIGHT[t] in param_names, (
                    f"{slug}/{variant}: target {t!r} kept without its weight"
                )
            # Study 1a asks no contrastive-only question: never reweighted.
            if slug == "food_inv_desire":
                assert not used, f"1a/{variant} should carry no eta"
    # 21 pairs: 18 preregistered, plus the exploratory `base_shared` in the three
    # studies whose `base` otherwise uses a relationship-free comparison set
    # (1a, 1b, 3a). 14 reweighted: base_shared inherits base's targets, which is
    # 'world' in the two joint_de studies and none in 1a.
    assert n_pairs == 21, f"expected 21 (study, variant) pairs, got {n_pairs}"
    assert n_reweighted == 14, f"expected 14 reweighted pairs, got {n_reweighted}"
    print(f"✓ reweighting scope rule gives {n_reweighted}/{n_pairs} reweighted pairs")


def test_reweighting_sensitivity_scores_are_feature_contrasts():
    """The two sensitivity scores are the leading-order feature contrasts named
    in the SI: the effort swing across world states, and |risk(a) - risk(a_obs)|
    with the observed action in slot 0 (so its own score is exactly zero)."""
    import _reweighting

    rng = np.random.default_rng(1)
    effort = jnp.asarray(rng.random((2, 3, 2, 5)))  # (..., effort_condition, slot)
    risk = jnp.asarray(rng.random((2, 3, 5)))
    v_world = _reweighting.sensitivity("world", {"effort_table": effort})
    assert np.allclose(
        np.asarray(v_world),
        np.abs(np.asarray(effort)[..., 1, :] - np.asarray(effort)[..., 0, :]),
    )
    v_int = _reweighting.sensitivity("intimacy", {"risk_table": risk})
    assert np.all(np.asarray(v_int)[..., 0] == 0.0), (
        "the observed action must have zero intimacy sensitivity (it is the "
        "contrast baseline)"
    )
    assert np.all(np.asarray(v_int) >= 0.0), "sensitivity scores are magnitudes"
    # Summing is parameter-free and order-independent.
    tk = {"effort_table": effort, "risk_table": risk}
    both = _reweighting.combined_sensitivity(("world", "intimacy"), tk)
    rev = _reweighting.combined_sensitivity(("intimacy", "world"), tk)
    assert np.allclose(np.asarray(both), np.asarray(rev))
    assert np.allclose(np.asarray(both), np.asarray(v_world) + np.asarray(v_int))
    try:
        _reweighting.combined_sensitivity((), tk)
        raise AssertionError("combined_sensitivity(()) should raise")
    except ValueError:
        pass
    print("✓ reweighting sensitivity scores are the documented feature contrasts")


def run_all_tests():
    print("=" * 60)
    print("Active model compliance tests")
    print("=" * 60)
    test_utility_ablation_algebra()
    test_discomfort_only_invariant_to_desire()
    test_observer_desire_posterior_sums_to_one()
    test_observer_joint_de_posterior_sums_to_one()
    test_joint_observers_match_memo_reference()
    test_single_latent_observers_match_memo_reference()
    test_sharpening_survives_high_alpha()
    test_single_latent_observer_gradients_are_finite()
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
    test_parse_run_config_args_defaults()
    test_parse_run_config_args_no_reweighting()
    test_alternatives_prompt_latent_awareness()
    test_reweighting_matches_numpy_oracle()
    test_reweighting_nests_the_unreweighted_model_at_eta_zero()
    test_reweighting_survives_an_all_inactive_row()
    test_reweighting_matches_the_diagnostics_prototype()
    test_reweighting_scope_rule_matches_the_reported_matrix()
    test_reweighting_sensitivity_scores_are_feature_contrasts()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
