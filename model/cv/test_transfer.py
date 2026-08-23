"""Tests for the cross-study transfer analysis (`model/cv/transfer.py`) and the
masked-fit machinery it rests on (`_helpers.fit_masked`, `RunOverride`).

The point of most of these is that the REPORTED path must not have moved: a
`free_mask` of None has to reach `_fit_multistart` with exactly the arguments it
saw before, and a `RunOverride` of None has to leave `_run_loso` as it was.

Run standalone: uv run python model/cv/test_transfer.py
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))
sys.path.insert(0, str(_project_root / "model" / "cv"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import _helpers  # noqa: E402
import transfer  # noqa: E402
from _inverse_dispatcher import RunOverride, _free_mask  # noqa: E402
from observers import VARIANT_PARAM_NAMES  # noqa: E402
from study_registry import STUDIES, reported_base  # noqa: E402


def check(name, cond):
    print(f"{'✓' if cond else '✗'} {name}")
    assert cond, name


# A convex toy loss with a known optimum, so a masked fit's answer is checkable
# without running an observer.
TARGET = jnp.array([1.0, 2.0, 3.0, 4.0])


def toy(p):
    return jnp.sum((p - TARGET) ** 2)


def test_none_mask_delegates():
    """free_mask=None must reproduce `_fit_multistart` exactly — this is the
    reported fits' path, and any divergence here moves published numbers."""
    kw = dict(n_restarts=2, lr=0.3, max_steps=300, verbose=False, seed_key="k")
    a, na, ra = _helpers.fit_masked(toy, 4, init_params=np.full(4, 9.0), **kw)
    b, nb, rb = _helpers._fit_multistart(toy, 4, init_params=np.full(4, 9.0), **kw)
    check(
        "free_mask=None delegates to _fit_multistart bit-for-bit",
        bool(np.array_equal(np.asarray(a), np.asarray(b))) and na == nb and ra == rb,
    )
    # ... including without an init, where the basin seeds fire.
    a, na, _ = _helpers.fit_masked(toy, 4, alpha_obs_index=0, **kw)
    b, nb, _ = _helpers._fit_multistart(toy, 4, alpha_obs_index=0, **kw)
    check(
        "free_mask=None delegates with basin seeds too",
        bool(np.array_equal(np.asarray(a), np.asarray(b))) and na == nb,
    )


def test_frozen_slots_do_not_move():
    init = np.array([9.0, 9.0, 9.0, 9.0])
    mask = np.array([False, False, True, True])
    p, nll, records = _helpers.fit_masked(
        toy,
        4,
        free_mask=mask,
        init_params=init,
        n_restarts=1,
        lr=0.3,
        max_steps=800,
        verbose=False,
        seed_key="k",
    )
    p = np.asarray(p)
    check("frozen slots keep their init value", bool(np.array_equal(p[:2], init[:2])))
    check(
        "free slots reach their optimum",
        bool(np.allclose(p[2:], [3.0, 4.0], atol=1e-2)),
    )
    check(
        "records are mapped back to full-length vectors",
        all(len(r["final_params"]) == 4 for r in records),
    )


def test_empty_mask_estimates_nothing():
    """The zero-free-parameter arm: no optimizer runs, and the returned NLL is
    the loss AT the transferred vector — otherwise the `frozen` arm would be
    quietly fitting something."""
    init = np.array([9.0, 8.0, 7.0, 6.0])
    p, nll, records = _helpers.fit_masked(
        toy, 4, free_mask=np.zeros(4, bool), init_params=init
    )
    check(
        "empty mask returns the init untouched",
        bool(np.array_equal(np.asarray(p), init)),
    )
    check("empty mask returns the loss at the init", abs(nll - float(toy(init))) < 1e-6)
    check("empty mask records a single non-fit", len(records) == 1)


def test_mask_validation():
    def raises(fn, needle):
        try:
            fn()
        except ValueError as e:
            return needle in str(e)
        return False

    check(
        "a mask without an init is rejected",
        raises(
            lambda: _helpers.fit_masked(toy, 4, free_mask=np.ones(4, bool)),
            "needs init_params",
        ),
    )
    check(
        "a wrong-length mask is rejected",
        raises(
            lambda: _helpers.fit_masked(
                toy, 4, free_mask=np.ones(3, bool), init_params=np.zeros(4)
            ),
            "expected (4,)",
        ),
    )


def test_run_override_defaults_are_inert():
    o = RunOverride()
    check(
        "an empty RunOverride selects every variant and overrides nothing",
        o.variants == ()
        and o.init_params is None
        and o.free_mask is None
        and o.outputs_dir is None
        and o.fingerprint is None,
    )
    # `_free_mask` reads worker state; with none set it must report "estimate
    # everything", which is what the reported fold refits do.
    from _inverse_dispatcher import _CV_W

    saved = _CV_W.get("free_mask")
    _CV_W["free_mask"] = None
    check(
        "no override means fold refits estimate every parameter",
        _free_mask("full") is None,
    )
    _CV_W["free_mask"] = saved


def test_pairs_are_well_formed():
    labels = {s.short_label for s in STUDIES.values()}
    check(
        "every pair names two real studies",
        all({d, r} <= labels for d, r, _ in transfer.PAIRS),
    )
    check(
        "every pair has a known kind",
        all(k in transfer.PAIR_KINDS for _, _, k in transfer.PAIRS),
    )
    check(
        "every pair is run in both directions",
        all((r, d, k) in transfer.PAIRS for d, r, k in transfer.PAIRS),
    )
    check("no study is paired with itself", all(d != r for d, r, _ in transfer.PAIRS))
    check(
        "pairs are unique",
        len({(d, r) for d, r, _ in transfer.PAIRS}) == len(transfer.PAIRS),
    )


def test_roles_transfer_the_same_utility_terms():
    """A role must price the same utility terms on both sides of every pair, or
    the donor's vector cannot be laid into the recipient's."""
    ok = True
    for donor_label, recipient_label, _ in transfer.PAIRS:
        donor, recipient = transfer._slug(donor_label), transfer._slug(recipient_label)
        for role in transfer.ROLES:
            d_names = VARIANT_PARAM_NAMES[_base_role(donor, role)]
            r_names = VARIANT_PARAM_NAMES[_base_role(recipient, role)]
            ok &= d_names == r_names
    check("each role prices the same utility terms in donor and recipient", ok)


def _base_role(slug, role):
    """The VARIANT_PARAM_NAMES key for a role in one study. `base_shared` shares
    `base`'s utility, so it is not a key of its own."""
    variant = reported_base(slug) if role == "base" else role
    return "base" if variant.startswith("base") else variant


def test_eta_policy():
    donor_with = {"param_eta": 2.5}
    donor_without = {}
    eta, rule = transfer._eta_policy(donor_with, recipient_has_eta=True)
    check("a donor eta transfers when the recipient's scope wants one", eta == 2.5)
    eta, rule = transfer._eta_policy(donor_without, recipient_has_eta=True)
    check(
        "a donor with no reweighting gives eta = 0, not a borrowed gain",
        eta == 0.0 and "preregistered" in rule,
    )
    eta, rule = transfer._eta_policy(donor_with, recipient_has_eta=False)
    check("a recipient with no reweighting drops the donor's eta", eta is None)


def test_free_mask_layout():
    """`refit` frees exactly the response layer; `frozen` frees nothing."""
    check(
        "frozen estimates nothing",
        not transfer._free_mask(4, 7, "frozen").any(),
    )
    m = transfer._free_mask(4, 7, "refit")
    check(
        "refit frees alpha_observer, sigma and eta but no utility weight",
        (~m[:4]).all() and m[4:].all(),
    )


def test_tags_are_distinct_per_donor_and_arm():
    tags = {
        transfer._tag(d, arm) for d, _, _ in transfer.PAIRS for arm in transfer.ARMS
    }
    check(
        "each (donor, arm) gets its own output tag",
        len(tags) == len({(d, a) for d, _, _ in transfer.PAIRS for a in transfer.ARMS}),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Cross-study transfer tests")
    print("=" * 60)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 - report, don't abort the suite
            failures.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print("=" * 60)
    if failures:
        print(f"{len(failures)} of {len(tests)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print(f"All {len(tests)} tests passed!")
    print("=" * 60)
