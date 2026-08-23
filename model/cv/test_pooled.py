"""Tests for the pooled cross-experiment fits (`model/cv/pooled.py` and
`model/inverse/_pooled.py`).

The load-bearing claims are that the pooled vector is carved up correctly --
a mis-sliced response block would silently give an experiment another
experiment's alpha_observer -- and that the loss factories the pooled objective
sums are the same ones the per-experiment fits use.

Run standalone: uv run python model/cv/test_pooled.py
"""

import sys

import numpy as np

from model.cv import pooled
from model.inverse._pooled import LOSS_FACTORY, build_layout, pooled_init
from study_registry import SLUGS, STUDIES


def check(name, cond):
    print(f"{'✓' if cond else '✗'} {name}")
    assert cond, name


def _layout(has_eta=(False, True, True)):
    slugs = ["a", "b", "c"][: len(has_eta)]
    return build_layout(slugs, 4, dict(zip(slugs, has_eta)))


def test_layout_sizes():
    lay = _layout()
    # 4 shared + (2) + (3) + (3)
    check("n_params counts each experiment's own response block", lay.n_params == 12)
    check(
        "blocks start after the shared utility and do not overlap",
        [b[0] for b in lay.blocks] == [4, 6, 9],
    )


def test_study_slice_is_a_real_fit_vector():
    lay = _layout()
    p = np.arange(lay.n_params, dtype=float)
    s0 = np.asarray(lay.study_slice(p, 0))
    s1 = np.asarray(lay.study_slice(p, 1))
    s2 = np.asarray(lay.study_slice(p, 2))
    check(
        "every slice starts with the SAME shared utility",
        all(np.array_equal(s[:4], p[:4]) for s in (s0, s1, s2)),
    )
    check(
        "an eta-free experiment gets [utility, alpha, sigma]",
        list(s0) == [0, 1, 2, 3, 4, 5],
    )
    check(
        "an eta experiment gets [utility, alpha, sigma, eta]",
        list(s1) == [0, 1, 2, 3, 6, 7, 8],
    )
    check(
        "the last experiment reads its own block, not off the end",
        list(s2) == [0, 1, 2, 3, 9, 10, 11],
    )


def test_slices_do_not_share_response_params():
    """The failure this guards against: an off-by-one in `blocks` handing one
    experiment another's alpha_observer while everything still runs."""
    lay = _layout()
    p = np.arange(lay.n_params, dtype=float)
    resp = [tuple(np.asarray(lay.study_slice(p, i))[4:]) for i in range(3)]
    check(
        "no two experiments share a response slot",
        len({x for r in resp for x in r}) == sum(len(r) for r in resp),
    )


def test_param_names_line_up():
    lay = _layout()
    names = lay.param_names(["w_v", "w_d", "w_e", "gamma"])
    check("one name per slot", len(names) == lay.n_params)
    check(
        "shared utility names come first", names[:4] == ["w_v", "w_d", "w_e", "gamma"]
    )
    check(
        "response names are namespaced by experiment",
        names[4:6] == ["a:alpha_observer", "a:sigma"] and names[8] == "b:eta",
    )


def test_pooled_init_averages_utility_keeps_response():
    lay = _layout(has_eta=(False, True))
    per_study = [
        np.array([1.0, 2.0, 3.0, 4.0, 10.0, 0.2]),
        np.array([3.0, 4.0, 5.0, 6.0, 20.0, 0.3, 7.0]),
    ]
    init = pooled_init(lay, per_study)
    check(
        "utility init is the mean of the separate fits",
        list(init[:4]) == [2.0, 3.0, 4.0, 5.0],
    )
    check(
        "each experiment keeps its own response params",
        list(init[4:]) == [10.0, 0.2, 20.0, 0.3, 7.0],
    )


def test_groups_and_rungs():
    check(
        "every group is a non-empty set of real slugs",
        all(g and set(g) <= set(SLUGS) for _, g in pooled.GROUPS.values()),
    )
    check(
        "rung 3 partitions the six experiments by domain",
        sorted(pooled.GROUPS["food"][1] + pooled.GROUPS["nonfood"][1]) == sorted(SLUGS),
    )
    check("rung 4 is all six", sorted(pooled.GROUPS["all"][1]) == sorted(SLUGS))
    check(
        "food and nonfood groups match the study registry's domains",
        all(STUDIES[s].domain == "food" for s in pooled.GROUPS["food"][1])
        and all(STUDIES[s].domain == "nonfood" for s in pooled.GROUPS["nonfood"][1]),
    )
    check(
        "every rung names known groups",
        all(g in pooled.GROUPS for gs in pooled.RUNGS.values() for g in gs),
    )


def test_loss_factory_covers_every_family():
    from model.inverse._fit_dispatcher import FAMILY_BY_SLUG

    check(
        "each experiment's observer family has a loss factory",
        {FAMILY_BY_SLUG[s] for s in SLUGS} <= set(LOSS_FACTORY),
    )


def test_fold_alignment():
    """Fold k must mean scenario index k in every experiment, so the rungs are
    comparable to each other and to the reported run on matched trials."""
    from model.tables import STUDY_SCENARIO_LABELS as L

    check(
        "every experiment has the same number of folds",
        {len(L[s]) for s in SLUGS} == {pooled.N_FOLDS},
    )
    check(
        "experiments in a domain share their scenario order",
        len({tuple(L[s]) for s in pooled.GROUPS["food"][1]}) == 1
        and len({tuple(L[s]) for s in pooled.GROUPS["nonfood"][1]}) == 1,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Pooled cross-experiment fit tests")
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
