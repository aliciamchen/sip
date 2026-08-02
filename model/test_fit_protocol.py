"""Tests for the shared full-data fit protocol (`model/inverse/_fit_dispatcher.py`).

The six `fit_<slug>.py` wrappers used to carry ~135 lines of identical protocol
each. Consolidating them removed the drift hazard but concentrated the risk: one
file now assembles every reported `fit_results.json`, so a mistake there is a
mistake in all six studies at once. These tests pin the surface that used to be
protected only by the duplication being visually comparable.

The consolidation itself was verified end-to-end by capturing what each wrapper
wrote before and after, with the optimizer stubbed -- byte-identical across all
six studies, and confirmed non-vacuous by injecting an `n_params` bug (5 of 6
studies flagged; 1a correctly unaffected since it carries no eta). That harness
is a one-off; what follows is the durable form.

Run: uv run python model/test_fit_protocol.py
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "model"))
sys.path.insert(0, str(_root / "model" / "inverse"))

import numpy as np  # noqa: E402

import _fit_dispatcher as fd  # noqa: E402
import _reweighting  # noqa: E402
from run_config import INFERRED_LATENTS, RunConfig  # noqa: E402

ROSTER = sorted(INFERRED_LATENTS)


def test_registry_covers_the_roster():
    """Every study routes to a family, and every family is reachable. A study
    missing here fails at fit time, not import time, so pin it."""
    assert sorted(fd.FAMILY_BY_SLUG) == ROSTER, sorted(fd.FAMILY_BY_SLUG)
    assert set(fd.FAMILY_BY_SLUG.values()) == set(fd._FAMILIES), (
        f"families in use {set(fd.FAMILY_BY_SLUG.values())} vs registered "
        f"{set(fd._FAMILIES)}"
    )
    for family, spec in fd._FAMILIES.items():
        for key in (
            "variants",
            "loader",
            "fitter",
            "table_kwargs",
            "tables_take_base",
            "data_names",
        ):
            assert key in spec, f"{family} missing {key!r}"
    print("✓ fit registry covers the roster and every family is complete")


def test_loader_arity_matches_data_names():
    """`data_names` maps the loader's trailing return values onto the fitter's
    keyword arguments. If a loader gains or loses an array, the two silently
    misalign and the fitter receives the wrong array under the right name --
    which produces a plausible fit of the wrong data rather than an error.

    Reads the committed CSVs, so it catches a real signature change.
    """
    for family, spec in fd._FAMILIES.items():
        slug = next(s for s, f in fd.FAMILY_BY_SLUG.items() if f == family)
        data, action, scenario_idx, *rest = spec["loader"](slug)
        assert len(rest) == len(spec["data_names"]), (
            f"{family} ({slug}): loader returns {len(rest)} arrays after "
            f"scenario_idx, data_names lists {len(spec['data_names'])}: "
            f"{spec['data_names']}"
        )
        n = len(action)
        for name, arr in zip(spec["data_names"], rest):
            assert len(arr) == n, f"{family}: {name} length {len(arr)} != {n}"
    print("✓ every family's loader arity matches its data_names")


def test_tables_take_base_matches_the_relationship_studies():
    """Only the given-relationship studies have a base-ablation alternatives
    vintage elicited without the relationship paragraph. 2a/2b/3b infer intimacy
    and never show one, so asking their builders for `base=True` would request a
    file that does not exist."""
    with_base = {
        s for s, f in fd.FAMILY_BY_SLUG.items() if fd._FAMILIES[f]["tables_take_base"]
    }
    assert with_base == {
        "food_inv_desire",
        "food_inv_joint_de",
        "nonfood_inv_joint_de",
    }, with_base
    print("✓ base-vintage families are exactly the given-relationship studies")


def test_domain_routing():
    for slug in ROSTER:
        want = "nonfood" if slug.startswith("nonfood_") else "food"
        assert fd._domain_for(slug) == want, slug
    print("✓ nonfood slugs route to the nonfood stimulus set")


def test_table_builder_passes_domain_and_base_only_where_meaningful():
    """The builder closure must pass `domain` only for nonfood and `base` only
    for families that have a base vintage -- passing either elsewhere is a
    TypeError at fit time."""
    seen = []

    def fake_builder(utility_names, **kwargs):
        seen.append(kwargs)
        return {"risk_table": np.zeros((1, 2, 3))}

    orig = fd._FAMILIES["joint_de"]["table_kwargs"]
    try:
        fd._FAMILIES["joint_de"]["table_kwargs"] = fake_builder
        fd._table_kwargs_builder("joint_de", "food_inv_joint_de")("full", ("w_v",))
        fd._table_kwargs_builder("joint_de", "nonfood_inv_joint_de")("base", ("w_v",))
    finally:
        fd._FAMILIES["joint_de"]["table_kwargs"] = orig
    assert seen[0] == {"base": False}, seen[0]
    assert seen[1] == {"domain": "nonfood", "base": True}, seen[1]

    seen.clear()
    orig = fd._FAMILIES["joint_ie"]["table_kwargs"]
    try:
        fd._FAMILIES["joint_ie"]["table_kwargs"] = fake_builder
        fd._table_kwargs_builder("joint_ie", "food_inv_joint_ie")("base", ("w_d",))
    finally:
        fd._FAMILIES["joint_ie"]["table_kwargs"] = orig
    assert seen[0] == {}, f"intimacy-inferring family must get no base=: {seen[0]}"
    print("✓ table builder passes domain/base only where meaningful")


def _params_for(n_util, extras):
    """A parameter vector laid out as the fitters pack it, with distinct values
    so a mis-sliced row shows up as a wrong number."""
    return [round(0.5 + 0.25 * i, 6) for i in range(n_util + 2 + extras)]


def test_result_row_layout_for_every_study_and_variant():
    """`_result_row` slices [*utility, alpha_observer, sigma, *extras]. Every
    reported parameter is read by offset, so an off-by-one here silently
    relabels sigma as a utility weight in all six studies."""
    for slug in ROSTER:
        family = fd.FAMILY_BY_SLUG[slug]
        for variant, (_fn, util) in fd._FAMILIES[family]["variants"].items():
            rw = _reweighting.config_for(slug, variant, list(util))
            n_util = len(util)
            extras = 1 if rw else 0
            params = _params_for(n_util, extras)
            row = fd._result_row(slug, variant, util, params, 9.5, False, rw)

            assert row["model"] == variant and row["experiment"] == slug
            assert row["nll"] == 9.5
            assert row["alpha_observer"] == params[n_util]
            assert row["param_sigma"] == params[n_util + 1]
            for i, name in enumerate(util):
                assert row[f"param_{name}"] == params[i], (slug, variant, name)
            # n_params must equal the vector the fitter actually optimized.
            assert row["n_params"] == len(params), (slug, variant)
            if rw:
                assert row["param_eta"] == params[-1]
                assert row["reweighting_targets"] == list(rw["targets"])
            else:
                assert "param_eta" not in row
                assert "reweighting_targets" not in row
            # prior_nu only appears in informative-prior runs.
            assert "param_prior_nu" not in row
    print("✓ result-row layout correct for all 18 (study, variant) pairs")


def test_result_row_with_informative_prior_adds_nu_before_eta():
    """Extras are packed (prior_nu, then eta); the row must read them in that
    order or an informative-prior reweighted fit mislabels both."""
    slug, variant = "food_inv_joint_de", "full"
    util = fd._FAMILIES["joint_de"]["variants"][variant][1]
    rw = _reweighting.config_for(slug, variant, list(util))
    assert rw, "1b/full should be reweighted"
    n_util = len(util)
    params = _params_for(n_util, 2)
    row = fd._result_row(slug, variant, util, params, 1.0, True, rw)
    assert row["param_prior_nu"] == params[n_util + 2]
    assert row["param_eta"] == params[-1]
    assert row["n_params"] == len(params)
    print("✓ informative-prior + reweighted row packs prior_nu before eta")


def test_alpha_observer_at_bound_is_false_while_the_bound_is_disabled():
    """The bound is off by default, so this flag must never report a
    constrained optimum -- a True here would be read as the fit hitting a
    ceiling that does not exist."""
    util = ("w_v", "w_d", "w_e", "gamma")
    row = fd._result_row(
        "food_inv_joint_de", "full", util, _params_for(4, 0), 1.0, False, None
    )
    assert fd.ALPHA_OBS_MAX is None, (
        "ALPHA_OBS_MAX is enabled; update this test and the reported "
        "alpha_observer_at_bound interpretation"
    )
    assert row["alpha_observer_at_bound"] is False
    print("✓ alpha_observer_at_bound is False while the bound is disabled")


def test_priors_k_alignment_tiles_k1_and_rejects_mismatch():
    """A K=1 priors vintage (the human-ceiling file) tiles up to the tables' K;
    any other mismatch must raise rather than broadcast, since mismatched K
    pairs each run's tables with another run's priors."""
    tk = {"risk_table": np.zeros((20, 2, 3))}

    pr = {"m_latent": np.ones((1, 4)), "effort_prior": np.ones((1, 2)), "other": None}
    fd._check_priors_k_alignment("s", "full", pr, tk)
    assert pr["m_latent"].shape[0] == 20, pr["m_latent"].shape
    assert pr["effort_prior"].shape[0] == 20
    assert pr["other"] is None

    bad = {"m_latent": np.ones((7, 4))}
    try:
        fd._check_priors_k_alignment("s", "full", bad, tk)
        raise AssertionError("K=7 against K=20 tables should raise")
    except ValueError as e:
        assert "priors K=7" in str(e) and "K=20" in str(e), str(e)

    fd._check_priors_k_alignment("s", "full", None, tk)  # uniform: no-op
    print("✓ priors K alignment tiles K=1 and rejects other mismatches")


def test_wrappers_are_thin_and_route_to_their_own_slug():
    """Each wrapper must do nothing but call the dispatcher with its own slug.
    A wrapper that reimplements any protocol step reintroduces exactly the
    duplication this consolidation removed."""
    import importlib

    for slug in ROSTER:
        path = _root / "model" / "inverse" / f"fit_{slug}.py"
        src = path.read_text()
        assert f'EXPERIMENT_SLUG = "{slug}"' in src, slug
        assert "_fit_dispatcher.main(EXPERIMENT_SLUG" in src, slug
        # No protocol logic: these names belong to the dispatcher now.
        for leaked in (
            "resolve_variant_table_kwargs",
            "build_priors_kwarg",
            "restart_records_to_rows",
            "write_fit_manifest",
            "n_params",
        ):
            assert leaked not in src, f"fit_{slug}.py still carries {leaked!r}"
        assert len(src.splitlines()) < 40, f"fit_{slug}.py is no longer thin"
        mod = importlib.import_module(f"fit_{slug}")
        assert mod.EXPERIMENT_SLUG == slug
    print("✓ all six wrappers are thin and route to their own slug")


def test_default_config_writes_the_study_root():
    """The preregistered config must keep writing outputs/<slug>/ -- the path
    every committed fit, CV warm start, and figure already reads."""
    cfg = RunConfig()
    for slug in ROSTER:
        assert cfg.outputs_dir(slug).name == slug, slug
        assert "alt" not in cfg.outputs_dir(slug).parts
    print("✓ preregistered config writes the study root")


def test_warm_start_round_trips_for_every_fitted_variant():
    """A CV fold rebuilds its warm start from fit_results.json, so every member
    of the optimizer vector must survive that round trip.

    `_read_fit_results` strips the `param_` prefix for an explicit list of names.
    When the reweighting added `eta` to the vector, that list was not updated, so
    CV raised KeyError on all 12 reweighted (study, variant) pairs -- loudly, but
    only once a CV run had already started. This closes the loop by asserting the
    round trip for every variant whose fit exists on disk.
    """
    import sys as _sys

    _sys.path.insert(0, str(_root / "model" / "cv"))
    import _inverse_dispatcher as D

    from _helpers import params_dict_to_array

    checked, skipped = 0, []
    for slug in ROSTER:
        fit_dir = _root / "model" / "outputs" / slug
        if not (fit_dir / "fit_results.json").exists():
            continue
        fits = D._read_fit_results(str(fit_dir))
        family = fd.FAMILY_BY_SLUG[slug]
        for variant, (_fn, util) in fd._FAMILIES[family]["variants"].items():
            if variant not in fits:
                continue
            extras = (
                ("eta",)
                if _reweighting.uses_reweighting(slug, list(util))
                else ()
            )
            # A fit written before the reweighting existed has no param_eta, so
            # it cannot round-trip. That is a stale artifact, not a code defect
            # (it resolves when the study is refit), so skip it and report --
            # while still requiring that SOMETHING was checked, so this can
            # never pass vacuously on an all-stale tree.
            if extras and "eta" not in fits[variant]:
                skipped.append(f"{slug}/{variant}")
                continue
            vec = params_dict_to_array(
                fits[variant], list(util), extra_param_names=extras
            )
            assert len(vec) == len(util) + 2 + len(extras), (
                f"{slug}/{variant}: warm vector is {len(vec)} long, expected "
                f"{len(util) + 2 + len(extras)}"
            )
            checked += 1
    assert checked, "no fits on disk to check the warm-start round trip against"
    note = f" ({len(skipped)} pre-reweighting fits skipped: {skipped})" if skipped else ""
    print(f"✓ warm start round-trips for {checked} fitted (study, variant) pairs{note}")


def run_all_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 - report, don't abort the suite
            failures.append((fn.__name__, f"{type(e).__name__}: {e}"))
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print("=" * 60)
    if failures:
        print(f"{len(failures)} of {len(tests)} fit-protocol tests FAILED")
        sys.exit(1)
    print(f"All {len(tests)} fit-protocol tests passed!")
    print("=" * 60)


def test_base_shared_uses_fulls_priors_vintage():
    """`base_shared` must NOT route to the relationship-free base priors.

    It is base's utility on full's relationship-conditioned comparison set, so
    its cells carry a relationship axis; the collapsed base priors vintage would
    misalign with that grid. Pins the exact-match in priors_base_variant against
    a well-meaning widening to startswith("base").
    """
    from _priors import priors_base_variant

    for slug in ("food_inv_desire", "food_inv_joint_de", "nonfood_inv_joint_de"):
        assert priors_base_variant(slug, "base") is True, slug
        assert priors_base_variant(slug, "base_shared") is False, slug
        assert priors_base_variant(slug, "full") is False, slug


def test_results_latex_macro_names_are_valid_and_unique():
    """The generated macro names must be letters-only and collision-free.

    LaTeX command names cannot contain digits or underscores, so every study
    label and parameter name has to be mapped to letters ("1a" -> "OneA"). A
    violation would surface as a LaTeX syntax error in the manuscript rather
    than here, so this pins it at the source: build the whole macro set and let
    Macros.add's own guards fire.

    Skipped when the comparison JSONs are absent (a clean tree), since the
    exporter reads them; the guards are exercised as soon as CV has run.
    """
    import export_results_latex as X

    have = [
        s
        for s in X.studies()
        if (_root / "model" / "outputs" / s.slug / "cv_model_comparison.json").exists()
    ]
    if not have:
        print("~ results-latex macros: no cv_model_comparison.json yet, skipped")
        return
    macros, _loaded = X.build(have)
    names = [name for name, _b, _c in macros._items]
    bad = [n for n in names if not n.isalpha()]
    assert not bad, f"macro names not letters-only: {bad}"
    assert len(names) == len(set(names)), "duplicate macro names"
    # Every study must contribute the contrasts the tables reference.
    for st in have:
        tok = X.token(st)
        for required in (f"llFull{tok}", f"statBase{tok}", f"statDisc{tok}"):
            assert required in names, f"missing {required}"
    print(f"✓ {len(names)} results-latex macros, all letters-only and unique")


if __name__ == "__main__":
    run_all_tests()
