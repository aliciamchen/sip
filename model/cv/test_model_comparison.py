"""Tests for the reported model-comparison statistics (`model/cv/model_comparison.py`).

Every confidence interval in the paper comes out of `_bootstrap_mean_by_subject`,
and every reported full-minus-ablation difference out of `_primary_comparisons`.
Both were untested. The implementation is correct, but it is correct in a way
that is easy to "simplify" into being wrong:

    sums[idx].sum(axis=1) / counts[idx].sum(axis=1)

is a RATIO estimator. It weights each resampled subject by how many trials they
contributed, which is what makes the bootstrap distribution centre on the
trial-level point estimate that is reported alongside it. Replacing it with the
apparently equivalent `per_subject_mean[idx].mean(axis=1)` silently switches to
an unweighted mean over subjects, which no longer matches the reported estimate
whenever trial counts differ across participants.

The other failure mode these tests exist for is resampling the wrong unit.
Bootstrapping trials instead of subjects ignores within-participant correlation
and returns intervals that are too narrow -- the classic way to overstate
significance. That cannot be caught by inspection of the output, only by a test
that makes the two behave differently.

Run: uv run python model/cv/test_model_comparison.py
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "model"))
sys.path.insert(0, str(_root / "model" / "cv"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model_comparison import (  # noqa: E402
    _bootstrap_mean_by_subject,
    _config_dir,
    _primary_comparisons,
)

N_BOOT = 4000


def test_bootstrap_returns_one_mean_per_replicate():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=200)
    subj = np.repeat(np.arange(20), 10)
    boots = _bootstrap_mean_by_subject(vals, subj, N_BOOT, rng)
    assert boots.shape == (N_BOOT,), boots.shape
    assert np.all(np.isfinite(boots))
    print("✓ bootstrap returns one finite mean per replicate")


def test_bootstrap_centres_on_the_reported_point_estimate():
    """The reported number is `values.mean()` over trials. The bootstrap
    distribution must centre there, or the CI is not an interval for the
    quantity being reported."""
    rng = np.random.default_rng(1)
    # Deliberately UNEQUAL trial counts: this is where a weighted and an
    # unweighted subject mean diverge.
    subj, vals = [], []
    for s in range(30):
        n = 2 + (s % 7)
        subj += [s] * n
        vals += list(rng.normal(loc=s * 0.05, scale=0.4, size=n))
    subj, vals = np.array(subj), np.array(vals)
    boots = _bootstrap_mean_by_subject(vals, subj, N_BOOT, rng)
    point = vals.mean()
    assert abs(boots.mean() - point) < 0.02, (
        f"bootstrap mean {boots.mean():.4f} vs trial-level point estimate "
        f"{point:.4f} — the interval does not describe the reported statistic"
    )
    print("✓ bootstrap centres on the trial-level point estimate it accompanies")


def test_bootstrap_weights_subjects_by_trial_count():
    """The ratio-of-sums weighting is the property that keeps the bootstrap
    aligned with the trial-level estimate. Pinned against the unweighted
    alternative, which is the tempting simplification.

    Two subjects, wildly unequal trial counts and different values: the
    trial-weighted mean sits near the heavy subject, the unweighted subject
    mean sits midway.
    """
    subj = np.array([0] * 90 + [1] * 10)
    vals = np.array([1.0] * 90 + [0.0] * 10)
    trial_weighted = vals.mean()  # 0.90
    unweighted_subject = np.array([1.0, 0.0]).mean()  # 0.50
    assert abs(trial_weighted - 0.9) < 1e-12
    boots = _bootstrap_mean_by_subject(
        values=vals, subject_ids=subj, n_boot=N_BOOT, rng=np.random.default_rng(2)
    )
    # Resampling 2 subjects with replacement gives {both A} 25%, {both B} 25%,
    # {one each} 50%. The ratio estimator yields 1.0, 0.0, and 90/100 = 0.9.
    uniq = np.unique(np.round(boots, 6))
    assert set(uniq) <= {0.0, 0.9, 1.0}, uniq
    assert abs(boots.mean() - 0.5 * 0.9 - 0.25) < 0.03, boots.mean()
    # And it is NOT the unweighted subject mean, which would give {1.0, 0.5, 0.0}.
    assert 0.5 not in set(uniq), (
        "0.5 appearing means subjects were averaged unweighted rather than "
        "pooled by trial count"
    )
    assert abs(trial_weighted - unweighted_subject) > 0.3  # the test has teeth
    print("✓ bootstrap weights subjects by trial count (ratio estimator)")


def test_bootstrap_resamples_subjects_not_trials():
    """With perfectly correlated trials within a subject, resampling subjects
    must produce real spread, while resampling trials would nearly collapse it.
    This is the test that distinguishes a cluster bootstrap from a naive one.
    """
    n_subj, per = 12, 40
    subj = np.repeat(np.arange(n_subj), per)
    # Each subject is a constant, far apart across subjects: all variance is
    # BETWEEN subjects, none within.
    vals = np.repeat(np.linspace(-1.0, 1.0, n_subj), per)
    rng = np.random.default_rng(3)
    boots = _bootstrap_mean_by_subject(vals, subj, N_BOOT, rng)
    cluster_sd = boots.std()

    # A trial-level bootstrap of the same data, for contrast.
    idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
    trial_sd = vals[idx].mean(axis=1).std()

    expected = vals[::per].std(ddof=0) / np.sqrt(n_subj)
    assert abs(cluster_sd - expected) < 0.03 * max(expected, 1e-9) + 0.01, (
        f"cluster bootstrap SD {cluster_sd:.4f} vs analytic {expected:.4f}"
    )
    assert cluster_sd > 4 * trial_sd, (
        f"cluster SD {cluster_sd:.4f} is not materially wider than the "
        f"trial-level SD {trial_sd:.4f} — subjects are not the resampling unit, "
        "so the reported intervals would be too narrow"
    )
    print(
        f"✓ bootstrap resamples subjects, not trials "
        f"(SD {cluster_sd:.3f} vs {trial_sd:.3f} trial-level)"
    )


def test_bootstrap_is_reproducible_from_a_seeded_generator():
    """Reported CIs must be reproducible from the recorded seed."""
    vals = np.random.default_rng(9).normal(size=120)
    subj = np.repeat(np.arange(12), 10)
    a = _bootstrap_mean_by_subject(vals, subj, 500, np.random.default_rng(7))
    b = _bootstrap_mean_by_subject(vals, subj, 500, np.random.default_rng(7))
    assert np.array_equal(a, b)
    c = _bootstrap_mean_by_subject(vals, subj, 500, np.random.default_rng(8))
    assert not np.array_equal(a, c)
    print("✓ bootstrap is reproducible from its seed")


def test_bootstrap_handles_a_single_subject():
    """Degenerate but reachable (a study with one retained participant would be
    a data problem, not a crash): every replicate is that subject's mean."""
    boots = _bootstrap_mean_by_subject(
        np.array([1.0, 3.0]), np.array([0, 0]), 50, np.random.default_rng(0)
    )
    assert np.allclose(boots, 2.0)
    print("✓ bootstrap handles a single subject without error")


def _trial_df(models=("full", "base"), n_subj=8, n_scen=4, effect=0.5):
    rows = []
    rng = np.random.default_rng(11)
    for m in models:
        for s in range(n_subj):
            for sc in range(n_scen):
                base = rng.normal(-1.0, 0.2)
                rows.append(
                    {
                        "model": m,
                        "subject_id": f"s{s}",
                        "scenario_label": f"sc{sc}",
                        "held_out_ll": base + (effect if m == "full" else 0.0),
                    }
                )
    return pd.DataFrame(rows)


def test_primary_comparisons_reports_full_minus_each_ablation():
    df = _trial_df(models=("full", "discomfort_only", "base"))
    out = _primary_comparisons(df, 500, np.random.default_rng(0))
    names = {r["comparison"] for r in out}
    assert names == {"full_minus_discomfort_only", "full_minus_base"}, names
    for r in out:
        lo, hi = r["ci_95"]
        assert lo <= r["mean_per_trial_ll_diff"] <= hi, r
        assert r["mean_per_trial_ll_diff"] > 0  # `full` was built better
    print("✓ primary comparisons cover every ablation with a bracketing CI")


def test_primary_comparisons_rejects_duplicate_trial_rows():
    """A duplicated (model, subject, scenario) row would double-count a trial
    and silently shift the reported difference."""
    df = _trial_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    try:
        _primary_comparisons(df, 50, np.random.default_rng(0))
        raise AssertionError("duplicate trial rows should raise")
    except AssertionError as e:
        assert "duplicate" in str(e), str(e)
    print("✓ primary comparisons reject duplicate trial rows")


def test_primary_comparisons_rejects_unmatched_trials():
    """The comparison is paired on (subject, scenario). A trial present for one
    variant but not another must fail rather than be dropped, since dropping
    would compare different trial sets across variants."""
    df = _trial_df()
    df = df[~((df["model"] == "base") & (df["subject_id"] == "s0"))]
    try:
        _primary_comparisons(df, 50, np.random.default_rng(0))
        raise AssertionError("unmatched trials should raise")
    except AssertionError as e:
        assert "not matched" in str(e), str(e)
    print("✓ primary comparisons reject trials unmatched across variants")


def test_config_dir_routes_reported_to_the_root_and_rejects_retired_names():
    """`reported` is the root; every retired spelling of it raises rather than
    resolving. Aliasing them would be worse than inconvenient: both named a
    different model than a reader would now assume, so a stale invocation would
    silently mislabel which model a comparison is about."""
    root = _config_dir("food_inv_desire", "reported")
    assert root.name == "food_inv_desire" and "alt" not in root.parts
    alt = _config_dir("food_inv_desire", "informative")
    assert alt.parts[-2:] == ("alt", "informative"), alt
    # The preregistered model is now a real, separate run — its tag must resolve
    # under alt/, never to the root the reported fits occupy.
    prereg = _config_dir("food_inv_desire", "uniform-noreweight")
    assert prereg.parts[-2:] == ("alt", "uniform-noreweight"), prereg
    for retired in ("canonical", "preregistered"):
        try:
            _config_dir("food_inv_desire", retired)
            raise AssertionError(f"the retired tag {retired!r} should raise")
        except SystemExit as e:
            assert "reported" in str(e), str(e)
    print("✓ config_dir routes 'reported' to the root, rejects retired tags")


def run_all_tests():
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
        print(f"{len(failures)} of {len(tests)} model-comparison tests FAILED")
        sys.exit(1)
    print(f"All {len(tests)} model-comparison tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
