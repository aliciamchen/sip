#!/usr/bin/env python3
"""Tests for the hypothesis-matched contrast statistics (cv/contrast_tests.py).

Every test builds synthetic data whose true decomposition is known by
construction, so the assertions are against ground truth rather than against
whatever the code currently returns. The bias-correction tests are the point of
the file: at ~20 observations per cell the uncorrected moments are inflated
enough to change what the paper would claim, so "the correction fires and lands
near the truth" is the property that has to hold.

Three study shapes are covered, because they exercise different code paths: the
1b shape (one given condition, 4 focal levels), the 1a shape (a SECOND given
condition, so the group grid gains an axis the focal contrast must pool over),
and the 2b shape (a 2-level focal condition, where the trend reduces to a plain
difference of means).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contrast_tests import (  # noqa: E402
    LEVEL_ORDER,
    condition_gradients,
    focal_condition,
    ordered_levels,
    trend_coefficients,
    variance_decomposition,
)


@dataclass(frozen=True)
class FakeStudy:
    """Minimal stand-in for study_registry.Study: only what these functions read."""

    slug: str
    given_conditions: tuple

    @property
    def cell_keys(self):
        return ["scenario_label", "action", *self.given_conditions]


STUDY_1B = FakeStudy("fake_1b", ("intimacy_condition",))
STUDY_1A = FakeStudy("fake_1a", ("intimacy_condition", "effort_condition"))
STUDY_2B = FakeStudy("fake_2b", ("desire_condition",))

_INTIMACY = LEVEL_ORDER["intimacy_condition"]


def _synth(
    study=STUDY_1B,
    *,
    n_scen=16,
    n_act=3,
    n_subj_per_cell=20,
    focal_slope=0.0,
    action_effect=0.0,
    scenario_specific_sd=0.0,
    within_sd=0.0,
    seed=0,
):
    """Trial-level data with a known variance decomposition.

    The cell mean is `action_effect * action + slope * z(level)`, where `z` is the
    focal level index centered on its mean -- so the focal effect has mean zero
    within every group, exactly the deviation `variance_decomposition` isolates,
    and `focal_slope` is its per-step size. With `scenario_specific_sd > 0` the
    slope varies by (scenario, action). Non-focal given conditions are crossed in
    but carry no effect, so pooling over them must not change the answer.
    """
    rng = np.random.default_rng(seed)
    focal = focal_condition(study)
    levels = LEVEL_ORDER[focal]
    z = np.arange(len(levels), dtype=float)
    z = z - z.mean()
    others = [c for c in study.given_conditions if c != focal]
    other_levels = [("low", "high") for _ in others]
    rows = []
    for s in range(n_scen):
        for a in range(n_act):
            slope = focal_slope + rng.normal(0.0, scenario_specific_sd)
            for li, lv in enumerate(levels):
                combos = [()] if not others else [(x,) for x in other_levels[0]]
                for combo in combos:
                    mu = action_effect * a + slope * z[li]
                    vals = mu + rng.normal(0.0, within_sd, size=n_subj_per_cell)
                    for j, v in enumerate(vals):
                        row = {
                            "subject_id": f"s{j}",
                            "scenario_label": f"sc{s}",
                            "action": a,
                            focal: lv,
                            "u": float(v),
                        }
                        for name, val in zip(others, combo):
                            row[name] = val
                        rows.append(row)
    return pd.DataFrame(rows), z


def _preds(data, study, delta_col, scale=1.0):
    """One variant's predictions, equal to `scale` x the human cell means."""
    cm = data.groupby(study.cell_keys, as_index=False)["u"].mean()
    return cm.assign(**{delta_col: cm["u"] * scale}).drop(columns="u")


def _grads(data, study, *, scale=1.0, n_boot=200, seed=0, models=("full",)):
    return condition_gradients(
        data,
        {m: _preds(data, study, "delta_desire", scale) for m in models},
        study,
        "u",
        "delta_desire",
        "desire",
        n_boot=n_boot,
        rng=np.random.default_rng(seed),
    )


# --------------------------------------------------------------------------
# contrast construction
# --------------------------------------------------------------------------


def test_trend_coefficients_are_centered_and_in_endpoint_units():
    for n in (2, 3, 4, 5):
        c = trend_coefficients(n)
        x = np.arange(n, dtype=float)
        assert abs(c.sum()) < 1e-12, f"n={n}: weights must annihilate an offset"
        # Under a linear trend the contrast must equal the total lowest-to-highest
        # change, which is what makes 2-level and 4-level studies comparable.
        assert abs(c @ x - (n - 1)) < 1e-12, f"n={n}: not in endpoint units"
    assert np.allclose(trend_coefficients(2), [-1.0, 1.0])
    print("✓ trend coefficients are centered and scaled to endpoint units")


def test_focal_condition_reads_the_design():
    assert focal_condition(STUDY_1B) == "intimacy_condition"
    assert focal_condition(STUDY_1A) == "intimacy_condition"
    assert focal_condition(STUDY_2B) == "desire_condition"
    try:
        focal_condition(FakeStudy("bad", ("effort_condition",)))
    except ValueError:
        pass
    else:
        raise AssertionError("a study giving neither latent should raise")
    print("✓ focal condition is derived from the given conditions")


def test_unknown_focal_level_raises_rather_than_being_dropped():
    """Silently dropping it would also silently rescale the contrast: the trend
    normalizes to however many levels survive, so a 3-level survivor would report
    a 2-step change under the same field name as a 4-level study's 3-step one."""
    data, _ = _synth(focal_slope=0.05)
    data.loc[data["intimacy_condition"] == "somewhat_formal", "intimacy_condition"] = (
        "neither"  # the pre-2026-06-19 spelling
    )
    try:
        ordered_levels(data, "intimacy_condition", "fake_1b")
    except ValueError as e:
        assert "neither" in str(e), str(e)
    else:
        raise AssertionError("an unrecognized level must raise")
    print("✓ an unrecognized focal level raises instead of being dropped")


# --------------------------------------------------------------------------
# variance decomposition
# --------------------------------------------------------------------------


def test_variance_decomposition_recovers_a_noiseless_design():
    slope, act = 0.05, 0.20
    data, z = _synth(focal_slope=slope, action_effect=act, within_sd=0.0)
    got = variance_decomposition(data, STUDY_1B, "u", "desire")
    true_focal = (slope**2) * float((z**2).mean())
    assert abs(got["within_cell_var"]) < 1e-18, got["within_cell_var"]
    assert abs(got["focal_var"] - true_focal) < 1e-12, (got["focal_var"], true_focal)
    assert abs(got["frac_explainable"] - 1.0) < 1e-9
    assert got["focal_scenario_specific_var"] < 1e-12
    print("✓ variance decomposition is exact on a noiseless design")


def test_second_given_condition_does_not_change_the_focal_component():
    """The 1a shape. Crossing in an inert second given condition adds cells but no
    signal, so the focal component must come out the same as the 1b shape."""
    slope = 0.05
    one, z = _synth(STUDY_1B, focal_slope=slope, action_effect=0.2, within_sd=0.0)
    two, _ = _synth(STUDY_1A, focal_slope=slope, action_effect=0.2, within_sd=0.0)
    a = variance_decomposition(one, STUDY_1B, "u", "desire")
    b = variance_decomposition(two, STUDY_1A, "u", "desire")
    assert abs(a["focal_var"] - b["focal_var"]) < 1e-12, (
        a["focal_var"],
        b["focal_var"],
    )
    assert abs(b["focal_var"] - (slope**2) * float((z**2).mean())) < 1e-12
    print("✓ a second given condition leaves the focal component unchanged")


def test_bias_correction_removes_a_spurious_focal_effect():
    """The correction's reason for existing: a TRUE focal effect of zero.

    Uncorrected, sampling noise in the cell means manufactures one; at 20
    observations per cell it is large enough to be reported as a real result.
    """
    data, _ = _synth(focal_slope=0.0, action_effect=0.2, within_sd=0.25, seed=3)
    got = variance_decomposition(data, STUDY_1B, "u", "desire")
    raw = got["bias_correction"]["uncorrected_focal_var"]
    assert raw > 1e-4, f"expected the raw moment to be inflated, got {raw}"
    assert got["focal_var"] < raw / 10, (got["focal_var"], raw)
    assert got["focal_frac_of_total"] < 0.01, got["focal_frac_of_total"]
    print("✓ bias correction removes a focal effect that is purely sampling noise")


def test_bias_correction_preserves_a_real_focal_effect():
    """Unbiasedness, which is a property of the estimator and not of one draw.

    Averaged over seeds rather than asserted on a single one: at 20 observations
    per cell a single estimate has an SD of roughly 15% of the true value, so a
    tight one-seed tolerance tests the seed, not the correction.
    """
    slope, n_seeds = 0.06, 20
    ests = []
    for seed in range(n_seeds):
        data, z = _synth(
            focal_slope=slope, action_effect=0.2, within_sd=0.25, seed=seed
        )
        ests.append(variance_decomposition(data, STUDY_1B, "u", "desire")["focal_var"])
    true_focal = (slope**2) * float((z**2).mean())
    assert abs(float(np.mean(ests)) / true_focal - 1.0) < 0.10, (
        np.mean(ests),
        true_focal,
    )
    print("✓ bias correction leaves a real focal effect near its true size (unbiased)")


def test_scenario_specific_share_is_bounded_in_the_clipping_regime():
    """The regression this test exists for.

    Clipping the three corrected components independently let the share exceed 1
    whenever the true focal effect is near zero and the consistent part clipped
    first -- 12 of 200 seeds at slope 0.01, once at 291%. The bound has to hold in
    exactly that regime, so this sweeps it rather than testing a comfortable one.
    """
    for slope in (0.0, 0.005, 0.01):
        for seed in range(40):
            got = variance_decomposition(
                _synth(focal_slope=slope, action_effect=0.2, within_sd=0.25, seed=seed)[
                    0
                ],
                STUDY_1B,
                "u",
                "desire",
            )
            f = got["focal_frac_scenario_specific"]
            assert f != f or 0.0 <= f <= 1.0, (slope, seed, f)
            parts = (
                got["focal_scenario_consistent_var"]
                + got["focal_scenario_specific_var"]
            )
            assert abs(parts - got["focal_var"]) < 1e-12, (parts, got["focal_var"])
    print("✓ scenario-specific share stays in [0, 1] in the clipping regime")


def test_scenario_specific_share_tracks_real_heterogeneity():
    common, _ = _synth(
        focal_slope=0.06, scenario_specific_sd=0.0, within_sd=0.25, seed=7
    )
    varied, _ = _synth(
        focal_slope=0.06, scenario_specific_sd=0.06, within_sd=0.25, seed=7
    )
    a = variance_decomposition(common, STUDY_1B, "u", "desire")
    b = variance_decomposition(varied, STUDY_1B, "u", "desire")
    assert a["focal_frac_scenario_specific"] < 0.25, a["focal_frac_scenario_specific"]
    assert b["focal_frac_scenario_specific"] > a["focal_frac_scenario_specific"]
    print("✓ scenario-specific share rises with real scenario heterogeneity")


def test_decomposition_bootstrap_brackets_the_point_estimate():
    data, _ = _synth(focal_slope=0.06, action_effect=0.2, within_sd=0.25, seed=9)
    got = variance_decomposition(
        data, STUDY_1B, "u", "desire", n_boot=200, rng=np.random.default_rng(0)
    )
    lo, hi = got["focal_frac_of_total_ci_95"]
    assert lo < got["focal_frac_of_total"] < hi, (lo, got["focal_frac_of_total"], hi)
    assert 0.0 <= lo, lo
    print("✓ the decomposition's bootstrap interval brackets its point estimate")


# --------------------------------------------------------------------------
# condition gradients
# --------------------------------------------------------------------------


def test_gradient_recovers_a_known_slope_and_a_perfect_model():
    slope = 0.05
    data, _ = _synth(focal_slope=slope, action_effect=0.2, within_sd=0.0)
    rows = _grads(data, STUDY_1B, scale=1.0)
    assert len(rows) == 3, rows
    expected = slope * (len(_INTIMACY) - 1)  # endpoint units: 4 levels, 3 steps
    for r in rows:
        assert abs(r["human_gradient"] - expected) < 1e-9, r
        assert abs(r["model_gradient"] - expected) < 1e-9, r
        assert abs(r["human_endpoint_gradient"] - expected) < 1e-9, r
        assert abs(r["human_minus_model"]) < 1e-9, r
        assert r["n_levels"] == 4, r
    print("✓ gradient recovers a known slope, and a perfect model recovers 100%")


def test_gradient_on_a_two_level_focal_condition():
    """The 2b/3b path: with two levels the trend IS the difference of means."""
    slope = 0.05
    data, _ = _synth(STUDY_2B, focal_slope=slope, action_effect=0.2, within_sd=0.0)
    rows = _grads(data, STUDY_2B, scale=1.0)
    for r in rows:
        assert r["n_levels"] == 2, r
        assert abs(r["human_gradient"] - slope) < 1e-9, r
        assert abs(r["human_gradient"] - r["human_endpoint_gradient"]) < 1e-12, r
    print("✓ a two-level focal condition reduces the trend to a difference of means")


def test_gradient_pools_over_a_second_given_condition():
    slope = 0.05
    data, _ = _synth(STUDY_1A, focal_slope=slope, action_effect=0.2, within_sd=0.0)
    rows = _grads(data, STUDY_1A, scale=1.0)
    expected = slope * (len(_INTIMACY) - 1)
    assert len(rows) == 3, rows
    for r in rows:
        assert abs(r["human_gradient"] - expected) < 1e-9, r
    print("✓ the gradient pools over a non-focal given condition")


def test_gradient_scores_a_half_size_model_at_half():
    data, _ = _synth(focal_slope=0.05, action_effect=0.2, within_sd=0.15, seed=11)
    for r in _grads(data, STUDY_1B, scale=0.5, n_boot=400, seed=1):
        assert abs(r["model_gradient"] / r["human_gradient"] - 0.5) < 1e-9, r
        assert abs(r["recovered_fraction"] - 0.5) < 1e-9, r
        lo, hi = r["human_ci_95"]
        assert lo < r["human_gradient"] < hi, r
    print("✓ a model at half the human gradient is scored at 50% recovered")


def test_human_interval_is_identical_across_model_variants():
    """The human gradient does not depend on the model, so neither may its CI.

    Seeding the bootstrap per variant published four different intervals for one
    identical human statistic (Study 1a action 0: -0.0937 with four CIs).
    """
    data, _ = _synth(focal_slope=0.05, action_effect=0.2, within_sd=0.2, seed=17)
    rows = condition_gradients(
        data,
        {
            "full": _preds(data, STUDY_1B, "delta_desire", 1.0),
            "base": _preds(data, STUDY_1B, "delta_desire", 0.0),
            "discomfort_only": _preds(data, STUDY_1B, "delta_desire", 0.3),
        },
        STUDY_1B,
        "u",
        "delta_desire",
        "desire",
        n_boot=300,
        rng=np.random.default_rng(5),
    )
    by_action = {}
    for r in rows:
        by_action.setdefault(r["action"], []).append(
            (r["human_gradient"], tuple(r["human_ci_95"]))
        )
    for action, seen in by_action.items():
        assert len(set(seen)) == 1, (action, seen)
    print("✓ the human gradient and its interval are identical across variants")


def test_recovered_fraction_is_withheld_when_the_human_gradient_is_null():
    """A ratio to a null denominator is noise; reported as a percentage it reads
    as a finding. One real cell produced '-2129% recovered' this way."""
    data, _ = _synth(focal_slope=0.0, action_effect=0.2, within_sd=0.25, seed=13)
    rows = _grads(data, STUDY_1B, scale=1.0, n_boot=400, seed=2)
    assert any(r["recovered_fraction"] is None for r in rows), (
        "a gradient whose CI spans zero must not report a recovered fraction"
    )
    for r in rows:
        if r["recovered_fraction"] is None:
            lo, hi = r["human_ci_95"]
            assert lo <= 0 <= hi, r
    print("✓ recovered fraction is withheld when the human gradient is not reliable")


def test_gradient_sign_follows_the_declared_level_order():
    """LEVEL_ORDER fixes what a positive gradient means (formal -> intimate)."""
    data, _ = _synth(focal_slope=-0.05, action_effect=0.0, within_sd=0.0)
    assert all(r["human_gradient"] < 0 for r in _grads(data, STUDY_1B, n_boot=100))
    print("✓ gradient sign follows the declared level order")


def test_missing_model_prediction_raises_naming_the_cells():
    """Inner-joining it away would leave the human contrast pooling cells the
    model contrast does not, with no warning."""
    data, _ = _synth(focal_slope=0.05, within_sd=0.1)
    preds = _preds(data, STUDY_1B, "delta_desire")
    try:
        condition_gradients(
            data,
            {"full": preds[preds["scenario_label"] != "sc0"]},
            STUDY_1B,
            "u",
            "delta_desire",
            "desire",
            n_boot=50,
            rng=np.random.default_rng(0),
        )
    except RuntimeError as e:
        assert "no matching" in str(e) and "sc0" in str(e), str(e)
    else:
        raise AssertionError("a missing model prediction must raise")
    print("✓ a missing model prediction raises and names the offending cells")


def test_multi_variant_prediction_frame_raises():
    """The documented precondition, enforced: one variant's rows per entry."""
    data, _ = _synth(focal_slope=0.05, within_sd=0.1)
    doubled = pd.concat(
        [
            _preds(data, STUDY_1B, "delta_desire", 1.0),
            _preds(data, STUDY_1B, "delta_desire", 0.5),
        ]
    )
    try:
        condition_gradients(
            data,
            {"full": doubled},
            STUDY_1B,
            "u",
            "delta_desire",
            "desire",
            n_boot=50,
            rng=np.random.default_rng(0),
        )
    except RuntimeError as e:
        assert "one prediction row per cell" in str(e), str(e)
    else:
        raise AssertionError("a multi-variant preds frame must raise")
    print("✓ a prediction frame holding more than one variant raises")


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
        print(f"{len(failures)} of {len(tests)} contrast-test checks FAILED")
        sys.exit(1)
    print(f"All {len(tests)} contrast-test checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
