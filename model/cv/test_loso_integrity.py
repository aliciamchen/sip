#!/usr/bin/env python3
"""Fast contract tests for leave-one-scenario-out fold isolation.

The artifact audit checks that the committed outputs cover every held-out
scenario exactly once. This test operates one level earlier: it instruments the
shared fold dispatcher and proves that the held-out scenario cannot enter the
fitter, while scoring sees only that scenario.

Run: uv run python model/cv/test_loso_integrity.py
"""

import sys

import numpy as np

from model.cv import _inverse_dispatcher as dispatcher
from model.run_config import RunConfig


def test_fold_masks_keep_the_held_out_scenario_out_of_the_fitter():
    held_out = 1
    scenarios = np.array([0, 0, 1, 1, 2, 2])
    arrays = {
        "scenario": scenarios,
        "subj": np.array([f"s{i}" for i in range(len(scenarios))]),
    }
    seen = {}

    def fitter(**kwargs):
        training = np.asarray(kwargs["training_scenarios"])
        seen["training"] = training
        assert held_out not in training
        return np.array([1.5, 0.25]), 12.0, None

    def train_kwargs(arrays, mask):
        assert mask.dtype == bool
        return {"training_scenarios": arrays["scenario"][mask]}

    def predictions(_tables, fold, _slug, scenario_label, variant):
        assert fold == held_out
        return [{"model": variant, "scenario_label": scenario_label}]

    def test_lls(_tables, arrays, indices, sigma):
        seen["test"] = arrays["scenario"][indices]
        assert np.all(arrays["scenario"][indices] == held_out)
        assert sigma == 0.25
        return np.full(len(indices), -0.75)

    family_name = "_test_fold_isolation"
    fake_family = {
        "variants": {"full": (object(), ())},
        "fitter": fitter,
        "train_kwargs": train_kwargs,
        "predictions": predictions,
        "test_lls": test_lls,
    }
    original_tk = dispatcher._tk_cached
    original_rw = dispatcher._rw_cached
    original_build = dispatcher._build_observer_tables_runs
    try:
        dispatcher._FAMILIES[family_name] = fake_family
        dispatcher._cv_worker_init(
            family_name, "food_inv_desire", arrays, RunConfig(), None
        )
        dispatcher._tk_cached = lambda *_args: {}
        dispatcher._rw_cached = lambda *_args: None
        dispatcher._build_observer_tables_runs = lambda *_args, **_kwargs: np.zeros(1)

        predictions_out, fold_row, trial_rows = dispatcher._cv_fold(
            "full", held_out, warm=None, patience=3
        )
    finally:
        dispatcher._tk_cached = original_tk
        dispatcher._rw_cached = original_rw
        dispatcher._build_observer_tables_runs = original_build
        dispatcher._FAMILIES.pop(family_name, None)
        dispatcher._CV_W.clear()

    assert set(seen["training"]) == {0, 2}
    assert set(seen["test"]) == {held_out}
    assert fold_row["n_train"] == 4 and fold_row["n_test"] == 2
    assert fold_row["test_nll"] == 1.5
    assert len(predictions_out) == 1 and len(trial_rows) == 2
    assert {row["scenario_label"] for row in trial_rows} == {
        fold_row["held_out_scenario"]
    }
    print("✓ held-out scenarios never reach the fitter and alone reach scoring")


def main():
    try:
        test_fold_masks_keep_the_held_out_scenario_out_of_the_fitter()
    except BaseException as error:
        print(f"FAIL: {type(error).__name__}: {error}")
        sys.exit(1)
    print("All 1 LOSO-integrity tests passed!")


if __name__ == "__main__":
    main()
