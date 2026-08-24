#!/usr/bin/env python3
"""Integrity checks on the committed, de-identified participant datasets.

These checks operate on the public CSVs rather than the private raw JSON.  The
converter uses a deterministic UUID5 mapping, so the public IDs are sufficient
to verify the manuscript's cross-study exclusion claim without exposing a
Prolific ID.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from study_registry import STUDIES
from utils import get_project_root


ROOT = get_project_root()
ACTION_LEVELS = {"no_share", "low_risk_share", "high_risk_share"}
CONDITION_COLS = {
    "food_inv_desire": ("action_condition", "effort", "intimacy"),
    "food_inv_joint_de": ("action_condition", "intimacy"),
    "food_inv_intimacy": ("action_condition", "desire", "effort"),
    "food_inv_joint_ie": ("action_condition", "desire"),
    "nonfood_inv_joint_de": ("action_condition", "intimacy"),
    "nonfood_inv_joint_ie": ("action_condition", "desire"),
}
RATING_COLS = {
    "food_inv_desire": ("response",),
    "food_inv_joint_de": ("desire_rating", "effort_rating"),
    "food_inv_intimacy": ("intimacy_rating",),
    "food_inv_joint_ie": ("intimacy_rating", "effort_rating"),
    "nonfood_inv_joint_de": ("desire_rating", "effort_rating"),
    "nonfood_inv_joint_ie": ("intimacy_rating", "effort_rating"),
}


def _paths(slug):
    root = ROOT / "data" / slug
    return (
        root / "main_trials.csv",
        root / "main_trials_long.csv",
        root / "exit_survey.csv",
    )


def _expected_retained(slug, survey):
    attention = survey["attention_passed"].astype("boolean").fillna(False)
    memory = pd.to_numeric(survey["memory_correct_count"], errors="raise")
    if slug == "food_inv_desire":
        excluded = (~attention) & (memory == 0)
    else:
        excluded = (~attention) | (memory == 0)
    return set(survey.loc[~excluded, "subject_id"])


def test_public_subject_ids_do_not_overlap_across_studies():
    recruited = {}
    for slug in STUDIES:
        main, _long, _survey = _paths(slug)
        recruited[slug] = set(pd.read_csv(main, usecols=["subject_id"])["subject_id"])
    overlaps = {
        (a, b): recruited[a] & recruited[b]
        for a, b in combinations(recruited, 2)
        if recruited[a] & recruited[b]
    }
    assert not overlaps, f"participants appear in multiple studies: {overlaps}"


def test_subject_sets_and_exclusions_recompute_from_public_csvs():
    for slug in STUDIES:
        main_path, long_path, survey_path = _paths(slug)
        main = pd.read_csv(main_path)
        long = pd.read_csv(long_path)
        survey = pd.read_csv(survey_path)
        main_subjects = set(main["subject_id"])
        survey_subjects = set(survey["subject_id"])
        retained_subjects = set(long["subject_id"])
        assert main_subjects == survey_subjects, (
            f"{slug}: trial/survey subject mismatch"
        )
        assert retained_subjects == _expected_retained(slug, survey), (
            f"{slug}: main_trials_long.csv does not implement the declared exclusion rule"
        )
        comparison = ROOT / "model" / "outputs" / slug / "cv_model_comparison.json"
        if comparison.exists():
            reported = pd.read_json(comparison, typ="series")
            assert int(reported["n_subjects"]) == len(retained_subjects)


def test_every_retained_participant_has_one_complete_randomized_design():
    for slug in STUDIES:
        _main_path, long_path, _survey_path = _paths(slug)
        data = pd.read_csv(long_path)
        assert set(data["stage"]) == {"prior", "posterior"}
        assert set(data["action_condition"]) == ACTION_LEVELS
        duplicate = data.duplicated(["subject_id", "scenario_label", "stage"])
        assert not duplicate.any(), f"{slug}: duplicate subject/scenario/stage rows"

        per_subject = data.groupby("subject_id").agg(
            n_rows=("scenario_label", "size"),
            n_scenarios=("scenario_label", "nunique"),
            n_positions=("stimulus_index", "nunique"),
            first_index=("stimulus_index", "min"),
            last_index=("stimulus_index", "max"),
        )
        assert (per_subject["n_rows"] == 32).all(), f"{slug}: incomplete trial rows"
        assert (per_subject["n_scenarios"] == 16).all(), f"{slug}: not 16 scenarios"
        assert (per_subject["n_positions"] == 16).all(), (
            f"{slug}: repeated trial position"
        )
        assert (per_subject["first_index"] == 0).all()
        assert (per_subject["last_index"] == 15).all()

        stable = data.groupby(["subject_id", "scenario_label"])[
            ["stimulus_index", *CONDITION_COLS[slug]]
        ].nunique()
        assert (stable == 1).all().all(), f"{slug}: prior/posterior metadata drift"

        for rating in RATING_COLS[slug]:
            values = pd.to_numeric(data[rating], errors="raise")
            assert values.notna().all(), f"{slug}: missing {rating}"
            assert values.between(0, 1).all(), f"{slug}: {rating} outside [0, 1]"


def test_retained_condition_assignment_remains_balanced():
    """Random assignment should not produce a grossly underrepresented cell."""
    for slug in STUDIES:
        _main_path, long_path, _survey_path = _paths(slug)
        data = pd.read_csv(long_path)
        one_row = data[data["stage"] == "posterior"]
        counts = one_row.groupby(["scenario_label", *CONDITION_COLS[slug]]).size()
        n_cells = int(np.prod([one_row[c].nunique() for c in CONDITION_COLS[slug]]))
        expected = one_row["subject_id"].nunique() / n_cells
        assert len(counts) == 16 * n_cells, f"{slug}: randomized cell is absent"
        # A three-Poisson-SD envelope is deliberately a gross-integrity check,
        # not a test of the randomizer. It catches missing counterbalance arms
        # while allowing ordinary random variation and participant exclusions.
        assert ((counts - expected).abs() <= 3 * np.sqrt(expected)).all(), (
            f"{slug}: retained condition counts are grossly imbalanced"
        )


def test_exit_survey_demographics_are_complete_and_add_up():
    for slug in STUDIES:
        _main_path, _long_path, survey_path = _paths(slug)
        survey = pd.read_csv(survey_path)
        assert survey["subject_id"].is_unique
        ages = pd.to_numeric(survey["age"], errors="coerce")
        assert ages.notna().all(), f"{slug}: missing/non-numeric age"
        assert np.isfinite(ages).all()
        assert survey["gender"].notna().all(), f"{slug}: missing gender row"


def main():
    tests = [
        test_public_subject_ids_do_not_overlap_across_studies,
        test_subject_sets_and_exclusions_recompute_from_public_csvs,
        test_every_retained_participant_has_one_complete_randomized_design,
        test_retained_condition_assignment_remains_balanced,
        test_exit_survey_demographics_are_complete_and_add_up,
    ]
    print("=" * 60)
    print("Participant-data integrity tests")
    print("=" * 60)
    for test in tests:
        test()
        print(f"✓ {test.__name__}")
    print("=" * 60)
    print(f"All {len(tests)} participant-data integrity tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
