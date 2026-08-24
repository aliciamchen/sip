#!/usr/bin/env python3
"""Content invariants for the committed LM elicitation tables.

Prompt-hash and resume behavior are tested in test_elicitation_guards.py.  This
file checks the scientific inputs that reached the reported model: complete
cell coverage, well-formed local action sets, valid feature scales, and the
manipulation orderings cited in the manuscript.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from study_registry import STUDIES
from utils import get_project_root


ROOT = get_project_root()
K_RUNS = 20
N_SCENARIOS = 16
OBSERVED_ACTIONS = {"no_share", "low_risk_share", "high_risk_share"}
GIVEN_RELATIONSHIP = {
    "food_inv_desire",
    "food_inv_joint_de",
    "nonfood_inv_joint_de",
}
RELATIONSHIP_ORDER = (
    "max_formal",
    "somewhat_formal",
    "somewhat_intimate",
    "max_intimate",
)


def _read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _runs(slug, base=False):
    suffix = "_base" if base else ""
    path = ROOT / "model" / "outputs" / "lm" / slug / f"lm_runs{suffix}.jsonl"
    return path, _read_jsonl(path)


def test_every_reported_table_has_exactly_twenty_complete_runs():
    for slug in STUDIES:
        path, rows = _runs(slug)
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["k_runs"] == K_RUNS
        assert manifest["n_scenarios"] == N_SCENARIOS
        assert manifest["n_records"] == len(rows)
        expected_per_run = (
            N_SCENARIOS * 3 * 2 * (4 if slug in GIVEN_RELATIONSHIP else 2)
        )
        counts = defaultdict(int)
        for row in rows:
            counts[row["run_id"]] += 1
        assert set(counts) == set(range(K_RUNS)), f"{slug}: missing run IDs"
        assert set(counts.values()) == {expected_per_run}, f"{slug}: incomplete runs"
        assert len(rows) == K_RUNS * expected_per_run


def test_every_local_action_set_is_nonempty_unique_and_well_formed():
    for slug in STUDIES:
        _path, rows = _runs(slug)
        for row in rows:
            actions = row["actions"]
            assert 2 <= len(actions) <= 12, f"{slug}: empty or oversized comparison set"
            assert [a["slot"] for a in actions] == list(range(len(actions)))
            observed = [a for a in actions if a.get("is_observed")]
            assert len(observed) == 1 and observed[0]["slot"] == 0
            texts = [a["action_text"].strip().casefold() for a in actions]
            assert len(texts) == len(set(texts)), f"{slug}: duplicate action text"
            for action in actions:
                for feature in ("risk", "effort", "g"):
                    value = action[feature]
                    assert math.isfinite(value) and 0 <= value <= 1


def test_given_magnitudes_have_the_declared_ordering():
    for slug in STUDIES:
        _path, rows = _runs(slug)
        if slug in GIVEN_RELATIONSHIP:
            values = defaultdict(list)
            for row in rows:
                assert "intimacy" in row and 0 <= row["intimacy"] <= 1
                values[row["intimacy_condition"]].append(row["intimacy"])
            means = [np.mean(values[level]) for level in RELATIONSHIP_ORDER]
            assert all(a < b for a, b in zip(means, means[1:])), (
                f"{slug}: rated intimacy is not strictly monotonic"
            )
        else:
            values = defaultdict(list)
            for row in rows:
                assert "desire" in row and 0 <= row["desire"] <= 1
                values[row["desire_condition"]].append(row["desire"])
            assert np.mean(values["high"]) > np.mean(values["low"]), (
                f"{slug}: high-desire descriptions do not rate above low"
            )


def test_observed_action_features_preserve_the_designed_structure():
    for slug in STUDIES:
        _path, rows = _runs(slug)
        features = defaultdict(lambda: defaultdict(list))
        low_risk_effort = defaultdict(list)
        for row in rows:
            action = row["actions"][0]
            label = row["observed_action"]
            assert label in OBSERVED_ACTIONS
            for feature in ("risk", "g"):
                features[feature][label].append(action[feature])
            if label == "low_risk_share":
                low_risk_effort[row["effort_condition"]].append(action["effort"])
        risk = {k: np.mean(v) for k, v in features["risk"].items()}
        goal = {k: np.mean(v) for k, v in features["g"].items()}
        assert risk["no_share"] < risk["low_risk_share"] < risk["high_risk_share"]
        assert goal["no_share"] < goal["low_risk_share"]
        assert goal["no_share"] < goal["high_risk_share"]
        assert np.mean(low_risk_effort["high"]) > np.mean(low_risk_effort["low"])


def test_base_tables_exist_only_where_declared_and_drop_relationship():
    lm_root = ROOT / "model" / "outputs" / "lm"
    present = {p.parent.name for p in lm_root.glob("*/lm_runs_base.jsonl")}
    assert present == GIVEN_RELATIONSHIP
    for slug in GIVEN_RELATIONSHIP:
        path, rows = _runs(slug, base=True)
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["k_runs"] == K_RUNS
        assert manifest["n_records"] == len(rows) == K_RUNS * N_SCENARIOS * 3 * 2
        assert {row["run_id"] for row in rows} == set(range(K_RUNS))
        for row in rows:
            assert "intimacy_condition" not in row
            assert "intimacy" not in row
            actions = row["actions"]
            assert 2 <= len(actions) <= 12
            assert [action["slot"] for action in actions] == list(range(len(actions)))
            assert sum(bool(action.get("is_observed")) for action in actions) == 1
            assert actions[0].get("is_observed")
            for action in actions:
                for feature in ("risk", "effort", "g"):
                    assert math.isfinite(action[feature]) and 0 <= action[feature] <= 1


def main():
    tests = [
        test_every_reported_table_has_exactly_twenty_complete_runs,
        test_every_local_action_set_is_nonempty_unique_and_well_formed,
        test_given_magnitudes_have_the_declared_ordering,
        test_observed_action_features_preserve_the_designed_structure,
        test_base_tables_exist_only_where_declared_and_drop_relationship,
    ]
    print("=" * 60)
    print("LM-table content tests")
    print("=" * 60)
    for test in tests:
        test()
        print(f"✓ {test.__name__}")
    print("=" * 60)
    print(f"All {len(tests)} LM-table content tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
