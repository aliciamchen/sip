"""
Unit tests for the CV fold checkpoint (model/cv/_checkpoint.py): the
fingerprint-guarded JSONL side file that lets a multi-hour LOSO run resume
from its completed (variant, fold) refits after an interrupt or crash.

Covered:
  - init/append round trip (completed folds come back typed and equal),
  - a stale fingerprint (any changed input or refit config) discards the
    checkpoint instead of resuming across vintages,
  - a truncated tail line (crash mid-append) is dropped and the file repaired,
  - a corrupt header discards the file,
  - clear_checkpoint is idempotent,
  - run_fingerprint tracks every input that determines fold results
    (data CSV, LM tables, warm-start fit, patience, restarts).

Run standalone:  uv run python model/cv/test_checkpoint.py
Or with pytest:  uv run python -m pytest model/cv/test_checkpoint.py -v
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _checkpoint import (  # noqa: E402
    append_fold,
    checkpoint_path,
    clear_checkpoint,
    init_checkpoint,
    run_fingerprint,
)

# Any JSON-safe dict works as a fingerprint: init_checkpoint compares it to the
# header verbatim. The real one comes from run_fingerprint (tested below).
FP_A = {"version": 1, "slug": "test_slug", "patience": 100}
FP_B = {"version": 1, "slug": "test_slug", "patience": 50}


def _fold_results(i):
    """A (pred_rows, fold_row, trial_ll_rows) triple shaped like a real fold's
    output: plain str/int/float and nested float lists (delta_*_runs)."""
    pred_rows = [
        {
            "experiment": "test_slug",
            "scenario_label": f"scenario_{i}",
            "action": 0,
            "delta_desire": 0.123456789 + i,
            "delta_desire_runs": [0.1 + i, 0.2, 0.30000000000004],
            "model": "full",
        }
    ]
    fold_row = {
        "experiment": "test_slug",
        "variant": "full",
        "fold": i,
        "alpha_observer": 1.0831778049468994,
        "train_nll": 401.5039978027344,
        "n_train": 7215,
    }
    trial_ll_rows = [
        {"subject_id": f"subj_{i}", "held_out_ll": -0.0577 - i, "model": "full"}
    ]
    return pred_rows, fold_row, trial_ll_rows


def test_init_creates_and_round_trips():
    """A missing checkpoint initializes empty; appended folds come back equal
    (JSON round-trips the plain-typed fold results losslessly) and keyed by
    (variant, fold with fold as int)."""
    with tempfile.TemporaryDirectory() as d:
        path = checkpoint_path(Path(d))
        assert init_checkpoint(path, FP_A) == {}, "fresh checkpoint should be empty"
        assert path.exists(), "init should create the header file"

        r0, r3 = _fold_results(0), _fold_results(3)
        append_fold(path, "full", 0, *r0)
        append_fold(path, "base", 3, *r3)

        got = init_checkpoint(path, FP_A)
        assert set(got) == {("full", 0), ("base", 3)}, f"keys: {set(got)}"
        assert got[("full", 0)] == r0, "fold results changed across round trip"
        assert got[("base", 3)] == r3, "fold results changed across round trip"
        assert isinstance(next(iter(got))[1], int), "fold index must come back int"
    print("✓ checkpoint init/append/reload round-trips fold results")


def test_stale_fingerprint_discards():
    """A checkpoint whose header fingerprint doesn't match the current run's
    is discarded (fresh header, no folds) — never resumed across vintages."""
    with tempfile.TemporaryDirectory() as d:
        path = checkpoint_path(Path(d))
        init_checkpoint(path, FP_A)
        append_fold(path, "full", 0, *_fold_results(0))

        assert init_checkpoint(path, FP_B) == {}, "stale checkpoint not discarded"
        append_fold(path, "full", 1, *_fold_results(1))
        got = init_checkpoint(path, FP_B)
        assert set(got) == {("full", 1)}, "discard should have dropped old folds"
        assert init_checkpoint(path, FP_A) == {}, "old fingerprint must not match"
    print("✓ stale fingerprint discards the checkpoint")


def test_truncated_tail_recovered():
    """A partial trailing line (the process died mid-append) is dropped: the
    complete folds load, and the file is repaired so later appends parse."""
    with tempfile.TemporaryDirectory() as d:
        path = checkpoint_path(Path(d))
        init_checkpoint(path, FP_A)
        r0 = _fold_results(0)
        append_fold(path, "full", 0, *r0)
        with open(path, "a") as f:
            f.write('{"kind": "fold", "variant": "full", "fo')  # no newline

        got = init_checkpoint(path, FP_A)
        assert set(got) == {("full", 0)}, "complete fold lost with truncated tail"
        assert got[("full", 0)] == r0

        append_fold(path, "base", 1, *_fold_results(1))
        got = init_checkpoint(path, FP_A)
        assert set(got) == {("full", 0), ("base", 1)}, "append after repair broke"
        with open(path) as f:
            for line in f:
                json.loads(line)  # every line must parse after the repair
    print("✓ truncated tail is dropped and the file repaired")


def test_corrupt_header_discards():
    """A file whose first line isn't a valid header is discarded wholesale."""
    with tempfile.TemporaryDirectory() as d:
        path = checkpoint_path(Path(d))
        with open(path, "w") as f:
            f.write("not json at all\n")
            f.write(json.dumps({"kind": "fold", "variant": "full", "fold": 0}) + "\n")

        assert init_checkpoint(path, FP_A) == {}, "corrupt header not discarded"
        append_fold(path, "full", 2, *_fold_results(2))
        assert set(init_checkpoint(path, FP_A)) == {("full", 2)}
    print("✓ corrupt header discards the checkpoint")


def test_non_dict_json_lines_handled():
    """Valid-JSON-but-not-an-object lines must hit the graceful paths, not
    raise: a non-dict header discards the file; a non-dict (or key-less) fold
    line is treated as a corrupt tail — the folds before it survive."""
    with tempfile.TemporaryDirectory() as d:
        path = checkpoint_path(Path(d))
        with open(path, "w") as f:
            f.write('"just a string"\n')  # valid JSON, wrong shape
        assert init_checkpoint(path, FP_A) == {}, "non-dict header not discarded"

        r0 = _fold_results(0)
        append_fold(path, "full", 0, *r0)
        with open(path, "a") as f:
            f.write("42\n")  # valid JSON, not a fold record
            f.write(json.dumps({"kind": "fold", "variant": "full"}) + "\n")  # no fold
        got = init_checkpoint(path, FP_A)
        assert set(got) == {("full", 0)}, "folds before a corrupt line were lost"
        append_fold(path, "base", 1, *_fold_results(1))
        assert set(init_checkpoint(path, FP_A)) == {("full", 0), ("base", 1)}
    print("✓ non-dict JSON lines are handled gracefully")


def test_append_rejects_nan():
    """A NaN in a fold's results must fail the append loudly (ValueError) —
    silently persisting a NaN would poison the final outputs, and R's
    jsonlite can't parse a bare NaN token anyway."""
    with tempfile.TemporaryDirectory() as d:
        path = checkpoint_path(Path(d))
        init_checkpoint(path, FP_A)
        pred_rows, fold_row, trial_ll_rows = _fold_results(0)
        fold_row = dict(fold_row, train_nll=float("nan"))
        try:
            append_fold(path, "full", 0, pred_rows, fold_row, trial_ll_rows)
        except ValueError:
            pass
        else:
            raise AssertionError("NaN fold result was not rejected")
        assert init_checkpoint(path, FP_A) == {}, "rejected fold left a record"
    print("✓ append_fold rejects NaN fold results")


def test_clear_checkpoint_idempotent():
    """clear_checkpoint removes the file and tolerates a missing one."""
    with tempfile.TemporaryDirectory() as d:
        path = checkpoint_path(Path(d))
        clear_checkpoint(path)  # missing: must not raise
        init_checkpoint(path, FP_A)
        append_fold(path, "full", 0, *_fold_results(0))
        clear_checkpoint(path)
        assert not path.exists(), "checkpoint not removed"
        clear_checkpoint(path)  # idempotent
    print("✓ clear_checkpoint is idempotent")


def _fake_project(root, slug):
    """Minimal on-disk project tree with every input run_fingerprint hashes."""
    (root / "data" / slug).mkdir(parents=True)
    (root / "data" / slug / "main_trials_long.csv").write_text("subject_id\ns1\n")
    (root / "model" / "outputs" / "lm" / slug).mkdir(parents=True)
    (root / "model" / "outputs" / "lm" / slug / "lm_runs.jsonl").write_text("{}\n")
    (root / "model" / "outputs" / slug).mkdir(parents=True)
    (root / "model" / "outputs" / slug / "fit_results.json").write_text("[]")
    (root / "model" / "inverse").mkdir(parents=True)
    (root / "model" / "inverse" / "_helpers.py").write_text("# fit logic\n")


def test_run_fingerprint_tracks_inputs():
    """The fingerprint is stable across calls and changes with every input
    that changes fold results: the data CSV, the LM tables (incl. a base file
    appearing), the warm-start fit, and the refit config (patience, restarts).
    A missing warm start still fingerprints (CV cold-starts in that case)."""
    slug = "test_slug"
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _fake_project(root, slug)
        fp = run_fingerprint(slug, "desire", 100, 2, project_root=root)
        assert fp == run_fingerprint(slug, "desire", 100, 2, project_root=root), (
            "fingerprint not deterministic"
        )
        assert fp != run_fingerprint(slug, "desire", 50, 2, project_root=root)
        assert fp != run_fingerprint(slug, "desire", 100, 1, project_root=root)

        (root / "data" / slug / "main_trials_long.csv").write_text("subject_id\ns2\n")
        fp_data = run_fingerprint(slug, "desire", 100, 2, project_root=root)
        assert fp != fp_data, "changed data CSV not detected"

        lm = root / "model" / "outputs" / "lm" / slug
        (lm / "lm_runs.jsonl").write_text('{"run_id": 1}\n')
        fp_lm = run_fingerprint(slug, "desire", 100, 2, project_root=root)
        assert fp_data != fp_lm, "changed lm_runs.jsonl not detected"

        (lm / "lm_runs_base.jsonl").write_text("{}\n")
        fp_base = run_fingerprint(slug, "desire", 100, 2, project_root=root)
        assert fp_lm != fp_base, "appearing lm_runs_base.jsonl not detected"

        fit = root / "model" / "outputs" / slug / "fit_results.json"
        fit.write_text('[{"model": "full"}]')
        fp_warm = run_fingerprint(slug, "desire", 100, 2, project_root=root)
        assert fp_base != fp_warm, "changed warm-start fit not detected"

        fit.unlink()
        fp_cold = run_fingerprint(slug, "desire", 100, 2, project_root=root)
        assert fp_warm != fp_cold, "missing warm start should change fingerprint"

        # A mid-run edit to the model-math code must invalidate the checkpoint
        # too — otherwise a resume splices folds from two code vintages.
        (root / "model" / "inverse" / "_helpers.py").write_text("# edited\n")
        fp_code = run_fingerprint(slug, "desire", 100, 2, project_root=root)
        assert fp_cold != fp_code, "changed model code not detected"
        json.dumps(fp)  # must be JSON-serializable for the header
    print("✓ run_fingerprint tracks data, LM tables, warm start, config, code")


def test_fingerprint_distinguishes_configs():
    """Two fingerprints differing only in `config_fields` (the reported run vs
    the preregistered no-reweighting run) must mismatch, so a checkpoint
    written under one run config is never resumed under another."""
    slug = "food_inv_desire"
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _fake_project(root, slug)
        fp_a = run_fingerprint(
            slug,
            "desire",
            100,
            2,
            {"tag": "reported", "runs": "lm_runs.jsonl"},
            project_root=root,
        )
        fp_b = run_fingerprint(
            slug,
            "desire",
            100,
            2,
            {
                "tag": "uniform-noreweight",
                "runs": "lm_runs.jsonl",
            },
            project_root=root,
        )
        assert fp_a != fp_b, "config_fields not reflected in the fingerprint"
        # The no-config default must still fingerprint (and equal the explicit
        # reported config), so a plain CV run keeps a stable checkpoint key.
        fp_default = run_fingerprint(slug, "desire", 100, 2, project_root=root)
        assert fp_default == fp_a, "default config_fields must be the reported one"
        json.dumps(fp_b)  # must stay JSON-serializable for the header
    print("✓ run_fingerprint distinguishes run configs")


def run_all_tests():
    print("=" * 60)
    print("CV checkpoint tests")
    print("=" * 60)
    test_init_creates_and_round_trips()
    test_stale_fingerprint_discards()
    test_truncated_tail_recovered()
    test_corrupt_header_discards()
    test_non_dict_json_lines_handled()
    test_append_rejects_nan()
    test_clear_checkpoint_idempotent()
    test_run_fingerprint_tracks_inputs()
    test_fingerprint_distinguishes_configs()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
