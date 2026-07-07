"""
Offline tests for the raw-JSON -> CSV converter (analysis/json_to_csv.py).

Run standalone:  uv run python analysis/test_json_to_csv.py

Each test builds tiny synthetic jsPsych-style JSON fixtures in a temp
directory and runs the converter's two stages (process_json_files +
create_main_trials_long) against them. Coverage:
  - happy-path conversion for both response schemas (single-slider
    food_inv_desire, two-slider survey-html-form food_inv_joint_de),
  - the legacy "neither" intimacy label normalizing to "somewhat_formal",
  - both exclusion-rule branches (1a's lax rule vs. the strict rule),
  - hard errors: trial/exit-survey subject-set mismatch, duplicate subject
    across files, missing subject_id, and zero parsed rows (which must abort
    before the long-CSV rebuild so stale CSVs are never reused).
"""

import contextlib
import csv
import io
import json
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from json_to_csv import (
    EXPERIMENT_CONFIGS,
    create_main_trials_long,
    process_json_files,
)

# Mirrors the fixed namespace in json_to_csv.generate_deterministic_id (defined
# inside process_json_files, so it can't be imported directly).
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def anon(original_id):
    """The deterministic anonymous ID the converter assigns to a subject."""
    return str(uuid.uuid5(_NAMESPACE, original_id))


def desire_trial(
    subject, scenario, stage, response, intimacy="max_formal", stimulus_index=0
):
    """One single-slider rating trial in the food_inv_desire raw schema."""
    return {
        "subject_id": subject,
        "response_type": "response",
        "scenario_label": scenario,
        "stimulus_index": stimulus_index,
        "action_condition": "no_share",
        "effort_condition": "low",
        "intimacy_condition": intimacy,
        "stage": stage,
        "response": response,
    }


def joint_de_trial(
    subject, scenario, stage, desire, effort, intimacy="max_formal", stimulus_index=0
):
    """One two-slider rating trial in the food_inv_joint_de raw schema (the
    survey-html-form response is an object with one key per slider)."""
    return {
        "subject_id": subject,
        "response_type": "response",
        "scenario_label": scenario,
        "stimulus_index": stimulus_index,
        "action_condition": "no_share",
        "intimacy_condition": intimacy,
        "stage": stage,
        "response": {"desire": desire, "effort": effort},
    }


def exit_trial(subject, attention_passed=True, memory_correct_count=2):
    """One exit-survey trial (attention/memory live on the trial itself)."""
    return {
        "subject_id": subject,
        "response_type": "exit_survey",
        "response": {
            "gender": "woman",
            "age": 30,
            "understood": "yes",
            "comments": "",
        },
        "attention_passed": attention_passed,
        "memory_correct_count": memory_correct_count,
        "comprehension_attempt": 1,
    }


def write_json(input_dir, name, trials):
    (Path(input_dir) / f"{name}.json").write_text(json.dumps(trials))


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert(input_dir, output_dir, config, quiet=True):
    """Run both converter stages, suppressing their progress prints."""
    sink = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(sink):
        process_json_files(input_dir, output_dir, config)
        create_main_trials_long(output_dir, config)
    return sink


def expect_exit(fn):
    """Run fn, requiring a non-zero SystemExit; returns (exit, stdout text)."""
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            fn()
    except SystemExit as e:
        assert e.code, f"expected a non-zero exit, got code {e.code!r}"
        return e, sink.getvalue()
    raise AssertionError("expected SystemExit, but the conversion succeeded")


def test_happy_path_single_slider():
    """food_inv_desire: raw 0-100 responses land on the 0-1 scale, condition
    columns are renamed in the long CSV, and subject IDs are anonymized."""
    config = EXPERIMENT_CONFIGS["food_inv_desire"]
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "raw", Path(tmp) / "out"
        inp.mkdir()
        write_json(
            inp,
            "s1",
            [
                desire_trial("s1", "apples", "prior", 73, stimulus_index=3),
                desire_trial("s1", "apples", "posterior", 40, stimulus_index=3),
                exit_trial("s1"),
            ],
        )
        write_json(
            inp,
            "s2",
            [
                desire_trial("s2", "oysters", "prior", 0),
                desire_trial("s2", "oysters", "posterior", 100),
                exit_trial("s2"),
            ],
        )
        convert(inp, out, config)

        main = read_csv_rows(out / "main_trials.csv")
        assert len(main) == 4, f"expected 4 main-trial rows, got {len(main)}"
        assert list(main[0].keys()) == config["main_trial_fields"]
        s1_prior = next(
            r for r in main if r["subject_id"] == anon("s1") and r["stage"] == "prior"
        )
        assert float(s1_prior["response"]) == 0.73, s1_prior["response"]
        assert s1_prior["stimulus_index"] == "3", s1_prior["stimulus_index"]

        exit_rows = read_csv_rows(out / "exit_survey.csv")
        assert len(exit_rows) == 2
        assert {r["subject_id"] for r in exit_rows} == {anon("s1"), anon("s2")}

        long = read_csv_rows(out / "main_trials_long.csv")
        assert len(long) == 4
        assert "effort" in long[0] and "intimacy" in long[0], list(long[0].keys())
        assert "effort_condition" not in long[0] and "intimacy_condition" not in long[0]
        assert "stimulus_index" in long[0], list(long[0].keys())
        assert {r["subject_id"] for r in long} == {anon("s1"), anon("s2")}
    print("✓ single-slider happy path (food_inv_desire)")


def test_happy_path_two_slider():
    """food_inv_joint_de: the response object splits into desire_rating and
    effort_rating, both normalized to 0-1."""
    config = EXPERIMENT_CONFIGS["food_inv_joint_de"]
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "raw", Path(tmp) / "out"
        inp.mkdir()
        write_json(
            inp,
            "s1",
            [
                joint_de_trial("s1", "apples", "prior", desire=40, effort=60),
                joint_de_trial("s1", "apples", "posterior", desire=80, effort=10),
                exit_trial("s1"),
            ],
        )
        convert(inp, out, config)

        main = read_csv_rows(out / "main_trials.csv")
        assert list(main[0].keys()) == config["main_trial_fields"]
        prior = next(r for r in main if r["stage"] == "prior")
        assert float(prior["desire_rating"]) == 0.4, prior["desire_rating"]
        assert float(prior["effort_rating"]) == 0.6, prior["effort_rating"]

        long = read_csv_rows(out / "main_trials_long.csv")
        assert len(long) == 2
        assert "intimacy" in long[0] and "intimacy_condition" not in long[0]
    print("✓ two-slider happy path (food_inv_joint_de)")


def test_legacy_neither_normalized():
    """Raw pre-2026-06-19 'neither' intimacy labels become 'somewhat_formal'
    in both output CSVs."""
    config = EXPERIMENT_CONFIGS["food_inv_joint_de"]
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "raw", Path(tmp) / "out"
        inp.mkdir()
        write_json(
            inp,
            "s1",
            [
                joint_de_trial("s1", "apples", "prior", 50, 50, intimacy="neither"),
                joint_de_trial("s1", "apples", "posterior", 60, 40, intimacy="neither"),
                exit_trial("s1"),
            ],
        )
        convert(inp, out, config)

        main = read_csv_rows(out / "main_trials.csv")
        assert all(r["intimacy_condition"] == "somewhat_formal" for r in main), main
        long = read_csv_rows(out / "main_trials_long.csv")
        assert all(r["intimacy"] == "somewhat_formal" for r in long), long
    print("✓ legacy 'neither' intimacy label normalized to 'somewhat_formal'")


def test_exclusion_rule_lax():
    """1a's lax rule excludes only failed-attention AND 0 memory questions."""
    config = EXPERIMENT_CONFIGS["food_inv_desire"]
    assert config["exclusion_rule"] == "lax"
    cases = {  # subject -> (attention_passed, memory_correct_count, retained?)
        "both_fail": (False, 0, False),
        "attn_fail_only": (False, 1, True),
        "memory_fail_only": (True, 0, True),
        "both_pass": (True, 2, True),
    }
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "raw", Path(tmp) / "out"
        inp.mkdir()
        for subj, (attn, mem, _) in cases.items():
            write_json(
                inp,
                subj,
                [
                    desire_trial(subj, "apples", "prior", 50),
                    desire_trial(subj, "apples", "posterior", 60),
                    exit_trial(subj, attention_passed=attn, memory_correct_count=mem),
                ],
            )
        convert(inp, out, config)
        retained = {
            r["subject_id"] for r in read_csv_rows(out / "main_trials_long.csv")
        }
        expected = {anon(s) for s, (_, _, keep) in cases.items() if keep}
        assert retained == expected, (
            f"lax rule retained {retained}, expected {expected}"
        )
    print("✓ lax exclusion rule (exclude only failed attention AND 0 memory)")


def test_exclusion_rule_strict():
    """The strict rule retains only passed-attention AND >=1 memory question."""
    config = EXPERIMENT_CONFIGS["food_inv_joint_de"]
    assert config["exclusion_rule"] == "strict"
    cases = {
        "both_fail": (False, 0, False),
        "attn_fail_only": (False, 2, False),
        "memory_fail_only": (True, 0, False),
        "both_pass": (True, 1, True),
    }
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "raw", Path(tmp) / "out"
        inp.mkdir()
        for subj, (attn, mem, _) in cases.items():
            write_json(
                inp,
                subj,
                [
                    joint_de_trial(subj, "apples", "prior", 50, 50),
                    joint_de_trial(subj, "apples", "posterior", 60, 40),
                    exit_trial(subj, attention_passed=attn, memory_correct_count=mem),
                ],
            )
        convert(inp, out, config)
        retained = {
            r["subject_id"] for r in read_csv_rows(out / "main_trials_long.csv")
        }
        expected = {anon(s) for s, (_, _, keep) in cases.items() if keep}
        assert retained == expected, (
            f"strict rule retained {retained}, expected {expected}"
        )
    print("✓ strict exclusion rule (retain only passed attention AND >=1 memory)")


def test_exit_survey_trial_mismatch_errors():
    """A subject with trials but no exit-survey record (or vice versa) must
    abort the long-CSV build, naming the difference."""
    config = EXPERIMENT_CONFIGS["food_inv_desire"]
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "raw", Path(tmp) / "out"
        inp.mkdir()
        write_json(
            inp,
            "complete",
            [
                desire_trial("complete", "apples", "prior", 50),
                desire_trial("complete", "apples", "posterior", 60),
                exit_trial("complete"),
            ],
        )
        # Trials but no exit-survey record: without the subject-set check this
        # participant would silently pass every exclusion filter.
        write_json(
            inp,
            "no_exit",
            [
                desire_trial("no_exit", "apples", "prior", 50),
                desire_trial("no_exit", "apples", "posterior", 60),
            ],
        )
        e, _ = expect_exit(lambda: convert(inp, out, config))
        assert "no exit-survey record" in str(e.code), e.code
        assert anon("no_exit") in str(e.code), e.code
    print("✓ trial/exit-survey subject-set mismatch is a hard error")


def test_duplicate_subject_across_files_errors():
    """The same subject_id in two JSON files must abort the conversion."""
    config = EXPERIMENT_CONFIGS["food_inv_desire"]
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "raw", Path(tmp) / "out"
        inp.mkdir()
        trials = [
            desire_trial("dupe", "apples", "prior", 50),
            desire_trial("dupe", "apples", "posterior", 60),
            exit_trial("dupe"),
        ]
        write_json(inp, "first_copy", trials)
        write_json(inp, "second_copy", trials)
        _, out_text = expect_exit(lambda: convert(inp, out, config, quiet=False))
        assert "already seen in" in out_text, out_text
        assert not (out / "main_trials.csv").exists(), "CSV written despite abort"
    print("✓ duplicate subject_id across files is a hard error")


def test_missing_subject_id_errors():
    """A file whose first trial has no subject_id must abort, naming the file
    (no silent 'unknown' fallback that merges participants)."""
    config = EXPERIMENT_CONFIGS["food_inv_desire"]
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "raw", Path(tmp) / "out"
        inp.mkdir()
        bad = [
            {k: v for k, v in t.items() if k != "subject_id"}
            for t in [
                desire_trial("x", "apples", "prior", 50),
                desire_trial("x", "apples", "posterior", 60),
                exit_trial("x"),
            ]
        ]
        write_json(inp, "anonymous_file", bad)
        _, out_text = expect_exit(lambda: convert(inp, out, config, quiet=False))
        assert "anonymous_file.json" in out_text, out_text
        assert "no subject_id" in out_text, out_text
    print("✓ missing subject_id is a hard error naming the file")


def test_zero_parsed_rows_aborts_before_long_rebuild():
    """If no rating trials parse, the conversion must exit non-zero before the
    long CSV is rebuilt, leaving any stale CSVs untouched (and unused)."""
    config = EXPERIMENT_CONFIGS["food_inv_desire"]
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "raw", Path(tmp) / "out"
        inp.mkdir()
        out.mkdir()
        # Stale outputs from an earlier run; the abort must not rebuild the
        # long CSV from the stale main_trials.csv.
        stale_main = "subject_id,stage\nstale,prior\n"
        stale_long = "subject_id,stage\nstale,prior\n"
        (out / "main_trials.csv").write_text(stale_main)
        (out / "main_trials_long.csv").write_text(stale_long)
        # Raw file with an exit survey but zero rating trials.
        write_json(inp, "s1", [exit_trial("s1")])
        e, _ = expect_exit(lambda: convert(inp, out, config))
        assert "No rows parsed" in str(e.code), e.code
        assert (out / "main_trials.csv").read_text() == stale_main
        assert (out / "main_trials_long.csv").read_text() == stale_long
    print("✓ zero parsed rows aborts before the long-CSV rebuild")


def run_all_tests():
    print("=" * 60)
    print("json_to_csv converter tests")
    print("=" * 60)
    test_happy_path_single_slider()
    test_happy_path_two_slider()
    test_legacy_neither_normalized()
    test_exclusion_rule_lax()
    test_exclusion_rule_strict()
    test_exit_survey_trial_mismatch_errors()
    test_duplicate_subject_across_files_errors()
    test_missing_subject_id_errors()
    test_zero_parsed_rows_aborts_before_long_rebuild()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
