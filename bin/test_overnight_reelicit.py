#!/usr/bin/env python3
"""Behavioral tests for the unattended re-elicitation driver.

The tests copy the real scripts into a temporary project and replace only the
paid or long-running external commands. They assert on the driver's exit status
and filesystem effects, not on its source text.

Run: uv run python bin/test_overnight_reelicit.py
"""

import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "bin" / "overnight-reelicit.sh"
MAIN_STUDIES = (
    "food_inv_desire",
    "food_inv_joint_de",
    "food_inv_intimacy",
    "food_inv_joint_ie",
    "nonfood_inv_joint_de",
    "nonfood_inv_joint_ie",
)


def _write_executable(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


@contextmanager
def _runner_sandbox(make_body="exit 0\n", uv_body="exit 0\n"):
    with tempfile.TemporaryDirectory() as d:
        sandbox = Path(d)
        (sandbox / "bin").mkdir()
        copied_runner = sandbox / "bin" / RUNNER.name
        copied_runner.write_bytes(RUNNER.read_bytes())
        copied_runner.chmod(0o755)
        for study in MAIN_STUDIES:
            (sandbox / "model" / "outputs" / "lm" / study).mkdir(parents=True)
        (sandbox / ".env").write_text("TOGETHER_API_KEY=sk-test\n")

        fake_bin = sandbox / "fake-bin"
        fake_bin.mkdir()
        _write_executable(fake_bin / "make", make_body)
        _write_executable(fake_bin / "uv", uv_body)
        _write_executable(fake_bin / "ps", "exit 0\n")
        env = {
            **os.environ,
            "_CAFFEINATED": "1",
            "OVERNIGHT_DIR": str(sandbox / "runs"),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        }
        yield sandbox, copied_runner, env


def _run(runner, sandbox, env, *args):
    return subprocess.run(
        [str(runner), *args],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_smoke_only_never_touches_the_canonical_artifact_set():
    """A smoke must use diagnostic targets and leave canonical files in place."""
    make_body = 'printf "%s\\n" "$*" >> "$CALL_LOG"\n'
    with _runner_sandbox(make_body=make_body) as (sandbox, runner, env):
        study_dir = sandbox / "model" / "outputs" / "lm" / "food_inv_desire"
        data = study_dir / "lm_alternatives.jsonl"
        manifest = study_dir / "lm_alternatives.manifest.json"
        data.write_text('{"vintage":"original"}\n')
        manifest.write_text('{"vintage":"original"}\n')
        call_log = sandbox / "make-calls.log"
        env["CALL_LOG"] = str(call_log)

        completed = _run(runner, sandbox, env, "--smoke-only", "--yes")

        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert data.read_text() == '{"vintage":"original"}\n'
        assert manifest.read_text() == '{"vintage":"original"}\n'
        targets = set(call_log.read_text().split())
        assert "lm-diag-food_inv_desire" in targets
        assert "lm-base-diag-food_inv_desire" in targets
        assert "lm-food_inv_desire" not in targets
        assert "lm-base-food_inv_desire" not in targets
        assert "Backup lm/ tables" not in completed.stdout
        assert "Restoring original tables" not in completed.stdout


def test_failed_parallel_chain_returns_nonzero_without_complete():
    """A failed study child must survive the xargs boundary."""
    make_body = r"""
case " $* " in
  *" K_RUNS=1 "*) exit 0 ;;
  *" lm-food_inv_desire "*) exit 7 ;;
  *) exit 0 ;;
esac
"""
    with _runner_sandbox(make_body=make_body) as (sandbox, runner, env):
        completed = _run(runner, sandbox, env, "--yes")

    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert "CHAIN FAILED: food_inv_desire" in completed.stdout
    assert "COMPLETE" not in completed.stdout


def test_failed_final_validation_returns_nonzero_without_complete():
    """A failed K=20 validator must gate the success message and exit code."""
    uv_body = r"""
case " $* " in
  *"_overnight_validate.py --k 20"*) exit 9 ;;
  *) exit 0 ;;
esac
"""
    with _runner_sandbox(uv_body=uv_body) as (sandbox, runner, env):
        completed = _run(runner, sandbox, env, "--yes")

    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert "FINAL VALIDATION FAILED" in completed.stdout
    assert "COMPLETE" not in completed.stdout


def test_failed_downstream_phase_returns_nonzero_without_complete():
    """The promised end-to-end run must not hide model-comparison/figure failure."""
    make_body = r"""
case " $* " in
  *" figures-results "*) exit 8 ;;
  *) exit 0 ;;
esac
"""
    with _runner_sandbox(make_body=make_body) as (sandbox, runner, env):
        completed = _run(runner, sandbox, env, "--yes")

    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert "DOWNSTREAM FAILED" in completed.stdout
    assert "COMPLETE" not in completed.stdout


def run_all_tests():
    tests = [
        value for name, value in sorted(globals().items()) if name.startswith("test_")
    ]
    failures = []
    for fn in tests:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - report every failure together
            failures.append((fn.__name__, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {fn.__name__}")
    print("=" * 60)
    if failures:
        print(f"{len(failures)} of {len(tests)} overnight tests FAILED:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        print("=" * 60)
        raise SystemExit(1)
    print(f"All {len(tests)} overnight tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
