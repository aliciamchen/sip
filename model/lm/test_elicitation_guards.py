"""Tests for the alternatives prompt and the elicitation vintage guard.

The LM tables are expensive to regenerate (a full six-study re-elicitation is
tens of dollars and hours of wall clock) and the failure mode they guard against
is silent: a resumed run skips already-done units, so resuming after a prompt
edit splices records from two wordings into one file, which then feeds fits and
CV as if it were a single vintage. Nothing downstream can detect that. So the
guard is tested here rather than trusted.

What's covered:

  - There is exactly ONE alternatives system prompt. Provenance rests on
    `prompts_sha256` — a hash of prompts.py — which can only stand in for "what
    text was sent" while each stage has a single prompt. A second, run-time-
    selectable variant would silently break that, so its absence is asserted.
  - That prompt reasons before answering (adopted 2026-07-28 for coverage), and
    the rating stages deliberately do not.
  - Every generation captures its reasoning prose, at the larger token budget.
  - `guard_resume_prompt_mismatch` refuses a resume across a prompts.py edit.
  - The elicitation is reachable for all six studies, and the diagnostic
    `--arm-output-only` path still routes to its own vintage.

Run: uv run python model/lm/test_elicitation_guards.py
"""

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts  # noqa: E402
from client import (  # noqa: E402
    RESUME_PROMPT_MISMATCH_ENV,
    _prompts_sha,
    guard_resume_prompt_mismatch,
)


class _Reached(Exception):
    """Raised by a stubbed dependency to prove control flow got that far."""


@contextmanager
def _expect(exc_type, contains=None):
    """Assert the block raises `exc_type`, optionally with `contains` in its
    message. Stands in for pytest.raises (this repo's tests are plain scripts).

    Catches BaseException so an unexpected SystemExit surfaces as a readable
    assertion instead of tearing down the interpreter mid-suite.
    """
    try:
        yield
    except exc_type as e:
        if contains is not None and contains not in str(e):
            raise AssertionError(
                f"expected {contains!r} in the {exc_type.__name__} message, got: {e}"
            ) from None
        return
    except BaseException as e:  # noqa: BLE001 - deliberate, see docstring
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(e).__name__}: {e}"
        ) from None
    raise AssertionError(f"expected {exc_type.__name__} but nothing was raised")


@contextmanager
def _manifest(**fields):
    """A temp dir holding a minimal alternatives manifest; yields the notional
    output JSONL path the guard is asked about."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        manifest = {"stage": "generate_alternatives", "study": "food_inv_joint_de"}
        manifest.update(fields)
        (d / "lm_alternatives.manifest.json").write_text(json.dumps(manifest))
        yield d / "lm_alternatives.jsonl"


@contextmanager
def _stub(module, name, fn):
    """Temporarily replace `module.name`."""
    real = getattr(module, name)
    setattr(module, name, fn)
    try:
        yield
    finally:
        setattr(module, name, real)


@contextmanager
def _env(key, value):
    prior = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if prior is None:
            del os.environ[key]
        else:
            os.environ[key] = prior


# ------------------------------------------------------ one prompt, not two


def test_there_is_exactly_one_alternatives_system_prompt():
    """Provenance for every elicited table is `prompts_sha256`, a hash of this
    file. That only identifies what text was sent while the stage has ONE
    prompt: two run-time-selectable variants would share a hash while producing
    different tables, leaving the artifacts ambiguous and the resume guard
    blind. If a variant is ever genuinely needed, it needs its own recorded
    provenance field — not just a second constant.
    """
    variants = [
        n
        for n in dir(prompts)
        if n.startswith("ALTERNATIVES") and n.endswith("SYSTEM_PROMPT") is False
    ]
    assert variants == [], f"unexpected alternatives prompt variants: {variants}"
    assert isinstance(prompts.ALTERNATIVES_SYSTEM_PROMPT, str)


def test_generate_alternatives_has_no_prompt_mode_flag():
    """The elicitation must not offer a choice of prompt: one prompt means the
    CLI has no mode to select, and `--cot` in particular used to be coupled to
    `--arm-output-only`, which made the adopted wording unreachable from the
    production path."""
    src = (Path(__file__).resolve().parent / "generate_alternatives.py").read_text()
    for gone in ("--cot", "cot=", "COT_MAX_TOKENS", "_cot_system_prompt"):
        assert gone not in src, f"{gone!r} still present in generate_alternatives.py"


# ------------------------------------------------ the prompt reasons first


def test_alternatives_prompt_reasons_before_answering():
    p = prompts.ALTERNATIVES_SYSTEM_PROMPT
    assert "step by step before answering" in p
    assert "First, briefly explain step by step" in p
    assert "with no other text after the array:" in p
    # The old respond-only-with-JSON close must be gone, or the model returns a
    # bare array and the reasoning sidecar is empty.
    assert "no explanation:" not in p


def test_rating_prompts_stay_unreasoned():
    """Only generation reasons. A rating is a snap judgment, mirroring the
    participant, so the rating preamble keeps its intuition wording and none of
    the rating prompts ask for step-by-step thought."""
    assert "just going off your intuition" in prompts._PREAMBLE_RATING
    assert "step by step" not in prompts._PREAMBLE_RATING
    rating_prompts = [
        prompts.DESIRE_SYSTEM_PROMPT,
        prompts.INTIMACY_SYSTEM_PROMPT,
        prompts.PRIOR_DESIRE_SYSTEM_PROMPT,
        prompts.PRIOR_EFFORT_SYSTEM_PROMPT,
        prompts.PRIOR_INTIMACY_SYSTEM_PROMPT,
    ] + [prompts.system_prompt(t) for t in ("g", "effort", "risk")]
    for p in rating_prompts:
        assert "step by step" not in p, p[:80]
        assert "just going off your intuition" in p, p[:80]


def test_alternatives_prompt_keeps_the_methodological_framing():
    """The substance the paper's argument rests on, which the reasoning close
    must not have displaced."""
    p = prompts.ALTERNATIVES_SYSTEM_PROMPT
    for required in (
        "some resource",
        "a piece of information",  # keeps the nonfood/disclosure mode available
        "The vignette omits some information",  # the unconditional-phrasing clause
        "an observer would compare with the action they actually took",
        '{"action":',
    ):
        assert required in p, required


# --------------------------------------------- reasoning is always captured


def test_every_generation_captures_reasoning_at_the_larger_budget():
    """Since the prompt always reasons, the elicitation must always request the
    raw text and always write the sidecar — the reasoning is the record of how
    each comparison set was arrived at, and a coverage audit reads it. A budget
    left at the rating stages' default would truncate the array away."""
    import generate_alternatives as ga

    assert ga.ALT_MAX_TOKENS > 800
    src = (Path(__file__).resolve().parent / "generate_alternatives.py").read_text()
    assert "with_raw=True" in src
    assert "max_tokens=ALT_MAX_TOKENS" in src
    # The sidecar write is unconditional (inside the checkpoint, not behind a flag).
    assert "write_jsonl_atomic(reasoning_path, reasoning_rows)" in src


# ------------------------------------------------------------ resume guard


def test_resume_refused_after_a_prompt_edit():
    """The case that matters right now: the six existing tables were elicited
    under an older prompts.py, so re-elicitation must not append onto them."""
    with _manifest(prompts_sha256="deadbeef0000") as out:
        with _expect(SystemExit, "would silently mix data from two prompt versions"):
            guard_resume_prompt_mismatch(out)


def test_resume_allowed_when_the_prompt_is_unchanged():
    with _manifest(prompts_sha256=_prompts_sha()) as out:
        guard_resume_prompt_mismatch(out)  # must not raise


def test_prompt_edit_is_overridable_by_env_for_a_deliberate_mixed_run():
    """The escape hatch exists so a deliberate K-extension across a trivial
    wording fix isn't impossible; the mixed provenance is then recorded in
    prompt_sha_history rather than hidden."""
    with _env(RESUME_PROMPT_MISMATCH_ENV, "allow"):
        with _manifest(prompts_sha256="deadbeef0000") as out:
            guard_resume_prompt_mismatch(out)  # warns, does not raise


def test_missing_manifest_is_not_an_error():
    """No manifest means nothing to contradict — the pre-manifest behavior."""
    with tempfile.TemporaryDirectory() as d:
        guard_resume_prompt_mismatch(Path(d) / "lm_alternatives.jsonl")


def test_the_live_tables_would_be_refused_today():
    """Not hypothetical: assert the real manifests on disk carry a superseded
    hash, so the planned re-elicitation is forced to start clean. If this ever
    fails because the hashes match, the tables are current and the vintage
    marker in prompts.py should come out."""
    root = Path(__file__).resolve().parents[2]
    live = list(
        (root / "model" / "outputs" / "lm").glob("*/lm_alternatives.manifest.json")
    )
    if not live:
        return  # nothing elicited yet
    cur = _prompts_sha()
    stale = [
        p.parent.name
        for p in live
        if json.loads(p.read_text()).get("prompts_sha256") != cur
    ]
    assert len(stale) == len(live), (
        "some studies' tables match the current prompt and some don't, which is "
        f"a mixed roster: stale={sorted(stale)} of {len(live)}"
    )


# ----------------------------------------------------- reachability / paths


def test_elicitation_is_reachable_for_every_study_in_the_roster():
    """All six must get past argument validation to the credential load, where
    a sentinel stops them before any spend. The K=1 smoke and the eventual
    re-elicitation both need 1a and 2a, which the old diagnostic-only path
    excluded."""
    import generate_alternatives as ga

    def _sentinel():
        raise _Reached

    assert len(ga._STUDY_CONFIG) == 6, sorted(ga._STUDY_CONFIG)
    with _stub(ga, "load_api_key", _sentinel):
        for slug in ga._STUDY_CONFIG:
            with _expect(_Reached):
                ga.main(slug)


def test_base_mode_is_reachable_for_the_given_relationship_studies():
    import generate_alternatives as ga

    def _sentinel():
        raise _Reached

    with _stub(ga, "load_api_key", _sentinel):
        for slug in ga._BASE_OVERRIDE:
            with _expect(_Reached):
                ga.main(slug, base=True)


def test_base_mode_rejected_where_there_is_no_relationship_paragraph():
    import generate_alternatives as ga

    with _stub(ga, "load_api_key", lambda: "sk-test"):
        with _expect(SystemExit, "--base is only defined for"):
            ga.main("food_inv_intimacy", base=True)


def test_arm_output_only_available_for_every_study_and_never_canonical():
    """A prompt-edit smoke must be able to check any study — including 1a and 2a,
    which the old effort-inferring-only restriction excluded — and must never
    write the canonical file, since the current fits and figures are built on
    those tables."""
    import generate_alternatives as ga

    assert set(ga._DIAG_STUDIES) == set(ga._STUDY_CONFIG)

    seen = []

    def _capture(study):
        seen.append(study)
        raise _Reached

    with _stub(ga, "load_api_key", lambda: "sk-test"):
        with _stub(ga, "load_scenarios", _capture):
            for slug in ga._STUDY_CONFIG:
                with _expect(_Reached):
                    ga.main(slug, arm_output_only=True)
    assert seen == list(ga._STUDY_CONFIG)
    # The canonical output name is what the arm mode must displace, not write.
    for slug in ga._STUDY_CONFIG:
        assert ga._STUDY_CONFIG[slug]["output"] == "lm_alternatives.jsonl"


def test_arm_output_only_rejects_the_base_overlay():
    """`--base` has its own vintage already; combining the two would be
    ambiguous about which file is being written."""
    import generate_alternatives as ga

    with _stub(ga, "load_api_key", lambda: "sk-test"):
        with _expect(SystemExit, "arm-output mode is defined for"):
            ga.main("food_inv_desire", base=True, arm_output_only=True)


def run_all_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 - report, don't abort the suite
            failures.append((fn.__name__, f"{type(e).__name__}: {e}"))
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"  ok    {fn.__name__}")
    print("=" * 60)
    if failures:
        print(f"{len(failures)} of {len(tests)} elicitation tests FAILED:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        print("=" * 60)
        sys.exit(1)
    print(f"All {len(tests)} elicitation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
