"""Tests for the alternatives prompt and the elicitation vintage guard.

The LM tables are expensive to regenerate (a full six-study re-elicitation is
tens of dollars and hours of wall clock) and the failure mode they guard against
is silent: a resumed run skips already-done units, so resuming after a prompt
edit splices records from two wordings into one file, which then feeds fits and
CV as if it were a single vintage. Nothing downstream can detect that. So the
guard is tested here rather than trusted.

What's covered:

  - There is exactly ONE alternatives system prompt. Provenance rests on
    a stage-specific prompt hash, so an unrelated rating-prompt edit does not
    make an alternatives artifact look stale.
  - That prompt requests an explanation before answering (adopted 2026-07-28
    for coverage), while the rating stages retain intuition-only instructions.
  - Every generation captures its generated rationale, at the larger token
    budget.
  - `guard_resume_prompt_mismatch` refuses a resume across a prompts.py edit.
  - The elicitation is reachable for all six studies, and the diagnostic
    `--arm-output-only` path routes both main and base variants to their own
    vintages.

Run: uv run python model/lm/test_elicitation_guards.py
"""

import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts  # noqa: E402
import client as client_module  # noqa: E402
from client import (  # noqa: E402
    RESUME_PROMPT_MISMATCH_ENV,
    _prompt_sha,
    _prompts_sha,
    guard_resume_prompt_mismatch,
    manifest_prompt_matches,
    write_run_manifest,
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
    """A run-time-selectable variant would need its own manifest field.

    The stage hash covers the stage's prompt surfaces, but it cannot identify
    which one was selected at run time. Keeping one live prompt avoids that
    ambiguity.
    """
    variants = [
        n
        for n in dir(prompts)
        if n.startswith("ALTERNATIVES") and n.endswith("SYSTEM_PROMPT")
    ]
    assert variants == ["ALTERNATIVES_SYSTEM_PROMPT"], (
        f"unexpected alternatives prompt variants: {variants}"
    )
    assert isinstance(prompts.ALTERNATIVES_SYSTEM_PROMPT, str)


def test_generate_alternatives_has_no_prompt_mode_flag():
    """The elicitation must not offer a choice of prompt: one prompt means the
    CLI has no mode to select, and `--cot` in particular used to be coupled to
    `--arm-output-only`, which made the adopted wording unreachable from the
    production path."""
    src = (Path(__file__).resolve().parent / "generate_alternatives.py").read_text()
    for gone in ("--cot", "cot=", "COT_MAX_TOKENS", "_cot_system_prompt"):
        assert gone not in src, f"{gone!r} still present in generate_alternatives.py"


# --------------------------------------- the prompt requests an explanation first


def test_alternatives_prompt_requests_explanation_before_answering():
    """The request lives in the closing instruction only. It used to be repeated
    in the preamble, an artifact of the chain-of-thought arm having been built by
    two separate string substitutions; the preamble copy was dropped as
    redundant, so assert the close carries it on its own."""
    p = prompts.ALTERNATIVES_SYSTEM_PROMPT
    assert "First, briefly explain step by step" in p
    assert "with no other text after the array:" in p
    assert "step by step" not in prompts._PREAMBLE_ALTERNATIVES
    # The old respond-only-with-JSON close must be gone, or the model returns a
    # bare array and the generated-rationale sidecar is empty.
    assert "no explanation:" not in p


def test_rating_prompts_stay_intuition_only():
    """A rating is a snap judgment, mirroring the participant, so its preamble
    keeps the intuition wording and does not request a step-by-step
    explanation."""
    assert "just going off your intuition" in prompts._PREAMBLE_RATING
    assert "step by step" not in prompts._PREAMBLE_RATING
    rating_prompts = [
        prompts.DESIRE_SYSTEM_PROMPT,
        prompts.INTIMACY_SYSTEM_PROMPT,
    ] + [prompts.system_prompt(t) for t in ("g", "effort", "risk")]
    for p in rating_prompts:
        assert "step by step" not in p, p[:80]
        assert "just going off your intuition" in p, p[:80]


def test_effort_is_total_executional_cost_for_the_dyad():
    """The scorer must count required work regardless of which person does it."""
    p = prompts.system_prompt("effort")
    assert "total" in p.lower()
    assert "either person" in p.lower()
    assert "social awkwardness" in p


def _condition_order(rendered):
    """The sequence of given-condition blocks in a rendered user prompt."""
    order = []
    for line in rendered.split("\n"):
        if line.startswith(prompts._RELATIONSHIP_PREFIX):
            order.append("intimacy")
        elif line.startswith("Scenario:"):
            order.append("vignette")
        elif line == "<DESIRE PARA>":
            order.append("desire")
        elif line == "<EFFORT PARA>":
            order.append("effort")
    return order


def test_condition_paragraph_order_matches_the_experiment_screens():
    """The LM must read the given conditions in the order participants see them.

    The human screens put the relationship sentence BEFORE the vignette (in
    1a/1b/3a it is also shown on its own page first) and the desire / effort
    paragraphs after it. The prompt used to append the relationship last, so the
    LM saw the trial in a different order than any participant did. Expected
    orders are transcribed from `experiments/*/trials.js`; the companion test
    below checks those files still say so.
    """
    cases = {
        "1a": (
            dict(
                effort_text="<EFFORT PARA>",
                intimacy_level="max_intimate",
                unknown_desire_object="the cake",
            ),
            ["intimacy", "vignette", "effort"],
        ),
        "1b/3a": (
            dict(
                intimacy_level="max_formal",
                effort_hypotheses=("<LOW>", "<HIGH>"),
                unknown_desire_object="the cake",
            ),
            ["intimacy", "vignette"],
        ),
        "2a": (
            dict(
                desire_text="<DESIRE PARA>",
                effort_text="<EFFORT PARA>",
                unknown_intimacy=True,
            ),
            ["vignette", "desire", "effort"],
        ),
        "2b/3b": (
            dict(
                desire_text="<DESIRE PARA>",
                effort_hypotheses=("<LOW>", "<HIGH>"),
                unknown_intimacy=True,
            ),
            ["vignette", "desire"],
        ),
    }
    for study, (kwargs, expected) in cases.items():
        got = _condition_order(
            prompts.alternatives_user_prompt("<VIGNETTE>", "<OBSERVED>", **kwargs)
        )
        assert got == expected, f"{study}: prompt order {got} != experiment {expected}"


def test_experiment_screens_still_show_the_order_the_prompts_assume():
    """Read the order back out of the jsPsych trial builders.

    The test above hardcodes the expected orders; this one keeps that transcription
    honest, so a change to a trial screen surfaces here instead of silently making
    the prompts wrong. Each study passes a `paragraphs` list to
    `scenarioStimulus`; the relationship descriptor must precede the vignette
    wherever it appears.
    """
    root = Path(__file__).resolve().parents[2]
    expect_intimacy_first = {
        "food_inv_desire",
        "food_inv_joint_de",
        "nonfood_inv_joint_de",
    }
    for slug in expect_intimacy_first:
        src = (root / "experiments" / slug / "trials.js").read_text()
        block = src[src.index("paragraphs:") :][:400]
        i_int, i_vig = block.find("intimacyDescriptor"), block.find("stimulus.vignette")
        assert i_int != -1, f"{slug}: no intimacyDescriptor in the paragraphs block"
        assert i_vig != -1, f"{slug}: no vignette in the paragraphs block"
        assert i_int < i_vig, (
            f"{slug}: the experiment now shows the vignette before the "
            "relationship — the prompts assume the opposite"
        )
    for slug in ("food_inv_intimacy", "food_inv_joint_ie", "nonfood_inv_joint_ie"):
        src = (root / "experiments" / slug / "trials.js").read_text()
        block = src[src.index("paragraphs:") :][:400]
        assert "intimacyDescriptor" not in block, (
            f"{slug} infers intimacy, so its screens must not reveal it"
        )


def test_alternatives_prompt_keeps_the_methodological_framing():
    """The substance the paper's argument rests on, which the explain-then-JSON
    close must not have displaced."""
    p = prompts.ALTERNATIVES_SYSTEM_PROMPT
    for required in (
        "some resource",
        "a piece of information",  # keeps the nonfood/disclosure mode available
        "The vignette omits some information",  # the unconditional-phrasing clause
        "an observer would compare with the action they actually took",
        '{"action":',
    ):
        assert required in p, required


def test_prompt_appendix_documents_generation_without_repeating_it():
    """The appendix reproduces the generation user prompt ONCE, as an example,
    and tabulates what differs per experiment.

    Six near-identical boxes (one per live branch) were repetitive without adding
    information. What must survive the collapse is the information itself: every
    live clause has to appear in the rendered example, and every experiment has
    to be accounted for in the table -- otherwise a reader cannot tell what any
    given study's observer actually saw.
    """
    import export_prompts_latex as exporter

    rendered = exporter.build_content()
    assert "User prompt (Example)" in rendered
    # Exactly one generation user-prompt box, not one per study.
    assert rendered.count("--- counterfactual action generation") == 2, (
        "expected one system-prompt box and one example user-prompt box"
    )
    # Which paragraphs each experiment reveals is the experiment design, stated
    # in the main text, so the appendix must POINT at it rather than restate it.
    assert "follows each experiment's design" in rendered
    assert "1b, 3a &" not in rendered, "the per-experiment table should be gone"
    assert "base-model" in rendered, "the base variant must still be described"
    # Every live clause still visible in the example.
    for live_clause in (
        "do not know how much the two people would like",
        "One of the following is true of the situation",
        "not whether it is available",
        "comparison set you would use to interpret their choice",
    ):
        assert live_clause in rendered, live_clause
    # The intimacy-inferred branch has no revealed relationship, so its clause
    # is named in the table rather than shown; make sure it is not simply lost.
    assert "relationship" in rendered


# --------------------------------------------- prompt-stage provenance


def test_generation_hash_ignores_an_unrelated_rating_prompt_edit():
    """Changing a scoring prompt must not invalidate generation artifacts."""
    before = _prompt_sha("generate_alternatives")
    with _stub(prompts, "DESIRE_SYSTEM_PROMPT", prompts.DESIRE_SYSTEM_PROMPT + " edit"):
        assert _prompt_sha("generate_alternatives") == before


def test_generation_hash_changes_with_the_generation_prompt():
    """Changing text sent during generation must invalidate its artifacts."""
    before = _prompt_sha("generate_alternatives")
    with _stub(
        prompts,
        "ALTERNATIVES_SYSTEM_PROMPT",
        prompts.ALTERNATIVES_SYSTEM_PROMPT + " edit",
    ):
        assert _prompt_sha("generate_alternatives") != before


def test_generation_resume_refuses_a_caller_routing_change():
    """The exact rendered user messages must guard resume even when
    prompts.py itself is unchanged."""
    import pandas as pd
    import generate_alternatives as ga

    base_cell = {
        "scenario_label": "hot-dog",
        "observed_action": "no_share",
        "effort_condition": "low",
        "intimacy_condition": "max_formal",
    }
    with tempfile.TemporaryDirectory() as d:
        output = Path(d) / "lm_alternatives.jsonl"
        cells = [{**base_cell, "user_prompt": "Rendered prompt A"}]

        def _run():
            with _stub(ga, "_output_path_for", lambda study, base: output):
                with _stub(
                    ga,
                    "load_scenarios",
                    lambda study: pd.DataFrame([{"scenario_label": "hot-dog"}]),
                ):
                    with _stub(ga, "_build_cells", lambda df, cfg, study: cells):
                        with _stub(ga, "load_api_key", lambda: "sk-test"):
                            with _stub(ga, "Together", lambda **kwargs: object()):
                                with _stub(ga, "N_RUNS_ALT", 1):
                                    with _stub(
                                        ga,
                                        "elicit_alternatives",
                                        lambda *args, **kwargs: (
                                            [{"action": "Use separate forks."}],
                                            "Generated rationale followed by JSON.",
                                        ),
                                    ):
                                        ga.main("food_inv_desire")

        _run()
        manifest_path = output.with_name("lm_alternatives.manifest.json")
        interrupted = json.loads(manifest_path.read_text())
        interrupted["status"] = "in_progress"
        manifest_path.write_text(json.dumps(interrupted))
        _run()
        assert json.loads(manifest_path.read_text())["status"] == "complete"

        cells[0] = {**base_cell, "user_prompt": "Rendered prompt B"}
        with _expect(SystemExit, "rendered generation prompts"):
            _run()


def test_score_hash_changes_with_the_effort_prompt():
    """An effort-rubric edit must invalidate feature-scoring artifacts."""
    before = _prompt_sha("score_merged")
    edited_bodies = {**prompts._BODIES, "effort": prompts._BODIES["effort"] + " edit"}
    with _stub(prompts, "_BODIES", edited_bodies):
        assert _prompt_sha("score_merged") != before


def test_score_hash_changes_with_the_upstream_generation_prompt():
    """Scored tables must be invalidated when their alternative sets are stale."""
    before = _prompt_sha("score_merged")
    with _stub(
        prompts,
        "ALTERNATIVES_SYSTEM_PROMPT",
        prompts.ALTERNATIVES_SYSTEM_PROMPT + " edit",
    ):
        assert _prompt_sha("score_merged") != before


def test_score_rendered_fingerprint_uses_the_production_message_builder():
    """Changing a caller-owned formatter must change the exact scoring-message
    fingerprint, even when prompts.py is untouched."""
    import pandas as pd
    import score_merged as sm

    scenarios = pd.DataFrame(
        [
            {
                "scenario_label": "hot-dog",
                "vignette": "Two people have one hot dog.",
                "no_share": "They do not share it.",
                "low_risk_share": "They cut it with a clean knife.",
                "high_risk_share": "They bite from opposite ends.",
                "low_risk_share_effort_low": "A clean knife is nearby.",
                "low_risk_share_effort_high": "A clean knife is far away.",
                "desire_object": "the hot dog",
                "desire_low": "They just ate.",
                "desire_high": "They are hungry.",
            }
        ]
    ).set_index("scenario_label", drop=False)
    alternatives = pd.DataFrame(
        [
            {
                "scenario_label": "hot-dog",
                "observed_action": "no_share",
                "effort_condition": "low",
                "intimacy_condition": "max_formal",
                "run_id": 0,
                "alt_idx": 0,
                "action_text": "Use separate forks.",
            }
        ]
    )
    cfg = sm._STUDY_CONFIG["food_inv_desire"]
    before = sm._scoring_prompt_sha(scenarios, alternatives, cfg)
    real = sm.format_risk_prompt_variable
    with _stub(
        sm,
        "format_risk_prompt_variable",
        lambda vignette, actions: real(vignette, actions) + "\nCaller edit",
    ):
        after = sm._scoring_prompt_sha(scenarios, alternatives, cfg)
    assert after != before


def test_scoring_completion_requires_every_unit_and_nonnull_feature():
    """A checkpoint manifest is complete only when the full scenario/run grid
    has valid records."""
    import score_merged as sm

    valid = {
        "scenario_label": "hot-dog",
        "run_id": 0,
        "actions": [{"risk": 0.1, "effort": 0.2, "g": 0.3}],
    }
    assert sm._scoring_grid_completed([valid], ["hot-dog"], [0])
    assert not sm._scoring_grid_completed([], ["hot-dog"], [0])
    assert not sm._scoring_grid_completed(
        [{**valid, "actions": [{"risk": 0.1, "effort": None, "g": 0.3}]}],
        ["hot-dog"],
        [0],
    )


def test_resume_uses_the_stage_hash_when_available():
    """A new-style manifest stays resumable across unrelated prompt edits."""
    with _manifest(prompt_sha256=_prompt_sha("generate_alternatives")) as out:
        with _stub(
            prompts, "DESIRE_SYSTEM_PROMPT", prompts.DESIRE_SYSTEM_PROMPT + " edit"
        ):
            guard_resume_prompt_mismatch(out)


# --------------------------------------------- rationale is always captured


def test_every_generation_captures_a_rationale_at_the_larger_budget():
    """Since the prompt requests an explanation, the elicitation must request
    the raw text and write the generated-rationale sidecar. A budget left at the
    rating stages' default would truncate the array away."""
    import generate_alternatives as ga

    assert ga.ALT_MAX_TOKENS > 800
    assert ga._rationale_path(Path("lm_alternatives.jsonl")).name == (
        "lm_alternatives.rationale.jsonl"
    )
    record = ga._rationale_record(
        {
            "scenario_label": "hot-dog",
            "observed_action": "low_risk_share",
            "intimacy_condition": "formal",
        },
        ["intimacy_condition"],
        run=3,
        raw_text="Generated explanation followed by JSON.",
    )
    assert record["rationale"] == "Generated explanation followed by JSON."
    assert "reasoning" not in record


def test_legacy_reasoning_sidecar_is_loaded_as_rationale():
    """A same-vintage resume must not discard audit records written before the
    sidecar was relabeled."""
    import generate_alternatives as ga

    with tempfile.TemporaryDirectory() as d:
        output = Path(d) / "lm_alternatives.jsonl"
        legacy = output.with_name("lm_alternatives.reasoning.jsonl")
        legacy.write_text(
            json.dumps(
                {
                    "scenario_label": "hot-dog",
                    "observed_action": "low_risk_share",
                    "intimacy_condition": "formal",
                    "run_id": 3,
                    "reasoning": "Generated explanation followed by JSON.",
                }
            )
            + "\n"
        )
        rows = ga._load_rationale_rows(output)

    assert rows == [
        {
            "scenario_label": "hot-dog",
            "observed_action": "low_risk_share",
            "intimacy_condition": "formal",
            "run_id": 3,
            "rationale": "Generated explanation followed by JSON.",
        }
    ]


def test_missing_rationale_requeues_the_checkpointed_generation_unit():
    """If the main JSONL landed but the rationale sidecar did not, the next
    resume must not permanently skip that unit."""
    import generate_alternatives as ga

    rows = [
        {
            "scenario_label": "hot-dog",
            "observed_action": "low_risk_share",
            "intimacy_condition": "formal",
            "run_id": 3,
            "alt_idx": 0,
            "action_text": "Use separate forks.",
        },
        {
            "scenario_label": "soup",
            "observed_action": "no_share",
            "intimacy_condition": "intimate",
            "run_id": 1,
            "alt_idx": 0,
            "action_text": "Share a clean spoon.",
        },
    ]
    rationales = [
        {
            "scenario_label": "soup",
            "observed_action": "no_share",
            "intimacy_condition": "intimate",
            "run_id": 1,
            "rationale": "Generated explanation followed by JSON.",
        }
    ]

    kept, requeued = ga._requeue_units_missing_rationales(
        rows, rationales, ["intimacy_condition"]
    )

    assert kept == [rows[1]]
    assert requeued == {
        ("hot-dog", "low_risk_share", "formal", 3),
    }


# ------------------------------------------------------------ resume guard


def test_resume_refused_after_a_prompt_edit():
    """The case that matters right now: the six existing tables were elicited
    under an older prompts.py, so re-elicitation must not append onto them."""
    with _manifest(prompts_sha256="deadbeef0000") as out:
        with _expect(SystemExit, "would silently mix data from two prompt versions"):
            guard_resume_prompt_mismatch(out)


def test_resume_allowed_when_the_prompt_is_unchanged():
    with _manifest(prompt_sha256=_prompt_sha("generate_alternatives")) as out:
        guard_resume_prompt_mismatch(out)  # must not raise


def test_legacy_resume_allowed_when_the_whole_source_hash_is_unchanged():
    """Legacy manifests use the whole prompts.py hash as their positive path."""
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


def test_legacy_manifest_history_migrates_to_the_source_hash_history():
    """Old ``prompt_sha_history`` entries were whole-file hashes; a new
    manifest must not relabel them as stage-prompt hashes."""
    with tempfile.TemporaryDirectory() as d:
        output = Path(d) / "lm_alternatives.jsonl"
        manifest_path = output.with_name("lm_alternatives.manifest.json")
        manifest_path.write_text(
            json.dumps(
                {
                    "stage": "generate_alternatives",
                    "study": "food_inv_desire",
                    "prompts_sha256": "legacy-current",
                    "prompt_sha_history": ["legacy-older"],
                }
            )
        )
        with _stub(client_module, "_prompt_sha", lambda stage: "stage-current"):
            with _stub(client_module, "_prompts_sha", lambda: "source-current"):
                write_run_manifest(output, "generate_alternatives", "food_inv_desire")
        migrated = json.loads(manifest_path.read_text())

    assert migrated.get("prompt_sha_history") is None
    assert migrated["prompts_sha_history"] == [
        "legacy-older",
        "legacy-current",
    ]


def test_score_refuses_alternatives_from_a_stale_generation_prompt():
    """Scoring must stop before client initialization when the alternatives
    manifest identifies a different generation prompt."""
    import score_merged as sm

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "experiments").mkdir()
        (root / "model" / "outputs" / "lm" / "food_inv_desire").mkdir(parents=True)
        (root / "experiments" / "scenarios.csv").write_text("scenario_label\nhot-dog\n")
        alts = (
            root
            / "model"
            / "outputs"
            / "lm"
            / "food_inv_desire"
            / "lm_alternatives.jsonl"
        )
        alts.write_text(
            json.dumps(
                {
                    "scenario_label": "hot-dog",
                    "observed_action": "no_share",
                    "run_id": 0,
                    "alt_idx": 0,
                    "action_text": "Use separate forks.",
                }
            )
            + "\n"
        )
        alts.with_name("lm_alternatives.manifest.json").write_text(
            json.dumps(
                {
                    "stage": "generate_alternatives",
                    "study": "food_inv_desire",
                    "prompt_sha256": "stale-stage",
                    "prompts_sha256": _prompts_sha(),
                }
            )
        )
        with _stub(sm, "get_project_root", lambda: root):
            with _stub(sm, "load_api_key", lambda: "sk-test"):
                with _stub(
                    sm,
                    "Together",
                    lambda **kwargs: (_ for _ in ()).throw(
                        AssertionError("client initialized before validation")
                    ),
                ):
                    with _expect(SystemExit, "alternatives prompt"):
                        sm.main("food_inv_desire")


def test_score_refuses_an_in_progress_alternatives_artifact():
    """A checkpoint manifest must not authorize scoring before generation has
    accounted for the complete cell-by-run grid."""
    import score_merged as sm

    with tempfile.TemporaryDirectory() as d:
        alts = Path(d) / "lm_alternatives.jsonl"
        alts.write_text('{"run_id":0}\n')
        alts.with_name("lm_alternatives.manifest.json").write_text(
            json.dumps(
                {
                    "stage": "generate_alternatives",
                    "study": "food_inv_desire",
                    "status": "in_progress",
                    "prompt_sha256": _prompt_sha("generate_alternatives"),
                    "prompts_sha256": _prompts_sha(),
                }
            )
        )
        with _expect(SystemExit, "not complete"):
            sm._alternatives_provenance(alts, "food_inv_desire")


def test_score_resume_refuses_a_replaced_alternatives_file():
    """A partial scoring file must remain bound to the exact alternatives
    content it began with."""
    import score_merged as sm

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "experiments").mkdir()
        study_dir = root / "model" / "outputs" / "lm" / "food_inv_desire"
        study_dir.mkdir(parents=True)
        (root / "experiments" / "scenarios.csv").write_text("scenario_label\nhot-dog\n")
        alts = study_dir / "lm_alternatives.jsonl"
        alts.write_text(
            json.dumps(
                {
                    "scenario_label": "hot-dog",
                    "observed_action": "no_share",
                    "run_id": 0,
                    "alt_idx": 0,
                    "action_text": "Use separate forks.",
                }
            )
            + "\n"
        )
        alts.with_name("lm_alternatives.manifest.json").write_text(
            json.dumps(
                {
                    "stage": "generate_alternatives",
                    "study": "food_inv_desire",
                    "prompt_sha256": _prompt_sha("generate_alternatives"),
                    "prompts_sha256": _prompts_sha(),
                }
            )
        )
        old_provenance = sm._alternatives_provenance(alts, "food_inv_desire")
        runs = study_dir / "lm_runs.jsonl"
        runs.write_text("{}\n")
        runs.with_name("lm_runs.manifest.json").write_text(
            json.dumps(
                {
                    "stage": "score_merged",
                    "study": "food_inv_desire",
                    "prompt_sha256": _prompt_sha("score_merged"),
                    "prompts_sha256": _prompts_sha(),
                    **old_provenance,
                }
            )
        )
        alts.write_text(
            json.dumps(
                {
                    "scenario_label": "hot-dog",
                    "observed_action": "no_share",
                    "run_id": 0,
                    "alt_idx": 0,
                    "action_text": "Use two plates instead.",
                }
            )
            + "\n"
        )
        with _stub(sm, "get_project_root", lambda: root):
            with _stub(sm, "load_api_key", lambda: "sk-test"):
                with _stub(
                    sm,
                    "Together",
                    lambda **kwargs: (_ for _ in ()).throw(
                        AssertionError("client initialized before validation")
                    ),
                ):
                    with _expect(SystemExit, "alternatives content"):
                        sm.main("food_inv_desire")


def test_the_reported_tables_match_the_current_prompt():
    """The tables the fits read must have been produced by the prompt in the tree.

    This replaces an earlier check that asserted the opposite — that every
    manifest was stale, so a planned re-elicitation would be forced to start
    clean. Once that re-elicitation ran (2026-07-31, K=20 across all six studies
    plus the three base overlays), the useful invariant inverted: a mismatch here
    now means the reported tables and the prompt source have drifted apart, which
    is exactly what must never ship. Scope is the reported path —
    `lm_alternatives*` and `lm_runs*`.
    """
    root = Path(__file__).resolve().parents[2]
    lm = root / "model" / "outputs" / "lm"
    live = [
        m
        for pat in ("*/lm_alternatives*.manifest.json", "*/lm_runs*.manifest.json")
        for m in lm.glob(pat)
        if not any(x in m.name for x in ("diag", "k20v5", "preclause"))
    ]
    if not live:
        return  # nothing elicited yet
    stale = [
        f"{m.parent.name}/{m.name}"
        for m in live
        if not manifest_prompt_matches(json.loads(m.read_text()))
    ]
    assert not stale, (
        f"{len(stale)} of {len(live)} reported tables were produced by a "
        f"different prompt than the one in the tree: {sorted(stale)}. Either "
        "re-elicit the affected studies or restore the prompt they were made "
        "with — do not fit across the two."
    )


def test_reported_tables_carry_the_stage_specific_hash():
    """New elicitations must record `prompt_sha256`, not just the whole-file
    `prompts_sha256`. The stage hash is what lets an unrelated rating-prompt edit
    stop invalidating an alternatives artifact; a table that only has the legacy
    field predates that and cannot be checked precisely."""
    root = Path(__file__).resolve().parents[2]
    lm = root / "model" / "outputs" / "lm"
    legacy = []
    for pat in ("*/lm_alternatives*.manifest.json", "*/lm_runs*.manifest.json"):
        for m in lm.glob(pat):
            if any(x in m.name for x in ("diag", "k20v5", "preclause")):
                continue
            if json.loads(m.read_text()).get("prompt_sha256") is None:
                legacy.append(f"{m.parent.name}/{m.name}")
    assert not legacy, (
        f"{len(legacy)} reported tables carry only the legacy whole-file hash: "
        f"{sorted(legacy)}"
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


def test_arm_output_only_routes_the_base_overlay_to_a_diagnostic_vintage():
    """Base smoke output must be isolated just like the six main variants."""
    import generate_alternatives as ga
    import pandas as pd

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        canonical = d / "lm_alternatives_base.jsonl"
        with _stub(ga, "load_api_key", lambda: "sk-test"):
            with _stub(ga, "load_scenarios", lambda study: pd.DataFrame()):
                with _stub(ga, "_build_cells", lambda scenarios, cfg, study: []):
                    with _stub(
                        ga,
                        "_output_path_for",
                        lambda study, base: canonical,
                    ):
                        ga.main(
                            "food_inv_desire",
                            base=True,
                            arm_output_only=True,
                        )

        assert not canonical.exists()
        assert (d / "lm_alternatives_base_diag.jsonl").exists()
        assert (d / "lm_alternatives_base_diag.empty_units.jsonl").exists()
        assert (d / "lm_alternatives_base_diag.rationale.jsonl").exists()


def test_score_arm_routes_the_base_overlay_to_a_diagnostic_vintage():
    """Base scoring must read and write only the base diagnostic filenames."""
    import score_merged as sm

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        study_dir = root / "model" / "outputs" / "lm" / "food_inv_desire"
        study_dir.mkdir(parents=True)
        expected = study_dir / "lm_alternatives_base_diag.jsonl"
        expected.write_text("{}\n")
        captured = []

        def _capture(path, study):
            captured.append((path, study))
            raise _Reached

        with _stub(sm, "get_project_root", lambda: root):
            with _stub(sm, "_alternatives_provenance", _capture):
                with _expect(_Reached):
                    sm.main("food_inv_desire", base=True, arm=True)

        assert captured == [(expected, "food_inv_desire")]


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
