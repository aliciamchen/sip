#!/usr/bin/env python3
"""Check that the manuscript is synchronized with reproducible artifacts.

The journal source is a separate, gitignored repository, so these tests skip
cleanly when ``SIP_journal/`` is absent. With ``--build``, the LaTeX check uses
the repository's Dropbox-safe rebuild script on a temporary copy; it never
writes build artifacts back into the working manuscript.

Run: uv run python model/test_manuscript_sync.py [--build]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from study_registry import STUDIES
from utils import get_project_root


ROOT = get_project_root()
JOURNAL = ROOT / "SIP_journal"
MAIN = JOURNAL / "main.tex"
RESULTS = JOURNAL / "results"
MACROS = RESULTS / "results_macros.tex"
MACRO_START = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{")
CONTROL_SEQUENCE = re.compile(r"\\([A-Za-z]+)")
INPUT = re.compile(r"\\input\{([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
PROVENANCE = re.compile(r"^% git .*?  generated .*?$", re.MULTILINE)
PLACEHOLDER = re.compile(r"\b(?:TODO|FIXME|XXX|TBD|TK)\b|\\todo\b", re.IGNORECASE)
RESULT_MACRO = re.compile(
    r"^(?:n(?:Recruited|Retained|Excluded|Female|Male|Nonconforming|Abstain|"
    r"Alts|Scenarios|Runs|Folds)|age(?:Min|Max|Mean|Sd)|ll|dll|ci|stat|"
    r"p(?:Wv|We|Wd|Gamma|AlphaObs|Sigma|Eta)|r(?:Study|Nonfood)|ceilStudy|"
    r"altTemperature|scoreTemperature|lmName)"
)
NUMBER_WORD = {"1": "One", "2": "Two", "3": "Three"}
MANUAL_FIGURES = {
    "main-schematic.pdf",
    "study-1-results.pdf",
    "study-2-results.pdf",
    "study-3-results.pdf",
    "study-3-domains.pdf",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _macro_bodies():
    bodies = {}
    for line in MACROS.read_text().splitlines():
        match = MACRO_START.search(line)
        if not match:
            continue
        depth = 1
        body = []
        for char in line[match.end() :]:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            body.append(char)
        assert depth == 0, f"multiline or malformed macro definition: {line}"
        assert match.group(1) not in bodies, f"duplicate macro {match.group(1)}"
        bodies[match.group(1)] = "".join(body)
    return bodies


def _plain_number(body):
    value = body.replace(r"\ensuremath{", "").replace("{,}", ",")
    value = value.replace("{", "").replace("}", "").replace(",", "")
    return float(value)


def _token(study):
    return NUMBER_WORD[study.number] + study.substudy.upper()


def _normalized_generated(text):
    return PROVENANCE.sub("% git <normalized>  generated <normalized>", text)


def test_results_latex_reexports_byte_for_byte():
    committed = sorted(RESULTS.glob("results*.tex"))
    with tempfile.TemporaryDirectory(prefix="sip-results-audit-") as directory:
        output = Path(directory)
        run = subprocess.run(
            [
                sys.executable,
                str(ROOT / "model" / "export_results_latex.py"),
                "--out-dir",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        regenerated = sorted(output.glob("results*.tex"))
        assert [path.name for path in regenerated] == [path.name for path in committed]
        for path in committed:
            actual = _normalized_generated(path.read_text())
            expected = _normalized_generated((output / path.name).read_text())
            assert actual == expected, f"stale generated manuscript file: {path.name}"


def test_generated_inputs_and_result_macros_have_no_orphans():
    main_text = MAIN.read_text()
    generated = {path.name for path in RESULTS.glob("results*.tex")}
    referenced_inputs = {
        Path(name).name
        for name in INPUT.findall(main_text)
        if Path(name).name.startswith("results")
    }
    assert referenced_inputs == generated, (
        f"generated inputs differ: missing={generated - referenced_inputs}, "
        f"orphaned={referenced_inputs - generated}"
    )

    bodies = _macro_bodies()
    defined = set(bodies)
    table_text = "\n".join(
        path.read_text() for path in RESULTS.glob("results_table_*.tex")
    )
    used = set(CONTROL_SEQUENCE.findall(main_text + "\n" + table_text))
    result_uses = {name for name in used if RESULT_MACRO.match(name)}
    assert result_uses <= defined, (
        f"undefined result macros: {sorted(result_uses - defined)}"
    )
    roots = result_uses
    reachable = set(roots)
    pending = list(roots)
    while pending:
        name = pending.pop()
        dependencies = set(CONTROL_SEQUENCE.findall(bodies[name])) & defined
        for dependency in dependencies - reachable:
            reachable.add(dependency)
            pending.append(dependency)
    assert reachable <= defined


def _journal_figure_names():
    names = set()
    for raw in GRAPHIC.findall(MAIN.read_text()):
        path = Path(raw)
        names.add(path.name if path.suffix else f"{path.name}.pdf")
    return names


def _makefile_figure_pairs():
    lines = (ROOT / "Makefile").read_text().splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("JOURNAL_FIGURES :=")
    )
    tokens = []
    for line in lines[start:]:
        segment = line.split(":=", 1)[1] if ":=" in line else line
        continued = segment.rstrip().endswith("\\")
        tokens.extend(segment.replace("\\", "").split())
        if not continued:
            break
    pairs = {}
    for token in tokens:
        source, destination = token.split(":", 1)
        assert destination not in pairs
        pairs[destination] = source
    return pairs


def test_every_figure_is_present_once_and_synced_from_its_declared_source():
    included = _journal_figure_names()
    present = {path.name for path in (JOURNAL / "figures").glob("*.pdf")}
    assert included == present, (
        f"figure set drift: not included={present - included}, missing={included - present}"
    )
    pairs = _makefile_figure_pairs()
    assert set(pairs) == included - MANUAL_FIGURES
    for destination, source in pairs.items():
        source_path = ROOT / "figures" / "si" / source
        destination_path = JOURNAL / "figures" / destination
        assert source_path.exists(), f"missing source figure: {source_path}"
        assert _sha256(source_path) == _sha256(destination_path), (
            f"journal figure is stale: {destination}"
        )


def _jsonl_values(path, key):
    values = set()
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                values.add(json.loads(line)[key])
    return values


def test_prose_constants_match_code_data_and_manifests():
    from model.inverse._reweighting import STUDY_CONTRASTIVE
    from model.lm.client import MODEL_ID, TEMPERATURE
    from model.lm.generate_alternatives import ALT_GEN_TEMPERATURE

    macros = _macro_bodies()
    n_runs = int(_plain_number(macros["nRuns"]))
    n_scenarios = int(_plain_number(macros["nScenarios"]))
    n_folds = int(_plain_number(macros["nFolds"]))
    assert n_folds == n_scenarios
    assert int(_plain_number(macros["nFoldsTrain"])) == n_folds - 1
    assert math.isclose(_plain_number(macros["altTemperature"]), ALT_GEN_TEMPERATURE)
    assert math.isclose(_plain_number(macros["scoreTemperature"]), TEMPERATURE)
    assert macros["lmName"] == MODEL_ID.rsplit("/", 1)[-1]

    recruited_total = retained_total = 0
    for slug, study in STUDIES.items():
        token = _token(study)
        scenario_csv = (
            ROOT
            / "experiments"
            / ("scenarios.csv" if study.domain == "food" else "scenarios_nonfood.csv")
        )
        assert pd.read_csv(scenario_csv)["scenario_label"].nunique() == n_scenarios

        lm_dir = ROOT / "model" / "outputs" / "lm" / slug
        scored_manifest = json.loads((lm_dir / "lm_runs.manifest.json").read_text())
        alt_manifest = json.loads(
            (lm_dir / "lm_alternatives.manifest.json").read_text()
        )
        assert scored_manifest["k_runs"] == alt_manifest["k_runs"] == n_runs
        assert scored_manifest["n_scenarios"] == n_scenarios
        assert scored_manifest["model"] == alt_manifest["model"] == MODEL_ID
        assert math.isclose(scored_manifest["score_temperature"], TEMPERATURE)
        assert math.isclose(alt_manifest["gen_temperature"], ALT_GEN_TEMPERATURE)
        assert _jsonl_values(lm_dir / "lm_runs.jsonl", "run_id") == set(range(n_runs))
        assert _jsonl_values(lm_dir / "lm_alternatives.jsonl", "run_id") == set(
            range(n_runs)
        )

        output = ROOT / "model" / "outputs" / slug
        folds = [
            json.loads(line)
            for line in (output / "cv_folds.jsonl").read_text().splitlines()
        ]
        assert {row["fold"] for row in folds} == set(range(n_folds))
        comparison = json.loads((output / "cv_model_comparison.json").read_text())
        assert (
            comparison["n_trials_per_model"] / comparison["n_subjects"] == n_scenarios
        )

        recruited = int(_plain_number(macros[f"nRecruited{token}"]))
        retained = int(_plain_number(macros[f"nRetained{token}"]))
        excluded = int(_plain_number(macros[f"nExcluded{token}"]))
        genders = sum(
            int(_plain_number(macros[f"n{label}{token}"]))
            for label in ("Female", "Male", "Nonconforming", "Abstain")
        )
        assert recruited == retained + excluded == genders
        recruited_total += recruited
        retained_total += retained
    assert recruited_total == int(_plain_number(macros["nRecruitedTotal"]))
    assert retained_total == int(_plain_number(macros["nRetainedTotal"]))

    labels = {
        "none": (),
        "physical state": ("world",),
        "intimacy": ("intimacy",),
        "both": ("intimacy", "world"),
    }
    table_rows = re.findall(
        r"^(\d[a-z])\s*&\s*[^&]+&\s*([^&]+?)\s*&", MAIN.read_text(), re.MULTILINE
    )
    manuscript_scope = {number: labels[target.strip()] for number, target in table_rows}
    expected_scope = {
        study.short_label: targets
        for slug, targets in STUDY_CONTRASTIVE.items()
        for study in [STUDIES[slug]]
    }
    assert manuscript_scope == expected_scope


def test_manuscript_contains_no_submission_placeholders():
    failures = []
    for number, line in enumerate(MAIN.read_text().splitlines(), 1):
        if PLACEHOLDER.search(line):
            failures.append(f"line {number}: {line.strip()}")
    assert not failures, "submission placeholders remain:\n" + "\n".join(failures)


def test_clean_latex_build_has_no_broken_references_or_floats():
    assert shutil.which("latexmk"), "latexmk is required for --build"
    script = (
        ROOT / ".agents" / "skills" / "fix-latex-build" / "rebuild-outside-dropbox.sh"
    )
    with tempfile.TemporaryDirectory(prefix="sip-manuscript-build-") as directory:
        manuscript_copy = Path(directory) / "SIP_journal"
        shutil.copytree(JOURNAL, manuscript_copy, ignore=shutil.ignore_patterns(".git"))
        run = subprocess.run(
            ["bash", str(script), str(manuscript_copy), "main.tex"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        log = (manuscript_copy / "main.log").read_text(errors="replace")
        problems = []
        forbidden = (
            r"^.*Citation.*undefined.*$",
            r"^.*LaTeX Warning: Reference.*undefined.*$",
            r"^.*File `[^']+' not found.*$",
            r"^.*Float too large.*$",
        )
        for pattern in forbidden:
            for match in re.finditer(pattern, log, re.MULTILINE):
                line = log.count("\n", 0, match.start()) + 1
                problems.append(f"main.log:{line}: {match.group(0)}")
        pdf = (manuscript_copy / "main.pdf").read_bytes()
        if not (pdf.startswith(b"%PDF-") and b"%%EOF" in pdf[-3000:]):
            problems.append("main.pdf is missing a valid header or EOF marker")

    keys = re.findall(
        r"^@[A-Za-z]+\{([^,]+)", (JOURNAL / "references.bib").read_text(), re.MULTILINE
    )
    folded = [key.casefold() for key in keys]
    duplicates = sorted({key for key in folded if folded.count(key) > 1})
    if duplicates:
        problems.append(f"duplicate bibliography keys: {duplicates}")
    assert not problems, "clean LaTeX build problems:\n" + "\n".join(problems)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build", action="store_true", help="also run a clean LaTeX build"
    )
    args = parser.parse_args()
    if not MAIN.exists():
        print("SKIP: SIP_journal/ is absent; manuscript-sync checks are local-only.")
        return

    tests = [
        test_results_latex_reexports_byte_for_byte,
        test_generated_inputs_and_result_macros_have_no_orphans,
        test_every_figure_is_present_once_and_synced_from_its_declared_source,
        test_prose_constants_match_code_data_and_manifests,
        test_manuscript_contains_no_submission_placeholders,
    ]
    if args.build:
        tests.append(test_clean_latex_build_has_no_broken_references_or_floats)
    failures = []
    print("=" * 72)
    print("Manuscript synchronization tests")
    print("=" * 72)
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except BaseException as error:
            failures.append(test.__name__)
            print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
    print("=" * 72)
    if failures:
        print(f"{len(failures)} of {len(tests)} manuscript tests failed")
        sys.exit(1)
    print(f"All {len(tests)} manuscript synchronization tests passed!")


if __name__ == "__main__":
    main()
