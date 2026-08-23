"""
Roster-consistency check across the hand-synced experiment registries.

The active experiment roster is declared independently in six places (the
Makefile, data_prep/json_to_csv.py, bin/deploy-experiment, the counterbalancing
generator, experiments/_lib/config.js, and the per-study directories), each
with a "keep in sync" comment. This test makes that sync a checked invariant
instead of a discipline problem: a study added, removed, or migrated in one
registry but not another (e.g. moving the Study 3 slugs into
EXPERIMENTS_INVERSE without adding them to bin/deploy-experiment) fails
`make test` and CI rather than being silently skipped by one pipeline stage.

Run standalone:  uv run python test_roster_sync.py
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_make_list(text, name):
    """Words assigned to a Makefile `NAME := ...` variable, joining
    backslash-continuation lines."""
    joined = text.replace("\\\n", " ")
    # Horizontal whitespace only around ':=' — a bare '\s*' would let an EMPTY
    # value (e.g. `EXPERIMENTS_NONFOOD :=` after the Study 3 roster move) swallow
    # the following newline and mis-read the next line as the value.
    m = re.search(rf"^{name}[ \t]*:=[ \t]*(.*)$", joined, re.MULTILINE)
    if m is None:
        raise AssertionError(f"Makefile variable {name} not found")
    return [w for w in m.group(1).split() if not w.startswith("$(")]


def parse_python_dict_keys(path, name):
    """Top-level string keys of a `NAME = {...}` assignment, via ast (the
    module is not imported, so its dependencies are not needed)."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                    and isinstance(node.value, ast.Dict)
                ):
                    return [
                        k.value for k in node.value.keys if isinstance(k, ast.Constant)
                    ]
    raise AssertionError(f"{name} dict not found in {path}")


def parse_bash_array(text, name):
    """Elements of a bash `NAME=( ... )` array (one element per word,
    comments stripped)."""
    m = re.search(rf"{name}=\(\s*(.*?)\)", text, re.DOTALL)
    if m is None:
        raise AssertionError(f"bash array {name} not found")
    body = re.sub(r"#.*", "", m.group(1))
    return body.split()


def parse_js_object_keys(text, name):
    """Bare-identifier keys of an `export const NAME = {...}` object."""
    m = re.search(rf"export const {name} = \{{(.*?)\n\}};", text, re.DOTALL)
    if m is None:
        raise AssertionError(f"JS object {name} not found")
    return re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", m.group(1), re.MULTILINE)


def run_all_checks():
    print("=" * 60)
    print("Experiment roster consistency checks")
    print("=" * 60)

    makefile = (ROOT / "Makefile").read_text()
    inverse = parse_make_list(makefile, "EXPERIMENTS_INVERSE")
    nonfood = parse_make_list(makefile, "EXPERIMENTS_NONFOOD")
    base = parse_make_list(makefile, "EXPERIMENTS_BASE")
    roster = set(inverse) | set(nonfood)

    failures = []

    def check(ok, label, detail=""):
        if ok:
            print(f"✓ {label}")
        else:
            failures.append(f"{label}: {detail}")

    check(
        not set(inverse) & set(nonfood),
        "EXPERIMENTS_INVERSE and EXPERIMENTS_NONFOOD are disjoint",
        f"overlap: {sorted(set(inverse) & set(nonfood))}",
    )
    check(
        set(base) <= roster,
        "EXPERIMENTS_BASE is a subset of the roster",
        f"unknown slugs: {sorted(set(base) - roster)}",
    )

    registries = {
        "data_prep/json_to_csv.py EXPERIMENT_CONFIGS": set(
            parse_python_dict_keys(
                ROOT / "data_prep" / "json_to_csv.py", "EXPERIMENT_CONFIGS"
            )
        ),
        "experiments/build/counterbalancing.py STUDY_CONFIGS": set(
            parse_python_dict_keys(
                ROOT / "experiments" / "build" / "counterbalancing.py",
                "STUDY_CONFIGS",
            )
        ),
        "bin/deploy-experiment ACTIVE_EXPERIMENTS": set(
            parse_bash_array(
                (ROOT / "bin" / "deploy-experiment").read_text(),
                "ACTIVE_EXPERIMENTS",
            )
        ),
        "experiments/_lib/config.js DATAPIPE_IDS": set(
            parse_js_object_keys(
                (ROOT / "experiments" / "_lib" / "config.js").read_text(),
                "DATAPIPE_IDS",
            )
        ),
        "experiments/<slug>/ directories": {
            d.name
            for d in (ROOT / "experiments").iterdir()
            if d.is_dir() and (d / "trials.js").exists()
        },
    }
    for label, slugs in registries.items():
        check(
            slugs == roster,
            f"{label} matches the Makefile roster",
            f"missing {sorted(roster - slugs)}, extra {sorted(slugs - roster)}",
        )

    # Completion codes may lag the roster (a study gets its code when it is
    # created on Prolific, and bootstrap refuses to launch without one), but
    # must never name a slug outside the roster.
    completion = set(
        parse_js_object_keys(
            (ROOT / "experiments" / "_lib" / "config.js").read_text(),
            "PROLIFIC_COMPLETION_CODES",
        )
    )
    check(
        completion <= roster,
        "PROLIFIC_COMPLETION_CODES only names roster slugs",
        f"unknown slugs: {sorted(completion - roster)}",
    )

    if failures:
        print("=" * 60)
        for f in failures:
            print(f"FAILED — {f}")
        print("=" * 60)
        sys.exit(1)
    print("=" * 60)
    print("All roster checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_checks()
