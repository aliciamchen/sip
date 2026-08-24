"""
Consistency checks across the repo's hand-synced registries.

The active experiment roster is declared independently in eight places (the
Makefile, data_prep/json_to_csv.py, study_registry.py, bin/deploy-experiment,
the counterbalancing generator, the entry-file generator, experiments/_lib/
config.js, and the per-study directories), each with a "keep in sync" comment.
This test makes that sync a checked invariant instead of a discipline problem:
a study added, removed, or migrated in one registry but not another (e.g.
moving the Study 3 slugs into EXPERIMENTS_INVERSE without adding them to
study_registry.STUDIES) fails `make test` and CI rather than silently dropping
that study from one pipeline stage — study_registry feeds the model
comparison, the LaTeX export, and every figure script.

Two further hand-synced invariants ride along: the OBSERVED_ACTIONS constant
(declared in plot_style.py, set_diagnostics.py, and score_merged.py, which
cannot share one source without dragging heavy imports across layers), and the
agent docs (the root AGENTS.md and .agents/skills are symlinks into .claude/,
so each guide and skill is one file under two names).

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


def _assignment_targets(node):
    """Names bound by a plain or annotated top-level assignment."""
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def parse_python_dict_keys(path, name):
    """Top-level string keys of a `NAME = {...}` (or annotated `NAME: T = {...}`)
    assignment, via ast (the module is not imported, so its dependencies are
    not needed)."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if name in _assignment_targets(node) and isinstance(node.value, ast.Dict):
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    raise AssertionError(f"{name} dict not found in {path}")


def parse_python_sequence(path, name):
    """Constant elements of a `NAME = [...]` or `NAME = (...)` assignment."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if name in _assignment_targets(node) and isinstance(
            node.value, (ast.List, ast.Tuple)
        ):
            return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    raise AssertionError(f"{name} sequence not found in {path}")


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


# The agent docs are single files under two names: the root `AGENTS.md` symlinks
# to `.claude/CLAUDE.md`, and `.agents/skills` symlinks to `.claude/skills`.
# Both used to be hand-maintained near-copies and both drifted within weeks, so
# the symlink itself is the invariant worth checking — a `cp`, an editor that
# writes through a symlink by replacing it, or a checkout on a filesystem
# without symlinks all bring the drift back.
_AGENT_DOC_SYMLINKS = [
    ("AGENTS.md", ".claude/CLAUDE.md"),
    (".agents/skills", ".claude/skills"),
]


def _agent_doc_symlink_problems():
    """Empty when both agent-doc paths are symlinks onto their targets."""
    problems = []
    for link, target in _AGENT_DOC_SYMLINKS:
        path = ROOT / link
        if not path.is_symlink():
            problems.append(f"{link} is not a symlink (should point at {target})")
        elif path.resolve() != (ROOT / target).resolve():
            problems.append(f"{link} points at {path.readlink()}, not {target}")
    return problems


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
        "study_registry.py STUDIES": set(
            parse_python_dict_keys(ROOT / "study_registry.py", "STUDIES")
        ),
        "experiments/build/sync_entry_files.py ACTIVE_SLUGS": set(
            parse_python_dict_keys(
                ROOT / "experiments" / "build" / "sync_entry_files.py",
                "ACTIVE_SLUGS",
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

    # OBSERVED_ACTIONS is declared in three layers that cannot share one source
    # without heavy cross-layer imports (plot_style must stay JAX-free, the LM
    # modules matplotlib-free); each copy carries a "keep in sync" comment.
    # Order matters (it maps to condition indices), so compare as sequences.
    observed_defs = {
        rel: list(parse_python_sequence(ROOT / rel, "OBSERVED_ACTIONS"))
        for rel in (
            "figures/scripts/plot_style.py",
            "model/lm/set_diagnostics.py",
            "model/lm/score_merged.py",
        )
    }
    reference = observed_defs["model/lm/score_merged.py"]
    check(
        all(v == reference for v in observed_defs.values()),
        "OBSERVED_ACTIONS agrees across plot_style / set_diagnostics / score_merged",
        f"definitions: {observed_defs}",
    )

    # The agent docs are one file under two names, kept that way by symlink.
    symlink_problems = _agent_doc_symlink_problems()
    check(
        not symlink_problems,
        "AGENTS.md and .agents/skills are symlinks into .claude/",
        "; ".join(symlink_problems),
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
