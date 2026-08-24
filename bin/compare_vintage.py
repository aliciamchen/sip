"""Acceptance gate for a regenerated output vintage: compare the new headline
JSON outputs against the committed ones (git HEAD) and report every numeric
leaf that moved by more than the reported-precision tolerance.

A graph-changing refactor shifts float32 results by ~1 ulp, which compounds
through the optimizer into small parameter drift — expected and fine as long as
nothing the paper prints moves at its quoted precision. The default tolerance
(5e-4) sits below the third decimal, so a clean report means every 2- and
3-decimal number in the manuscript is unchanged; anything larger (e.g. a
multistart hopping to a different optimum) is listed for eyeballing.

Usage: uv run python bin/compare_vintage.py [--tol 5e-4] [--ref HEAD]
"""

import argparse
import json
import subprocess
import sys

from study_registry import SLUGS
from utils import get_project_root

# Headline outputs, repo-relative. Manifests are excluded (their hashes and
# timestamps change by construction); per-trial JSONLs are covered through the
# aggregates in cv_model_comparison.json.
FILES = (
    [f"model/outputs/{s}/fit_results.json" for s in SLUGS]
    + [f"model/outputs/{s}/cv_model_comparison.json" for s in SLUGS]
    + [
        "model/outputs/group_correlations.json",
        "model/outputs/transfer/transfer_summary.json",
        "model/outputs/pooled/pooled_summary.json",
        "model/outputs/generalization_primary.json",
    ]
)

SKIP_KEYS = {"timestamp", "git_sha", "sha256", "path", "n_boot_seed"}


def _leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            yield from _leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare(rel_path, ref, tol):
    root = get_project_root()
    new_path = root / rel_path
    if not new_path.exists():
        return None, f"missing (not regenerated yet): {rel_path}"
    try:
        old_text = subprocess.run(
            ["git", "show", f"{ref}:{rel_path}"],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        ).stdout
    except subprocess.CalledProcessError:
        return None, f"not in {ref} (new file): {rel_path}"
    old = dict(_leaves(json.loads(old_text)))
    new = dict(_leaves(json.loads(new_path.read_text())))
    moved, max_dev = [], 0.0
    for key in sorted(set(old) | set(new)):
        if key not in old or key not in new:
            moved.append((key, old.get(key, "<absent>"), new.get(key, "<absent>")))
            continue
        a, b = old[key], new[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            dev = abs(float(a) - float(b))
            max_dev = max(max_dev, dev)
            if dev > tol:
                moved.append((key, a, b))
        elif a != b:
            moved.append((key, a, b))
    return (moved, max_dev), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--ref", default="HEAD")
    args = ap.parse_args()

    n_clean = n_flagged = 0
    notes = []
    for rel in FILES:
        result, note = compare(rel, args.ref, args.tol)
        if note:
            notes.append(note)
            continue
        moved, max_dev = result
        if moved:
            n_flagged += 1
            print(f"MOVED   {rel} (max numeric dev {max_dev:.2e}):")
            for key, a, b in moved[:20]:
                print(f"    {key}: {a} -> {b}")
            if len(moved) > 20:
                print(f"    ... and {len(moved) - 20} more")
        else:
            n_clean += 1
            print(f"clean   {rel} (max numeric dev {max_dev:.2e})")
    for note in notes:
        print(f"skip    {note}")
    print(
        f"\n{n_clean} clean, {n_flagged} with moves beyond tol={args.tol:g}, "
        f"{len(notes)} skipped"
    )
    # Missing files are a FAILURE, not a pass: the gate must never report a
    # vintage verified when an upstream stage silently produced nothing.
    if n_clean == 0 or any(note.startswith("missing") for note in notes):
        print(
            "GATE FAILED: expected outputs are missing — the regeneration is incomplete."
        )
        sys.exit(2)
    sys.exit(1 if n_flagged else 0)


if __name__ == "__main__":
    main()
