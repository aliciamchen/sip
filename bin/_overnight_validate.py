#!/usr/bin/env python3
"""Validate re-elicited lm_runs*.jsonl files before/after an overnight run.

Used by bin/overnight-reelicit.sh as the smoke-test gate and the post-run check.
It is deliberately loader-independent (raw-record checks only) so it stays robust
to per-study loader kwargs; the checks it makes are exactly the systematic
failures a prompt/schema mismatch or a bad cell config would produce:

  - record count vs the known structural baseline (scaled to K) — catches a wrong
    cell grid or large-scale unit failures;
  - every observed-action (slot 0) feature present and a number in [0, 1] —
    catches null/NaN slot-0 features (the same thing the model loaders reject
    fail-fast), which would poison every fit gradient;
  - no JSON leakage inside action_text — catches grammar-constrained decoding
    jamming intended JSON into free-text fields (the is_share incident).

Exit 0 = all requested studies pass; exit 1 = at least one failed.

Usage:
    _overnight_validate.py --k 1  --studies food_inv_desire ...   # smoke (K=1)
    _overnight_validate.py --k 20 --studies food_inv_desire ...   # post full run
    _overnight_validate.py --k 20                                 # all six + bases
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
LM_DIR = _ROOT / "model" / "outputs" / "lm"

# Structural record counts at K=20 (measured from the known-good tables; these
# depend only on the scenario/condition grid, not on the prompt, so they are the
# same for the diffuse and the tight elicitation). Expected at K=k is baseline*k/20.
BASELINE_K20_MAIN = {
    "food_inv_desire": 7680,
    "food_inv_joint_de": 7680,
    "food_inv_intimacy": 3840,
    "food_inv_joint_ie": 3840,
    "nonfood_inv_joint_de": 7680,
    "nonfood_inv_joint_ie": 3840,
}
BASELINE_K20_BASE = {
    "food_inv_desire": 1920,
    "food_inv_joint_de": 1920,
    "nonfood_inv_joint_de": 1920,
}
ALL_STUDIES = list(BASELINE_K20_MAIN)

# Fraction of expected records that must be present (a few units can validly fail
# all parse retries on any given run and get re-filled by a re-invocation; a
# systematic problem shows up as a gross shortfall, not one or two missing units).
COUNT_TOLERANCE = 0.98

# Substrings that must never appear inside an action_text — the signature of JSON
# structure leaking into a free-text field.
LEAK_MARKERS = ('{"', '"action_', '"risk"', '"effort"', '":', "action_0")


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _check_file(path, expected, k, label):
    """Return (ok, [messages]) for one lm_runs*.jsonl file."""
    msgs = []
    if not path.exists():
        return False, [f"MISSING: {path}"]
    try:
        records = _load_jsonl(path)
    except (json.JSONDecodeError, ValueError) as e:
        return False, [f"UNPARSEABLE ({e}): {path}"]

    n = len(records)
    floor = int(expected * COUNT_TOLERANCE)
    if n < floor:
        msgs.append(
            f"count {n} < {floor} (expected {expected} at K={k}) — "
            "gross shortfall, likely a cell-config or systematic-failure problem"
        )
    elif n > expected:
        msgs.append(
            f"count {n} > expected {expected} at K={k} — unexpected extra units"
        )

    null_slot0 = 0
    leaks = 0
    example_leak = None
    for r in records:
        actions = r.get("actions") or []
        # Observed action = slot 0 / is_observed. Its features must never be null.
        obs = next(
            (a for a in actions if a.get("is_observed") or a.get("slot") == 0), None
        )
        if obs is None:
            null_slot0 += 1
        else:
            for feat in ("risk", "effort", "g"):
                v = obs.get(feat)
                if (
                    v is None
                    or not isinstance(v, (int, float))
                    or not (0.0 <= v <= 1.0)
                ):
                    null_slot0 += 1
                    break
        # JSON-leakage scan across every action's text.
        for a in actions:
            txt = a.get("action_text", "")
            if isinstance(txt, str) and any(m in txt for m in LEAK_MARKERS):
                leaks += 1
                if example_leak is None:
                    example_leak = txt[:120]
                break

    if null_slot0:
        msgs.append(f"{null_slot0} record(s) with null/out-of-range slot-0 features")
    if leaks:
        msgs.append(
            f"{leaks} record(s) with JSON leakage in action_text, e.g. {example_leak!r}"
        )

    ok = not msgs
    head = f"{'PASS' if ok else 'FAIL'}  {label:<24} n={n:<6} (expect {expected})"
    return ok, [head] + [f"       - {m}" for m in msgs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--k", type=int, required=True, help="K_RUNS the files were elicited at."
    )
    ap.add_argument(
        "--studies",
        nargs="*",
        default=ALL_STUDIES,
        help="Studies to validate (default: all six).",
    )
    args = ap.parse_args()

    all_ok = True
    for study in args.studies:
        if study not in BASELINE_K20_MAIN:
            print(f"FAIL  unknown study: {study}")
            all_ok = False
            continue
        exp_main = BASELINE_K20_MAIN[study] * args.k // 20
        ok, lines = _check_file(
            LM_DIR / study / "lm_runs.jsonl", exp_main, args.k, study
        )
        all_ok &= ok
        print("\n".join(lines))
        if study in BASELINE_K20_BASE:
            exp_base = BASELINE_K20_BASE[study] * args.k // 20
            ok_b, lines_b = _check_file(
                LM_DIR / study / "lm_runs_base.jsonl",
                exp_base,
                args.k,
                study + " (base)",
            )
            all_ok &= ok_b
            print("\n".join(lines_b))

    print()
    print("=" * 60)
    print("ALL PASS" if all_ok else "VALIDATION FAILED — do not proceed")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
