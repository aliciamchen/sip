#!/usr/bin/env python3
"""Three-arm K=1 smoke report for the alternatives prompt change (spec:
notes/2026-07-18-informative-priors-refusal-alts-design.md, smoke gate).

Prints per-arm: refusal (low-g) coverage on share-observed cells, alternative
set sizes, empty-generation units, and effort-paragraph contamination. Run
after `generate_alternatives.py --arm ... ` + `score_merged.py --alts-suffix ...`
at K_RUNS=1 for each arm.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tables import (
    load_padded_lm_tables_desire,
    load_padded_lm_tables_intimacy,
    load_padded_lm_tables_joint_de,
    load_padded_lm_tables_joint_ie,
)

# The desire/intimacy loaders accept `runs_filename=` directly; the joint
# loaders additionally take `slug=` (Study 3a/3b share these designs), so the
# lambdas hardcode the food slug and pass `runs_filename=` through.
_LOADERS = {
    "food_inv_desire": load_padded_lm_tables_desire,
    "food_inv_joint_de": lambda **kw: load_padded_lm_tables_joint_de(
        slug="food_inv_joint_de", **kw
    ),
    "food_inv_intimacy": load_padded_lm_tables_intimacy,
    "food_inv_joint_ie": lambda **kw: load_padded_lm_tables_joint_ie(
        slug="food_inv_joint_ie", **kw
    ),
}


def _ngrams(text, n=4):
    toks = re.findall(r"[a-z']+", text.lower())
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def report(study, arms):
    scenarios = pd.read_csv(
        Path(__file__).resolve().parent.parent.parent
        / "experiments"
        / ("scenarios_nonfood.csv" if study.startswith("nonfood") else "scenarios.csv")
    )
    eff_grams = {
        row["scenario_label"]: _ngrams(row["low_risk_share_effort_low"])
        | _ngrams(row["low_risk_share_effort_high"])
        for _, row in scenarios.iterrows()
    }
    lm_dir = Path(__file__).resolve().parent.parent / "outputs" / "lm" / study
    print(f"\n=== {study} smoke report ===")
    print(
        f"{'arm':20s}{'lowg-coverage(share)':>22s}{'set size mean/max':>18s}{'empty units':>12s}{'contamination':>14s}"
    )
    for arm in arms:
        suffix = "" if arm == "current" else "_" + arm
        runs = lm_dir / f"lm_runs{suffix}.jsonl"
        if not runs.exists():
            print(f"{arm:20s}{'(not elicited)':>22s}")
            continue
        t = _LOADERS[study](runs_filename=runs.name)
        if t is None:
            print(f"{arm:20s}{'(unscored)':>22s}")
            continue
        g, pr = np.asarray(t["g"]), np.asarray(t["prior"])
        mask = pr > 1e-6
        gmin = np.where(mask, g, np.inf).min(-1)
        # observed-action axis is axis 2 in every padded family
        # (ACTION_COLS = [no_share, low_risk_share, high_risk_share]; slot 0 is
        # no_share, so [1:] keeps the two share-observed cells).
        share = gmin[:, :, 1:, ...]
        coverage = float((share <= 1 / 6 + 1e-9).mean())
        sizes = mask.sum(-1)
        # Alternatives text field is `action_text` (generate_alternatives.py
        # writes `row["action_text"] = alt["action"]`); each record also carries
        # `scenario_label`.
        alts = lm_dir / f"lm_alternatives{suffix}.jsonl"
        n_contam, n_alts = 0, 0
        with open(alts) as f:
            for line in f:
                rec = json.loads(line)
                n_alts += 1
                if _ngrams(rec["action_text"]) & eff_grams[rec["scenario_label"]]:
                    n_contam += 1
        empties = lm_dir / f"lm_alternatives{suffix}.empty_units.jsonl"
        n_empty = sum(1 for _ in open(empties)) if empties.exists() else 0
        print(
            f"{arm:20s}{coverage:>21.1%} {sizes.mean():>10.1f}/{int(sizes.max()):<6d}"
            f"{n_empty:>12d}{n_contam / max(n_alts, 1):>13.1%}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True, choices=tuple(_LOADERS))
    p.add_argument(
        "--arms", nargs="+", default=["current", "refusal_hint", "refusal_hint_hyp"]
    )
    a = p.parse_args()
    report(a.study, a.arms)
