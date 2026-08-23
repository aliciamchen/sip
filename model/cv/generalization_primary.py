#!/usr/bin/env python3
"""The cross-experiment generalization analyses, on the PRIMARY metric.

`transfer.py` and `pooled.py` score a shared utility by held-out log-likelihood.
That is the preregistered metric, and the paper demotes it for a stated reason:
the manipulated condition is only ~1-3% of trial-level variance, so a global
likelihood is close to blind to the modulation these studies test. An
*equivalence* claim ("the food-fitted utility predicts nonfood judgments about as
well") argued from a small dLL therefore does not support itself -- a transferred
utility could preserve the bulk of the response and lose the relationship
modulation entirely, and dLL would still read about zero.

This module scores the same arms on the two measures that are sensitive to it:

  correlation  the condition-averaged model-vs-human Pearson r the main text
               reports, computed exactly as `model_comparison.study_group_correlations`
               computes it, so a transferred arm and the reported run are
               comparable number for number.
  gradient     the median fraction of the human modulation the arm recovers,
               over the cells whose human gradient is reliable -- the same
               statistic and the same reliability screen `tab:gradients` uses
               for the ablations.

Nothing is refitted: the transfer and pooled runs already wrote standard CV
output sets under `outputs/<slug>/alt/<tag>/`, so this reads their held-out
predictions.

Writes `model/outputs/generalization_primary.json`.

Usage:
    uv run python model/cv/generalization_primary.py
"""

import argparse
import json
import statistics

import numpy as np
import pandas as pd

from model.inverse._helpers import write_json
from study_registry import STUDIES, SLUGS, study
from utils import get_project_root

from model.cv.contrast_tests import condition_gradients
from model.cv.model_comparison import (
    STUDY_SPECS,
    N_PAIR_BOOT,
    _condition_cells,
    _prepare_data,
    _seed_for,
    _verify_cv_manifest,
    merge_condition_cells,
    pair_bootstrap_corr,
)

#: The arms the SI generalization section reports, as
#: (key, config tag, human-readable description). `None` is the reported run --
#: each experiment's own utility -- which every arm is compared against.
#:
#: Tags must match what `transfer.py` / `pooled.py` wrote; a study missing an
#: arm's directory is skipped for that arm (the food experiments have no
#: food-to-nonfood transfer, by construction).
ARMS = (
    ("own", None, "the experiment's own fitted utility"),
    (
        "food",
        "transfer-pooled-food-refit",
        "one utility fit to the four food experiments",
    ),
    ("pooled", "pooled-all", "one utility fit to all six experiments"),
)


def _load_preds(slug, tag):
    """The full model's held-out per-cell predictions for one config, or None.

    The reported run (`tag is None`) is manifest-verified, the same check
    `model_comparison.run_study` and `study_group_correlations` make, so a stale
    or mixed-vintage CV output cannot silently become the baseline every arm is
    compared against. The alt arms carry no manifest of their own -- `transfer.py`
    and `pooled.py` write the CV file set but not one -- so they are read as-is;
    they are exploratory and their vintage is the run that produced them.
    """
    outputs = get_project_root() / "model" / "outputs" / slug
    if tag is None:
        _verify_cv_manifest(slug, outputs)
    path = (outputs if tag is None else outputs / "alt" / tag) / "cv_preds_summary.json"
    if not path.exists():
        return None
    preds = pd.DataFrame(json.loads(path.read_text()))
    full = preds[preds["model"] == "full"]
    return full if len(full) else None


def _condition_points(slug, data, preds):
    """(model x, human y) at condition grain, concatenated over the study's DVs
    in registry order -- the same construction and the same order
    `study_group_correlations` uses, so the numbers are comparable."""
    keys = _condition_cells(slug)
    xs, ys = [], []
    for update_col, delta_col, _dv in STUDY_SPECS[slug]["dvs"]:
        human = data.groupby(keys, as_index=False)[update_col].mean()
        x, y = merge_condition_cells(
            human,
            preds,
            keys,
            update_col,
            delta_col,
            slug,
            "in this arm -- its CV run is incomplete or stale.",
        )
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def _gradient_recovery(slug, data, preds_by_arm, n_boot):
    """{arm: (median recovered fraction, n reliable cells)}.

    The reliability screen is a property of the human data, so it is identical
    across arms -- `condition_gradients` computes the human side once and shares
    it. The median, not the mean, for the reason `_modulation_macros` takes a
    median: a recovered fraction is a ratio and the cells with a small human
    denominator would otherwise dominate.
    """
    st = study(slug)
    per_arm = {}
    for update_col, delta_col, dv in STUDY_SPECS[slug]["dvs"]:
        rows = condition_gradients(
            data,
            preds_by_arm,
            st,
            update_col,
            delta_col,
            dv,
            n_boot=n_boot,
            # The SAME seed key run_study uses for this statistic. The human
            # bootstrap decides which cells pass the reliability screen and the
            # median is taken over exactly those, so a separate stream could
            # select a different cell set from tab:gradients and make the `own`
            # column disagree with \gradFull* with nothing raising.
            rng=np.random.default_rng(_seed_for(f"{slug}|{delta_col}|gradient")),
        )
        by = {(r["model"], r["dv"], r["action"]): r for r in rows}
        reliable = [
            (r["dv"], r["action"])
            for r in rows
            if r["model"] == "own" and r["recovered_fraction"] is not None
        ]
        for arm in preds_by_arm:
            for dv_name, action in reliable:
                got = by.get((arm, dv_name, action))
                ref = by[("own", dv_name, action)]
                if got is None:
                    continue
                per_arm.setdefault(arm, []).append(
                    abs(got["model_gradient"] / ref["human_gradient"])
                )
    return {
        arm: (statistics.median(vals), len(vals))
        for arm, vals in per_arm.items()
        if vals
    }


def run(n_boot=N_PAIR_BOOT, slugs=None):
    """Per-study correlation and gradient recovery under each generalization arm,
    plus the nonfood pair pooled (what the main text quotes)."""
    rows, nonfood_points = [], {}
    for slug in slugs or SLUGS:
        data = _prepare_data(slug)
        preds_by_arm = {}
        for arm, tag, _desc in ARMS:
            preds = _load_preds(slug, tag)
            if preds is not None:
                preds_by_arm[arm] = preds
        if "own" not in preds_by_arm:
            print(f"[{slug}] no reported CV outputs — skipped.")
            continue

        grads = _gradient_recovery(slug, data, preds_by_arm, n_boot)
        for arm, preds in preds_by_arm.items():
            x, y = _condition_points(slug, data, preds)
            corr = pair_bootstrap_corr(
                x, y, seed_key=f"gen|{slug}|{arm}|pair_ci", n_boot=n_boot
            )
            median, n_cells = grads.get(arm, (None, 0))
            rows.append(
                {
                    "slug": slug,
                    "experiment": STUDIES[slug].short_label,
                    "arm": arm,
                    "r": corr["r"],
                    "ci_95": corr["ci_95"],
                    "n_points": corr["n_cells"],
                    "grad_median_recovered": median,
                    "n_grad_cells": n_cells,
                }
            )
            if STUDIES[slug].domain != "food":
                nonfood_points.setdefault(arm, ([], [], set()))
                nonfood_points[arm][0].append(x)
                nonfood_points[arm][1].append(y)
                nonfood_points[arm][2].add(slug)

    # The main text quotes the nonfood pair as one number, since the claim is
    # about the domain rather than about either experiment. Which means it must
    # actually BE the pair: an arm whose directory exists for 3a but not 3b (a
    # partial `make transfer`) would otherwise pool 3a alone and read as the
    # domain result, and `\rNonfoodFoodFit` is the main text's generalization
    # number. The contributing slugs are recorded too, so the claim is auditable
    # from the artifact rather than only enforced here.
    combined = []
    n_nonfood = sum(1 for s in (slugs or SLUGS) if STUDIES[s].domain != "food")
    for arm, (xs, ys, arm_slugs) in nonfood_points.items():
        if len(arm_slugs) != n_nonfood:
            raise RuntimeError(
                f"the `{arm}` arm covers {sorted(arm_slugs)} but the nonfood "
                f"domain has {n_nonfood} experiments — a combined number from a "
                f"subset would be quoted as the domain result. Finish the run "
                f"that writes this arm, or drop it."
            )
        corr = pair_bootstrap_corr(
            np.concatenate(xs),
            np.concatenate(ys),
            seed_key=f"gen|nonfood|{arm}|pair_ci",
            n_boot=n_boot,
        )
        combined.append(
            {"group": "nonfood", "arm": arm, "slugs": sorted(arm_slugs), **corr}
        )
    return {"per_experiment": rows, "combined": combined}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=N_PAIR_BOOT)
    args = ap.parse_args()

    result = run(n_boot=args.n_boot)
    out_path = get_project_root() / "model" / "outputs" / "generalization_primary.json"
    write_json(out_path, result)

    print("\n=== generalization on the primary metric (condition grain) ===")
    for row in result["per_experiment"]:
        lo, hi = row["ci_95"]
        grad = (
            f", recovers {row['grad_median_recovered']:.0%} of the modulation "
            f"over {row['n_grad_cells']} cells"
            if row["grad_median_recovered"] is not None
            else ""
        )
        print(
            f"  {row['experiment']:3s} {row['arm']:7s} r = {row['r']:.3f} "
            f"[{lo:.3f}, {hi:.3f}]{grad}"
        )
    for row in result["combined"]:
        lo, hi = row["ci_95"]
        print(
            f"  nonfood combined {row['arm']:7s} r = {row['r']:.3f} [{lo:.3f}, {hi:.3f}]"
        )
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
