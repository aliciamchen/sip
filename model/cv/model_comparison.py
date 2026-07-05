#!/usr/bin/env python3
"""
Model comparison and evaluation from the LOSO CV outputs (the paper's numbers).

PRIMARY metric (manuscript "Model comparison and evaluation"): the difference
between the full model and each ablation in per-trial held-out log-likelihood,
with a 95% CI from bootstrap resampling of participants (default 1,000
resamples). Trials are matched across model variants on (subject_id,
scenario_label) — each participant sees each scenario exactly once — from
`outputs/<slug>/cv_trial_ll.jsonl`.

SECONDARY (descriptive): the Pearson correlation between the model's
out-of-sample per-cell belief-update predictions (`delta_<latent>` in
`cv_preds_summary.json`) and the condition-averaged human belief updates, with
a 95% CI from a subject-cluster bootstrap (resample participants, recompute the
cell means, re-correlate against the fixed model predictions).

Writes `outputs/<slug>/cv_model_comparison.json` and prints a summary.

Usage:
    uv run python model/cv/model_comparison.py                  # all studies with CV outputs
    uv run python model/cv/model_comparison.py --study food_inv_desire
    uv run python model/cv/model_comparison.py --n-boot 1000 --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

from _helpers import _load_long, read_jsonl, sha256_file, write_json  # noqa: E402
from utils import get_project_root  # noqa: E402

# The three CV output files written together by the dispatcher's _write_outputs
# and hashed into cv_manifest.json (must match CV_OUTPUT_NAMES there).
_CV_OUTPUT_NAMES = ("cv_preds_summary.json", "cv_folds.jsonl", "cv_trial_ll.jsonl")

# Per-study cell grid and DV mapping. `keys` are the columns that identify a
# scenario × condition cell in BOTH the human data (after _prepare_data below)
# and cv_preds_summary.json; `dvs` maps each human belief-update column to its
# model delta column.
STUDY_SPECS = {
    "food_inv_desire": {
        "keys": ["scenario_label", "action", "intimacy_condition", "effort_condition"],
        "dvs": [("response_update", "delta_desire", "desire")],
    },
    "food_inv_joint_de": {
        "keys": ["scenario_label", "action", "intimacy_condition"],
        "dvs": [
            ("desire_rating_update", "delta_desire", "desire"),
            ("effort_rating_update", "delta_effort", "effort"),
        ],
    },
    "food_inv_intimacy": {
        "keys": ["scenario_label", "action", "desire_condition", "effort_condition"],
        "dvs": [("intimacy_rating_update", "delta_intimacy", "intimacy")],
    },
    "food_inv_joint_ie": {
        "keys": ["scenario_label", "action", "desire_condition"],
        "dvs": [
            ("intimacy_rating_update", "delta_intimacy", "intimacy"),
            ("effort_rating_update", "delta_effort", "effort"),
        ],
    },
    # Study 3 (nonfood stimulus set): 3a mirrors 1b, 3b mirrors 2b.
    "nonfood_inv_joint_de": {
        "keys": ["scenario_label", "action", "intimacy_condition"],
        "dvs": [
            ("desire_rating_update", "delta_desire", "desire"),
            ("effort_rating_update", "delta_effort", "effort"),
        ],
    },
    "nonfood_inv_joint_ie": {
        "keys": ["scenario_label", "action", "desire_condition"],
        "dvs": [
            ("intimacy_rating_update", "delta_intimacy", "intimacy"),
            ("effort_rating_update", "delta_effort", "effort"),
        ],
    },
}

_LEVEL_STR = {0: "low", 1: "high"}


def _verify_cv_manifest(slug, outputs_dir):
    """Require the CV provenance manifest and verify the three CV output files
    still hash to the values recorded when they were written together. The
    files are only meaningful as one CV run's outputs — a mixed-vintage
    combination (e.g. a re-run cv_trial_ll.jsonl next to an older
    cv_preds_summary.json) would silently combine incompatible predictions."""
    manifest_path = outputs_dir / "cv_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"missing {manifest_path} — stale or mixed-vintage CV outputs for "
            f"{slug} (written before provenance manifests existed, or partially "
            f"deleted); re-run `make cv-{slug}`."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    stale = [
        name
        for name in _CV_OUTPUT_NAMES
        if sha256_file(outputs_dir / name) != manifest.get("outputs", {}).get(name)
    ]
    if stale:
        raise RuntimeError(
            f"CV output file(s) {stale} do not match cv_manifest.json — stale "
            f"or mixed-vintage CV outputs for {slug}; re-run `make cv-{slug}`."
        )


def _prepare_data(slug):
    """Per-trial belief updates with cell-key columns matching cv_preds_summary:
    `action` as an int, `intimacy_condition` as the verbal slug, and
    `desire_condition` / `effort_condition` as 'low'/'high' strings."""
    data = _load_long(slug)
    if "intimacy" in data.columns:
        data["intimacy_condition"] = data["intimacy"]
    for col in ("desire_condition", "effort_condition"):
        if col in data.columns and data[col].dtype != object:
            data[col] = data[col].map(_LEVEL_STR)
    return data


def _bootstrap_mean_by_subject(values, subject_ids, n_boot, rng):
    """Bootstrap the mean of `values` (one per trial) by resampling subjects
    with replacement. Returns (n_boot,) bootstrap means."""
    df = pd.DataFrame({"subject_id": subject_ids, "v": values})
    per_subj = df.groupby("subject_id")["v"].agg(["sum", "count"])
    sums = per_subj["sum"].to_numpy()
    counts = per_subj["count"].to_numpy()
    n_subj = len(per_subj)
    idx = rng.integers(0, n_subj, size=(n_boot, n_subj))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def _primary_comparisons(trial_df, n_boot, rng):
    """Full-minus-ablation per-trial held-out LL differences with participant-
    bootstrap CIs. Trials are matched across variants on (subject, scenario)."""
    assert not trial_df.duplicated(["model", "subject_id", "scenario_label"]).any(), (
        "cv_trial_ll.jsonl has duplicate (model, subject, scenario) rows"
    )
    wide = trial_df.pivot(
        index=["subject_id", "scenario_label"], columns="model", values="held_out_ll"
    )
    assert not wide.isna().any().any(), "trials not matched across model variants"
    subject_ids = wide.index.get_level_values("subject_id").to_numpy()

    out = []
    for ablation in [m for m in wide.columns if m != "full"]:
        diff = (wide["full"] - wide[ablation]).to_numpy()
        boots = _bootstrap_mean_by_subject(diff, subject_ids, n_boot, rng)
        out.append(
            {
                "comparison": f"full_minus_{ablation}",
                "mean_per_trial_ll_diff": float(diff.mean()),
                "ci_95": [
                    float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5)),
                ],
            }
        )
    return out


def _secondary_correlation(slug, data, preds, keys, update_col, delta_col, n_boot, rng):
    """Pearson r between condition-averaged human updates and the model's
    per-cell delta, with a subject-cluster bootstrap CI. Cells that lose all
    trials in a bootstrap resample are dropped pairwise from that resample's
    correlation."""
    cell_mean = data.groupby(keys, as_index=False)[update_col].mean()
    merged = cell_mean.merge(preds[keys + [delta_col]], on=keys, how="inner")
    if len(merged) != len(cell_mean):
        # An inner merge would silently drop human cells with no matching model
        # prediction (e.g. a stale condition label on either side).
        missing = cell_mean.merge(preds[keys + [delta_col]], on=keys, how="left")
        missing = missing.loc[missing[delta_col].isna(), keys]
        raise RuntimeError(
            f"{len(missing)} human cell(s) in {slug} have no matching model "
            f"prediction in cv_preds_summary.json. First offenders:\n"
            f"{missing.head(5)}\nStale CV outputs or a condition-label "
            f"mismatch; re-run `make cv-{slug}`."
        )
    if len(merged) < 3:
        return None
    r = float(np.corrcoef(merged[update_col], merged[delta_col])[0, 1])

    # Per-(subject, cell) sums/counts aligned to the merged cell order.
    cell_key = merged[keys].apply(tuple, axis=1)
    cell_pos = {c: i for i, c in enumerate(cell_key)}
    sc = data.groupby(["subject_id", *keys])[update_col].agg(["sum", "count"])
    subjects = sorted(data["subject_id"].unique())
    subj_pos = {s: i for i, s in enumerate(subjects)}
    mat_sum = np.zeros((len(subjects), len(merged)))
    mat_cnt = np.zeros((len(subjects), len(merged)))
    for row_key, row in sc.iterrows():
        subj, cell = row_key[0], tuple(row_key[1:])
        if cell not in cell_pos:  # cell missing from preds (shouldn't happen)
            continue
        mat_sum[subj_pos[subj], cell_pos[cell]] = row["sum"]
        mat_cnt[subj_pos[subj], cell_pos[cell]] = row["count"]

    delta = merged[delta_col].to_numpy()
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(subjects), size=len(subjects))
        s = mat_sum[idx].sum(axis=0)
        c = mat_cnt[idx].sum(axis=0)
        mask = c > 0
        boots[b] = np.corrcoef(s[mask] / c[mask], delta[mask])[0, 1]

    return {
        "r": r,
        # Percentile interval over the subject-cluster bootstrap. Conservative
        # for r: resampling participants adds sampling noise to the cell means,
        # which attenuates the bootstrapped correlations, so the interval sits
        # below the point estimate when per-cell trial counts are small.
        "ci_95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_cells": int(len(merged)),
    }


def run_study(slug, n_boot, seed):
    spec = STUDY_SPECS[slug]
    outputs_dir = get_project_root() / "model" / "outputs" / slug
    trial_path = outputs_dir / "cv_trial_ll.jsonl"
    preds_path = outputs_dir / "cv_preds_summary.json"
    if not trial_path.exists() or not preds_path.exists():
        print(f"[{slug}] missing CV outputs — run `make cv-{slug}` first; skipping.")
        return None
    _verify_cv_manifest(slug, outputs_dir)

    rng = np.random.default_rng(seed)
    trial_df = pd.DataFrame(read_jsonl(trial_path))
    with open(preds_path) as f:
        preds = pd.DataFrame(json.load(f))
    data = _prepare_data(slug)

    result = {
        "experiment": slug,
        "n_boot": n_boot,
        "seed": seed,
        "n_subjects": int(trial_df["subject_id"].nunique()),
        "n_trials_per_model": int(len(trial_df) / trial_df["model"].nunique()),
        "mean_held_out_ll_per_trial": {
            m: float(v)
            for m, v in trial_df.groupby("model")["held_out_ll"].mean().items()
        },
        "primary": _primary_comparisons(trial_df, n_boot, rng),
        "secondary_correlations": [],
    }

    for model in sorted(preds["model"].unique()):
        pm = preds[preds["model"] == model]
        for update_col, delta_col, dv in spec["dvs"]:
            corr = _secondary_correlation(
                slug, data, pm, spec["keys"], update_col, delta_col, n_boot, rng
            )
            if corr is not None:
                result["secondary_correlations"].append(
                    {"model": model, "dv": dv, **corr}
                )

    out_path = outputs_dir / "cv_model_comparison.json"
    write_json(out_path, result)

    print(f"\n=== {slug} (n = {result['n_subjects']} subjects) ===")
    for row in result["primary"]:
        lo, hi = row["ci_95"]
        print(
            f"  {row['comparison']}: {row['mean_per_trial_ll_diff']:+.4f} "
            f"per-trial LL, 95% CI [{lo:+.4f}, {hi:+.4f}]"
        )
    for row in result["secondary_correlations"]:
        lo, hi = row["ci_95"]
        print(
            f"  r({row['model']}, {row['dv']}) = {row['r']:.3f} "
            f"[{lo:.3f}, {hi:.3f}] over {row['n_cells']} cells"
        )
    print(f"  wrote {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--study",
        choices=[*STUDY_SPECS.keys(), "all"],
        default="all",
        help="Which experiment to evaluate (default: every study whose CV "
        "outputs exist; studies without them are skipped with a message).",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    slugs = list(STUDY_SPECS) if args.study == "all" else [args.study]
    for slug in slugs:
        run_study(slug, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
