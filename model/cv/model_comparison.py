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

from _helpers import (  # noqa: E402
    _load_long,
    read_jsonl,
    sha256_file,
    verify_fit_manifest,
    write_json,
)
from study_registry import STUDIES, reported_base  # noqa: E402
from utils import get_project_root  # noqa: E402

# The three CV output files written together by the dispatcher's _write_outputs
# and hashed into cv_manifest.json (must match CV_OUTPUT_NAMES there).
_CV_OUTPUT_NAMES = ("cv_preds_summary.json", "cv_folds.jsonl", "cv_trial_ll.jsonl")

# Per-study cell grid and DV mapping, derived from the shared study registry
# (study_registry.py) so the model-comparison cells and the figures never
# disagree. `keys` are the columns that identify a scenario × condition cell in
# BOTH the human data (after _prepare_data below) and cv_preds_summary.json;
# `dvs` maps each human belief-update column to its model delta column (short
# DV id).
STUDY_SPECS = {
    slug: {
        "keys": s.cell_keys,
        "dvs": [(dv.update_col, dv.delta_col, dv.name) for dv in s.dvs],
    }
    for slug, s in STUDIES.items()
}

_LEVEL_STR = {0: "low", 1: "high"}


def _verify_cv_manifest(slug, outputs_dir):
    """Provenance check for a study's CV outputs, with the same asymmetry as the
    fit check: a *present* cv_manifest.json that no longer matches (the three CV
    files were not written together, or the input data changed) is a hard error
    — that is the mixed-vintage combination the manifest exists to catch. A
    *missing* manifest only warns and proceeds: CV outputs produced before
    provenance tracking can't be verified, but blocking would refuse to compare
    a study you already ran. Re-run `make cv-<slug>` to record provenance before
    trusting the final published numbers."""
    manifest_path = outputs_dir / "cv_manifest.json"
    if not manifest_path.exists():
        print(
            f"WARNING: no cv_manifest.json for {slug} — these CV outputs predate "
            f"provenance tracking and can't be verified as a single consistent "
            f"run over the current data. Proceeding; re-run `make cv-{slug}` to "
            f"record provenance.",
            file=sys.stderr,
        )
        return
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
    data_csv = get_project_root() / "data" / slug / "main_trials_long.csv"
    if sha256_file(data_csv) != manifest.get("input_data", {}).get("sha256"):
        raise RuntimeError(
            f"data/{slug}/main_trials_long.csv changed since CV ran — stale CV "
            f"outputs for {slug}; re-run `make fit-{slug}` and `make cv-{slug}`."
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


def _wide_trials(trial_df):
    """Per-trial held-out LL as (subject, scenario) x variant, with every
    variant present for every trial — the matched form all contrasts need."""
    assert not trial_df.duplicated(["model", "subject_id", "scenario_label"]).any(), (
        "cv_trial_ll.jsonl has duplicate (model, subject, scenario) rows"
    )
    wide = trial_df.pivot(
        index=["subject_id", "scenario_label"], columns="model", values="held_out_ll"
    )
    assert not wide.isna().any().any(), "trials not matched across model variants"
    return wide


def _contrast(wide, a, b, n_boot, rng, label=None):
    """One per-trial held-out LL contrast (a - b) with a participant-bootstrap
    95% CI. Positive favors `a`."""
    subject_ids = wide.index.get_level_values("subject_id").to_numpy()
    diff = (wide[a] - wide[b]).to_numpy()
    boots = _bootstrap_mean_by_subject(diff, subject_ids, n_boot, rng)
    return {
        "comparison": label or f"{a}_minus_{b}",
        "mean_per_trial_ll_diff": float(diff.mean()),
        "ci_95": [
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        ],
    }


def _deviation_contrasts(wide, slug, n_boot, rng):
    """Preregistration-deviation statistics for one study, or [].

    The main text reports `base_shared` as Base (see study_registry.
    reported_base). These are the two numbers the deviation section needs: the
    preregistered contrast, and the comparison-set change on its own — which is
    what the deviation consists of, since the promoted and preregistered bases
    share an identical utility and parameter count and differ only in the
    alternative set they are scored against.
    """
    promoted = reported_base(slug)
    if promoted == "base" or promoted not in wide.columns:
        return []
    # Only the contrast that is NOT already in `primary`. The preregistered
    # comparison is primary's own `full_minus_base` (raw keys throughout this
    # file), so recomputing it here would put the same statistic in the file
    # twice with two different bootstrap draws and two slightly different CIs.
    return [
        {
            **_contrast(wide, promoted, "base", n_boot, rng),
            "meaning": "the deviation itself: comparison set alone, utility and "
            "parameter count held fixed. Pairs with primary's full_minus_base "
            "(the preregistered contrast, which moves both) and "
            f"full_minus_{promoted} (the reported one, utility alone)",
        }
    ]


def _primary_comparisons(trial_df, n_boot, rng):
    """Full-minus-ablation per-trial held-out LL differences with participant-
    bootstrap CIs. Trials are matched across variants on (subject, scenario)."""
    wide = _wide_trials(trial_df)
    out = []
    for ablation in [m for m in wide.columns if m != "full"]:
        out.append(_contrast(wide, "full", ablation, n_boot, rng))
    return out


def subject_cell_matrices(data, keys, update_col, cells):
    """Per-(subject, cell) sums and counts of `update_col`, each
    (n_subjects, n_cells) with columns in `cells`' row order.

    The two together are all any subject-level resampling or splitting needs: a
    subset's cell mean is `Σ sums / Σ counts` over the chosen subjects, which is
    the trial-level mean the observed cell mean is, not a mean of subject means.
    """
    cell_pos = {c: i for i, c in enumerate(cells.apply(tuple, axis=1))}
    sc = data.groupby(["subject_id", *list(keys)])[update_col].agg(["sum", "count"])
    subj_pos = {s: i for i, s in enumerate(sorted(data["subject_id"].unique()))}
    mat_sum = np.zeros((len(subj_pos), len(cell_pos)))
    mat_cnt = np.zeros((len(subj_pos), len(cell_pos)))
    for row_key, row in sc.iterrows():
        subj, cell = row_key[0], tuple(row_key[1:])
        if cell not in cell_pos:  # cell missing from preds (shouldn't happen)
            continue
        mat_sum[subj_pos[subj], cell_pos[cell]] = row["sum"]
        mat_cnt[subj_pos[subj], cell_pos[cell]] = row["count"]
    return mat_sum, mat_cnt


#: Random splits averaged when estimating a noise ceiling. A single split is noisy
#: at these cell counts; the estimate is stable well before 400.
N_CEILING_SPLITS = 400


def split_half_ceiling(blocks, *, n_split=N_CEILING_SPLITS, rng=None):
    """Upper bound on the correlation ANY model can reach with these cell means.

    The cell means are noisy estimates of the true cell means, so even a perfect
    model cannot correlate with them at 1. Split-half reliability measures how
    much of them is signal: split the participants in half, correlate one half's
    cell means against the other's across cells, average over splits in Fisher-z,
    then Spearman-Brown correct from half- to full-sample length.

    The ceiling on the CORRELATION is the SQUARE ROOT of that reliability, because
    the reliability is the maximum explained *variance* of a perfect model. Using
    the reliability itself is the error van Bree, Styrnal & Hebart (2025) document
    as the prevalent one in this literature; it gives a too-low ceiling and makes
    models look closer to it than they are.

    What is split is participants, because participants are the unit whose
    sampling generates the noise being bounded -- the same unit the primary
    bootstrap and the human error bars cluster on. Scenarios are deliberately NOT
    the split unit: cell means are averaged over scenarios, so two scenario-halves
    differ in stimulus content rather than only in noise, and correlating them
    would charge the model for real scenario heterogeneity it is not being asked
    to predict at this grain (that is a generalization ceiling, not a noise one).

    `blocks` is one entry per independent participant pool (i.e. per study), each
    a list of that pool's (sums, counts) matrices from `subject_cell_matrices` --
    one per DV, all sharing the pool's subject axis. Every DV of a study is split
    on the SAME draw, so the pooled half-vectors come from the same people; the
    matrices are concatenated across blocks in the order given, which must match
    the order the correlation itself pools its cells in.

    Returns {"split_half", "reliability", "ceiling", "n_splits"}.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    zs = []
    for _ in range(n_split):
        halves = ([], [])
        for mats in blocks:
            n_subj = mats[0][0].shape[0]
            pick = rng.permutation(n_subj) < n_subj // 2
            for sums, counts in mats:
                for sel, acc in ((pick, halves[0]), (~pick, halves[1])):
                    num, den = sums[sel].sum(0), counts[sel].sum(0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        acc.append(np.where(den > 0, num / den, np.nan))
        a, b = np.concatenate(halves[0]), np.concatenate(halves[1])
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() > 2 and np.std(a[ok]) > 1e-12 and np.std(b[ok]) > 1e-12:
            zs.append(
                np.arctanh(np.clip(np.corrcoef(a[ok], b[ok])[0, 1], -0.9999, 0.9999))
            )
    if not zs:
        return None
    r_half = float(np.tanh(np.mean(zs)))
    # Spearman-Brown: a half-sample split understates the full sample's reliability.
    reliability = 2 * r_half / (1 + r_half)
    return {
        "split_half": r_half,
        "reliability": float(reliability),
        "ceiling": float(np.sqrt(max(reliability, 0.0))),
        "n_splits": len(zs),
    }


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

    mat_sum, mat_cnt = subject_cell_matrices(data, keys, update_col, merged[keys])
    subjects = range(mat_sum.shape[0])

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
    # The fit the CV warm-started from must also match its manifest and the
    # current data CSV; both manifests validating against the same CSV
    # guarantees the fit and the CV share one data vintage.
    verify_fit_manifest(slug, output_dir=outputs_dir)

    rng = np.random.default_rng(seed)
    trial_df = pd.DataFrame(read_jsonl(trial_path))
    with open(preds_path) as f:
        preds = pd.DataFrame(json.load(f))
    data = _prepare_data(slug)

    result = {
        "experiment": slug,
        "n_boot": n_boot,
        "seed": seed,
        # Which variant key the paper's "Base" column refers to. Every key in
        # this file is a raw variant name, so in the desire-inferring studies
        # `full_minus_base` is the PREREGISTERED broadcast-set contrast while
        # the main text reports `full_minus_base_shared` as full - base (the
        # preregistered one moves the comparison set as well as the utility;
        # see study_registry.reported_base). Annotation only —
        # nothing here is renamed, so both contrasts stay quotable and the
        # deviation section can cite either.
        "reported_base": reported_base(slug),
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

    # Computed LAST, and off a dedicated stream, so adding it cannot perturb the
    # primary/secondary bootstrap draws: those are already-published numbers, and
    # inserting a draw ahead of them shifted their CIs in the 3rd decimal.
    result["prereg_deviation"] = _deviation_contrasts(
        _wide_trials(trial_df), slug, n_boot, np.random.default_rng(seed)
    )

    # Likewise off its own stream. One ceiling per DV, not per (model, DV): it is a
    # property of the human data alone, so every model in a study is judged against
    # the same bound. Cells come from the `full` predictions purely to fix the cell
    # ORDER -- the ceiling never reads a model's values.
    ceil_rng = np.random.default_rng(seed + 1)
    result["noise_ceilings"] = []
    ref = preds[preds["model"] == "full"]
    for update_col, _delta_col, dv in spec["dvs"]:
        cells = data.groupby(spec["keys"], as_index=False)[update_col].mean()
        cells = cells.merge(ref[spec["keys"]].drop_duplicates(), on=spec["keys"])
        mats = subject_cell_matrices(
            data, spec["keys"], update_col, cells[spec["keys"]]
        )
        got = split_half_ceiling([[mats]], rng=ceil_rng)
        if got is not None:
            result["noise_ceilings"].append({"dv": dv, **got})

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
    best = {
        r["dv"]: r["r"]
        for r in result["secondary_correlations"]
        if r["model"] == "full"
    }
    for row in result["noise_ceilings"]:
        r_full = best.get(row["dv"])
        frac = f", full model at {r_full / row['ceiling']:.1%} of it" if r_full else ""
        print(
            f"  noise ceiling ({row['dv']}) = {row['ceiling']:.4f} "
            f"(split-half {row['split_half']:.4f} -> reliability "
            f"{row['reliability']:.4f}){frac}"
        )
    print(f"  wrote {out_path}")
    return result


#: Retired spellings of the root tag, each with what to pass instead. Rejected
#: loudly rather than aliased: every one of them names a *different* model than a
#: reader would now assume, so silently resolving them would mislabel a
#: comparison rather than merely inconvenience the caller.
_RETIRED_CONFIG_TAGS = {
    "canonical": (
        "'canonical' (retired 2026-07-30) read as 'the authoritative model', "
        "which the study root is not by itself — the reported fits add the "
        "comparison-set reweighting on top. Pass 'reported'."
    ),
    "preregistered": (
        "'preregistered' (retired 2026-08-03) named the study root, but the "
        "root holds the REPORTED model, which deviates from the "
        "preregistration by reweighting the comparison set. Pass 'reported' for "
        "the root; the preregistered model is the --no-reweighting run, tag "
        "'uniform-noreweight'."
    ),
}


def _config_dir(slug, tag):
    """Outputs directory for one run-config tag. `reported` is the default
    config, which writes the study root; anything else is an alt/ tag. Mirrors
    `RunConfig.outputs_dir` (and `RunConfig.tag`, which generates the alt tags)."""
    if tag in _RETIRED_CONFIG_TAGS:
        raise SystemExit(f"run-config tag {_RETIRED_CONFIG_TAGS[tag]}")
    root = get_project_root() / "model" / "outputs" / slug
    return root if tag == "reported" else root / "alt" / tag


def compare_configs(slug, tag_a, tag_b, n_boot, seed):
    """Matched-trial held-out-LL comparison between two run configs of the
    same study (the attribution grid). Both configs' CV
    manifests are verified against the same data CSV, then per-variant
    (subject, scenario)-matched LL differences get the standard participant
    bootstrap."""
    rng = np.random.default_rng(seed)
    frames, source = {}, {}
    for tag in (tag_a, tag_b):
        d = _config_dir(slug, tag)
        path = d / "cv_trial_ll.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run CV for config {tag} first")
        _verify_cv_manifest(slug, d)
        frames[tag] = pd.DataFrame(read_jsonl(path))
        # Hash of each side's primary CV output, so a consumer of this file (the
        # results-LaTeX exporter) can tell that one of the two configs has been
        # re-run since — which a verified manifest inside each dir cannot reveal,
        # each being self-consistent on its own.
        source[tag] = sha256_file(path)
    result = {
        "experiment": slug,
        "comparison": f"{tag_b}_minus_{tag_a}",
        "n_boot": n_boot,
        "seed": seed,
        "source": {f"{tag}/cv_trial_ll.jsonl": sha for tag, sha in source.items()},
        "per_variant": [],
    }
    common = sorted(set(frames[tag_a]["model"]) & set(frames[tag_b]["model"]))
    for variant in common:
        a = frames[tag_a][frames[tag_a]["model"] == variant]
        b = frames[tag_b][frames[tag_b]["model"] == variant]
        wide = a.merge(
            b, on=["subject_id", "scenario_label"], suffixes=("_a", "_b"), how="inner"
        )
        if len(wide) != len(a) or len(wide) != len(b):
            raise RuntimeError(
                f"{slug}/{variant}: trial sets differ between configs "
                f"({len(a)} vs {len(b)}, matched {len(wide)}) — different data "
                "vintages; re-run CV."
            )
        diff = (wide["held_out_ll_b"] - wide["held_out_ll_a"]).to_numpy()
        boots = _bootstrap_mean_by_subject(
            diff, wide["subject_id"].to_numpy(), n_boot, rng
        )
        result["per_variant"].append(
            {
                "variant": variant,
                "mean_ll_a": float(wide["held_out_ll_a"].mean()),
                "mean_ll_b": float(wide["held_out_ll_b"].mean()),
                "mean_per_trial_ll_diff": float(diff.mean()),
                "ci_95": [
                    float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5)),
                ],
            }
        )
    out_dir = get_project_root() / "model" / "outputs" / slug / "alt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_{tag_a}_vs_{tag_b}.json"
    write_json(out_path, result)
    print(f"\n=== {slug}: {tag_b} − {tag_a} (per-trial held-out LL) ===")
    for row in result["per_variant"]:
        lo, hi = row["ci_95"]
        print(
            f"  {row['variant']}: {row['mean_per_trial_ll_diff']:+.4f} "
            f"[{lo:+.4f}, {hi:+.4f}]  ({row['mean_ll_a']:.4f} -> {row['mean_ll_b']:.4f})"
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
    parser.add_argument(
        "--compare-configs",
        nargs=2,
        metavar=("TAG_A", "TAG_B"),
        default=None,
        help="Matched-trial held-out-LL comparison between two run configs of "
        "one study, reported as TAG_B minus TAG_A. Each tag is `reported` "
        "(outputs/<slug>) or an alt/ tag (outputs/<slug>/alt/<tag>) — e.g. "
        "`uniform-noreweight reported` for the reported model's gain over the "
        "preregistered one. Requires --study to name a single study.",
    )
    args = parser.parse_args()

    if args.compare_configs is not None:
        if args.study == "all":
            parser.error("--compare-configs requires --study to name a single study")
        tag_a, tag_b = args.compare_configs
        compare_configs(args.study, tag_a, tag_b, n_boot=args.n_boot, seed=args.seed)
        return

    slugs = list(STUDY_SPECS) if args.study == "all" else [args.study]
    for slug in slugs:
        run_study(slug, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
