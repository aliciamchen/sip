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
`cv_preds_summary.json`) and the human cell means, with a 95% CI bootstrapped
over the cells. Alongside it a split-half noise CEILING per DV, which is the
maximum correlation any model could reach given the noise in those cell means --
the quantity that says whether a given r is close to attainable.

HYPOTHESIS-MATCHED (secondary, `cv/contrast_tests.py`): the preregistered
primary is a global fit index, and the paper's claim is about a modulation worth
1-3% of trial-level variance, which such an index is close to blind to. Two
blocks quantify that and test the claim directly: `variance_decomposition` (where
the variance lives, bias-corrected, with participant-bootstrap intervals) and
`condition_gradients` (the belief-update change across the focal condition's
ordered levels, human against each variant's held-out predictions). Reported
BESIDE the preregistered primary, never in place of it.

Writes `outputs/<slug>/cv_model_comparison.json` and prints a summary.

Usage:
    uv run python model/cv/model_comparison.py                  # all studies with CV outputs
    uv run python model/cv/model_comparison.py --study food_inv_desire
    uv run python model/cv/model_comparison.py --n-boot 1000 --seed 0
"""

import argparse
import hashlib
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
from study_registry import STUDIES, reported_base, study, study_groups  # noqa: E402
from utils import get_project_root  # noqa: E402

from contrast_tests import condition_gradients, variance_decomposition  # noqa: E402

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


#: Resamples for the cell/pair bootstrap on a correlation. Matches the panels'
#: N_BOOT_AGG so the two report intervals of the same Monte-Carlo precision.
N_PAIR_BOOT = 1000


def _seed_for(key):
    """Deterministic 32-bit seed from a purpose string (the repo's SHA-256
    convention; Python's builtin hash is salted per process)."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


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


def _secondary_correlation(slug, data, preds, keys, update_col, delta_col, model=""):
    """Pearson r between per-cell human updates and the model's per-cell delta,
    with the interval bootstrapped over the PLOTTED POINTS (the cells).

    The interval was a subject-cluster bootstrap until 2026-08-04, and it was
    mislocated: a resample holds ~63% unique participants, so its cell means carry
    extra noise, noise in y attenuates r against a fixed x, and the whole bootstrap
    distribution sits below the observed r. At these scenario-level cells — a
    handful of judgments each — that put the interval below its own point estimate
    for most correlations. Resampling the cells reuses the observed means untouched,
    injects no noise, and is unbiased; it also matches both the panels
    (`_agg.corr_with_pair_ci`) and what comparable papers report. See
    `notes/2026-08-03-correlation-ci-audit.md`.

    What the interval therefore means: how far r would move with a different sample
    of *cells*, not of *participants*. The participant-driven limit on r is
    reported separately as the noise ceiling (`split_half_ceiling`), which is the
    quantity the old interval was gesturing at and getting wrong.

    Seeded from (slug, model, delta_col) rather than drawing from the caller's
    shared generator, so a correlation's interval no longer depends on how many
    other (model, DV) pairs happened to be computed before it.
    """
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
    return pair_bootstrap_corr(
        merged[delta_col].to_numpy(),
        merged[update_col].to_numpy(),
        seed_key=f"{slug}|{model}|{delta_col}|pair_ci",
    )


#: Below this, a variant's per-cell predictions are treated as CONSTANT and its
#: correlation as undefined rather than computed.
#:
#: An ablation missing a utility term cannot infer the latent that term carries,
#: so its posterior stays exactly the prior and every cell gets the same
#: prediction. That should surface as `n/a` -- it is the visible ablation
#: contrast, not a fit. The guard was 1e-12, which only catches it when the
#: constant is bit-exact: the intimacy posterior of the vanilla model in 2b/3b is
#: flat to float32 rounding and spans about 1e-7, so 1e-12 let a correlation of
#: pure numerical noise through and the paper would have printed r = 0.265 for a
#: model the same paragraph says cannot infer intimacy at all. The predictions
#: are belief updates on a [-0.5, 0.5] scale, so 1e-6 is six orders of magnitude
#: below anything a model means to say and far above float32 noise.
CONSTANT_PREDICTION_TOL = 1e-6


def pair_bootstrap_corr(x, y, *, seed_key, n_boot=N_PAIR_BOOT):
    """Pearson r over (x, y) with a 95% CI bootstrapped over the PAIRS.

    The one implementation behind every correlation the paper reports -- the
    per-study `secondary_correlations` here, the pooled per-study-number ones in
    `study_group_correlations`, the generalization arms in
    `generalization_primary.py`, and the panel annotations (`_agg.corr_with_pair_ci`
    delegates here), so a number in the text cannot disagree with the same number
    on a figure.

    Making that hold takes more than sharing the code, because a bootstrap draws
    INDICES: two callers holding the same points in a different order get
    different resamples and so different intervals from the same seed. The
    figure side concatenates its cells grouped by display label and this module
    groups by the raw condition columns, which sort differently -- that is how
    Study 2's vanilla interval came out [0.228, 0.691] here and [0.200, 0.703] on
    the panel. The points are therefore sorted canonically before resampling, so
    the interval depends on the SET of points and not on the order a caller
    happened to build them in. The point estimate never depended on order.

    `r` is NaN when the model's predictions are constant (see
    `CONSTANT_PREDICTION_TOL`) -- the same shape an exactly-constant variant
    already produced on disk, so downstream consumers keep rendering those as
    `n/a` rather than losing the entry.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x) < CONSTANT_PREDICTION_TOL or np.std(y) < CONSTANT_PREDICTION_TOL:
        return {
            "r": float("nan"),
            "ci_95": [float("nan"), float("nan")],
            "ci_method": "undefined (constant predictions)",
            "n_cells": int(len(x)),
        }
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(_seed_for(seed_key))
    n = len(x)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if (
            np.std(x[i]) > CONSTANT_PREDICTION_TOL
            and np.std(y[i]) > CONSTANT_PREDICTION_TOL
        ):
            boots.append(np.corrcoef(x[i], y[i])[0, 1])
    boots = np.asarray(boots)
    boots = boots[np.isfinite(boots)]
    ci = (
        [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        if boots.size
        else [float("nan"), float("nan")]
    )
    return {
        "r": r,
        "ci_95": ci,
        # Named on disk, because an interval whose construction is unstated is
        # exactly what made six comparable papers' intervals un-auditable.
        "ci_method": "percentile bootstrap over cells",
        "n_cells": int(n),
    }


#: Variant order for the pooled panels and the group correlations, after the
#: reported-base promotion below (so "base" here always means the variant the
#: paper calls the vanilla model).
GROUP_MODEL_ORDER = ("base", "discomfort_only", "full")


def _condition_cells(slug):
    """Condition-grain grouping columns: the study's cell keys without the
    scenario. The main text and the results figures quote a correlation over
    these cells -- each averaged over the \\nScenarios{} scenarios -- while
    `secondary_correlations` above is at scenario x condition grain. Two
    different numbers, so the paper names the grain wherever it quotes one."""
    return [k for k in STUDY_SPECS[slug]["keys"] if k != "scenario_label"]


def _promote_reported_base(slug, preds):
    """Rename this study's reported base variant to `base` and drop the other.

    The desire-inferring studies report `base_shared` as the vanilla model
    because the preregistered `base` also swaps the comparison set (see
    study_registry.reported_base). Doing it here means the pooled correlation
    describes the same three models the paper's tables and figures do.
    """
    promoted = reported_base(slug)
    if promoted == "base":
        return preds
    present = set(preds["model"].unique())
    if promoted not in present:
        raise KeyError(
            f"{slug}: cv_preds_summary.json has no `{promoted}` variant, but "
            f"study_registry.reported_base says the paper reports it as the "
            f"vanilla model. Re-run `make fit-{slug} cv-{slug}`."
        )
    out = preds[preds["model"] != "base"].copy()
    out.loc[out["model"] == promoted, "model"] = "base"
    return out


def group_corr_seed_key(number, model, seed=0):
    """The bootstrap seed key for one pooled study-group correlation.

    Exported because `figures/scripts/_agg.py` annotates its panels with these
    same correlations and must draw the same resamples to print the same
    interval; with `pair_bootstrap_corr` sorting its points canonically, a shared
    seed key is the remaining thing the two sides need to agree on.
    """
    return f"group|{seed}|{number}|{model}|pair_ci"


def study_group_correlations(seed):
    """Pooled model-vs-human correlation per paper study number, at condition grain.

    This is the number the main text quotes and the number the results figures'
    scatter columns annotate: one point per (sub-study x condition x inferred
    latent), the model's out-of-sample prediction against the human mean, both
    averaged over the scenarios. Pooling the sub-studies is what makes it one
    correlation per study rather than one per DV per sub-study.

    Reported beside a split-half noise ceiling computed at the SAME grain, since
    a ceiling from the scenario-grain cells would not bound these. Participants
    are split within each sub-study independently (separate participant pools)
    and every DV of a sub-study uses that sub-study's draw, matching how the
    correlation pools its cells.

    A variant whose predictions are constant gets `r: null` -- see
    `CONSTANT_PREDICTION_TOL`.

    `seed` enters every stream this computes, so `--seed` moves these intervals
    the way it moves the primary ones. It was accepted and ignored until
    2026-08-16, which made a seed-sensitivity check wrongly conclude the pooled
    correlations were seed-independent.

    Each entry records the SHA-256 of the `cv_preds_summary.json` it read, so the
    exporter can refuse to quote these beside per-study numbers from a newer CV
    run -- this file is written only by the all-studies pass, so re-running one
    study leaves it a vintage behind with nothing else to notice.
    """
    out = []
    for number, members in study_groups():
        loaded, sources = [], {}
        for st in members:
            outputs_dir = get_project_root() / "model" / "outputs" / st.slug
            preds_path = outputs_dir / "cv_preds_summary.json"
            if not preds_path.exists():
                loaded = None
                break
            _verify_cv_manifest(st.slug, outputs_dir)
            sources[st.slug] = sha256_file(preds_path)
            with open(preds_path) as f:
                preds = _promote_reported_base(st.slug, pd.DataFrame(json.load(f)))
            loaded.append((st, _prepare_data(st.slug), preds))
        if not loaded:
            print(f"[Study {number}] missing CV outputs — skipping group correlation.")
            continue

        entry = {
            "study": number,
            "slugs": [st.slug for st, _d, _p in loaded],
            "grain": "condition (averaged over scenarios)",
            "seed": seed,
            "source": sources,
            "correlations": [],
        }
        # Human cells and the ceiling blocks first: both are properties of the
        # data alone, so every variant is judged against the same points.
        human, blocks = {}, []
        for st, data, _preds in loaded:
            keys = _condition_cells(st.slug)
            mats = []
            for update_col, _delta_col, dv in STUDY_SPECS[st.slug]["dvs"]:
                cells = data.groupby(keys, as_index=False)[update_col].mean()
                human[(st.slug, dv)] = (keys, cells, update_col)
                mats.append(subject_cell_matrices(data, keys, update_col, cells[keys]))
            blocks.append(mats)
        ceiling = split_half_ceiling(
            blocks,
            rng=np.random.default_rng(_seed_for(f"group|{seed}|{number}|ceiling")),
        )
        entry["noise_ceiling"] = ceiling

        for model in GROUP_MODEL_ORDER:
            xs, ys = [], []
            for st, _data, preds in loaded:
                pm = preds[preds["model"] == model]
                if pm.empty:
                    continue
                for update_col, delta_col, dv in STUDY_SPECS[st.slug]["dvs"]:
                    keys, cells, _u = human[(st.slug, dv)]
                    model_cells = pm.groupby(keys, as_index=False)[delta_col].mean()
                    merged = cells.merge(model_cells, on=keys, how="inner")
                    if len(merged) != len(cells):
                        raise RuntimeError(
                            f"{st.slug}: {len(cells) - len(merged)} condition "
                            f"cell(s) have no {model} prediction — stale CV "
                            f"outputs; re-run `make cv-{st.slug}`."
                        )
                    xs.append(merged[delta_col].to_numpy())
                    ys.append(merged[update_col].to_numpy())
            if not xs:
                continue
            got = pair_bootstrap_corr(
                np.concatenate(xs),
                np.concatenate(ys),
                seed_key=group_corr_seed_key(number, model, seed),
            )
            entry["correlations"].append({"model": model, **got})
        out.append(entry)
    return out


def fmt_ci(ci, pct=True):
    """`[lo, hi]` for the console, or an empty string when absent/undefined."""
    if not ci or any(v != v for v in ci):
        return ""
    return f"[{ci[0]:.1%}, {ci[1]:.1%}]" if pct else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


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
                slug, data, pm, spec["keys"], update_col, delta_col, model
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

    # Hypothesis-matched statistics (see cv/contrast_tests.py). Secondary to the
    # preregistered primary above and reported beside it, never in place of it.
    # Each draws from its own `_seed_for` stream for the same reason the ceiling
    # does: the primary and secondary numbers are already published, and sharing
    # a generator would move their CIs in the third decimal.
    st = study(slug)
    result["variance_decomposition"] = []
    result["condition_gradients"] = []
    # One entry per variant, so the gradient code can compute the model-free
    # human bootstrap once and reuse it across variants rather than redrawing a
    # different interval for the same human statistic under each.
    preds_by_model = {
        m: preds[preds["model"] == m] for m in sorted(preds["model"].unique())
    }
    for update_col, delta_col, dv in spec["dvs"]:
        vd = variance_decomposition(
            data,
            st,
            update_col,
            dv,
            n_boot=n_boot,
            rng=np.random.default_rng(_seed_for(f"{slug}|{dv}|variance")),
        )
        if vd is not None:
            result["variance_decomposition"].append(vd)
        result["condition_gradients"].extend(
            condition_gradients(
                data,
                preds_by_model,
                st,
                update_col,
                delta_col,
                dv,
                n_boot=n_boot,
                rng=np.random.default_rng(_seed_for(f"{slug}|{delta_col}|gradient")),
            )
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
    for row in result["variance_decomposition"]:
        print(
            f"  variance ({row['dv']}): {row['frac_explainable']:.0%} explainable; "
            f"{row['focal_condition'].split('_')[0]} effect is "
            f"{row['focal_frac_of_total']:.1%} of trial variance "
            f"({row['focal_frac_of_explainable']:.0%} of explainable) "
            f"CI {fmt_ci(row.get('focal_frac_of_total_ci_95'))}, "
            f"{row['focal_frac_scenario_specific']:.0%} of it scenario-specific"
        )
    for row in result["condition_gradients"]:
        lo, hi = row["human_ci_95"]
        rec = row["recovered_fraction"]
        rec_s = f"{rec:+.0%} recovered" if rec is not None else "human gradient n.s."
        print(
            f"  gradient ({row['model']}, {row['dv']}, action {row['action']}): "
            f"human {row['human_gradient']:+.4f} [{lo:+.4f}, {hi:+.4f}] vs "
            f"model {row['model_gradient']:+.4f} -- {rec_s}"
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

    # The pooled per-study-number correlations pool ACROSS studies, so they can
    # only be computed once every study has been run. Written to their own
    # artifact rather than into any one study's file for the same reason.
    if args.study == "all":
        groups = study_group_correlations(seed=args.seed)
        if groups:
            out_path = (
                get_project_root() / "model" / "outputs" / "group_correlations.json"
            )
            write_json(out_path, groups)
            print("\n=== pooled correlations by study (condition grain) ===")
            for entry in groups:
                ceil = entry.get("noise_ceiling")
                for row in entry["correlations"]:
                    r, (lo, hi) = row["r"], row["ci_95"]
                    if r != r:
                        print(
                            f"  Study {entry['study']} {row['model']}: n/a (constant)"
                        )
                        continue
                    frac = (
                        f", {r / ceil['ceiling']:.1%} of ceiling"
                        if ceil and row["model"] == "full"
                        else ""
                    )
                    print(
                        f"  Study {entry['study']} {row['model']}: r = {r:.3f} "
                        f"[{lo:.3f}, {hi:.3f}] over {row['n_cells']} cells{frac}"
                    )
                if ceil:
                    print(
                        f"  Study {entry['study']} noise ceiling = "
                        f"{ceil['ceiling']:.4f}"
                    )
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
