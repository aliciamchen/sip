#!/usr/bin/env python3
"""Aggregate model-vs-human panel: the pooled scatter and its bootstrap r.

Helper module, not a figure script. `figure_paper_panels.py` calls `agg_points`
and `draw_agg_panel` to build panel_model_vs_humans; the assembled 3x3
correlation grids this file used to emit were dropped when the paper moved to
Illustrator-assembled panels (see git history for figure_model_corr.py).
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import DV_MARKERS, apply_style, savefig  # noqa: E402
from study_registry import study_groups  # noqa: E402

import _data as data  # noqa: E402
import _panels as panels  # noqa: E402

# Same construct colors as figure_model_scatter.py (dark anchor per DV). Used by
# the per-study grids; the poster aggregate encodes DV by MARKER SHAPE instead.
DV_COLORS = {"desire": "#7A4A5A", "effort": "#4A7A4A", "intimacy": "#274D77"}
DV_LABELS = {"desire": "Desire", "effort": "Effort", "intimacy": "Intimacy"}
DV_LEGEND_ORDER = ["desire", "intimacy", "effort"]
# Aggregate poster figure: DV -> marker shape (from plot_style, shared with the
# points panels), one point color for all.
AGG_POINT_COLOR = "#333333"
# Markers and their human-CI error bars are fully opaque, so a crowded cluster
# cannot darken into a blob: overlap is shown by the white seam each marker
# carries rather than by accumulated tint. The seam is deliberately thinner than
# the 0.5 the points panels use — at 0.7pt on a 4.6pt marker it ate enough of the
# face that a cluster read as mostly white. The bars are one step lighter than
# the marker faces so a long bar does not out-weigh the estimate it belongs to.
AGG_POINT_MS = 5.2
AGG_POINT_EDGE = "white"
AGG_POINT_EDGEWIDTH = 0.2
AGG_BAR_COLOR = "#5A5A5A"
AGG_BAR_LINEWIDTH = 1.2
# Size of the r / CI annotation on the aggregate panels. Both lines share it;
# larger and the bracketed interval runs over the x = 0 column of points.
R_LABEL_FS = 13

# (figure stem, study title, [(slug, paper label)]); columns are the ablations.
# Derived from the study registry rather than restating the roster.
STUDY_GROUPS = [
    (
        f"study{number}",
        f"Study {number}",
        [(s.slug, s.short_label) for s in group],
    )
    for number, group in study_groups()
]


N_BOOT_AGG = 1000


def agg_points(slugs=None):
    """({model: [(dv, x, y, y_lo, y_hi), ...]}, {model: (r, lo, hi)}) over the
    given studies at condition level (averaged over the 16 scenarios). `slugs`
    None pools all six; a subset gives the same quantities for one study number.

    y is the human condition mean with its 95% subject-cluster bootstrap CI; the
    model x is the out-of-sample CV delta averaged over scenarios (a point
    estimate, no error bar). The pooled Pearson r carries a CI from the same
    resamples: participants are resampled within each study independently (they
    are different participant pools), every DV of a study reuses that study's
    draw, and r is recomputed over all pooled points per resample. This mirrors
    model_comparison._secondary_correlation's per-study convention.

    Restricting `slugs` cannot perturb the studies that remain: each study's
    resampling seed is derived from its own slug, so a per-study-number panel
    reuses exactly the draws the pooled panel used for those studies.
    """
    out = {m: [] for m in data.MODEL_ORDER}
    boot_y = {m: [] for m in data.MODEL_ORDER}  # per-model list of (n_boot, k)
    boot_x = {m: [] for m in data.MODEL_ORDER}  # matching model deltas
    obs_y = {m: [] for m in data.MODEL_ORDER}  # observed human means
    for _stem, _name, members in STUDY_GROUPS:
        for slug, _paper in members:
            if slugs is not None and slug not in slugs:
                continue
            trials = data.load_trials(slug)
            preds = data.load_cv_preds(slug)
            if trials is None or preds is None:
                continue
            data.warn_if_stale(slug, trials, data.load_comparison(slug))
            spec = data.STUDY_SPECS[slug]
            keys = data.condition_cols(slug)
            trials = trials.copy()
            trials["action_label"] = data.action_label_col(trials)
            preds = preds.copy()
            preds["action_label"] = data.action_label_col(preds)
            # One draw per study, shared by its DVs.
            cells, boots = data.bootstrap_cell_means(
                trials,
                [u for u, _d, _dv in spec["dvs"]],
                keys,
                n_boot=N_BOOT_AGG,
                seed=data.seed_for(f"figures:agg:{slug}"),
                return_boots=True,
            )
            for update_col, delta_col, dv in spec["dvs"]:
                for model in data.MODEL_ORDER:
                    pm = (
                        preds[preds["model"] == model]
                        .groupby(keys, as_index=False)[delta_col]
                        .mean()
                    )
                    merged = pm.merge(cells, on=keys, how="inner")
                    out[model].append(
                        (
                            dv,
                            merged[delta_col].to_numpy(),
                            merged[update_col].to_numpy(),
                            merged[f"{update_col}_ci_lower"].to_numpy(),
                            merged[f"{update_col}_ci_upper"].to_numpy(),
                        )
                    )
                    # Align the resampled human means to `merged`'s row order.
                    # `merged` is pm.merge(cells), so the merge must run in that
                    # same direction -- merging cells into pm the other way round
                    # returns rows in `cells` order and silently mispairs x and y.
                    order = pm.merge(cells.reset_index(), on=keys, how="inner")[
                        "index"
                    ].to_numpy()
                    boot_y[model].append(boots[update_col][:, order])
                    boot_x[model].append(merged[delta_col].to_numpy())
                    obs_y[model].append(merged[update_col].to_numpy())
    return out, _agg_correlations(boot_x, boot_y, obs_y)


def _agg_correlations(boot_x, boot_y, obs_y):
    """{model: (r, lo, hi)} -- the correlation pooled over every study and DV,
    from the per-resample pooled human means (see `corr_with_ci`)."""
    cis = {}
    for model in data.MODEL_ORDER:
        if not boot_y[model]:
            continue
        cis[model] = corr_with_ci(
            np.concatenate(boot_x[model]),
            np.concatenate(obs_y[model]),
            np.concatenate(boot_y[model], axis=1),  # (n_boot, n_points)
        )
    return cis


def corr_with_ci(x, y, ys):
    """(r, lo, hi) for the model-vs-human correlation: the observed Pearson r
    over (x, y), with a percentile CI recomputed over the resampled human means
    `ys` (n_boot, n_points). Resamples that lose a cell drop it pairwise.

    An ablation whose predictions have no spread (base or discomfort-only on a
    latent it cannot infer) yields NaN rather than a spurious r.
    """
    if np.std(x) < 1e-12:
        return (np.nan, np.nan, np.nan)
    r = float(np.corrcoef(x, y)[0, 1])
    rs = []
    for b in range(ys.shape[0]):
        ok = ~np.isnan(ys[b])
        if ok.sum() > 2 and np.std(x[ok]) > 1e-12:
            rs.append(np.corrcoef(x[ok], ys[b][ok])[0, 1])
    rs = np.asarray(rs)
    rs = rs[np.isfinite(rs)]
    lo, hi = np.percentile(rs, [2.5, 97.5]) if rs.size else (np.nan, np.nan)
    return (r, float(lo), float(hi))


def _r_label(ci, x, y):
    """The panel annotation: "r = 0.96" over its bootstrap CI, or the bare
    observed r when no CI was supplied."""
    if ci is None or not np.isfinite(ci[0]):
        return f"$r = {np.corrcoef(x, y)[0, 1]:.2f}$"
    r, lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"$r = {r:.2f}$"
    return f"$r = {r:.2f}$\n[{lo:.2f}, {hi:.2f}]"


def draw_agg_panel(ax, groups, lim, ci=None):
    """groups: list of (dv, x, y, y_lo, y_hi). DV is encoded by MARKER SHAPE (one
    point color for all); vertical human-CI error bars sit behind the points; the
    panel shows a single pooled Pearson r over all points. Sized for a poster."""
    by_dv = OrderedDict()
    for dv, x, y, ylo, yhi in groups:
        by_dv.setdefault(dv, []).append((x, y, ylo, yhi))

    all_x, all_y, all_lo, all_hi, all_marker = [], [], [], [], []
    for dv, parts in by_dv.items():
        x = np.concatenate([p[0] for p in parts])
        all_x.append(x)
        all_y.append(np.concatenate([p[1] for p in parts]))
        all_lo.append(np.concatenate([p[2] for p in parts]))
        all_hi.append(np.concatenate([p[3] for p in parts]))
        all_marker.extend([DV_MARKERS[dv]] * len(x))
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    yerr = np.clip(
        np.vstack([all_y - np.concatenate(all_lo), np.concatenate(all_hi) - all_y]),
        0,
        None,
    )

    # Everything is opaque, so overlap is resolved by draw order and by the white
    # border around each marker rather than by tint. Two consequences handled
    # here: the bars go down as one layer beneath every point, and the points are
    # drawn in a deterministic shuffle -- drawn DV by DV, one marker shape would
    # sit systematically on top of another and the panel would misreport which
    # latent occupies a crowded region.
    ax.errorbar(
        all_x,
        all_y,
        yerr=yerr,
        fmt="none",
        ecolor=AGG_BAR_COLOR,
        elinewidth=AGG_BAR_LINEWIDTH,
        zorder=1,
    )
    order = np.random.default_rng(data.seed_for("figures:agg:draw_order")).permutation(
        len(all_x)
    )
    for i in order:
        ax.plot(
            all_x[i],
            all_y[i],
            linestyle="none",
            marker=all_marker[i],
            markersize=AGG_POINT_MS,
            markerfacecolor=AGG_POINT_COLOR,
            markeredgecolor=AGG_POINT_EDGE,
            markeredgewidth=AGG_POINT_EDGEWIDTH,
            zorder=3,
        )

    # The zero rules take the identity line's dashed style: all three are
    # reference geometry rather than data, and a solid rule reads as a series.
    ax.plot(lim, lim, **panels.IDENTITY_LINE)
    ax.axhline(0, **panels.ZERO_LINE)
    ax.axvline(0, **panels.ZERO_LINE)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=13)
    ax.text(
        0.04,
        0.96,
        _r_label(ci, all_x, all_y),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=R_LABEL_FS,
        color="0.1",
        linespacing=1.25,
        # Translucent plate: the CI line reaches the x = 0 column of points in
        # the ablation panels, and this keeps both readable over them.
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.8),
        zorder=6,
    )
