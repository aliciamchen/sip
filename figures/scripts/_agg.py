#!/usr/bin/env python3
"""Aggregate model-vs-human panel: the pooled scatter and its bootstrap r.

Helper module, not a figure script. `figure_paper_panels.py` calls `agg_points`
and `draw_agg_panel` to build panel_model_vs_humans.
"""

import json
from collections import OrderedDict

import numpy as np

from plot_style import DV_MARKERS
from study_registry import study, study_groups

import _data as data
import _panels as panels
import _points as points

# Aggregate poster figure: DV -> marker shape (from plot_style, shared with the
# points panels). Point color is the given condition, per study, so the two
# encodings match the results panels -- `agg_points` resolves it there.
# Markers and their human-CI error bars are fully opaque, so a crowded cluster
# cannot darken into a blob: overlap is shown by the white seam each marker
# carries rather than by accumulated tint. The seam is thinner than the points
# panels' so the faces stay dominant at this marker size, and the bars stay a
# neutral gray: they are uncertainty about the estimate, and at this density
# coloring them too turns a crowded region into hatching.
AGG_POINT_MS = 7.0
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

    Restricting `slugs` cannot perturb the error bars of the studies that remain:
    each study's resampling seed is derived from its own slug, so a
    per-study-number panel reuses exactly the draws the pooled panel used for
    those studies. The r interval DOES change with `slugs`, by design -- it is
    bootstrapped over the points the panel plots, and a per-study panel plots
    fewer of them.
    """
    out = {m: [] for m in data.MODEL_ORDER}
    model_x = {m: [] for m in data.MODEL_ORDER}  # per-model predictions
    obs_y = {m: [] for m in data.MODEL_ORDER}  # observed human means
    contributed = set()
    for _stem, _name, members in STUDY_GROUPS:
        for slug, _paper in members:
            if slugs is not None and slug not in slugs:
                continue
            trials = data.load_trials(slug)
            preds = data.load_cv_preds(slug)
            if trials is None or preds is None:
                continue
            data.warn_if_stale(slug, trials, data.load_comparison(slug))
            contributed.add(study(slug).number)
            spec = data.STUDY_SPECS[slug]
            keys = data.condition_cols(slug)
            trials = trials.copy()
            trials["action_label"] = data.action_label_col(trials)
            preds = preds.copy()
            preds["action_label"] = data.action_label_col(preds)
            # Subject-cluster bootstrap for the human error bars. One draw per
            # study, shared by its DVs. The r interval no longer reads these
            # resamples (see `corr_with_pair_ci`), but the y error bars still do,
            # and the bootstrap IS unbiased for a cell mean.
            cells = data.bootstrap_cell_means(
                trials,
                [u for u, _d, _dv in spec["dvs"]],
                keys,
                n_boot=N_BOOT_AGG,
                seed=data.seed_for(f"figures:agg:{slug}"),
            )
            # Point color is the study's own given condition, the same axis its
            # results panel colors by (`_points.fill_spec`) -- relationship
            # where relationship is given, desire otherwise. A panel pooling
            # studies from both families therefore carries both palettes; they
            # are far enough apart in hue to read as two axes rather than one.
            color_col, _levels, palette, _title = points.fill_spec(slug)
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
                            merged[color_col].map(palette).to_numpy(),
                        )
                    )
                    model_x[model].append(merged[delta_col].to_numpy())
                    obs_y[model].append(merged[update_col].to_numpy())
    # One paper study number -> that group's seed key, so the panel annotation
    # and the manuscript's `\rStudyOne`-family macros are the same numbers.
    number = contributed.pop() if len(contributed) == 1 else None
    return out, _agg_correlations(model_x, obs_y, number)


def _group_corr_seed(number):
    """The seed recorded in group_correlations.json for this study group, so the
    panel annotation reuses exactly the bootstrap stream the manuscript quotes.
    0 — model_comparison's default — when the artifact or the entry is missing."""
    path = data.get_project_root() / "model" / "outputs" / "group_correlations.json"
    if not path.exists():
        return 0
    with open(path) as f:
        for entry in json.load(f):
            if entry.get("study") == number:
                return entry.get("seed", 0)
    return 0


def _agg_correlations(model_x, obs_y, number=None):
    """{model: (r, lo, hi)} -- the correlation pooled over the panel's studies and
    DVs, with the interval bootstrapped over the plotted points.

    Both the estimate and its interval come from exactly the points the panel
    draws, which is what makes the annotation readable off the panel. When the
    panel covers one paper study number, `number` names it and the bootstrap runs
    on that group's seed key, so the annotation is byte-identical to the
    `group_correlations.json` entry the manuscript quotes."""
    cis = {}
    for model in data.MODEL_ORDER:
        if not obs_y[model]:
            continue
        seed_key = (
            data._mc.group_corr_seed_key(number, model, _group_corr_seed(number))
            if number is not None
            else f"figures:agg:pooled|{model}|pair_ci"
        )
        cis[model] = corr_with_pair_ci(
            np.concatenate(model_x[model]),
            np.concatenate(obs_y[model]),
            seed_key=seed_key,
        )
    return cis


def corr_with_pair_ci(x, y, *, n_boot=None, seed_key="figures:agg:pair_ci"):
    """(r, lo, hi) with the interval bootstrapped over the PLOTTED POINTS.

    Delegates to `model_comparison.pair_bootstrap_corr`, which is the single
    implementation behind every correlation the paper reports. It kept its own
    copy until 2026-08-16, and the copies disagreed: same point estimates, but
    Study 2's vanilla interval printed [0.20, 0.70] on the panel against
    [0.23, 0.69] in the artifact, because the two sides concatenate their cells
    in different orders and a bootstrap draws indices. `pair_bootstrap_corr` now
    sorts its points canonically, so passing the same seed key gives the same
    interval.

    Why the pair bootstrap at all: resampling the (x, y) pairs is the textbook
    bootstrap for a correlation, and what this literature reports beside an r.
    Inverting the published intervals of six comparable papers gives an effective
    sample size that stays ~constant across correlations spanning r = 0.01 to
    0.93 — the signature of resampling points rather than participants. It
    replaced a subject-cluster
    interval that was *mislocated*: a resample holds ~63% unique participants, so
    its cell means carry extra noise, noise in y attenuates r against a fixed x,
    and the whole bootstrap distribution sat below the observed r -- for the
    pooled panel the percentile interval excluded its own point estimate.

    Note what this interval therefore means: how far r would move with a different
    sample of *condition cells*, not with a different sample of *participants*.
    The per-cell participant uncertainty is carried by the y error bars, and the
    ceiling on r set by that noise is reported separately (`noise_ceilings` in
    each study's cv_model_comparison.json).
    """
    got = data._mc.pair_bootstrap_corr(
        x, y, seed_key=seed_key, n_boot=n_boot or N_BOOT_AGG
    )
    return (got["r"], got["ci_95"][0], got["ci_95"][1])


def _r_label(ci, x, y):
    """The panel annotation: "r = 0.96" over its bootstrap CI, or the bare
    observed r when no CI was supplied.

    Only the symbol `r` is mathtext; every number is plain text. The two lines used
    to disagree -- `r = -0.18` was inside math, so its sign became a true minus and
    its digits the math font, while the interval below it was text and got a
    hyphen -- which showed up as two different minus signs stacked once r went
    negative. Keeping the numbers in text puts hyphens on both lines and matches
    the tick labels, which are text too.
    """
    if ci is None or not np.isfinite(ci[0]):
        # An ablation with no term for the latent predicts the same value in
        # every cell, so there is no correlation to report -- printing one (or a
        # NaN) beside a column that is visibly a vertical stripe reads as a fit.
        # `n/a` is what tab:scenario-correlations shows for the same cases.
        tol = data._mc.CONSTANT_PREDICTION_TOL
        if np.std(x) < tol or np.std(y) < tol:
            return "$r$ = n/a"
        return f"$r$ = {np.corrcoef(x, y)[0, 1]:.2f}"
    r, lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"$r$ = {r:.2f}"
    return f"$r$ = {r:.2f}\n[{lo:.2f}, {hi:.2f}]"


def draw_agg_panel(ax, groups, lim, ci=None, *, zero_lw=None, style=None):
    """groups: list of (dv, x, y, y_lo, y_hi, colors). DV is encoded by MARKER
    SHAPE and the given condition by Color, the same two encodings the results
    panels use, so a reader carries one key between the rows; vertical human-CI
    error bars sit behind the points; the panel shows a single pooled Pearson r
    over all points.

    Pass the caller's points `style` to draw the points and their CIs exactly as
    the results panels draw theirs -- same marker size, same bar weight, and the
    bar in the point's own color where the style says so. Without it the panel
    falls back to the AGG_* constants, which are the same design at its own
    scale. `zero_lw` likewise matches the results panels' zero rule."""
    ms = style.markersize if style else AGG_POINT_MS
    bar_lw = (
        style.errbar.get("elinewidth", AGG_BAR_LINEWIDTH)
        if style
        else (AGG_BAR_LINEWIDTH)
    )
    bar_from_point = bool(style and style.errbar_from_point)
    by_dv = OrderedDict()
    for dv, x, y, ylo, yhi, colors in groups:
        by_dv.setdefault(dv, []).append((x, y, ylo, yhi, colors))

    all_x, all_y, all_lo, all_hi, all_marker, all_color = [], [], [], [], [], []
    for dv, parts in by_dv.items():
        x = np.concatenate([p[0] for p in parts])
        all_x.append(x)
        all_y.append(np.concatenate([p[1] for p in parts]))
        all_lo.append(np.concatenate([p[2] for p in parts]))
        all_hi.append(np.concatenate([p[3] for p in parts]))
        all_marker.extend([DV_MARKERS[dv]] * len(x))
        all_color.extend(np.concatenate([p[4] for p in parts]))
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
    # Colored bars go down one palette level at a time rather than per point, so
    # the panel carries a handful of artists instead of one per cell; they all
    # share zorder, so grouping them cannot change what covers what.
    all_color = np.asarray(all_color)
    if bar_from_point:
        for color in dict.fromkeys(all_color):
            sel = all_color == color
            ax.errorbar(
                all_x[sel],
                all_y[sel],
                yerr=yerr[:, sel],
                fmt="none",
                ecolor=color,
                elinewidth=bar_lw,
                zorder=1,
            )
    else:
        ax.errorbar(
            all_x,
            all_y,
            yerr=yerr,
            fmt="none",
            ecolor=AGG_BAR_COLOR,
            elinewidth=bar_lw,
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
            markersize=ms,
            markerfacecolor=all_color[i],
            markeredgecolor=AGG_POINT_EDGE,
            markeredgewidth=AGG_POINT_EDGEWIDTH,
            zorder=3,
        )

    # The zero rules take the identity line's dashed style: all three are
    # reference geometry rather than data, and a solid rule reads as a series.
    # `zero_lw` matches whatever the caller's points style uses for its own zero
    # rule (`_points.draw_points` does the same), so these panels and the results
    # panels don't sit side by side in one figure with reference lines of two
    # different weights.
    ident = dict(panels.IDENTITY_LINE)
    zero = dict(panels.ZERO_LINE)
    if zero_lw:
        ident["linewidth"] = zero["linewidth"] = zero_lw
    ax.plot(lim, lim, **ident)
    ax.axhline(0, **zero)
    ax.axvline(0, **zero)
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
