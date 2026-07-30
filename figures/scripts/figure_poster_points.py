#!/usr/bin/env python3
"""Poster figures: one per experiment, four columns Base | Discomfort-only |
Full | Humans, points = out-of-sample model / human belief update by condition.

x is the observed action; the inferred target(s) are distinguished by MARKER
SHAPE (circle = desire, square = intimacy, triangle = effort); color = the given
relationship/desire condition. The joint studies (1b/2b/3a/3b) show both targets
in one panel by shape; the single-DV studies (1a/2a) additionally encode their
given effort condition as filled (low) vs open (high) markers. Human panels carry
95% subject-cluster bootstrap CIs. Legends are written as standalone files
(poster_legend_*) for one-time placement in the poster layout. Larger fonts than
the paper figures, for a conference poster.

Usage:
    uv run python figures/scripts/figure_poster_points.py [--study <slug>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import (  # noqa: E402
    DESIRE_COLORS,
    DV_MARKERS,
    INTIMACY_COLORS,
    INTIMACY_LEVELS,
    OBSERVED_ACTIONS,
    apply_style,
    savefig,
)
from study_registry import study  # noqa: E402

import _data as data  # noqa: E402
import _panels as panels  # noqa: E402

# (slug, paper label, output stem)
STUDIES = [
    ("food_inv_desire", "1a", "study1a"),
    ("food_inv_joint_de", "1b", "study1b"),
    ("food_inv_intimacy", "2a", "study2a"),
    ("food_inv_joint_ie", "2b", "study2b"),
    ("nonfood_inv_joint_de", "3a", "study3a"),
    ("nonfood_inv_joint_ie", "3b", "study3b"),
]

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 18, 16, 13, 16
MARKERSIZE = 10
# Thicker, capless human-CI whiskers and a taller zero-stub, for poster legibility.
POSTER_ERRBAR = dict(ecolor="black", elinewidth=1.3, capsize=0, zorder=4)
STUB_HALF_HEIGHT = 0.010


def _fill_spec(slug):
    """(fill_col, fill_levels, fill_colors, legend_handles, legend_title) for
    the study's given relationship/desire condition (the bar color axis)."""
    given = data.condition_cols(slug)[1:]
    if "intimacy_condition" in given:
        return (
            "intimacy_condition",
            INTIMACY_LEVELS,
            INTIMACY_COLORS,
            panels.intimacy_handles(),
            "Relationship",
        )
    return (
        "desire_condition",
        panels.DESIRE_LEVELS,
        DESIRE_COLORS,
        panels.desire_handles(),
        "Desire",
    )


def _build(slug):
    """(human cells with CIs, model cells) aggregated to the condition grid for
    every DV of the study; either side None when its inputs are missing."""
    cell_cols = data.condition_cols(slug)
    dvs = data.dvs_display(slug)
    trials = data.load_trials(slug)
    human = None
    if trials is not None:
        trials = trials.assign(action_label=data.action_label_col(trials))
        human = data.bootstrap_cell_means(
            trials,
            [h for h, _d, _l in dvs],
            cell_cols,
            seed=data.seed_for(f"figures:poster:{slug}"),
        )
    preds = data.load_cv_preds(slug)
    model = None
    if preds is not None:
        preds = preds.assign(action_label=data.action_label_col(preds))
        model = preds.groupby(["model", *cell_cols], as_index=False)[
            [d for _h, d, _l in dvs]
        ].mean()
        data.warn_if_stale(slug, trials, data.load_comparison(slug))
    return human, model


def build_study(slug, paper, stem):
    human, model = _build(slug)
    if human is None and model is None:
        print(f"[{slug}] nothing to draw yet")
        return
    _render_points(slug, stem, human, model)


def _points_panel(
    ax,
    cells,
    *,
    value_cols,
    color_col,
    fill_levels,
    colors,
    lim,
    ci,
    style_col=None,
    style_levels=("low", "high"),
):
    """Points panel: x = observed action, y = belief update. Inferred DVs are
    distinguished by MARKER SHAPE (value_cols = [(value_col, marker), ...]); color
    = given condition; a given `style_col` (effort in 1a/2a) is filled (first
    level) vs open (last level); optional vertical CI bars (Humans). Points dodge
    by DV, then condition, then style within each action slot."""
    ax.axhline(0, **panels.ZERO_LINE)
    styles = list(style_levels) if style_col else [None]
    n_pts = len(value_cols) * len(fill_levels) * len(styles)
    total_w = 0.72
    step = total_w / n_pts
    for ai, action in enumerate(OBSERVED_ACTIONS):
        for di, (vcol, marker) in enumerate(value_cols):
            for cj, cond in enumerate(fill_levels):
                for si, style in enumerate(styles):
                    mask = (cells["action_label"] == action) & (
                        cells[color_col] == cond
                    )
                    if style_col:
                        mask = mask & (cells[style_col] == style)
                    row = cells[mask]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    gi = (di * len(fill_levels) + cj) * len(styles) + si
                    x = ai - total_w / 2 + (gi + 0.5) * step
                    y = row[vcol]
                    open_pt = style_col is not None and style == styles[-1]
                    mfc = "white" if open_pt else colors[cond]
                    mec = colors[cond] if open_pt else "white"
                    if ci:
                        lo, hi_ = row[f"{vcol}_ci_lower"], row[f"{vcol}_ci_upper"]
                        ax.errorbar(
                            x,
                            y,
                            yerr=[[y - lo], [hi_ - y]],
                            fmt="none",
                            ecolor="black",
                            elinewidth=1.0,
                            capsize=0,
                            zorder=2,
                        )
                    ax.plot(
                        x,
                        y,
                        marker,
                        markerfacecolor=mfc,
                        markeredgecolor=mec,
                        markeredgewidth=1.3 if open_pt else 0.5,
                        markersize=MARKERSIZE,
                        zorder=3,
                    )
    ax.set_xticks(range(3), panels.ACTION_AXIS_LABELS)
    ax.tick_params(axis="x", length=3.5, width=0.8)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-lim, lim)


def _render_points(slug, stem, human, model):
    """Points-by-action panel row (Base | Discomfort-only | Full | Humans) for
    any study. Marker shape = inferred target; color = given relationship/desire;
    for the single-DV studies (1a/2a) the given effort condition is filled (low)
    vs open (high)."""
    dvs = data.dvs_display(slug)  # one entry (1a/2a) or two (joint)
    fill_col, fill_levels, fill_colors, fill_handles, _ft = _fill_spec(slug)
    style_col = (
        "effort_condition"
        if "effort_condition" in data.condition_cols(slug)[1:]
        else None
    )
    panel_keys = (data.MODEL_ORDER if model is not None else []) + (
        ["humans"] if human is not None else []
    )
    # Shape per inferred latent, keyed off the registry's DV names rather than
    # parsed out of the display label (which a label edit would silently break).
    markers = [DV_MARKERS[dv.name] for dv in study(slug).dvs]

    vals = []
    if model is not None:
        vals += [model[d].abs().max() for _h, d, _l in dvs]
    if human is not None:
        for h, _d, _l in dvs:
            vals += [
                human[h].abs().max(),
                human[f"{h}_ci_lower"].abs().max(),
                human[f"{h}_ci_upper"].abs().max(),
            ]
    lim = float(max(vals)) * 1.12

    fig, axes = plt.subplots(
        1,
        len(panel_keys),
        figsize=(2.5 * len(panel_keys), 2.5),
        sharey=True,
        constrained_layout=True,
    )
    axes = list(axes)
    for ax, key in zip(axes, panel_keys):
        if key == "humans":
            value_cols = [(h, m) for (h, _d, _l), m in zip(dvs, markers)]
            cells, ci = human, True
        else:
            value_cols = [(d, m) for (_h, d, _l), m in zip(dvs, markers)]
            cells, ci = model[model["model"] == key], False
        _points_panel(
            ax,
            cells,
            value_cols=value_cols,
            color_col=fill_col,
            fill_levels=fill_levels,
            colors=fill_colors,
            lim=lim,
            ci=ci,
            style_col=style_col,
        )
        ax.set_title(data.PANEL_LABELS[key], fontsize=TITLE_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
    ylab = (
        "Belief update"
        if len(dvs) > 1
        else f"Belief update ({dvs[0][2].split()[0].lower()})"
    )
    axes[0].set_ylabel(ylab, fontsize=LABEL_FS)

    # Legends are rendered as standalone files (see _save_legends) so they can be
    # placed once in the poster layout rather than repeated on every panel.
    out = savefig(fig, f"poster_points_{stem}")
    print(f"wrote {out}")


def _save_legend(handles, stem, title, ncol=None):
    """Render one legend to its own tight-cropped file for Illustrator assembly."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=ncol or len(handles),
        title=title,
        frameon=False,
        fontsize=LEGEND_FS,
        title_fontsize=LEGEND_FS,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    out = savefig(fig, f"poster_legend_{stem}", png=True)
    print(f"wrote {out}")


def _save_legends():
    """The four standalone legends the poster panels share (placed once, not
    repeated per figure)."""
    _save_legend(panels.intimacy_handles(), "relationship", "Relationship")
    _save_legend(panels.desire_handles(), "desire", "Desire")
    target = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=DV_MARKERS[k],
            color="0.3",
            markersize=MARKERSIZE + 1,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=lbl,
        )
        for k, lbl in (
            ("desire", "Desire"),
            ("intimacy", "Intimacy"),
            ("effort", "Effort"),
        )
    ]
    _save_legend(target, "target", "Target of inference")
    effort = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="0.3",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=MARKERSIZE + 1,
            label="Low effort",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="0.3",
            markeredgewidth=1.3,
            markersize=MARKERSIZE + 1,
            label="High effort",
        ),
    ]
    _save_legend(effort, "effort", "Effort of low-risk share")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", help="Render one slug only (default: all six).")
    args = ap.parse_args()
    apply_style("si")
    for slug, paper, stem in STUDIES:
        if args.study and slug != args.study:
            continue
        build_study(slug, paper, stem)
    _save_legends()


if __name__ == "__main__":
    main()
