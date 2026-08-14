#!/usr/bin/env python3
"""Illustrator-bound results panels: one file per sub-study, legends separate.

The manuscript's results figures are assembled by hand in Illustrator, so this
script writes the pieces rather than finished figures. Each of the six studies
gets its own four-column row (Base | Discomfort-only | Full | Humans) at a
square panel aspect, and the legends every row shares are written once as
standalone files for one-time placement:

    figures/panels/results/panel_study1a.pdf ... panel_study3b.pdf
    figures/panels/results/panel_model_vs_humans.pdf
    figures/panels/legends/legend_{relationship,desire,target}.pdf
    figures/panels/legends/legend_target_study{1,2,3}.pdf

Panels keep their column titles, action tick labels and y-axis label, so each
one reads standalone; delete in Illustrator whatever the assembled layout
doesn't need. Text stays editable in Illustrator (`pdf.fonttype: 42`, set in plot_style), and
the axis/tick lines are heavier than print weight so they survive placement and
rescaling.

The points design itself is `_points.py`, shared with the SI per-scenario grids,
so the two cannot disagree. A gitignored PNG is written beside each PDF purely
as a preview.

Usage:
    uv run python figures/scripts/figure_paper_panels.py [--study <slug>]
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import (  # noqa: E402
    PANELS_LEGENDS,
    PANELS_RESULTS,
    apply_style,
    savefig,
)
from study_registry import studies  # noqa: E402
from utils import get_project_root  # noqa: E402

import _agg as corr  # noqa: E402
import _points as points  # noqa: E402

OUT_DIR = PANELS_RESULTS
LEGEND_DIR = PANELS_LEGENDS

# Square panels at the poster's scale (the aspect that reads best), with heavier
# axis furniture for Illustrator. Fonts are poster-scale: the artboard is ~10 in
# wide, so text lands at print size once the row is scaled to column width.
# Markers are smaller than the poster's because they trade against the human
# CIs: the whiskers are drawn behind the marker in its own color with no caps,
# so any CI narrower than the marker disappears under it. Measured over all six
# studies' Humans columns, the share of whisker ends the marker hides runs 37%
# at markersize 9.5, 43% at 10, 51% at 10.5 -- the size below is the last one
# that keeps most of them, so raising it further should come with a change to
# how the CIs are drawn (caps, or a darker whisker) rather than on its own.
STYLE = replace(
    points.POSTER,
    panel_w=2.9,
    panel_h=3.3,
    markersize=10.0,
    tick_len=4.5,
    tick_w=1.4,
    xtick_fs=12.5,
    ytick_len=4.5,
    ytick_w=1.4,
    zero_lw=1.3,
    errbar=dict(elinewidth=2.2, capsize=0, zorder=2),
    errbar_from_point=True,
)

# Axes rectangle shared by every panel, as figure fractions. Sized for the
# busiest decoration set so panels with fewer decorations simply leave the space
# empty rather than growing their plot area.
PANEL_MARGINS = dict(left=0.075, right=0.995, top=0.824, bottom=0.236, wspace=0.08)

# Columns PANEL_MARGINS is written against: the fractions above describe a
# four-column row, so anything sized from them has to know that.
REF_NCOLS = 4

# Marker size for the pooled model-vs-humans panel, which is the only scatter
# that draws all six studies at once (see `draw_scatter_panel`). Every other
# panel here uses STYLE.markersize.
POOLED_POINT_MS = 9.0

# The model-vs-human scatter is laid out FROM the points row rather than tuned on
# its own, so the two stack: `scatter_layout` gives each square box the same
# width, column pitch and left offset as one points column, and lets the canvas
# come out however wide that makes it. Sizing it independently is what left the
# boxes narrower than the panels above them -- the equal aspect squares the box
# to the shorter side, so the scatter's width was really being set by its height.
#
# Only the outward reserves below are the scatter's own, because its labels
# overflow where the points row's do not: the first and last x ticks sit ON the
# axis limits, so their centered labels stick out half a label width past each
# side, and the column title, the x axis label and the rotated y label each need
# a line outside the box. In inches, converted once from the fractions they were
# tuned as (against the fixed 8.70 x 3.30 canvas this replaces), which were
# tightened until no ink landed within 3 px of any canvas edge at 150 dpi --
# these are placed in Illustrator, where content flush to the artboard reads as
# cropped. The left reserve is the points row's, which is the larger of the two.
SCATTER_RESERVE_IN = dict(right=0.218, top=0.380, bottom=0.578)

PANEL_RC = {
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
}

# (slug, output stem), in paper order, from the study registry.
STUDIES = [(s.slug, f"study{s.short_label}") for s in studies()]

TARGET_TITLE = "Target of inference"
#: Canonical order for the inferred-target legend entries: the two continuous
#: latents, then the two-state world state. Fixed here rather than taken from
#: whatever order a study's `dvs` happen to be in, so every legend lists the
#: shapes the same way down the page.
TARGET_ORDER = ["desire", "intimacy", "effort"]


def targets_by_study_number():
    """{study number: its targets of inference, in TARGET_ORDER}.

    The union over the number's sub-studies, because the manuscript assembles one
    results figure per number. Derived from the registry rather than written out,
    so adding a sub-study or changing a DV cannot leave a legend claiming a shape
    the figure no longer draws.
    """
    by_number = {}
    for s in studies():
        by_number.setdefault(s.number, set()).update(dv.name for dv in s.dvs)
    unknown = set().union(*by_number.values()) - set(TARGET_ORDER)
    if unknown:
        raise ValueError(
            f"inferred target(s) {sorted(unknown)} are missing from TARGET_ORDER, "
            "so they would be silently dropped from the per-study legends"
        )
    return {
        number: [n for n in TARGET_ORDER if n in names]
        for number, names in sorted(by_number.items())
    }


def _save(fig, stem):
    """Write the PDF Illustrator links against, plus a PNG preview.

    PDF only: Illustrator places these as PDFs, and committing a parallel SVG of
    every panel just doubles the repo's figure weight for no use. Text stays
    editable there via `pdf.fonttype: 42`.
    """
    # Legends are shared components placed once, so they live beside the panels
    # rather than among them -- one artboard per legend, not per study.
    is_legend = stem.startswith("legend_")
    out = LEGEND_DIR if is_legend else OUT_DIR
    # Panels save the full canvas so their axes boxes land identically across
    # studies and can be stacked without nudging. Legends MUST crop to content:
    # each is drawn on a 0.1x0.1 in figure and relies on the tight bbox to grow
    # to its entries -- saving the canvas gives a 20x20 px stub.
    savefig(fig, stem, out_dir=out, tight=is_legend)
    print(f"wrote {out.parent.name}/{out.name}/{stem}.pdf")


def draw_panel(slug, stem):
    """One study's four-column row, on its own artboard."""
    human, model = points.build_cells(slug)
    if human is None and model is None:
        print(f"[{slug}] nothing to draw yet — skipped")
        return False
    keys = points.panel_keys(human, model)
    fig, axes = plt.subplots(
        1,
        len(keys),
        figsize=(STYLE.panel_w * len(keys), STYLE.panel_h),
        sharey=True,
        squeeze=False,
    )
    # Fixed margins, not constrained_layout: every study's axes box then lands on
    # the same rectangle of the same canvas whether or not it carries a top axis,
    # so the six panels stack without nudging. The reserve is sized for the
    # busiest case (a title clearing a top axis, two-line action labels below).
    fig.subplots_adjust(**PANEL_MARGINS)
    points.draw_row(
        list(axes[0]),
        slug,
        human,
        model,
        style=STYLE,
        keys=keys,
        lim=points.symmetric_limit(slug, human, model),
        titles=True,
    )
    axes[0][0].set_ylabel(points.ylabel_for(slug), fontsize=STYLE.label_fs)
    fig.supxlabel(points.X_AXIS_LABEL, fontsize=STYLE.label_fs, y=0.015)
    _save(fig, f"panel_{stem}")
    return True


def points_column_geometry():
    """(left offset, axes width, inter-column gap) of a points-panel column, in
    inches -- PANEL_MARGINS read back out as absolute lengths."""
    fig_w = STYLE.panel_w * REF_NCOLS
    span = (PANEL_MARGINS["right"] - PANEL_MARGINS["left"]) * fig_w
    axes_w = span / (REF_NCOLS + (REF_NCOLS - 1) * PANEL_MARGINS["wspace"])
    return PANEL_MARGINS["left"] * fig_w, axes_w, axes_w * PANEL_MARGINS["wspace"]


def scatter_layout(ncols):
    """(figsize, margins) for an `ncols`-column scatter row that stacks under a
    points row: same left offset, same box width, same column pitch.

    The axes rectangle is made SQUARE (height = the points row's column width),
    because `set_aspect("equal")` otherwise shrinks the box to the shorter side
    and the width asked for here would not be the width drawn.
    """
    left, axes_w, gap = points_column_geometry()
    fig_w = left + ncols * axes_w + (ncols - 1) * gap + SCATTER_RESERVE_IN["right"]
    fig_h = SCATTER_RESERVE_IN["top"] + axes_w + SCATTER_RESERVE_IN["bottom"]
    return (fig_w, fig_h), dict(
        left=left / fig_w,
        right=1 - SCATTER_RESERVE_IN["right"] / fig_w,
        top=1 - SCATTER_RESERVE_IN["top"] / fig_h,
        bottom=SCATTER_RESERVE_IN["bottom"] / fig_h,
        wspace=PANEL_MARGINS["wspace"],
    )


def visible_ticks(lim, nbins=5):
    """The ticks `MaxNLocator` would draw inside `lim`, as fixed positions."""
    return [t for t in plt.MaxNLocator(nbins).tick_values(*lim) if lim[0] < t < lim[1]]


def draw_scatter_panel(
    stem="panel_model_vs_humans", slugs=None, label="scatter", lim=None
):
    """Model-vs-humans scatter on its own artboard, over `slugs` (all six when
    None).

    One point per (study x condition x latent): the model's out-of-sample LOSO-CV
    prediction on x against the human condition mean on y, with the human 95%
    bootstrap CI, one column per ablation and a pooled Pearson r per column.
    Marker shape is the inferred latent, so `legend_target` covers it (or, for a
    single study number, `legend_target_study<n>`) -- the legend is left off the
    artboard like every other panel here.

    Aggregation and rendering live in `_agg`, which is also where the pooled
    bootstrap r is computed, so the panel and the reported correlation cannot
    disagree.

    Pass `lim` to force the axis range; without it the range is computed from the
    points drawn and returned, so the caller can compute it once on the pooled
    panel (whose points are every study's) and hand the same range to each
    per-study-number panel. They are meant to be stacked into one figure, and
    axes that differed between rows would invite exactly the cross-row comparison
    they would then be misleading about. The cost is that a study with smaller
    updates uses less of its axes; that is the correct trade when the rows sit
    together.
    """
    # The pooled panel draws every study's points into the box a per-study-number
    # panel gives one study's, so it takes a smaller marker than the results
    # panels' -- at the shared size its clusters merge into blocks of color, and
    # it also hides more of the CI bars, which are drawn in the point's color.
    style = replace(STYLE, markersize=POOLED_POINT_MS) if slugs is None else STYLE
    agg, agg_cis = corr.agg_points(slugs)
    if not any(agg.values()):
        print(f"[{label}] no CV predictions yet — skipped")
        return None
    if lim is None:
        vals = np.concatenate(
            [
                arr
                for m in points.data.MODEL_ORDER
                for _dv, x, y, ylo, yhi, _colors in agg[m]
                for arr in (x, y, ylo, yhi)
            ]
        )
        lim_hi = float(np.nanmax(np.abs(vals))) * 1.05
        lim = (-lim_hi, lim_hi)

    keys = points.data.MODEL_ORDER
    figsize, margins = scatter_layout(len(keys))
    fig, axes = plt.subplots(
        1,
        len(keys),
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    # Fixed margins for the same reason `draw_panel` uses them, and one this panel
    # needs even more: these axes are `set_aspect("equal")`, and under
    # constrained_layout the reserved space depends on how wide the tick labels
    # are. A study whose updates span a smaller range gets fewer x-tick labels, so
    # the layout hands it more room, the square box grows to fill it, and it grows
    # UPWARD into the title -- which is exactly how Study 2's column titles came to
    # be clipped while Studies 1 and 3 were fine. A fixed rectangle makes the box
    # identical across studies, so the titles land in the same place and the three
    # artboards stack.
    fig.subplots_adjust(**margins)
    for ax, model in zip(axes, keys):
        corr.draw_agg_panel(
            ax, agg[model], lim, agg_cis.get(model), zero_lw=STYLE.zero_lw, style=style
        )
        ax.set_title(points.data.MODEL_LABELS[model], fontsize=STYLE.title_fs)
        ax.tick_params(labelsize=STYLE.tick_fs)
        # The x axis keeps its full labeled range, at the cost of the two end
        # labels not being centered on their ticks: they sit at the ends of the
        # axis, so centered they hang half their width into the gap between
        # columns -- which is the points row's gap now, not one this row can
        # widen for itself -- and one column's "0.4" lands on the next column's
        # "-0.4". Aligned inward, no label crosses its own box edge. Ticks are
        # fixed rather than left to the locator so the two end labels can be
        # addressed at all.
        ax.set_xticks(visible_ticks(lim))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        xlabels = ax.get_xticklabels()
        xlabels[0].set_horizontalalignment("left")
        xlabels[-1].set_horizontalalignment("right")
    axes[0].set_ylabel("Human belief update", fontsize=STYLE.label_fs)
    # Parallel with the y label ("Human belief update"), so the axes read as the
    # same quantity from two sources rather than as two different quantities.
    axes[len(keys) // 2].set_xlabel("Model belief update", fontsize=STYLE.label_fs)
    _save(fig, stem)
    return lim


def draw_legend(handles, stem, title, ncol=None, **legend_kw):
    """One legend on its own tight-cropped artboard."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=ncol or len(handles),
        title=title,
        frameon=False,
        fontsize=STYLE.legend_fs,
        title_fontsize=STYLE.legend_fs,
        **{"columnspacing": 1.5, "handletextpad": 0.5, **legend_kw},
    )
    _save(fig, f"legend_{stem}")


def draw_legends():
    """The legends the panels draw from between them: the two given-condition
    palettes, and the inferred-target shapes in four versions.

    The target legend comes in a full set plus one per study number. The full set
    is for the pooled model-vs-humans panel, which really does span all three
    targets; each assembled results figure covers a single study number and so
    infers only two of them, unless the number spans both continuous latents
    (Study 3 does). Placing the full legend on a figure that draws two shapes
    leaves the reader looking for a third, so the per-number versions exist to be
    placed instead.

    There is deliberately no world-state legend. The panels that have a given
    world state label both of its levels on their own top axis, so the only thing
    left to say -- what the two states are states of -- goes in the figure
    caption rather than another artboard.
    """
    for condition, stem in (
        ("intimacy_condition", "relationship"),
        ("desire_condition", "desire"),
    ):
        handles, title = points.condition_color_handles(condition, STYLE)
        # A bar swatch fills its whole handle box, where a marker sits centered in
        # one, so the shared handle length and text pad that suit the target
        # legends leave these entries running into their labels. Shortened and
        # padded here rather than in the defaults: only the color legends draw
        # bars.
        # One entry per line, relationship's four included: the levels are
        # ordered (formal to intimate), and a two-column block reads down one
        # column and back up the other, which loses that order.
        draw_legend(
            handles,
            stem,
            title,
            ncol=1,
            handlelength=1.1,
            handletextpad=0.8,
        )
    # One entry per line: the effort entry names a probability, so the entries
    # read as a list rather than a row of unequal-width chips.
    n = 2
    # The full set, for the pooled model-vs-humans panel, which pools all six
    # studies and so really does infer all three.
    draw_legend(
        points.target_handles(TARGET_ORDER, STYLE), "target", TARGET_TITLE, ncol=1
    )
    n += 1
    # Per assembled figure: the manuscript's results figures are one per study
    # NUMBER (study-1-results.pdf pairs 1a with 1b, and so on), and a family
    # infers only two of the three targets unless it spans both continuous
    # latents. Listing a shape the figure never draws invites the reader to hunt
    # for it, so each family gets a legend holding exactly its own targets.
    for number, names in targets_by_study_number().items():
        draw_legend(
            points.target_handles(names, STYLE),
            f"target_study{number}",
            TARGET_TITLE,
            ncol=1,
        )
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", help="Render one slug only (default: all six).")
    args = ap.parse_args()
    apply_style("si")
    with plt.rc_context(PANEL_RC):
        drawn = []
        for slug, stem in STUDIES:
            if args.study and slug != args.study:
                continue
            if draw_panel(slug, stem):
                drawn.append(stem)
        # The pooled scatter, plus one per study number -- the grain the
        # manuscript's results figures are assembled at, so a study's
        # correlation can be reported beside its own panels if we decide to
        # report them separately rather than only pooled.
        # The pooled panel's range spans every study's points, so reuse it for
        # the per-number panels rather than letting each pick its own -- they are
        # stacked into one figure and have to share axes to be comparable.
        shared_lim = draw_scatter_panel()
        for group_stem, name, members in corr.STUDY_GROUPS:
            draw_scatter_panel(
                stem=f"panel_model_vs_humans_{group_stem}",
                slugs=[slug for slug, _paper in members],
                label=name,
                lim=shared_lim,
            )
        n_legends = draw_legends()
    print(f"\n{len(drawn)} panel(s) + {n_legends} legend(s) in {OUT_DIR}")


if __name__ == "__main__":
    main()
