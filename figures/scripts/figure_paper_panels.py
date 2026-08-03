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
# Markers are smaller than the poster's so the human CIs stay visible: the
# whiskers are drawn behind the marker in its own colour with no caps, so a
# marker wider than its CI would swallow them (at markersize 10 it did; 9.5
# still clears them, checked against the Humans columns).
STYLE = replace(
    points.POSTER,
    panel_w=2.9,
    panel_h=3.3,
    markersize=9.5,
    tick_len=4.5,
    tick_w=1.4,
    xtick_fs=12,
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


def draw_scatter_panel():
    """Model-vs-humans scatter pooling all six studies, on its own artboard.

    One point per (study x condition x latent): the model's out-of-sample LOSO-CV
    prediction on x against the human condition mean on y, with the human 95%
    bootstrap CI, one column per ablation and a pooled Pearson r per column.
    Marker shape is the inferred latent, so `legend_target` covers it -- the
    legend is left off the artboard like every other panel here.

    Aggregation and rendering live in `_agg`, which is also where the pooled
    bootstrap r is computed, so the panel and the reported correlation cannot
    disagree.
    """
    agg, agg_cis = corr.agg_points()
    if not any(agg.values()):
        print("[scatter] no CV predictions yet — skipped")
        return
    vals = np.concatenate(
        [
            arr
            for m in points.data.MODEL_ORDER
            for _dv, x, y, ylo, yhi in agg[m]
            for arr in (x, y, ylo, yhi)
        ]
    )
    lim_hi = float(np.nanmax(np.abs(vals))) * 1.05
    lim = (-lim_hi, lim_hi)

    keys = points.data.MODEL_ORDER
    fig, axes = plt.subplots(
        1,
        len(keys),
        figsize=(STYLE.panel_w * len(keys), STYLE.panel_h),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, model in zip(axes, keys):
        corr.draw_agg_panel(ax, agg[model], lim, agg_cis.get(model))
        ax.set_title(points.data.MODEL_LABELS[model], fontsize=STYLE.title_fs)
        ax.tick_params(labelsize=STYLE.tick_fs)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    axes[0].set_ylabel("Human belief update", fontsize=STYLE.label_fs)
    axes[len(keys) // 2].set_xlabel(
        "Model predicted belief update", fontsize=STYLE.label_fs
    )
    _save(fig, "panel_model_vs_humans")


def draw_legend(handles, stem, title, ncol=None):
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
        columnspacing=1.5,
        handletextpad=0.5,
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
        handles, title = points.condition_point_handles(condition, STYLE)
        draw_legend(handles, stem, title, ncol=2 if len(handles) > 2 else 1)
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
        draw_scatter_panel()
        n_legends = draw_legends()
    print(f"\n{len(drawn)} panel(s) + {n_legends} legend(s) in {OUT_DIR}")


if __name__ == "__main__":
    main()
