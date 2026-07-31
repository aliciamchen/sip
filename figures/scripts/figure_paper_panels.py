#!/usr/bin/env python3
"""Illustrator-bound results panels: one file per sub-study, legends separate.

The manuscript's results figures are assembled by hand in Illustrator, so this
script writes the pieces rather than finished figures. Each of the six studies
gets its own four-column row (Base | Discomfort-only | Full | Humans) at a
square panel aspect, and the legends every row shares are written once as
standalone files for one-time placement:

    figures/paper_panels/panel_study1a.pdf   ... panel_study3b.pdf
    figures/paper_panels/panel_model_vs_humans.pdf
    figures/paper_panels/legend_relationship.pdf
    figures/paper_panels/legend_desire.pdf
    figures/paper_panels/legend_target.pdf
    figures/paper_panels/legend_effort.pdf

Panels keep their column titles, action tick labels and y-axis label, so each
one reads standalone; delete in Illustrator whatever the assembled layout
doesn't need. Text stays editable in Illustrator (`pdf.fonttype: 42`, set in plot_style), and
the axis/tick lines are heavier than print weight so they survive placement and
rescaling.

The points design itself is `_points.py`, shared with the assembled per-study
figures (`figure_study1a.py` and friends), so panels and preview figures cannot
disagree. A PNG is written beside each file purely as a preview.

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
from plot_style import apply_style, savefig  # noqa: E402
from study_registry import studies  # noqa: E402
from utils import get_project_root  # noqa: E402

import figure_model_corr as corr  # noqa: E402
import _points as points  # noqa: E402

OUT_DIR = get_project_root() / "figures" / "paper_panels"

# Square panels at the poster's scale (the aspect that reads best), with heavier
# axis furniture for Illustrator. Fonts are poster-scale: the artboard is ~10 in
# wide, so text lands at print size once the row is scaled to column width.
# Markers are smaller than the poster's so the human CIs stay visible: the
# whiskers are drawn behind the marker in its own colour with no caps, so a
# marker wider than its CI would swallow them (at markersize 10 it did).
STYLE = replace(
    points.POSTER,
    panel_w=2.9,
    panel_h=2.9,
    markersize=8,
    tick_len=4.5,
    tick_w=1.4,
    xtick_fs=12,
    ytick_len=4.5,
    ytick_w=1.4,
    zero_lw=1.3,
    errbar=dict(elinewidth=2.2, capsize=0, zorder=2),
    errbar_from_point=True,
)

PANEL_RC = {
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
}

# (slug, output stem), in paper order, from the study registry.
STUDIES = [(s.slug, f"study{s.short_label}") for s in studies()]


def _save(fig, stem):
    """Write the PDF Illustrator links against, plus a PNG preview.

    PDF only: Illustrator places these as PDFs, and committing a parallel SVG of
    every panel just doubles the repo's figure weight for no use. Text stays
    editable there via `pdf.fonttype: 42`.
    """
    savefig(fig, stem, out_dir=OUT_DIR)
    print(f"wrote {OUT_DIR.name}/{stem}.pdf")


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
        constrained_layout=True,
    )
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
    _save(fig, f"panel_{stem}")
    return True


def draw_scatter_panel():
    """Model-vs-humans scatter pooling all six studies, on its own artboard.

    One point per (study x condition x latent): the model's out-of-sample LOSO-CV
    prediction on x against the human condition mean on y, with the human 95%
    bootstrap CI, one column per ablation and a pooled Pearson r per column.
    Marker shape is the inferred latent, so `legend_target` covers it -- the
    legend is left off the artboard like every other panel here.

    Reuses figure_model_corr's aggregation and panel renderer, so this and
    `model_corr_all_conditions.pdf` cannot disagree about the numbers.
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
    """The four legends the six panels draw from between them: the two given
    condition palettes, the inferred-target shapes, and the filled/open effort
    encoding used where effort is given (1a and 2a)."""
    for condition, stem in (
        ("intimacy_condition", "relationship"),
        ("desire_condition", "desire"),
    ):
        handles, title = points.condition_point_handles(condition, STYLE)
        draw_legend(handles, stem, title, ncol=2 if len(handles) > 2 else 1)
    draw_legend(
        points.target_handles(["desire", "intimacy", "effort"], STYLE),
        "target",
        "Target of inference",
    )
    draw_legend(points.effort_fill_handles(STYLE), "effort", "Effort of low-risk share")


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
        draw_legends()
    print(f"\n{len(drawn)} panel(s) + 4 legends in {OUT_DIR}")


if __name__ == "__main__":
    main()
