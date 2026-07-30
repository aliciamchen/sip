#!/usr/bin/env python3
"""Illustrator-bound results panels: one file per sub-study, legends separate.

The manuscript's results figures are assembled by hand in Illustrator, so this
script writes the pieces rather than finished figures. Each of the six studies
gets its own four-column row (Base | Discomfort-only | Full | Humans) at a
square panel aspect, and the legends every row shares are written once as
standalone files for one-time placement:

    figures/paper_panels/panel_study1a.{svg,pdf}   ... panel_study3b
    figures/paper_panels/legend_relationship.{svg,pdf}
    figures/paper_panels/legend_desire.{svg,pdf}
    figures/paper_panels/legend_target.{svg,pdf}
    figures/paper_panels/legend_effort.{svg,pdf}

Panels keep their column titles, action tick labels and y-axis label, so each
one reads standalone; delete in Illustrator whatever the assembled layout
doesn't need. Text stays editable in both formats (`svg.fonttype: none` and
`pdf.fonttype: 42`, set in plot_style), and the axis/tick lines are heavier than
print weight so they survive placement and rescaling.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import apply_style, savefig  # noqa: E402
from study_registry import studies  # noqa: E402
from utils import get_project_root  # noqa: E402

import _points as points  # noqa: E402

OUT_DIR = get_project_root() / "figures" / "paper_panels"

# Square panels at the poster's scale (the aspect that reads best), with heavier
# axis furniture for Illustrator. Fonts are poster-scale: the artboard is ~10 in
# wide, so text lands at print size once the row is scaled to column width.
# Markers are smaller than the poster's: at markersize 10 a cell's bootstrap CI
# is shorter than the marker radius, so the whiskers vanished inside the point.
STYLE = replace(
    points.POSTER,
    panel_h=2.5,
    markersize=7.5,
    tick_len=4.5,
    tick_w=1.4,
    ytick_len=4.5,
    ytick_w=1.4,
    zero_lw=1.3,
    errbar=dict(ecolor="0.15", elinewidth=1.0, capsize=2.0, capthick=1.0, zorder=5),
)

PANEL_RC = {
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
}

# (slug, output stem), in paper order, from the study registry.
STUDIES = [(s.slug, f"study{s.short_label}") for s in studies()]


def _save(fig, stem):
    """Write the vector pair Illustrator links against, plus a PNG preview."""
    savefig(fig, stem, out_dir=OUT_DIR, formats=("svg", "pdf"))
    print(f"wrote {OUT_DIR.name}/{stem}.svg + .pdf")


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
        draw_legends()
    print(f"\n{len(drawn)} panel(s) + 4 legends in {OUT_DIR}")


if __name__ == "__main__":
    main()
