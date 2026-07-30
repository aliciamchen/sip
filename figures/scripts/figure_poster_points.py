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
from plot_style import DV_MARKERS, apply_style, savefig  # noqa: E402
from study_registry import studies  # noqa: E402

import _panels as panels  # noqa: E402
import _points as points  # noqa: E402

# Not referenced directly any more (the shared renderer reads it), but kept in
# this module's namespace on purpose: figure_gated_points.py swaps the panel set
# by patching `poster.data.load_cv_preds` and the MODEL_ORDER/PANEL_LABELS
# tables on it. It is the same module object `_points` holds, so the patch still
# reaches the renderer.
import _data as data  # noqa: E402,F401

# (slug, paper label, output stem), in paper order, from the study registry.
STUDIES = [(s.slug, s.short_label, f"study{s.short_label}") for s in studies()]

# Panel scale and marker/CI weights come from the shared poster style; only the
# standalone legend files are drawn here, so those two sizes stay local.
STYLE = points.POSTER
LEGEND_FS = STYLE.legend_fs
MARKERSIZE = STYLE.markersize


def _build(slug):
    """(human cells with CIs, model cells) on the condition grid, via the shared
    builder.

    Keeps the poster's own bootstrap seed rather than the shared one: the CI
    extremes set each panel's y limit, so re-seeding would rescale every
    already-produced poster panel for no visible gain (the whiskers sit behind
    the poster's large markers).
    """
    return points.build_cells(slug, seed_tag="figures:poster")


def build_study(slug, paper, stem):
    human, model = _build(slug)
    if human is None and model is None:
        print(f"[{slug}] nothing to draw yet")
        return
    _render_points(slug, stem, human, model)


def _render_points(slug, stem, human, model):
    """Points-by-action panel row (Base | Discomfort-only | Full | Humans) for
    any study, at poster scale. The drawing is the shared renderer in
    `_points.py`; only the artboard size and the output name are poster-specific,
    and the legends live in standalone files (see `_save_legends`) rather than
    being repeated on every panel."""
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
