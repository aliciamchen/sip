#!/usr/bin/env python3
"""Study 1b results figure (figures/study1b_results.pdf).

Joint inference over desire and effort given intimacy. Panel (a): human joint
belief updates per observed action x given intimacy cell in (desire update,
effort update) space, with 95% subject-cluster bootstrap crosshair CIs. Panel
(b): the same cell means split by inferred variable (rows) with one column per
ablation's out-of-sample predictions plus the Humans column — the
manuscript's explaining-away panel (fig:study1b-results b).

Renders whatever inputs exist: human panels from data/food_inv_joint_de/,
model columns once `make cv-food_inv_joint_de` has produced predictions.

Usage:
    uv run python figures/results/figure_study1b.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import INTIMACY_COLORS, INTIMACY_LEVELS, apply_style, savefig  # noqa: E402

import _data as data  # noqa: E402
import _joint as joint  # noqa: E402
import _panels as panels  # noqa: E402

SLUG = "food_inv_joint_de"
CELL_COLS = data.condition_cols(SLUG)  # ["action_label", "intimacy_condition"]
DVS = data.dvs_display(
    SLUG
)  # [(desire cols, "Desire"), (effort cols, "Effort of low-risk share")]


def main():
    apply_style("si")
    human, model = joint.build_joint_cells(SLUG, DVS, CELL_COLS)
    if human is None:
        print(f"[{SLUG}] nothing to draw yet")
        return

    fig = plt.figure(figsize=(6.5, 5.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.02], hspace=0.02)

    joint.draw_vector_row(
        top,
        human,
        x_col="desire_rating_update",
        y_col="effort_rating_update",
        x_label="Desire belief update",
        y_label="Effort belief update\n(low-risk share)",
        color_col="intimacy_condition",
        colors=INTIMACY_COLORS,
        color_handles=panels.intimacy_line_handles(),
        color_title="Relationship",
        letter="a",
    )
    joint.draw_dv_grid(
        bottom,
        human,
        model,
        dvs=DVS,
        color_col="intimacy_condition",
        color_levels=INTIMACY_LEVELS,
        colors=INTIMACY_COLORS,
        letter="b",
    )

    out = savefig(fig, "study1b_results")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
