#!/usr/bin/env python3
"""Study 3 results figure (figures/study3_results.pdf).

The nonfood generalization: Study 3a mirrors 1b (joint desire + effort given
intimacy) and Study 3b mirrors 2b (joint intimacy + effort given desire), on
the 16 nonfood scenarios. Rows: (a) 3a human joint updates in 2D update space,
(b) 3a by-DV grid of model columns + Humans, (c) 3b joint updates, (d) 3b
by-DV grid.

Data collection hasn't started, so today this script prints skip notes and
writes nothing; the figure appears automatically once
data/nonfood_inv_joint_de/ / data/nonfood_inv_joint_ie/ CSVs and their CV
predictions exist.

Usage:
    uv run python figures/results/figure_study3.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import (  # noqa: E402
    DESIRE_COLORS,
    INTIMACY_COLORS,
    INTIMACY_LEVELS,
    apply_style,
    savefig,
)

import _joint as joint  # noqa: E402
import _panels as panels  # noqa: E402

SLUG_A = "nonfood_inv_joint_de"  # Study 3a (mirrors 1b)
SLUG_B = "nonfood_inv_joint_ie"  # Study 3b (mirrors 2b)
DVS_A = [
    ("desire_rating_update", "delta_desire", "Desire"),
    ("effort_rating_update", "delta_effort", "Effort of low-risk share"),
]
DVS_B = [
    ("intimacy_rating_update", "delta_intimacy", "Intimacy"),
    ("effort_rating_update", "delta_effort", "Effort of low-risk share"),
]
DESIRE_LEVELS = ["low", "high"]


def main():
    apply_style("si")
    human_a, model_a = joint.build_joint_cells(
        SLUG_A, DVS_A, ["action_label", "intimacy_condition"]
    )
    human_b, model_b = joint.build_joint_cells(
        SLUG_B, DVS_B, ["action_label", "desire_condition"]
    )
    if human_a is None and human_b is None:
        print("[study3] nothing to draw yet")
        return

    rows = [
        ("a", human_a is not None),
        ("b", human_a is not None),
        ("c", human_b is not None),
        ("d", human_b is not None),
    ]
    live = [r for r, ok in rows if ok]

    fig = plt.figure(figsize=(6.5, 2.8 * len(live)), layout="constrained")
    subfigs = fig.subfigures(len(live), 1)
    subfigs = [subfigs] if len(live) == 1 else list(subfigs)
    for key, sub in zip(live, subfigs):
        if key == "a":
            joint.draw_vector_row(
                sub,
                human_a,
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
        elif key == "b":
            joint.draw_dv_grid(
                sub,
                human_a,
                model_a,
                dvs=DVS_A,
                color_col="intimacy_condition",
                color_levels=INTIMACY_LEVELS,
                colors=INTIMACY_COLORS,
                letter="b",
            )
        elif key == "c":
            joint.draw_vector_row(
                sub,
                human_b,
                x_col="intimacy_rating_update",
                y_col="effort_rating_update",
                x_label="Intimacy belief update",
                y_label="Effort belief update\n(low-risk share)",
                color_col="desire_condition",
                colors=DESIRE_COLORS,
                color_handles=panels.desire_handles("line"),
                color_title="Desire",
                letter="c",
            )
        else:
            joint.draw_dv_grid(
                sub,
                human_b,
                model_b,
                dvs=DVS_B,
                color_col="desire_condition",
                color_levels=DESIRE_LEVELS,
                colors=DESIRE_COLORS,
                letter="d",
            )

    out = savefig(fig, "study3_results")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
