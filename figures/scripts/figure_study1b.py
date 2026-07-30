#!/usr/bin/env python3
"""Study 1b results figure (figures/outputs/study1b_results.pdf).

Joint inference over desire and the physical effort of low-risk sharing, with
the relationship given. A single observed action updates beliefs about both
latents at once, so both are drawn in one panel and distinguished by marker
shape (circle = desire, triangle = effort); color = the given relationship.
Columns are the ablations' out-of-sample LOSO-CV predictions (Base |
Discomfort-only | Full) plus Humans with 95% subject-cluster bootstrap CIs.

Points-by-action design shared with the other per-study results figures and the
poster set (see `_points.py`).

Usage:
    uv run python figures/scripts/figure_study1b.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import apply_style  # noqa: E402

import _points as points  # noqa: E402

SLUG = "food_inv_joint_de"


def main():
    apply_style("si")
    points.render_paper_figure([SLUG], "study1b_results")


if __name__ == "__main__":
    main()
