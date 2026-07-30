#!/usr/bin/env python3
"""Study 2 results figure (figures/outputs/study2_results.pdf).

Relationship inference from an observed action, with desire given -- the reverse
direction from Study 1. Row (a), Study 2a: belief update in intimacy per
observed action x given desire x given effort cell, where the effort of the
low-risk share is also given (filled = low effort, open = high). Row (b), Study
2b: joint inference over intimacy and that effort, so both latents appear in one
panel by marker shape (square = intimacy, triangle = effort). Columns are the
ablations' out-of-sample LOSO-CV predictions (Base | Discomfort-only | Full)
plus Humans with 95% subject-cluster bootstrap CIs.

Points-by-action design shared with the other per-study results figures and the
poster set (see `_points.py`). The two rows keep independent y scales, since 2b
puts two latents on one axis.

Usage:
    uv run python figures/scripts/figure_study2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import apply_style  # noqa: E402

import _points as points  # noqa: E402

SLUG_A = "food_inv_intimacy"  # Study 2a
SLUG_B = "food_inv_joint_ie"  # Study 2b


def main():
    apply_style("si")
    points.render_paper_figure([SLUG_A, SLUG_B], "study2_results")


if __name__ == "__main__":
    main()
