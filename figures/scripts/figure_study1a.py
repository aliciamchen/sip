#!/usr/bin/env python3
"""Study 1a results figure (figures/outputs/study1a_results.pdf).

Desire inference from an observed action when the relationship and the physical
effort of low-risk sharing are both given. Out-of-sample model predictions next
to the human data: belief update in desire per observed action x given
relationship x given effort cell, one panel per ablation (Base |
Discomfort-only | Full) plus Humans with 95% subject-cluster bootstrap CIs.
Model points are the LOSO-CV per-cell delta_desire means
(cv_preds_summary.json).

Points-by-action design shared with the other per-study results figures and the
poster set (see `_points.py`): color = given relationship, filled vs open =
given effort of the low-risk share.

Usage:
    uv run python figures/scripts/figure_study1a.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import apply_style  # noqa: E402

import _points as points  # noqa: E402

SLUG = "food_inv_desire"


def main():
    apply_style("si")
    points.render_paper_figure([SLUG], "study1a_results")


if __name__ == "__main__":
    main()
