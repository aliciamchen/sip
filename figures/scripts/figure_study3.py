#!/usr/bin/env python3
"""Study 3 results figure (figures/outputs/study3_results.pdf).

The nonfood generalization, on the 16 nonfood scenarios: row (a), Study 3a
mirrors 1b (joint desire + effort, relationship given, so color = relationship);
row (b), Study 3b mirrors 2b (joint intimacy + effort, desire given, so color =
desire). Both are joint studies, so each panel carries two latents distinguished
by marker shape (circle = desire, square = intimacy, triangle = effort). Columns
are the ablations' out-of-sample LOSO-CV predictions (Base | Discomfort-only |
Full) plus Humans with 95% subject-cluster bootstrap CIs.

Points-by-action design shared with the other per-study results figures and the
poster set (see `_points.py`). The two rows carry different given conditions, so
each gets its own color legend and its own y scale.

Usage:
    uv run python figures/scripts/figure_study3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import apply_style  # noqa: E402

import _points as points  # noqa: E402

SLUG_A = "nonfood_inv_joint_de"  # Study 3a
SLUG_B = "nonfood_inv_joint_ie"  # Study 3b


def main():
    apply_style("si")
    points.render_paper_figure([SLUG_A, SLUG_B], "study3_results")


if __name__ == "__main__":
    main()
