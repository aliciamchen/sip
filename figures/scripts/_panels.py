"""Shared labels, line styles, and legend swatches for the results figures.

The per-study results figures draw their panels through `_points.py` (points by
observed action); this module holds the pieces those panels and the correlation
figures share -- the action axis labels, the zero/identity guide lines, the
condition level orders and labels, and the patch-swatch legend handles the
poster's standalone legend files use. Palettes themselves stay in `plot_style`
(the visual source of truth), keyed by the same condition names.

The bar-panel, dodged-line, and 2D joint-vector panel functions that used to
live here were removed along with `_joint.py` when the figures moved to the
points design; `git log` has them if that layout is ever wanted back.
"""

import sys
from pathlib import Path

from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import (  # noqa: E402
    ACTION_LABELS,
    DESIRE_COLORS,
    INTIMACY_COLORS,
    INTIMACY_LABELS,
    INTIMACY_LEVELS,
    OBSERVED_ACTIONS,
)

# Two-line action labels, so the x tick text stays horizontal across a panel row.
ACTION_AXIS_LABELS = [ACTION_LABELS[a].replace(" ", "\n", 1) for a in OBSERVED_ACTIONS]

DESIRE_LEVELS = ["low", "high"]
DESIRE_LABELS = {"low": "Low desire", "high": "High desire"}
EFFORT_LABELS = {"low": "Low effort", "high": "High effort"}

ZERO_LINE = dict(color="0.75", linestyle=(0, (4, 3)), linewidth=0.8, zorder=0)
IDENTITY_LINE = dict(color="0.75", linestyle=(0, (4, 3)), linewidth=0.8, zorder=0)


# ------------------------------------------------------------------ legends


def intimacy_handles():
    """Filled swatches for the relationship palette, in level order."""
    return [
        Patch(
            facecolor=INTIMACY_COLORS[lvl],
            edgecolor="white",
            linewidth=0.4,
            label=INTIMACY_LABELS[lvl],
        )
        for lvl in INTIMACY_LEVELS
    ]


def desire_handles():
    """Filled swatches for the given-desire palette, in level order."""
    return [
        Patch(
            facecolor=DESIRE_COLORS[lvl],
            edgecolor="white",
            linewidth=0.4,
            label=DESIRE_LABELS[lvl],
        )
        for lvl in DESIRE_LEVELS
    ]
