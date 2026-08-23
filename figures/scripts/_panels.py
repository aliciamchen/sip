"""Shared labels and line styles for the results figures.

The per-study results figures draw their panels through `_points.py` (points by
observed action); this module holds the pieces those panels and the correlation
figures share -- the action axis labels, the zero/identity guide lines, and the
condition level orders and labels. Palettes themselves stay in `plot_style`
(the visual source of truth), keyed by the same condition names.

The legend swatch builders that used to live here went with the poster scripts
they served: the standalone legend artboards are built in `figure_paper_panels.py`
now, from `_points.py`'s handle builders. The bar-panel, dodged-line, and 2D
joint-vector panel functions went with `_joint.py` when the figures moved to the
points design; `git log` has both if they are ever wanted back.
"""

from plot_style import ACTION_LABELS, OBSERVED_ACTIONS

# Two-line action labels, so the x tick text stays horizontal across a panel row.
ACTION_AXIS_LABELS = [ACTION_LABELS[a].replace(" ", "\n", 1) for a in OBSERVED_ACTIONS]

DESIRE_LEVELS = ["low", "high"]
DESIRE_LABELS = {"low": "Low desire", "high": "High desire"}
# "Easy"/"Hard", not "Low"/"High": the action axis already reads "Low-risk
# share" / "High-risk share", so low/high there would name risk and effort
# with the same word in one axis.
EFFORT_LABELS = {"low": "Easy", "high": "Hard"}

# Guide lines: dashed and light enough to sit under the data without competing
# with it -- these mark a reference value (no belief update; model == human),
# not a quantity the reader is meant to read off.
ZERO_LINE = dict(color="0.85", linestyle=(0, (4, 3)), linewidth=0.8, zorder=0)
IDENTITY_LINE = dict(color="0.85", linestyle=(0, (4, 3)), linewidth=0.8, zorder=0)
