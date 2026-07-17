"""Reusable Axes-level panel functions for the main results figures.

Each panel draws one Axes from a tidy cell-level DataFrame; the per-figure
scripts (figure_study*.py) assemble panels into the paper layouts. Palettes and
labels come from plot_style (the visual source of truth). Shared conventions:

- x axis of condition panels: the three observed actions, two-line labels.
- effort is encoded as bar hatching (solid = low, white stripes = high),
  the matplotlib port of the R figures' ggpattern stripes.
- observed action as marker shape: no share = circle, low-risk = triangle,
  high-risk = square (the R figures' shapes 16/17/15).
- CIs are 95% subject-cluster bootstrap intervals, drawn only where the
  DataFrame carries <col>_ci_lower / <col>_ci_upper (the Humans panels).
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
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

ACTION_AXIS_LABELS = [ACTION_LABELS[a].replace(" ", "\n", 1) for a in OBSERVED_ACTIONS]
ACTION_MARKERS = {"no_share": "o", "low_risk_share": "^", "high_risk_share": "s"}

DESIRE_LEVELS = ["low", "high"]
DESIRE_LABELS = {"low": "Low desire", "high": "High desire"}
EFFORT_LEVELS = ["low", "high"]
EFFORT_LABELS = {"low": "Low effort", "high": "High effort"}

HATCH_HIGH_EFFORT = "//////"
ZERO_LINE = dict(color="0.75", linestyle=(0, (4, 3)), linewidth=0.8, zorder=0)
IDENTITY_LINE = dict(color="0.75", linestyle=(0, (4, 3)), linewidth=0.8, zorder=0)
ERRBAR_KW = dict(ecolor="0.15", elinewidth=0.9, capsize=1.6, capthick=0.9, zorder=4)

# Near-zero model predictions (the discomfort-only desire panels) would render
# as invisible zero-height bars; draw them as a thin stub straddling zero
# instead (the R figure's geom_tile trick). Same threshold as the qmd.
ZERO_STUB_THRESHOLD = 0.012
ZERO_STUB_HALF_HEIGHT = 0.006


def grouped_bars(
    ax,
    cells,
    *,
    value_col,
    fill_col,
    fill_levels,
    fill_colors,
    hatch_col="effort_condition",
    hatch_levels=EFFORT_LEVELS,
    action_col="action_label",
    ci=False,
    zero_stub=False,
):
    """Dodged bar panel: three observed actions on x, one bar per
    fill level x hatch level, optional bootstrap-CI whiskers."""
    ax.axhline(0, **ZERO_LINE)
    n_groups = len(fill_levels) * len(hatch_levels)
    total_w = 0.85
    bar_w = total_w / n_groups
    for ai, action in enumerate(OBSERVED_ACTIONS):
        for fi, fill in enumerate(fill_levels):
            for hi, hatch_lvl in enumerate(hatch_levels):
                # effort is the outer dodge block (all solid bars, then all
                # striped), matching the R original's interaction() order
                gi = hi * len(fill_levels) + fi
                row = cells[
                    (cells[action_col] == action)
                    & (cells[fill_col] == fill)
                    & (cells[hatch_col] == hatch_lvl)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]
                x = ai - total_w / 2 + (gi + 0.5) * bar_w
                v = row[value_col]
                kw = dict(
                    facecolor=fill_colors[fill],
                    edgecolor="white",
                    linewidth=0.4,
                    hatch=HATCH_HIGH_EFFORT if hatch_lvl == hatch_levels[-1] else None,
                    zorder=2,
                )
                if zero_stub and abs(v) < ZERO_STUB_THRESHOLD:
                    ax.bar(
                        x,
                        2 * ZERO_STUB_HALF_HEIGHT,
                        bar_w,
                        bottom=-ZERO_STUB_HALF_HEIGHT,
                        **kw,
                    )
                else:
                    ax.bar(x, v, bar_w, **kw)
                if ci:
                    lo, hi_ = row[f"{value_col}_ci_lower"], row[f"{value_col}_ci_upper"]
                    ax.errorbar(
                        x, v, yerr=[[v - lo], [hi_ - v]], fmt="none", **ERRBAR_KW
                    )
    ax.set_xticks(range(3), ACTION_AXIS_LABELS)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 2.6)


def dodged_lines(
    ax,
    cells,
    *,
    value_col,
    color_col,
    color_levels,
    colors,
    action_col="action_label",
    ci=False,
    dodge=0.3,
    marker="o",
):
    """Point + line panel: three observed actions on x, one dodged line per
    color level, optional bootstrap-CI whiskers."""
    ax.axhline(0, **ZERO_LINE)
    n = len(color_levels)
    for li, lvl in enumerate(color_levels):
        sub = cells[cells[color_col] == lvl]
        xs, ys, los, his = [], [], [], []
        for ai, action in enumerate(OBSERVED_ACTIONS):
            row = sub[sub[action_col] == action]
            if row.empty:
                continue
            row = row.iloc[0]
            xs.append(ai - dodge / 2 + (li + 0.5) * dodge / n)
            ys.append(row[value_col])
            if ci:
                los.append(row[f"{value_col}_ci_lower"])
                his.append(row[f"{value_col}_ci_upper"])
        c = colors[lvl]
        ax.plot(xs, ys, color=c, linewidth=1.1, zorder=2)
        if ci and los:
            ys_a, los_a, his_a = np.array(ys), np.array(los), np.array(his)
            ax.errorbar(
                xs,
                ys,
                yerr=[ys_a - los_a, his_a - ys_a],
                fmt="none",
                ecolor=c,
                elinewidth=0.9,
                capsize=1.6,
                capthick=0.9,
                alpha=0.85,
                zorder=3,
            )
        ax.plot(
            xs,
            ys,
            marker,
            color=c,
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linestyle="none",
            zorder=4,
        )
    ax.set_xticks(range(3), ACTION_AXIS_LABELS)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.45, 2.45)


def joint_vector(
    ax,
    cells,
    *,
    x_col,
    y_col,
    color_col,
    colors,
    action_col="action_label",
    ci=True,
    lim=None,
):
    """2D belief-update panel: one point per cell in (x update, y update)
    space, segments from the origin, crosshair CIs, equal aspect."""
    if lim is None:
        vals = [cells[x_col], cells[y_col]]
        if ci:
            vals += [
                cells[f"{x_col}_ci_lower"],
                cells[f"{x_col}_ci_upper"],
                cells[f"{y_col}_ci_lower"],
                cells[f"{y_col}_ci_upper"],
            ]
        lim = (
            float(
                np.nanmax(
                    np.abs(np.concatenate([np.asarray(v, dtype=float) for v in vals]))
                )
            )
            * 1.1
        )
    ax.axhline(0, color="0.8", linestyle=":", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.8", linestyle=":", linewidth=0.8, zorder=0)
    for _, row in cells.iterrows():
        c = colors[row[color_col]]
        x, y = row[x_col], row[y_col]
        ax.plot([0, x], [0, y], color=c, alpha=0.55, linewidth=0.9, zorder=1)
        if ci:
            ax.plot(
                [row[f"{x_col}_ci_lower"], row[f"{x_col}_ci_upper"]],
                [y, y],
                color=c,
                alpha=0.7,
                linewidth=0.9,
                zorder=2,
            )
            ax.plot(
                [x, x],
                [row[f"{y_col}_ci_lower"], row[f"{y_col}_ci_upper"]],
                color=c,
                alpha=0.7,
                linewidth=0.9,
                zorder=2,
            )
        ax.plot(
            x,
            y,
            ACTION_MARKERS[row[action_col]],
            color=c,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    return lim


# ------------------------------------------------------------------ legends


def intimacy_handles():
    return [
        Patch(
            facecolor=INTIMACY_COLORS[lvl],
            edgecolor="white",
            linewidth=0.4,
            label=INTIMACY_LABELS[lvl],
        )
        for lvl in INTIMACY_LEVELS
    ]


def intimacy_line_handles():
    return [
        Line2D(
            [],
            [],
            color=INTIMACY_COLORS[lvl],
            marker="o",
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.1,
            label=INTIMACY_LABELS[lvl],
        )
        for lvl in INTIMACY_LEVELS
    ]


def desire_handles(kind="patch"):
    if kind == "patch":
        return [
            Patch(
                facecolor=DESIRE_COLORS[lvl],
                edgecolor="white",
                linewidth=0.4,
                label=DESIRE_LABELS[lvl],
            )
            for lvl in DESIRE_LEVELS
        ]
    return [
        Line2D(
            [],
            [],
            color=DESIRE_COLORS[lvl],
            marker="o",
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.1,
            label=DESIRE_LABELS[lvl],
        )
        for lvl in DESIRE_LEVELS
    ]


def effort_hatch_handles():
    """Solid vs. white-striped swatches for the effort-of-low-risk-share
    encoding, drawn in neutral grey so they read as pattern, not palette."""
    return [
        Patch(
            facecolor="0.35",
            edgecolor="white",
            linewidth=0.4,
            label=EFFORT_LABELS["low"],
        ),
        Patch(
            facecolor="0.35",
            edgecolor="white",
            linewidth=0.4,
            hatch=HATCH_HIGH_EFFORT,
            label=EFFORT_LABELS["high"],
        ),
    ]


def action_marker_handles(color="0.35"):
    return [
        Line2D(
            [],
            [],
            color=color,
            marker=ACTION_MARKERS[a],
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linestyle="none",
            label=ACTION_LABELS[a],
        )
        for a in OBSERVED_ACTIONS
    ]
