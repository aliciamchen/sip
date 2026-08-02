"""Points-by-action renderer shared by the paper and poster results figures.

This is the design the manuscript's per-study results figures use: one row of
panels per study, columns Base | Discomfort-only | Full | Humans, with

- x = the observed action (no share / low-risk share / high-risk share),
- y = belief update (out-of-sample LOSO-CV cell means for the model columns,
  observed cell means with 95% subject-cluster bootstrap CIs for Humans),
- MARKER SHAPE = which latent is being inferred (circle = desire,
  square = intimacy, triangle = effort), so the joint studies show both of
  their targets in one panel,
- COLOR = the given condition (relationship for 1a/1b/3a, desire for 2a/2b/3b),
- FILL = the given effort condition where effort is given rather than inferred
  (filled = low effort, open = high effort, in 1a and 2a only).

Two styles remain. `PAPER` renders at `si` scale with the legends inline, for a
figure that has to be self-contained (`figure_si_scenarios.py`). `POSTER` is the
larger-marker scale with legends omitted, which `figure_paper_panels.py` uses
for the Illustrator components, since those place the legends as separate
artboards. The name is historical -- it came from the 2026-07 poster, whose
scripts were removed on 2026-08-02. Palettes stay in `plot_style.py`.
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_lowercase

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import (  # noqa: E402
    DESIRE_COLORS,
    DV_MARKERS,
    INTIMACY_COLORS,
    INTIMACY_LABELS,
    INTIMACY_LEVELS,
    OBSERVED_ACTIONS,
    panel_label,
    savefig,
)
from study_registry import study  # noqa: E402

import _data as data  # noqa: E402
import _panels as panels  # noqa: E402

# Shape per inferred latent comes from plot_style (the visual source of truth).
# Note figure_model_scatter.py reads against this convention -- there the shape is
# the sub-study and the colour is the latent.
DV_LEGEND_LABELS = {"desire": "Desire", "intimacy": "Intimacy", "effort": "Effort"}


@dataclass(frozen=True)
class PointsStyle:
    """Scale-dependent drawing parameters. Font sizes of None fall through to
    the active rcParams (the `si` profile for paper figures)."""

    markersize: float
    panel_w: float  # axes width per panel, inches
    panel_h: float  # axes height per panel row, inches
    title_fs: float | None = None
    label_fs: float | None = None
    tick_fs: float | None = None
    legend_fs: float | None = None
    # The two-line action labels sit close together across a 4-panel row, so
    # they get their own (smaller) size than the y ticks.
    xtick_fs: float | None = None
    open_edgewidth: float = 1.1
    filled_edgewidth: float = 0.5
    # Axis furniture. Illustrator-bound panels want these heavier than print.
    # The y pair is opt-in (None -> leave the rcParams alone), because the
    # poster figures predate this module and set only the x geometry.
    tick_len: float = 3.5
    tick_w: float = 0.8
    ytick_len: float | None = None
    ytick_w: float | None = None
    zero_lw: float | None = None  # None -> panels.ZERO_LINE's own width
    # Human CI whiskers, as errorbar() kwargs.
    errbar: dict = field(
        default_factory=lambda: dict(
            ecolor="0.15", elinewidth=0.8, capsize=1.4, capthick=0.8, zorder=5
        )
    )
    # Draw each whisker in its own point's condition colour instead of `errbar`'s
    # ecolor, so the CI reads as that point's stem. Combined with a zorder below
    # the marker this hides the bar inside the marker and shows only the extent,
    # which also keeps the open (high-effort) markers' white fill clean.
    errbar_from_point: bool = False
    dodge_width: float = 0.72  # fraction of an action slot the points span


# Height one legend entry-row adds to the figure, inches (at `si` font sizes).
LEGEND_ROW_H = 0.22
# Entry rows every legend group is laid out over, so their titles align.
LEGEND_ENTRY_ROWS = 2
# Purpose string the human-CI bootstrap is seeded from. Every figure drawn from
# these panels shares it, so the assembled previews and the Illustrator panels
# report the same intervals for the same data instead of two resample vintages.
BOOTSTRAP_SEED_TAG = "figures"
# x axis label for every points figure (the tick labels are the three actions).
X_AXIS_LABEL = "Observed action"

PAPER = PointsStyle(
    markersize=5.5,
    panel_w=1.55,
    panel_h=1.95,
    xtick_fs=7.5,
    ytick_len=3.5,
    ytick_w=0.8,
    errbar=dict(elinewidth=1.5, capsize=0, zorder=2),
    errbar_from_point=True,
)

POSTER = PointsStyle(
    markersize=10,
    panel_w=2.5,
    panel_h=2.5,
    title_fs=18,
    label_fs=16,
    tick_fs=13,
    legend_fs=16,
    open_edgewidth=1.3,
    errbar=dict(ecolor="black", elinewidth=1.0, capsize=0, zorder=2),
)


# --------------------------------------------------------------- study config


# The two given conditions a study can colour its points by:
#   condition column -> (levels in plot order, palette, labels, legend title)
CONDITION_COLOR_AXES = {
    "intimacy_condition": (
        INTIMACY_LEVELS,
        INTIMACY_COLORS,
        INTIMACY_LABELS,
        "Relationship",
    ),
    "desire_condition": (
        panels.DESIRE_LEVELS,
        DESIRE_COLORS,
        panels.DESIRE_LABELS,
        "Desire",
    ),
}


def color_axis(slug):
    """Which given condition this study colours its points by. Relationship
    where it is given (1a/1b/3a), desire otherwise (2a/2b/3b)."""
    given = data.condition_cols(slug)[1:]
    return "intimacy_condition" if "intimacy_condition" in given else "desire_condition"


def fill_spec(slug):
    """(column, levels, colors, legend_title) for the study's colour axis."""
    col = color_axis(slug)
    levels, colors, _labels, title = CONDITION_COLOR_AXES[col]
    return col, levels, colors, title


def style_col(slug):
    """The given effort column when effort is given rather than inferred (1a,
    2a), else None. Encoded as filled vs open markers."""
    return (
        "effort_condition"
        if "effort_condition" in data.condition_cols(slug)[1:]
        else None
    )


def markers_for(slug):
    """Marker shape per inferred DV, in the study's DV order."""
    return [DV_MARKERS[dv.name] for dv in study(slug).dvs]


def build_cells(slug, *, seed_tag=BOOTSTRAP_SEED_TAG):
    """(human cells with bootstrap CIs, model cells) aggregated to the
    condition grid for every DV of the study. Either side is None when its
    inputs are missing, so a figure can render the half that exists."""
    cell_cols = data.condition_cols(slug)
    dvs = data.dvs_display(slug)
    trials = data.load_trials(slug)
    human = None
    if trials is not None:
        trials = trials.assign(action_label=data.action_label_col(trials))
        human = data.bootstrap_cell_means(
            trials,
            [h for h, _d, _l in dvs],
            cell_cols,
            seed=data.seed_for(f"{seed_tag}:{slug}"),
        )
        print(
            f"[{slug}] humans: {trials['subject_id'].nunique()} subjects, "
            f"{len(human)} cells"
        )
    preds = data.load_cv_preds(slug)
    model = None
    if preds is not None:
        preds = preds.assign(action_label=data.action_label_col(preds))
        model = preds.groupby(["model", *cell_cols], as_index=False)[
            [d for _h, d, _l in dvs]
        ].mean()
        data.warn_if_stale(slug, trials, data.load_comparison(slug))
    return human, model


def panel_keys(human, model):
    """Panel column keys present given which inputs loaded."""
    return (data.MODEL_ORDER if model is not None else []) + (
        ["humans"] if human is not None else []
    )


def symmetric_limit(slug, human, model, pad=1.12):
    """Symmetric y limit covering every point and CI end the row will draw."""
    dvs = data.dvs_display(slug)
    vals = []
    if model is not None:
        vals += [model[d].abs().max() for _h, d, _l in dvs]
    if human is not None:
        for h, _d, _l in dvs:
            vals += [
                human[h].abs().max(),
                human[f"{h}_ci_lower"].abs().max(),
                human[f"{h}_ci_upper"].abs().max(),
            ]
    return float(max(vals)) * pad


def value_cols_for(slug, key, human, model):
    """([(column, marker), ...], cells, draw_ci) for one panel column."""
    dvs = data.dvs_display(slug)
    marks = markers_for(slug)
    if key == "humans":
        return [(h, m) for (h, _d, _l), m in zip(dvs, marks)], human, True
    return (
        [(d, m) for (_h, d, _l), m in zip(dvs, marks)],
        model[model["model"] == key],
        False,
    )


# ------------------------------------------------------------------- drawing


def iter_cells(
    cells,
    *,
    value_cols,
    color_col,
    fill_levels,
    style,
    style_col=None,
    style_levels=("low", "high"),
):
    """Yield (x, row, vcol, marker, cond, open_pt) for every cell a panel draws.

    The dodge is: DV, then given condition, then the filled/open style level,
    within each action slot -- so a cell's x is a stable function of its
    condition across panels, and anything overlaid on a panel (the per-scenario
    model dashes, say) lands on exactly the same positions as the points.
    """
    styles = list(style_levels) if style_col else [None]
    n_pts = len(value_cols) * len(fill_levels) * len(styles)
    step = style.dodge_width / n_pts
    for ai, action in enumerate(OBSERVED_ACTIONS):
        for di, (vcol, marker) in enumerate(value_cols):
            for cj, cond in enumerate(fill_levels):
                for si, lvl in enumerate(styles):
                    mask = (cells["action_label"] == action) & (
                        cells[color_col] == cond
                    )
                    if style_col:
                        mask = mask & (cells[style_col] == lvl)
                    row = cells[mask]
                    if row.empty:
                        continue
                    gi = (di * len(fill_levels) + cj) * len(styles) + si
                    x = ai - style.dodge_width / 2 + (gi + 0.5) * step
                    yield (
                        x,
                        row.iloc[0],
                        vcol,
                        marker,
                        cond,
                        (style_col is not None and lvl == styles[-1]),
                    )


def draw_points(
    ax,
    cells,
    *,
    value_cols,
    color_col,
    fill_levels,
    colors,
    lim,
    ci,
    style,
    style_col=None,
    style_levels=("low", "high"),
    xticklabels=True,
):
    """One points panel: marker per cell, optional human CI stems."""
    zero_kw = dict(panels.ZERO_LINE)
    if style.zero_lw:
        zero_kw["linewidth"] = style.zero_lw
    ax.axhline(0, **zero_kw)
    for x, row, vcol, marker, cond, open_pt in iter_cells(
        cells,
        value_cols=value_cols,
        color_col=color_col,
        fill_levels=fill_levels,
        style=style,
        style_col=style_col,
        style_levels=style_levels,
    ):
        y = row[vcol]
        ax.plot(
            x,
            y,
            marker,
            markerfacecolor="white" if open_pt else colors[cond],
            markeredgecolor=colors[cond] if open_pt else "white",
            markeredgewidth=(
                style.open_edgewidth if open_pt else style.filled_edgewidth
            ),
            markersize=style.markersize,
            zorder=3,
        )
        if ci:
            lo, hi = row[f"{vcol}_ci_lower"], row[f"{vcol}_ci_upper"]
            errbar = dict(style.errbar)
            if style.errbar_from_point:
                errbar["ecolor"] = colors[cond]
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", **errbar)
    if xticklabels:
        ax.set_xticks(range(3), panels.ACTION_AXIS_LABELS)
    else:
        ax.set_xticks(range(3), [""] * 3)
    xtick_kw = {"labelsize": style.xtick_fs} if style.xtick_fs else {}
    ax.tick_params(axis="x", length=style.tick_len, width=style.tick_w, **xtick_kw)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-lim, lim)


def draw_row(axes, slug, human, model, *, style, keys, lim, titles, xticklabels=True):
    """Draw one study's panel row into `axes` (already length len(keys))."""
    fcol, flevels, fcolors, _title = fill_spec(slug)
    scol = style_col(slug)
    for ax, key in zip(axes, keys):
        vcols, cells, ci = value_cols_for(slug, key, human, model)
        draw_points(
            ax,
            cells,
            value_cols=vcols,
            color_col=fcol,
            fill_levels=flevels,
            colors=fcolors,
            lim=lim,
            ci=ci,
            style=style,
            style_col=scol,
            xticklabels=xticklabels,
        )
        if titles:
            ax.set_title(data.PANEL_LABELS[key], fontsize=style.title_fs)
        ytick_kw = {"labelsize": style.tick_fs} if style.tick_fs else {}
        if style.ytick_len is not None:
            ytick_kw["length"] = style.ytick_len
        if style.ytick_w is not None:
            ytick_kw["width"] = style.ytick_w
        ax.tick_params(axis="y", **ytick_kw)


def _center_xlabel(bottom_axes, style):
    """Put the x axis label under the centre of the bottom row.

    fig.supxlabel would be laid out in the same band as the outside-lower
    legends and collide with them, so the label rides on a bottom-row Axes:
    on the right edge of the left-of-centre panel when the row has an even
    number of columns, and mid-panel when it is odd.
    """
    n = len(bottom_axes)
    if n % 2:
        bottom_axes[n // 2].set_xlabel(X_AXIS_LABEL, fontsize=style.label_fs)
    else:
        bottom_axes[n // 2 - 1].set_xlabel(
            X_AXIS_LABEL, x=1.0, ha="center", fontsize=style.label_fs
        )


def _group(handles, title):
    """Lay a legend group's entries over LEGEND_ENTRY_ROWS rows."""
    return (handles, title, math.ceil(len(handles) / LEGEND_ENTRY_ROWS))


def legend_rows(legends):
    """Height of the bottom legend strip, in entry-rows: the tallest group's
    entry rows plus its title line."""
    if not legends:
        return 0
    return 1 + max(math.ceil(len(hs) / ncol) for hs, _t, ncol in legends)


def place_legends(fig, legends, style):
    """Lay the groups out along the bottom strip, left to right, titles above
    their entries and aligned across groups."""
    locs = {
        1: ["outside lower left"],
        2: ["outside lower left", "outside lower right"],
        3: ["outside lower left", "outside lower center", "outside lower right"],
    }.get(len(legends))
    if locs is None:  # more groups than slots: stack them all at the left
        locs = ["outside lower left"] * len(legends)
    for (handles, title, ncol), loc in zip(legends, locs):
        fig.legend(
            handles=handles,
            loc=loc,
            ncol=ncol,
            title=title,
            alignment="left",
            borderaxespad=0.2,
            **({"fontsize": style.legend_fs} if style.legend_fs else {}),
        )


def ylabel_for(slug):
    """ "Belief update" alone for the joint studies (two latents share the
    axis), qualified by the latent for the single-DV studies."""
    dvs = study(slug).dvs
    if len(dvs) > 1:
        return "Belief update"
    return f"Belief update ({dvs[0].name})"


# ------------------------------------------------------------------- legends


def _grey_marker(marker, label, style, *, open_pt=False):
    return Line2D(
        [],
        [],
        linestyle="none",
        marker=marker,
        markerfacecolor="white" if open_pt else "0.35",
        markeredgecolor="0.35" if open_pt else "white",
        markeredgewidth=style.open_edgewidth if open_pt else style.filled_edgewidth,
        markersize=style.markersize,
        label=label,
    )


def target_handles(names, style):
    """Shape legend for the inferred latents, in neutral grey so the shapes
    read as shape rather than palette."""
    return [_grey_marker(DV_MARKERS[n], DV_LEGEND_LABELS[n], style) for n in names]


def effort_fill_handles(style):
    """Filled vs open legend for the given effort-of-low-risk-share condition."""
    return [
        _grey_marker("o", panels.EFFORT_LABELS["low"], style),
        _grey_marker("o", panels.EFFORT_LABELS["high"], style, open_pt=True),
    ]


def condition_point_handles(condition, style):
    """Colour legend for one given condition ("intimacy_condition" or
    "desire_condition"), drawn as filled points rather than patches so it
    matches how the panels encode colour. Returns (handles, title)."""
    levels, colors, labels, title = CONDITION_COLOR_AXES[condition]
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor=colors[lvl],
            markeredgecolor="white",
            markeredgewidth=style.filled_edgewidth,
            markersize=style.markersize,
            label=labels[lvl],
        )
        for lvl in levels
    ]
    return handles, title
