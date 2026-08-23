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
- the GIVEN EFFORT CONDITION, where effort is given rather than inferred (1a and
  2a), splits each action into two labeled sub-groups on the axis, low then
  high, with each condition's pair joined by a line. It is deliberately NOT a
  marker style: a study that infers effort still has to draw its markers
  somehow, and any style they take then also reads as one of the levels -- which
  is exactly what made 1b look like it was all low-effort when it shared a
  figure with 1a. Fill, size and marker shape all collide this way; the axis and
  the connector do not. The slope of each line is the effort effect.

Two styles remain. `PAPER` renders at `si` scale, for the SI figures that draw
these panels (`figure_si_scenarios.py` and the prereg/prior-posterior set).
`POSTER` is the larger-marker scale `figure_paper_panels.py` uses for the
Illustrator components; the name is historical -- it came from the 2026-07
poster, whose scripts were removed on 2026-08-02. Neither draws a legend into a
panel: the handle builders at the foot of this module serve both consumers, the
Illustrator components as standalone artboards (`figure_paper_panels.py`) and
the SI grids as a band under the figure (`legend_band`). Palettes stay in
`plot_style.py`.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_style import (
    DESIRE_COLORS,
    DV_MARKERS,
    INTIMACY_COLORS,
    INTIMACY_LABELS,
    INTIMACY_LEVELS,
    OBSERVED_ACTIONS,
)
from study_registry import study

import _data as data
import _panels as panels

# Shape per inferred latent comes from plot_style (the visual source of truth).
# The effort entry names the quantity, not the construct: what 1b/2b/3a/3b
# infer is the probability that the world is the harder of the two states,
# which is the same variable 1a/2a put on the axis as Easy | Hard.
DV_LEGEND_LABELS = {
    "desire": "Desire",
    "intimacy": "Intimacy",
    "effort": "P(low-risk share is hard)",
}


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
    # Draw each whisker in its own point's condition color instead of `errbar`'s
    # ecolor, so the CI reads as that point's stem. Combined with a zorder below
    # the marker this hides the bar inside the marker and shows only the extent,
    # which also keeps the open (high-effort) markers' white fill clean.
    errbar_from_point: bool = False
    dodge_width: float = 0.72  # fraction of an action slot the points span
    # Where a study's effort condition is GIVEN, each action splits into two
    # labeled sub-groups (low | high) and the pair is joined by a line. These
    # set that geometry, as fractions of dodge_width: the sub-group centres'
    # separation, and the span the conditions occupy inside one sub-group.
    split_sep_frac: float = 0.60
    split_within_frac: float = 0.34
    split_linewidth: float = 1.5
    split_sublabel_frac: float = 0.85  # state names vs the action labels
    split_sublabel_pad: float = -3.0  # state names' gap above the plot
    # Title height in axes coords, applied to EVERY panel whether or not it has
    # a top axis. Uniform so a stack of panels aligns on the title as well as on
    # the axes box; a study without a top axis just leaves that strip empty.
    title_y: float = 1.06


# Purpose string the human-CI bootstrap is seeded from. Every figure drawn from
# these panels shares it, so the assembled previews and the Illustrator panels
# report the same intervals for the same data instead of two resample vintages.
BOOTSTRAP_SEED_TAG = "figures"
# x axis label for every points figure (the tick labels are the three actions).
X_AXIS_LABEL = "Observed action"

PAPER = PointsStyle(
    markersize=5.8,
    panel_w=1.55,
    panel_h=1.95,
    xtick_fs=7.8,
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


# The two given conditions a study can color its points by:
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
    """Which given condition this study colors its points by. Relationship
    where it is given (1a/1b/3a), desire otherwise (2a/2b/3b)."""
    given = data.condition_cols(slug)[1:]
    return "intimacy_condition" if "intimacy_condition" in given else "desire_condition"


def fill_spec(slug):
    """(column, levels, colors, legend_title) for the study's color axis."""
    col = color_axis(slug)
    levels, colors, _labels, title = CONDITION_COLOR_AXES[col]
    return col, levels, colors, title


def style_col(slug):
    """The given effort column when effort is given rather than inferred (1a,
    2a), else None. Where set, the action splits into two labeled sub-groups
    joined by a line (see `iter_cells`); where None the study just has three
    action groups."""
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
    """Yield (x, row, vcol, marker, cond, level) for every cell a panel draws.

    `level` is the given style level ("low"/"high") or None where the study has
    no such condition.

    Two layouts. Without a style column the dodge is DV, then given condition,
    across the action slot. WITH one -- the given effort condition of 1a and 2a
    -- the action splits into two sub-groups, low then high, each holding the
    full set of conditions, and `draw_points` joins each condition's pair with a
    line. The pair needs that separation to be joinable at all: interleaving the
    two levels puts them one step apart, and a connector then hides behind its
    own markers.

    Either way a cell's x is a stable function of its condition across panels, so
    anything overlaid on a panel lands on exactly the same positions.
    """
    styles = list(style_levels) if style_col else [None]
    n_inner = len(value_cols) * len(fill_levels)
    if style_col:
        sep = style.dodge_width * style.split_sep_frac
        within = style.dodge_width * style.split_within_frac
        step = within / n_inner
    else:
        step = style.dodge_width / n_inner
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
                    gi = di * len(fill_levels) + cj
                    inner = (
                        -(within if style_col else style.dodge_width) / 2
                        + (gi + 0.5) * step
                    )
                    x = ai + inner
                    if style_col:
                        x += (si - (len(styles) - 1) / 2) * sep
                    yield (x, row.iloc[0], vcol, marker, cond, lvl)


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
    state_labels=True,
):
    """One points panel: marker per cell, optional human CI stems.

    `state_labels=False` keeps the split layout and its connectors but omits the
    secondary top axis naming the two states. Needed by the per-scenario SI
    grids: 16 facets x 3 actions would repeat "Easy Hard" 48 times per figure,
    at a size where it is unreadable anyway, so those name the states in the
    caption instead.
    """
    zero_kw = dict(panels.ZERO_LINE)
    if style.zero_lw:
        zero_kw["linewidth"] = style.zero_lw
    ax.axhline(0, **zero_kw)
    # Collected so each given-condition pair can be joined after the fact; the
    # line is what marks the pairing now that fill no longer does.
    pairs = {}
    for x, row, vcol, marker, cond, level in iter_cells(
        cells,
        value_cols=value_cols,
        color_col=color_col,
        fill_levels=fill_levels,
        style=style,
        style_col=style_col,
        style_levels=style_levels,
    ):
        y = row[vcol]
        if level is not None:
            # Keyed by action too: a (vcol, cond) bucket spans all three action
            # slots, and joining across them would draw one line through the
            # whole panel instead of one per action.
            pairs.setdefault((row["action_label"], vcol, cond), []).append(
                (x, y, level)
            )
        ax.plot(
            x,
            y,
            marker,
            markersize=style.markersize,
            zorder=3,
            **marker_fill(colors[cond], style),
        )
        if ci:
            lo, hi = row[f"{vcol}_ci_lower"], row[f"{vcol}_ci_upper"]
            errbar = dict(style.errbar)
            if style.errbar_from_point:
                errbar["ecolor"] = colors[cond]
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", **errbar)
    for (_action, _vcol, cond), pts in pairs.items():
        if len(pts) != len(style_levels):
            continue  # a level with no data -- nothing to join
        pts.sort(key=lambda t: list(style_levels).index(t[2]))
        ax.plot(
            [q[0] for q in pts],
            [q[1] for q in pts],
            "-",
            color=colors[cond],
            linewidth=style.split_linewidth,
            solid_capstyle="round",
            zorder=2,
        )

    if xticklabels:
        ax.set_xticks(range(3), panels.ACTION_AXIS_LABELS)
    else:
        ax.set_xticks(range(3), [""] * 3)
    xtick_kw = {"labelsize": style.xtick_fs} if style.xtick_fs else {}
    ax.tick_params(axis="x", length=style.tick_len, width=style.tick_w, **xtick_kw)

    if state_labels and xticklabels and style_col is not None:
        # The given world state goes on a secondary TOP axis, so the bottom axis
        # is identical to a study that has no such condition -- these panels get
        # stacked on one, and two categorical labels sharing the space under a
        # plot compete for which one names the action. No tick marks: the labels
        # sit over their sub-groups, and marks would read as data positions.
        sep = style.dodge_width * style.split_sep_frac
        offsets = [
            (si - (len(style_levels) - 1) / 2) * sep for si in range(len(style_levels))
        ]
        top = ax.secondary_xaxis("top")
        top.set_xticks(
            [a + off for a in range(3) for off in offsets],
            # style_levels are raw data values ("low"/"high"), not display words.
            [
                panels.EFFORT_LABELS.get(lvl, lvl)
                for _a in range(3)
                for lvl in style_levels
            ],
        )
        base_fs = style.xtick_fs or plt.rcParams["xtick.labelsize"]
        top.tick_params(
            axis="x",
            length=0,
            pad=style.split_sublabel_pad,
            labelsize=base_fs * style.split_sublabel_frac,
        )
        top.spines["top"].set_visible(False)

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
            # A split study carries a secondary top axis (state ticks + its own
            # label), so the column title has to clear both.
            # y is given explicitly to defeat matplotlib's automatic title
            # placement: it lifts a title that is wide enough to collide with the
            # top-axis labels and leaves narrow ones alone, so "Discomfort-only"
            # floated above "Base"/"Full"/"Humans" in the same row.
            ax.set_title(
                data.PANEL_LABELS[key],
                fontsize=style.title_fs,
                y=style.title_y,
            )
        ytick_kw = {"labelsize": style.tick_fs} if style.tick_fs else {}
        if style.ytick_len is not None:
            ytick_kw["length"] = style.ytick_len
        if style.ytick_w is not None:
            ytick_kw["width"] = style.ytick_w
        ax.tick_params(axis="y", **ytick_kw)


def ylabel_for(slug):
    """ "Belief update" alone for the joint studies (two latents share the
    axis), qualified by the latent for the single-DV studies."""
    dvs = study(slug).dvs
    if len(dvs) > 1:
        return "Belief update"
    return f"Belief update ({dvs[0].name})"


# ------------------------------------------------------------------- legends


def marker_fill(color, style):
    """Face/edge kwargs for one marker.

    Every marker is solid. Fill deliberately carries NO meaning: the given
    effort condition is shown by the split axis and the joining line instead
    (see `iter_cells`). Encoding it in the marker collides as soon as a
    given-effort study shares a figure with an inferred-effort one, because the
    latter's markers must still be drawn in some style and that style then reads
    as one of the levels.
    """
    return dict(
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=style.filled_edgewidth,
    )


#: Legend markers are drawn a little larger than the panels' own. A shape has to
#: be told apart from two others here, on its own beside a line of text, where a
#: panel marker is read in a field of its neighbours; at panel size the square
#: and the circle are hard to separate in the legend.
LEGEND_MARKER_SCALE = 1.4


def _grey_marker(marker, label, style):
    """A neutral-gray swatch drawn as the panels draw their markers, up to
    `LEGEND_MARKER_SCALE`, so the legend cannot describe an encoding the panels
    do not use."""
    return Line2D(
        [],
        [],
        linestyle="none",
        marker=marker,
        markersize=style.markersize * LEGEND_MARKER_SCALE,
        label=label,
        **marker_fill("0.35", style),
    )


def target_handles(names, style):
    """Shape legend for the inferred latents, in neutral gray so the shapes
    read as shape rather than palette."""
    return [_grey_marker(DV_MARKERS[n], DV_LEGEND_LABELS[n], style) for n in names]


def condition_color_handles(condition, style):
    """Color legend for one given condition ("intimacy_condition" or
    "desire_condition"), drawn as a rounded bar rather than a marker. Returns
    (handles, title).

    Deliberately shape-free: marker shape is the inferred target (`DV_MARKERS`),
    so a swatch drawn as a point would spend one of those shapes on a palette
    entry -- a circle in the relationship legend reads as the desire marker. The
    bar is as thick as a marker is wide, and its round caps echo the connector
    the panels draw between a condition's easy/hard pair in this same color, so
    the swatch still names something the panels draw.
    """
    levels, colors, labels, title = CONDITION_COLOR_AXES[condition]
    handles = [
        Line2D(
            [],
            [],
            linestyle="-",
            color=colors[lvl],
            linewidth=style.markersize,
            solid_capstyle="round",
            label=labels[lvl],
        )
        for lvl in levels
    ]
    return handles, title


#: Titles for the two encodings a study's panels share with the results figures.
#: Held here rather than in a consumer because both the standalone artboards
#: (`figure_paper_panels.py`) and the SI grids' own bands draw them.
TARGET_TITLE = "Target of inference"
GIVEN_STATE_TITLE = "Given physical state"
#: Canonical order for the inferred-target entries: the two continuous latents,
#: then the two-state world state. Fixed here rather than taken from whatever
#: order a study's `dvs` happen to be in, so every legend that spans more than
#: one study lists the shapes the same way.
TARGET_ORDER = ["desire", "intimacy", "effort"]


def state_handles(style):
    """Legend entry for the given world state of 1a and 2a.

    Those studies show the state by splitting each action into two sub-groups
    joined by a connector, never by styling the markers (see `iter_cells`), so
    what the legend has to name is the split itself rather than a swatch. The
    handle is drawn as the panels draw one pair -- two markers joined by the
    connector -- and the label says which side is which, since position is the
    encoding. Neutral gray for the same reason `target_handles` is: in the
    panels the connector takes its condition's color.
    """
    return [
        Line2D(
            [],
            [],
            linestyle="-",
            color="0.35",
            linewidth=style.split_linewidth,
            solid_capstyle="round",
            marker="o",
            markersize=style.markersize * LEGEND_MARKER_SCALE,
            label=(
                f"{panels.EFFORT_LABELS['low']} (left), "
                f"{panels.EFFORT_LABELS['high']} (right)"
            ),
            **marker_fill("0.35", style),
        )
    ]


def legend_groups(slug, style):
    """`legend_groups_for` for a single study."""
    return legend_groups_for([slug], style)


def legend_groups_for(slugs, style):
    """(handles, title, legend kwargs) per encoding the panels of `slugs` use,
    in the order they should be read: the given conditions' colors, the inferred
    latents' shapes, and the split that carries a given world state.

    A union, so one figure spanning several studies gets one legend covering all
    of them rather than a stack of per-study ones. The shape group is dropped
    where the whole set infers a single latent (a figure of 1a alone, say): its
    name is already in the y axis label there (`ylabel_for`), so the group would
    be a one-entry legend naming the only shape on the figure, and it invites the
    reader to look for a second one.
    """
    groups = []
    for condition in CONDITION_COLOR_AXES:  # relationship, then desire
        if any(color_axis(slug) == condition for slug in slugs):
            handles, title = condition_color_handles(condition, style)
            # A bar swatch fills its whole handle box where a marker sits
            # centered in one, so it needs a shorter handle and a wider text pad
            # than the marker groups to keep the swatch off its label.
            groups.append((handles, title, dict(handlelength=1.1, handletextpad=0.8)))
    names = {dv.name for slug in slugs for dv in study(slug).dvs}
    if len(names) > 1:
        ordered = [n for n in TARGET_ORDER if n in names]
        if len(ordered) != len(names):
            raise ValueError(
                f"inferred target(s) {sorted(names - set(TARGET_ORDER))} are "
                "missing from TARGET_ORDER, so the legend would drop them"
            )
        groups.append((target_handles(ordered, style), TARGET_TITLE, {}))
    if any(style_col(slug) for slug in slugs):
        # numpoints=2 is what puts a marker at each end of the connector, so
        # the handle shows a pair rather than one point on a line.
        groups.append(
            (
                state_handles(style),
                GIVEN_STATE_TITLE,
                dict(numpoints=2, handlelength=3.2),
            )
        )
    return groups


def legend_band(
    fig, groups, *, fontsize, pad_in=0.12, gap_in=0.45, row_gap_in=0.10, inside=False
):
    """Draw `groups` as a row of columns in a band under the figure.

    Each group is one COLUMN of entries under its title, and the groups sit side
    by side, top-aligned and centered as a set. One entry per line because the
    relationship palette is an ordered ramp: a row, or a multi-column block,
    breaks that order for a reader whose eye goes down a list. Side by side
    because the groups are independent encodings -- stacking them reads as one
    long list that happens to have headings in it. Groups that would run past
    the figure's own width wrap onto another line, so a figure spanning every
    study can carry all four groups without being widened for them.

    By default the band hangs BELOW the drawn canvas, anchored to whatever the
    figure's content already reaches down to (the x axis label), and `savefig`'s
    tight crop grows the saved page to hold it. Reserving space inside the canvas
    instead would mean taking it off the panels, which are sized against what
    they print at.

    `inside=True` is for a figure saved with `tight=False`, where the canvas IS
    the page and anything hanging off it is simply cut: the canvas grows by the
    band's height and every axes shifts up by the same amount, so the panels keep
    their inch geometry and the band lands in the space that opens under them.
    The axes are repositioned directly, so this cannot be used on a figure whose
    layout engine will run again at draw time (constrained or tight layout).
    """
    fig.canvas.draw()  # tight bbox and window extents both need a renderer
    w_in, h_in = fig.get_size_inches()
    # What the figure's own artists reach, in inches from its lower left. The
    # band hangs off that rather than off the canvas edge, so the gap under the
    # x axis label is the one set here whatever slack the layout left below it.
    content = fig.get_tightbbox()
    legends = [
        fig.legend(
            handles=handles,
            title=title,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.0),
            frameon=False,
            fontsize=fontsize,
            title_fontsize=fontsize,
            **{"handletextpad": 0.5, **kw},
        )
        for handles, title, kw in groups
    ]
    fig.canvas.draw()
    boxes = [
        leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        for leg in legends
    ]
    rows = [[]]
    for entry in zip(legends, boxes):
        row = rows[-1] + [entry]
        if rows[-1] and _span(row, gap_in) > content.width:
            rows.append([entry])
        else:
            rows[-1] = row

    top_in = content.y0 - pad_in
    if inside:
        # Grow by exactly what the band needs, then slide every axes up by the
        # same distance. The band then sits below the content with `pad_in` above
        # it and the figure's own bottom margin (`content.y0`) below it, so the
        # page keeps the margins it was laid out with.
        band_in = sum(max(b.height for _l, b in row) for row in rows)
        band_in += row_gap_in * (len(rows) - 1) + pad_in
        fig.set_size_inches(w_in, h_in + band_in)
        scale, offset = h_in / (h_in + band_in), band_in / (h_in + band_in)
        for ax in fig.axes:
            pos = ax.get_position()
            ax.set_position(
                [pos.x0, pos.y0 * scale + offset, pos.width, pos.height * scale]
            )
        h_in += band_in
        top_in += band_in
    for row in rows:
        # Centered on the content rather than the canvas: the tight crop that
        # makes room for the band also trims the right margin, which has no
        # counterpart for the y axis label on the left, so a canvas-centered row
        # would come out sitting left of the panels it describes.
        x_in = (content.x0 + content.x1 - _span(row, gap_in)) / 2
        for leg, box in row:
            # Explicitly against transFigure: the default anchor transform binds
            # to `fig.bbox`, which `savefig(bbox_inches="tight")` REPLACES when
            # it grows the page, leaving anything anchored to it drawn off the
            # new canvas. transFigure is mutated in place by the same code, so it
            # survives.
            leg.set_bbox_to_anchor(
                (x_in / w_in, top_in / h_in), transform=fig.transFigure
            )
            x_in += box.width + gap_in
        top_in -= max(box.height for _leg, box in row) + row_gap_in


def _span(row, gap_in):
    """Width in inches of one packed row of legends, gaps included."""
    return sum(box.width for _leg, box in row) + gap_in * (len(row) - 1)
