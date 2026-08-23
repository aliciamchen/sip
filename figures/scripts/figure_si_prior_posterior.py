#!/usr/bin/env python3
"""SI prior/posterior figure: the rating LEVELS behind the belief-update DV.

Every number the paper reports is a difference -- posterior rating minus prior
rating -- so nothing in it says where participants started, how much room they
had left, or whether they used the whole slider. This is that missing view, and
it is human data only: no model predictions, no CV outputs.

Levels (si_prior_posterior_levels). Each action slot splits into the prior and
    posterior rating, joined by a line whose slope IS the belief update the
    results figures plot. Deliberately the same grammar as the results panels (x
    is the observed action, color is the given condition, marker is the rated
    latent), so the only new thing a reader has to absorb is that y is a level
    rather than a difference. The dashed rule at 0.5 is the uniform prior mean
    the model assumes, which is NOT where participants sit. One facet per given
    world state in 1a and 2a -- see `facets` for why it is not averaged over.

Distributions (si_prior_posterior_distributions). A DIAGNOSTIC, cut from the SI
    on 2026-08-03: twelve histogram panels carried only two facts -- the share of
    ratings at a slider endpoint, and the residual spike at 0.5 where the slider
    was left at its default -- and both are stated as ranges in the SI text
    instead. It stays runnable because those ranges have to remain reproducible,
    and because the shape it shows (the world-state ratings are bimodal rather
    than merely spread) is not recoverable from the numbers. Neither synced nor
    cited; its output is gitignored.

Usage:
    uv run python figures/scripts/figure_si_prior_posterior.py
"""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_style import (
    ALT_GREY,
    DESIRE_COLORS,
    DV_MARKERS,
    INTIMACY_COLORS,
    INTIMACY_LEVELS,
    PANELS_LEGENDS,
    apply_style,
    savefig,
)
from study_registry import studies, study
from utils import get_project_root

import _data as data
import _panels as panels
import _points as points

STAGES = ("prior", "posterior")
# Facet columns per figure, chosen so each lands at \textwidth without shrinking:
# the levels panels need width for three action groups of dodged pairs, the
# histograms do not.
NCOLS_LEVELS = 4
NCOLS_DIST = 4
N_BOOT = 1000
# The prior mean the reported model assumes: uniform on the latent grid. Drawn as
# a reference because the point of panel (a) is that participants are not there.
MODEL_PRIOR_MEAN = 0.5
POSTERIOR_COLOR = "#333333"
BINS = np.linspace(0, 1, 21)
# A rating this close to a slider end counts as "at the end"; the sliders report
# 0-100 integers scaled to 0-1, so this catches exactly 0 and 100.
END_EPS = 0.01

# Markers and type are set against what the main text's panels print at -- ~5.3pt
# markers, once `figure_paper_panels.py`'s artboard is scaled into the column --
# rather than at the smaller sizes these facets used to carry. Both figures are
# full-page [p] floats that were giving up around 2.5in of page height, so the
# facets can take the size without anything else moving. They take it as width
# and marker size rather than as height: a facet's y axis is the fixed 0-1 rating
# scale, and stretching it vertically lengthens the prior-to-posterior connectors
# without separating any two of them, so the facets are held near square.
STYLE = replace(points.PAPER, markersize=7.0)

#: Facet titles and the axis labels, at that same scale. Named once because the
#: levels and distribution grids share them and are read as a pair.
FACET_TITLE_FS = 9.0
AXIS_LABEL_FS = 9.5
FACET_TICK_FS = 8.0
#: The action labels get their own, smaller size. They are the one piece of type
#: here that is width-bound rather than height-bound: "Low-risk" and "High-risk"
#: sit one x unit apart, which is about 0.43in across a facet of this grid, so at
#: the y ticks' size the two run into each other.
FACET_XTICK_FS = 7.0
#: The levels figure's legend band, a step below the facet titles: it is read
#: once on the way in, where the titles are read against every facet.
LEGEND_FS = 8.5
STAGE_TITLE = "Rating"


def condition_spec(slug):
    """(long-CSV condition column, ordered levels, level -> color) for the given
    condition that colors a study's points -- the relationship descriptor or the
    desire level. A study's second given condition (the world state, in 1a and
    2a) gets its own facet instead of a color; see `facets`."""
    given = study(slug).given_conditions[0]
    col = given.removesuffix("_condition")
    if given == "intimacy_condition":
        return col, list(INTIMACY_LEVELS), INTIMACY_COLORS
    return col, ["low", "high"], DESIRE_COLORS


def load_long(slug):
    """The study's per-rating rows (one per prior/posterior elicitation), or None
    when the study has no data yet."""
    csv = get_project_root() / "data" / slug / "main_trials_long.csv"
    if not csv.exists():
        print(f"[{slug}] no data yet ({csv.name}) — skipped")
        return None
    df = pd.read_csv(csv)
    return df.rename(columns={"action_condition": "action_label"})


def facets():
    """[(slug, dv, rating column, effort level or None), ...] -- one panel per
    study x rated latent x given world state, in registry order.

    Studies 1a and 2a are given the physical world state as well, and it is a
    manipulated condition rather than a nuisance: it shifts the prior rating
    itself (1a's prior desire is 0.704 when the low-risk share is easy against
    0.661 when it is hard, consistently across all three actions, so the
    world-state paragraph is read before the prior elicitation) and it changes
    the update. Those studies therefore get one panel per state instead of one
    panel averaging over both -- the results figures split on exactly this
    variable, and collapsing it here would hide a level difference the reader is
    being shown the levels for.

    The rating column is the belief-update column without its `_update` suffix:
    `response` in Study 1a, `<latent>_rating` elsewhere.
    """
    out = []
    for s in studies():
        splits = (
            list(panels.EFFORT_LABELS)
            if "effort_condition" in s.given_conditions
            else [None]
        )
        for dv in s.dvs:
            for eff in splits:
                out.append((s.slug, dv, dv.update_col.removesuffix("_update"), eff))
    return out


def facet_rows(df, effort):
    """The facet's slice of a study's ratings: one world state, or all of them
    where the study infers the state rather than giving it."""
    return df if effort is None else df[df["effort"] == effort]


def facet_title(slug, dv, effort):
    """ "Study 1a: desire + easy" -- the two manipulated factors joined with "+",
    matching how tab:model-comparison names a study's pair ("desire + physical"),
    rather than with a separator that reads as mere punctuation."""
    suffix = "" if effort is None else f" + {panels.EFFORT_LABELS[effort].lower()}"
    return f"{study(slug).paper_label}: {dv.name}{suffix}"


def cell_means(df, rating_col, cond_col, slug, effort):
    """Per (action, condition, stage) mean rating with a 95% subject-cluster
    bootstrap CI, averaged over scenarios within this facet's world state."""
    return data.bootstrap_cell_means(
        facet_rows(df, effort),
        [rating_col],
        ["action_label", cond_col, "stage"],
        n_boot=N_BOOT,
        seed=data.seed_for(f"figures:prior_posterior:{slug}:{rating_col}:{effort}"),
    )


def draw_levels(ax, cells, rating_col, dv, cond_col, levels, colors, *, xticklabels):
    """Panel (a) facet: prior and posterior markers per action, joined per
    condition."""
    ax.axhline(MODEL_PRIOR_MEAN, **panels.ZERO_LINE)
    marker = DV_MARKERS[dv.name]
    pairs = {}
    for x, row, vcol, mk, cond, stage in points.iter_cells(
        cells,
        value_cols=[(rating_col, marker)],
        color_col=cond_col,
        fill_levels=levels,
        style=STYLE,
        style_col="stage",
        style_levels=STAGES,
    ):
        y = row[vcol]
        pairs.setdefault((row["action_label"], cond), []).append((x, y, stage))
        lo, hi = row[f"{vcol}_ci_lower"], row[f"{vcol}_ci_upper"]
        ax.errorbar(
            x,
            y,
            yerr=[[max(y - lo, 0)], [max(hi - y, 0)]],
            fmt="none",
            ecolor=colors[cond],
            elinewidth=1.3,
            capsize=0,
            zorder=2,
        )
        # Prior hollow, posterior filled: the pair is the same quantity at two
        # times, so it reads as one object moving rather than two series.
        filled = stage == "posterior"
        ax.plot(
            x,
            y,
            mk,
            markersize=STYLE.markersize,
            markerfacecolor=colors[cond] if filled else "white",
            markeredgecolor=colors[cond],
            markeredgewidth=1.3,
            zorder=3,
        )
    for (_action, cond), pts in pairs.items():
        if len(pts) != len(STAGES):
            continue
        pts.sort(key=lambda t: STAGES.index(t[2]))
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "-",
            color=colors[cond],
            linewidth=1.6,
            zorder=2,
            solid_capstyle="round",
        )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(panels.ACTION_AXIS_LABELS if xticklabels else [])
    ax.tick_params(axis="y", labelsize=FACET_TICK_FS)
    ax.tick_params(axis="x", labelsize=FACET_XTICK_FS)


def draw_distribution(ax, df, rating_col):
    """Panel (b) facet: the raw ratings by stage, as % of ratings per bin."""
    ends = {}
    for stage, color in zip(STAGES, (ALT_GREY, POSTERIOR_COLOR)):
        v = df.loc[df["stage"] == stage, rating_col].dropna().to_numpy()
        if not len(v):
            continue
        ax.hist(
            v,
            bins=BINS,
            weights=np.full(len(v), 100 / len(v)),
            histtype="step",
            color=color,
            lw=1.2,
            zorder=3 if stage == "posterior" else 2,
        )
        ends[stage] = 100 * float(((v <= END_EPS) | (v >= 1 - END_EPS)).mean())
    # Each stage's endpoint share is printed in that stage's own color, rather
    # than as one "prior -> posterior" string: the arrow glyph is missing from the
    # house font, and the color says which is which without a key.
    for k, stage in enumerate(s for s in STAGES if s in ends):
        ax.annotate(
            f"{ends[stage]:.0f}% at ends",
            (0.03, 0.95 - 0.12 * k),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=7.5,
            color=ALT_GREY if stage == "prior" else POSTERIOR_COLOR,
            # Several studies pile mass in the first bin, which reaches the
            # annotation; the plate keeps both readable.
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8),
        )
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(labelsize=FACET_TICK_FS)


def _bottom_of_column(n, ncols):
    """{column -> index of the lowest facet in it}. The facet count is not a
    multiple of the column count, so "the last row" is not the bottom of every
    column -- tick labels keyed on the final facets would leave a column's lowest
    panel unlabelled while its neighbours carry labels."""
    return {c: max(i for i in range(n) if i % ncols == c) for c in range(min(n, ncols))}


def _grid(rows, ncols, panel_w, panel_h, pad_w, pad_h, hspace=0.45, wspace=0.30):
    """A figure sized so its panels land at `panel_w` x `panel_h` inches at 1:1.
    Both figures are laid out for \\textwidth so the 7-9.5pt type is read at the
    size it is set -- a single 5-column figure holding both blocks came out 11in
    wide, which \\textwidth would have shrunk to roughly 4pt.

    `hspace` is a fraction of the panel height, so it has to come down as the
    panels grow taller or the row gaps grow with them and eat the height the
    facets were given."""
    nrows = -(-len(rows) // ncols)
    fig = plt.figure(figsize=(panel_w * ncols + pad_w, panel_h * nrows + pad_h))
    gs = fig.add_gridspec(nrows, ncols, hspace=hspace, wspace=wspace)
    return fig, gs, nrows


def build_levels(rows, loaded):
    """The paired prior/posterior levels, one facet per study x rated latent."""
    fig, gs, nrows = _grid(
        rows, NCOLS_LEVELS, 1.55, 1.52, 0.80, 0.72, hspace=0.28, wspace=0.24
    )
    bottom = _bottom_of_column(len(rows), NCOLS_LEVELS)
    for i, (slug, dv, rating_col, eff) in enumerate(rows):
        cond_col, levels, colors = condition_spec(slug)
        r, c = divmod(i, NCOLS_LEVELS)
        ax = fig.add_subplot(gs[r, c])
        draw_levels(
            ax,
            cell_means(loaded[slug], rating_col, cond_col, slug, eff),
            rating_col,
            dv,
            cond_col,
            levels,
            colors,
            xticklabels=(i == bottom[c]),
        )
        ax.set_title(facet_title(slug, dv, eff), fontsize=FACET_TITLE_FS, pad=3)
        if c == 0:
            ax.set_ylabel("Rating", fontsize=AXIS_LABEL_FS)
    for j in range(len(rows), nrows * NCOLS_LEVELS):
        fig.add_subplot(gs[j // NCOLS_LEVELS, j % NCOLS_LEVELS]).set_axis_off()

    fig.subplots_adjust(bottom=0.075, top=0.955, left=0.075, right=0.99)
    # `inside`, because this figure is saved untrimmed (`tight=False` in `main`)
    # so a band hanging off the canvas would be cut at the page edge.
    points.legend_band(fig, legend_groups(), fontsize=LEGEND_FS, inside=True)
    return fig


def stage_handles():
    """The prior/posterior marker pair, in the neutral gray the facets' own
    colors stand in for."""
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=STYLE.markersize,
            markerfacecolor="white" if stage == "prior" else "0.35",
            markeredgecolor="0.35",
            markeredgewidth=1.1,
            label=f"{stage.capitalize()} ({when} the action)",
        )
        for stage, when in zip(STAGES, ("before", "after"))
    ]


def legend_groups():
    """The three encodings the levels facets use: the two given-condition
    palettes, and the prior/posterior marker pair.

    No group for the rated latent, though its shape varies across facets. Every
    facet is titled with the latent it rates, so the shape says nothing the panel
    does not; and here, unlike the results panels, marker FILL already means
    something (`draw_levels`), so a solid swatch in a shape legend would read as
    a posterior rating.
    """
    groups = []
    for condition in ("intimacy_condition", "desire_condition"):
        handles, title = points.condition_color_handles(condition, STYLE)
        groups.append((handles, title, dict(handlelength=1.1, handletextpad=0.8)))
    return [*groups, (stage_handles(), STAGE_TITLE, {})]


def build_legend():
    """The stage legend on its own tight-cropped artboard, for placing by hand
    beside a panel that is not this figure -- the levels grid carries its own
    band. Only the marker pair: the connector and the 0.5 rule are explained in
    the caption, where they cost a clause rather than an artboard."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.legend(
        handles=stage_handles(),
        title=STAGE_TITLE,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=STYLE.legend_fs,
        title_fontsize=STYLE.legend_fs,
        handletextpad=0.5,
    )
    return fig


def build_distributions(rows, loaded):
    """The raw rating distributions by stage, on the same facet set."""
    fig, gs, nrows = _grid(rows, NCOLS_DIST, 1.52, 1.75, 0.80, 1.10)
    bottom = _bottom_of_column(len(rows), NCOLS_DIST)
    for i, (slug, dv, rating_col, eff) in enumerate(rows):
        r, c = divmod(i, NCOLS_DIST)
        ax = fig.add_subplot(gs[r, c])
        draw_distribution(ax, facet_rows(loaded[slug], eff), rating_col)
        ax.set_title(facet_title(slug, dv, eff), fontsize=FACET_TITLE_FS, pad=3)
        if c == 0:
            ax.set_ylabel("% of ratings", fontsize=AXIS_LABEL_FS)
        if i == bottom[c]:
            ax.set_xlabel("Rating", fontsize=AXIS_LABEL_FS)
    for j in range(len(rows), nrows * NCOLS_DIST):
        fig.add_subplot(gs[j // NCOLS_DIST, j % NCOLS_DIST]).set_axis_off()

    fig.legend(
        handles=[
            Line2D([], [], color=ALT_GREY, lw=1.2, label="Prior ratings"),
            Line2D([], [], color=POSTERIOR_COLOR, lw=1.2, label="Posterior ratings"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=AXIS_LABEL_FS,
        handletextpad=0.5,
        columnspacing=1.8,
        borderaxespad=0.3,
    )
    fig.subplots_adjust(bottom=0.155, top=0.935, left=0.075, right=0.99)
    return fig


def main():
    apply_style("si")
    rows = facets()
    loaded = {}
    for slug, _dv, _col, _eff in rows:
        if slug not in loaded:
            loaded[slug] = load_long(slug)
    rows = [r for r in rows if loaded[r[0]] is not None]
    if not rows:
        print("no study has data yet — nothing to draw")
        return
    for fig, stem in (
        (build_levels(rows, loaded), "si_prior_posterior_levels"),
        (build_distributions(rows, loaded), "si_prior_posterior_distributions"),
    ):
        print(f"wrote {savefig(fig, stem, tight=False)}")
    # The legend is a placed-by-hand component, so it goes beside the results
    # legends and MUST crop to content -- the full 0.1in canvas would be a
    # 20x20px sliver.
    print(
        f"wrote {savefig(build_legend(), 'legend_prior_posterior', out_dir=PANELS_LEGENDS)}"
    )


if __name__ == "__main__":
    main()
