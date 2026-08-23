#!/usr/bin/env python3
"""SI per-scenario figures: the human cell means behind each averaged panel.

The main results figures average over the 16 scenarios. These show the same
belief updates scenario by scenario, one 4x4 facet grid per study, so a reader
can see how much of the averaged pattern each individual scenario carries.

Human data only -- the model's per-scenario predictions are deliberately not
overlaid, so these read as the data behind the averages rather than as a fit
check. Within a facet the encoding is the results figures' (see `_points.py`):
x is the observed action, marker shape is the inferred latent, and color is the
given relationship/desire condition. Where the effort of the low-risk share is
given rather than inferred (1a and 2a) each action splits into its two world
states, joined by a line, named in the legend rather than on every facet.
Error bars are 95% subject-cluster bootstrap CIs; per-scenario cells
are thin (often one observation per participant), so they are wide by design.

A legend band under the x axis names the encodings the facets cannot label
themselves: the given condition's colors, the inferred latents' shapes, and --
for 1a and 2a -- which side of a split action is which state.

Usage:
    uv run python figures/scripts/figure_si_scenarios.py [--study <slug>]
"""

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt

from plot_style import apply_style, savefig
from study_registry import studies

import _data as data
import _points as points

NCOLS = 4
# Sized against what the main text's panels print at -- ~5.3pt markers on a
# 1.55 x 1.76in panel, once `figure_paper_panels.py`'s artboard is scaled into
# the column -- rather than against SI density for its own sake. Sixteen facets
# of three action slots each used to be the reason to shrink everything, but the
# figure is a full-page [p] float carrying a one-line caption, so it was giving
# up around 2in of page height to stay wide-and-short. Spending part of that
# height on the facets squares them up and pays for markers and type at the
# sizes the results panels are read at. Only part: a facet holds three action
# slots of dodged points inside roughly half its y range, so past about square
# the extra height goes to empty margin rather than to separating anything.
STYLE = replace(
    points.PAPER,
    markersize=6.0,
    panel_w=1.5,
    panel_h=1.45,
    xtick_fs=7.2,
    tick_fs=8.0,
    errbar=dict(elinewidth=1.4, capsize=0, zorder=2),
)
#: Facet (scenario name), figure-level axis label, and legend sizes, at the
#: same scale. The legend sits a step below the axis labels: it is read once on
#: the way in, where the axis labels are read against every facet.
FACET_TITLE_FS = 9.0
AXIS_LABEL_FS = 10.0
LEGEND_FS = 9.0


def build_scenario_cells(slug):
    """Human cells with bootstrap CIs on the scenario x condition grid, or None
    when the study has no data yet."""
    trials = data.load_trials(slug)
    if trials is None:
        return None
    trials = trials.assign(action_label=data.action_label_col(trials))
    return data.bootstrap_cell_means(
        trials,
        [h for h, _d, _l in data.dvs_display(slug)],
        ["scenario_label", *data.condition_cols(slug)],
        seed=data.seed_for(f"figures:si_scenarios:{slug}"),
    )


def draw_facet(ax, slug, cells, lim, *, xticklabels):
    """One scenario's facet of human markers with CIs."""
    fcol, flevels, fcolors, _title = points.fill_spec(slug)
    dvs = data.dvs_display(slug)
    points.draw_points(
        ax,
        cells,
        value_cols=[(h, m) for (h, _d, _l), m in zip(dvs, points.markers_for(slug))],
        color_col=fcol,
        fill_levels=flevels,
        colors=fcolors,
        lim=lim,
        ci=True,
        style=STYLE,
        style_col=points.style_col(slug),
        xticklabels=xticklabels,
        # 48 repetitions of "Easy Hard" across a 16-facet grid, at 5pt,
        # would be noise; the legend band names the two states once instead.
        state_labels=False,
    )


def build_study(slug, stem):
    cells = build_scenario_cells(slug)
    if cells is None:
        print(f"[{slug}] no data yet — skipped")
        return False
    scenarios = sorted(cells["scenario_label"].unique())
    nrows = -(-len(scenarios) // NCOLS)

    lim = 1.08 * float(
        max(
            cells[c].abs().max()
            for h, _d, _l in data.dvs_display(slug)
            for c in (h, f"{h}_ci_lower", f"{h}_ci_upper")
        )
    )

    fig, axes = plt.subplots(
        nrows,
        NCOLS,
        figsize=(STYLE.panel_w * NCOLS + 0.55, STYLE.panel_h * nrows + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for i, scenario in enumerate(scenarios):
        ax = axes[i // NCOLS][i % NCOLS]
        draw_facet(
            ax,
            slug,
            cells[cells["scenario_label"] == scenario],
            lim,
            xticklabels=(i // NCOLS == nrows - 1),
        )
        ax.set_title(scenario, fontsize=FACET_TITLE_FS, pad=2.5)
    for j in range(len(scenarios), nrows * NCOLS):
        axes[j // NCOLS][j % NCOLS].set_axis_off()

    fig.supylabel(points.ylabel_for(slug), fontsize=AXIS_LABEL_FS)
    fig.supxlabel(points.X_AXIS_LABEL, fontsize=AXIS_LABEL_FS)
    points.legend_band(fig, points.legend_groups(slug, STYLE), fontsize=LEGEND_FS)
    out = savefig(fig, f"si_scenarios_{stem}")
    print(f"wrote {out}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", help="Render one slug only (default: all six).")
    args = ap.parse_args()
    apply_style("si")
    drawn = 0
    for s in studies():
        if args.study and s.slug != args.study:
            continue
        drawn += build_study(s.slug, f"study{s.short_label}")
    print(f"\n{drawn} per-scenario figure(s)")


if __name__ == "__main__":
    main()
