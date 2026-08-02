#!/usr/bin/env python3
"""SI per-scenario figures: the human cell means behind each averaged panel.

The main results figures average over the 16 scenarios. These show the same
belief updates scenario by scenario, one 4x4 facet grid per study, so a reader
can see how much of the averaged pattern each individual scenario carries.

Human data only -- the model's per-scenario predictions are deliberately not
overlaid, so these read as the data behind the averages rather than as a fit
check. Within a facet the encoding is the results figures' (see `_points.py`):
x is the observed action, marker shape is the inferred latent, colour is the
given relationship/desire condition, and filled vs open is the given effort
condition where effort is given rather than inferred (1a and 2a). Error bars are
95% subject-cluster bootstrap CIs; per-scenario cells are thin (often one
observation per participant), so they are wide by design.

No legend is drawn: these are assembled in Illustrator, where the legend is
placed by hand. `figures/panels/legends/legend_*.pdf` carry the same encoding.

Usage:
    uv run python figures/scripts/figure_si_scenarios.py [--study <slug>]
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import apply_style, savefig  # noqa: E402
from study_registry import studies  # noqa: E402

import _data as data  # noqa: E402
import _points as points  # noqa: E402

NCOLS = 4
# Denser than the averaged panels: 16 facets of three action slots each, so the
# markers and CI stems come down to keep a facet legible.
STYLE = replace(
    points.PAPER,
    markersize=3.6,
    panel_w=1.5,
    panel_h=1.15,
    xtick_fs=6.5,
    errbar=dict(elinewidth=1.0, capsize=0, zorder=2),
)


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
        ax.set_title(scenario, fontsize=7.5, pad=2.5)
    for j in range(len(scenarios), nrows * NCOLS):
        axes[j // NCOLS][j % NCOLS].set_axis_off()

    fig.supylabel(points.ylabel_for(slug), fontsize=9)
    fig.supxlabel(points.X_AXIS_LABEL, fontsize=9)
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
