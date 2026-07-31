#!/usr/bin/env python3
"""SI per-scenario figures: the cell means behind each study's averaged panel.

The main results figures average over the 16 scenarios. These show the same
belief updates scenario by scenario, one 4x4 facet grid per study, so a reader
can see how much of the averaged pattern each individual scenario carries.

Within a facet the encoding is the results figures' (see `_points.py`): x is the
observed action, marker shape is the inferred latent, colour is the given
relationship/desire condition, and filled vs open is the given effort condition
where effort is given rather than inferred (1a and 2a).

Humans are the markers, with 95% subject-cluster bootstrap CIs. The full model's
out-of-sample LOSO prediction for the same cell is a short horizontal dash at
the same x -- a separate visual channel, so it does not compete with the shape,
colour, or fill the conditions already use. Per-scenario cells are thin (one
observation per participant in most studies), so the CIs are wide by design;
that is the point of showing them.

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
MODEL_DASH = dict(marker="_", markeredgewidth=1.3, zorder=4)
# x span of a panel in data units (_points.draw_points' xlim), used to size the
# model dash so it never runs into the neighbouring dodge slot.
X_SPAN = 3.2


def _dash_size(slug):
    """Dash width in points: just under one dodge step, so each dash sits under
    its own marker instead of bridging across to the next condition."""
    n_pts = len(data.dvs_display(slug)) * len(points.fill_spec(slug)[1])
    if points.style_col(slug):
        n_pts *= 2
    step = STYLE.dodge_width / n_pts
    return 0.85 * step * (STYLE.panel_w * 72 / X_SPAN)


def build_scenario_cells(slug):
    """(human cells with CIs, full-model cells) on the scenario x condition
    grid, or (None, None) when the study's inputs are missing."""
    keys = ["scenario_label", *data.condition_cols(slug)]
    dvs = data.dvs_display(slug)
    trials = data.load_trials(slug)
    if trials is None:
        return None, None
    trials = trials.assign(action_label=data.action_label_col(trials))
    human = data.bootstrap_cell_means(
        trials,
        [h for h, _d, _l in dvs],
        keys,
        seed=data.seed_for(f"figures:si_scenarios:{slug}"),
    )
    preds = data.load_cv_preds(slug)
    model = None
    if preds is not None:
        preds = preds.assign(action_label=data.action_label_col(preds))
        model = (
            preds[preds["model"] == "full"]
            .groupby(keys, as_index=False)[[d for _h, d, _l in dvs]]
            .mean()
        )
        data.warn_if_stale(slug, trials, data.load_comparison(slug))
    return human, model


def draw_facet(ax, slug, human_cells, model_cells, lim, *, xticklabels):
    """One scenario's facet: human markers + CIs, full-model dashes behind."""
    fcol, flevels, fcolors, _title = points.fill_spec(slug)
    scol = points.style_col(slug)
    dvs = data.dvs_display(slug)
    marks = points.markers_for(slug)

    points.draw_points(
        ax,
        human_cells,
        value_cols=[(h, m) for (h, _d, _l), m in zip(dvs, marks)],
        color_col=fcol,
        fill_levels=flevels,
        colors=fcolors,
        lim=lim,
        ci=True,
        style=STYLE,
        style_col=scol,
        xticklabels=xticklabels,
    )
    if model_cells is None or model_cells.empty:
        return
    for x, row, vcol, _marker, cond, _open_pt in points.iter_cells(
        model_cells,
        value_cols=[(d, m) for (_h, d, _l), m in zip(dvs, marks)],
        color_col=fcol,
        fill_levels=flevels,
        style=STYLE,
        style_col=scol,
    ):
        ax.plot(
            x,
            row[vcol],
            color=fcolors[cond],
            markersize=_dash_size(slug),
            **MODEL_DASH,
        )


def build_study(slug, stem):
    human, model = build_scenario_cells(slug)
    if human is None:
        print(f"[{slug}] no data yet — skipped")
        return False
    scenarios = sorted(human["scenario_label"].unique())
    nrows = -(-len(scenarios) // NCOLS)

    dvs = data.dvs_display(slug)
    vals = [
        human[f"{h}_ci_{b}"].abs().max()
        for h, _d, _l in dvs
        for b in ("lower", "upper")
    ]
    vals += [human[h].abs().max() for h, _d, _l in dvs]
    if model is not None:
        vals += [model[d].abs().max() for _h, d, _l in dvs]
    lim = float(max(vals)) * 1.08

    legends = points.legend_groups([(slug, human, model)], STYLE)
    fig, axes = plt.subplots(
        nrows,
        NCOLS,
        figsize=(
            STYLE.panel_w * NCOLS + 0.55,
            STYLE.panel_h * nrows
            + 0.75
            + points.LEGEND_ROW_H * points.legend_rows(legends),
        ),
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
            human[human["scenario_label"] == scenario],
            None if model is None else model[model["scenario_label"] == scenario],
            lim,
            xticklabels=(i // NCOLS == nrows - 1),
        )
        ax.set_title(scenario, fontsize=7.5, pad=2.5)
    for j in range(len(scenarios), nrows * NCOLS):
        axes[j // NCOLS][j % NCOLS].set_axis_off()

    fig.supylabel(points.ylabel_for(slug), fontsize=9)
    points.place_legends(fig, legends, STYLE)
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
