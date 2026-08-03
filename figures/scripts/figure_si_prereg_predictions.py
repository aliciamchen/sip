#!/usr/bin/env python3
"""What the comparison-set reweighting changes in the predictions, all six studies.

`tab:prereg-deviation` and `fig:si-prereg-deviation` say how much the reweighting
buys in held-out likelihood -- one number per study, and a mixed one: it helps in
Studies 1b, 2a and 3a, ties in 2b, and costs a little in 3b. A likelihood
difference alone cannot say *where* the two models part company, or whether the
studies it does not help are studies where it changes nothing or studies where it
changes something in the wrong direction. This shows the predictions themselves.

One row per study, three columns: the reported model, the preregistered ($\\eta =
0$) model, and the human means. Both model columns are the `full` variant's
out-of-sample leave-one-scenario-out predictions, so they are directly comparable
and neither has seen the cell it predicts. Within a panel the encoding is the
results figures' (`_points.py`): x is the observed action, marker shape is the
inferred latent, colour is the given relationship/desire condition, and where the
world state is given rather than inferred (Studies 1a and 2a) each action splits
into its two states joined by a line. Error bars appear on the human column only
-- the model columns are point predictions -- and the two states are named in the
caption rather than on six rows of ticks.

Reads the reported CV outputs and the `alt/uniform-noreweight/` ones, which
`bin/prereg-eta0.sh` produces. Studies missing either side are skipped with a
message rather than drawn half-empty, since a blank column in one row of a
six-row grid reads as a model that predicts zero.

Usage:
    uv run python figures/scripts/figure_si_prereg_predictions.py
"""

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import apply_style, savefig  # noqa: E402
from study_registry import studies  # noqa: E402

import _data as data  # noqa: E402
import _points as points  # noqa: E402

#: The run-config tag `--no-reweighting` writes under (`RunConfig.tag()`).
PREREG_TAG = "uniform-noreweight"

#: Columns, left to right: the two models then the humans they are judged against.
#: Reported first because it is the model the paper's other figures show, so the
#: eye lands on the familiar column and reads the second as the departure.
COLUMNS = [
    ("reported", "Reported"),
    ("prereg", "Preregistered ($\\eta = 0$)"),
    ("humans", "Humans"),
]

# Denser than the paper panels: six stacked rows, so markers and CI stems come
# down to keep a row legible at SI width. `panel_h` is deliberately short of
# square -- six rows plus a caption that has to carry the whole encoding (these
# SI grids draw no legend) is more than a float page holds at a square aspect,
# and the wasted space in a panel is vertical: the updates occupy well under half
# the symmetric y range.
STYLE = replace(
    points.PAPER,
    markersize=4.6,
    panel_w=1.72,
    panel_h=1.13,
    xtick_fs=7.0,
    label_fs=8.0,
    title_fs=9.0,
    errbar=dict(elinewidth=1.0, capsize=0, zorder=2),
)


def _model_cells(slug, config_tag, name):
    """The `full` variant's per-cell predictions from one run config, relabelled
    `name` in the `model` column so both configs can share one frame and the
    column-selection logic stays the shared one."""
    preds = data.load_cv_preds(slug, config_tag=config_tag)
    if preds is None:
        return None
    preds = preds[preds["model"] == "full"]
    if preds.empty:
        print(
            f"[{slug}] config {config_tag or 'reported'} has no `full` rows — skipped"
        )
        return None
    preds = preds.assign(action_label=data.action_label_col(preds))
    cells = preds.groupby(["model", *data.condition_cols(slug)], as_index=False)[
        [d for _h, d, _l in data.dvs_display(slug)]
    ].mean()
    return cells.assign(model=name)


def build_row(slug):
    """(human cells, both models' cells) for one study, or None when either
    model side is missing."""
    human, _reported_all = points.build_cells(slug)
    reported = _model_cells(slug, None, "reported")
    prereg = _model_cells(slug, PREREG_TAG, "prereg")
    if human is None or reported is None or prereg is None:
        return None
    return human, pd.concat([reported, prereg], ignore_index=True)


def draw_cell(ax, slug, key, human, model, lim, *, title, xticklabels):
    """One panel: one study's cells under one column key."""
    fcol, flevels, fcolors, _t = points.fill_spec(slug)
    vcols, cells, ci = points.value_cols_for(slug, key, human, model)
    points.draw_points(
        ax,
        cells,
        value_cols=vcols,
        color_col=fcol,
        fill_levels=flevels,
        colors=fcolors,
        lim=lim,
        ci=ci,
        style=STYLE,
        style_col=points.style_col(slug),
        xticklabels=xticklabels,
        # Six rows of "Easy Hard" would repeat the same two words up to twelve
        # times; the caption names the states once instead.
        state_labels=False,
    )
    if title:
        ax.set_title(title, fontsize=STYLE.title_fs, y=STYLE.title_y)


def build(figname="si_prereg_predictions"):
    rows = []
    for st in studies():
        built = build_row(st.slug)
        if built is None:
            print(f"[{st.slug}] skipped — needs both configs' CV predictions")
            continue
        rows.append((st, *built))
    if not rows:
        print("skipping prereg-predictions figure: no study has both configs")
        return None

    ncols = len(COLUMNS)
    fig, axes = plt.subplots(
        len(rows),
        ncols,
        figsize=(STYLE.panel_w * ncols + 0.75, STYLE.panel_h * len(rows) + 0.75),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for r, (st, human, model) in enumerate(rows):
        # y limits are shared across a row but not down the figure: the studies
        # differ severalfold in update magnitude, and one global limit would
        # flatten the smaller ones into a line.
        lim = points.symmetric_limit(st.slug, human, model)
        for c, (key, label) in enumerate(COLUMNS):
            draw_cell(
                axes[r][c],
                st.slug,
                key,
                human,
                model,
                lim,
                title=label if r == 0 else None,
                xticklabels=(r == len(rows) - 1),
            )
            if c:
                axes[r][c].tick_params(axis="y", labelleft=False)
        axes[r][0].set_ylabel(
            f"{st.paper_label}\n{points.ylabel_for(st.slug)}", fontsize=STYLE.label_fs
        )
    fig.supxlabel(points.X_AXIS_LABEL, fontsize=STYLE.label_fs)
    return savefig(fig, figname, png=False)


if __name__ == "__main__":
    apply_style("si")
    out = build()
    if out:
        print(f"wrote {out}")
