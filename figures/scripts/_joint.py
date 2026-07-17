"""Row builders shared by the joint two-latent studies (1b, 2b, 3a, 3b).

Each joint study's results figure is built from the same two rows — a 2D
joint-belief-update vector panel and a by-DV grid of model columns plus the
Humans column — parameterized by which latent pairs with effort and which
given condition supplies the color. 3a mirrors 1b (desire + effort given
intimacy) and 3b mirrors 2b (intimacy + effort given desire), so they reuse
these builders on the nonfood data unchanged.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import panel_label  # noqa: E402

import _data as data  # noqa: E402
import _panels as panels  # noqa: E402


def build_joint_cells(slug, dvs, cell_cols):
    """(human cells with CIs, model cells) for a joint study, either side None
    when its inputs are missing."""
    trials = data.load_trials(slug)
    human = None
    if trials is not None:
        trials = trials.assign(action_label=data.action_label_col(trials))
        human = data.bootstrap_cell_means(
            trials,
            [dv[0] for dv in dvs],
            cell_cols,
            seed=data.seed_for(f"figures:{slug}"),
        )
        print(
            f"[{slug}] humans: {trials['subject_id'].nunique()} subjects, {len(human)} cells"
        )
    preds = data.load_cv_preds(slug)
    model = None
    if preds is not None:
        preds = preds.assign(action_label=data.action_label_col(preds))
        model = preds.groupby(["model", *cell_cols], as_index=False)[
            [dv[1] for dv in dvs]
        ].mean()
        data.warn_if_stale(slug, trials, data.load_comparison(slug))
    return human, model


def draw_vector_row(
    subfig,
    human,
    *,
    x_col,
    y_col,
    x_label,
    y_label,
    color_col,
    colors,
    color_handles,
    color_title,
    letter,
):
    """One subfigure row: the joint 2D update panel with its legends."""
    ax = subfig.add_subplot(111)
    panels.joint_vector(
        ax,
        human,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        colors=colors,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    panel_label(ax, letter, dx=-0.55, dy=1.0)
    subfig.legend(
        handles=color_handles,
        loc="outside right upper",
        title=color_title,
        alignment="left",
    )
    subfig.legend(
        handles=panels.action_marker_handles(),
        loc="outside right lower",
        title="Observed action",
        alignment="left",
    )
    return ax


def draw_dv_grid(
    subfig,
    human,
    model,
    *,
    dvs,
    color_col,
    color_levels,
    colors,
    letter,
):
    """One subfigure row: DV rows x (model columns + Humans) of dodged lines."""
    grid_cols = (data.MODEL_ORDER if model is not None else []) + (
        ["humans"] if human is not None else []
    )
    if not grid_cols:
        return None
    axes = subfig.subplots(2, len(grid_cols), sharey=True, sharex=True, squeeze=False)
    for ri, (human_col, delta_col, dv_label) in enumerate(dvs):
        for ci, key in enumerate(grid_cols):
            ax = axes[ri][ci]
            if key == "humans":
                cells, value_col, ci_flag = human, human_col, True
            else:
                cells = model[model["model"] == key]
                value_col, ci_flag = delta_col, False
            panels.dodged_lines(
                ax,
                cells,
                value_col=value_col,
                color_col=color_col,
                color_levels=color_levels,
                colors=colors,
                ci=ci_flag,
            )
            ax.tick_params(axis="x", labelsize=7)
            if ri == 0:
                ax.set_title(data.PANEL_LABELS[key])
            if ci == len(grid_cols) - 1:
                ax.text(
                    1.06,
                    0.5,
                    dv_label,
                    transform=ax.transAxes,
                    rotation=270,
                    va="center",
                    ha="left",
                    fontsize=plt.rcParams["axes.labelsize"],
                )
        axes[ri][0].set_ylabel("Belief update")
    panel_label(axes[0][0], letter, dx=-0.42, dy=1.02)
    return axes
