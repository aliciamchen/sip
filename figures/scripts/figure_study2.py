#!/usr/bin/env python3
"""Study 2 results figure (figures/study2_results.pdf).

Intimacy inference from observed actions. Panel (a), Study 2a: belief update
in intimacy per observed action x given desire x given effort cell, one column
per ablation's out-of-sample predictions plus Humans (mirrors the Study 1a
figure with desire as the fill palette). Panel (b), Study 2b: human joint
belief updates in (intimacy update, effort update) space per observed action x
given desire cell. Panel (c), Study 2b: by-DV cell means with one column per
model plus Humans.

Renders whatever inputs exist (2a data are in; 2b panels appear once its data
and CV predictions land).

Usage:
    uv run python figures/results/figure_study2.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import DESIRE_COLORS, apply_style, panel_label, savefig  # noqa: E402

import _data as data  # noqa: E402
import _joint as joint  # noqa: E402
import _panels as panels  # noqa: E402

SLUG_A = "food_inv_intimacy"  # Study 2a
SLUG_B = "food_inv_joint_ie"  # Study 2b
CELLS_A = data.condition_cols(
    SLUG_A
)  # ["action_label", "desire_condition", "effort_condition"]
CELLS_B = data.condition_cols(SLUG_B)  # ["action_label", "desire_condition"]
HUMAN_COL_A, DELTA_COL_A, _ = data.dvs_display(SLUG_A)[0]  # 2a intimacy
DVS_B = data.dvs_display(SLUG_B)  # [(intimacy cols, "Intimacy"), (effort cols, ...)]
DESIRE_LEVELS = ["low", "high"]


def build_2a():
    trials = data.load_trials(SLUG_A)
    human = None
    if trials is not None:
        trials = trials.assign(action_label=data.action_label_col(trials))
        human = data.bootstrap_cell_means(
            trials,
            [HUMAN_COL_A],
            CELLS_A,
            seed=data.seed_for(f"figures:{SLUG_A}"),
        )
        print(
            f"[{SLUG_A}] humans: {trials['subject_id'].nunique()} subjects, {len(human)} cells"
        )
    preds = data.load_cv_preds(SLUG_A)
    model = None
    if preds is not None:
        preds = preds.assign(action_label=data.action_label_col(preds))
        model = preds.groupby(["model", *CELLS_A], as_index=False)[DELTA_COL_A].mean()
        data.warn_if_stale(SLUG_A, trials, data.load_comparison(SLUG_A))
    return human, model


def draw_2a(subfig, human, model):
    grid_cols = (data.MODEL_ORDER if model is not None else []) + (
        ["humans"] if human is not None else []
    )
    axes = subfig.subplots(1, len(grid_cols), sharey=True, squeeze=False)[0]
    for ax, key in zip(axes, grid_cols):
        if key == "humans":
            cells, value_col, ci, stub = human, HUMAN_COL_A, True, False
        else:
            cells = model[model["model"] == key]
            # the base model has no intimacy representation, so its predicted
            # updates are ~0 everywhere — draw them as visible zero stubs
            value_col, ci, stub = DELTA_COL_A, False, key == "base"
        panels.grouped_bars(
            ax,
            cells,
            value_col=value_col,
            fill_col="desire_condition",
            fill_levels=DESIRE_LEVELS,
            fill_colors=DESIRE_COLORS,
            ci=ci,
            zero_stub=stub,
        )
        ax.tick_params(axis="x", labelsize=7)
        ax.margins(y=0.06)
        ax.set_title(data.PANEL_LABELS[key])
    axes[0].set_ylabel("Belief update\n(intimacy)")
    panel_label(axes[0], "a", dx=-0.42, dy=1.02)
    subfig.legend(
        handles=panels.desire_handles("patch"),
        loc="outside lower left",
        ncol=2,
        title="Desire",
        alignment="left",
        borderaxespad=0.2,
    )
    subfig.legend(
        handles=panels.effort_hatch_handles(),
        loc="outside lower right",
        ncol=2,
        title="Effort of low-risk share",
        alignment="left",
        borderaxespad=0.2,
    )


def main():
    apply_style("si")
    human_a, model_a = build_2a()
    human_b, model_b = joint.build_joint_cells(SLUG_B, DVS_B, CELLS_B)
    if human_a is None and human_b is None:
        print("[study2] nothing to draw yet")
        return

    rows = [
        ("a", human_a is not None or model_a is not None),
        ("b", human_b is not None),
        ("c", human_b is not None),
    ]
    live = [r for r, ok in rows if ok]

    heights = {"a": 1.05, "b": 1.0, "c": 1.0}
    fig = plt.figure(figsize=(6.5, 2.8 * len(live)), layout="constrained")
    subfigs = fig.subfigures(len(live), 1, height_ratios=[heights[r] for r in live])
    subfigs = [subfigs] if len(live) == 1 else list(subfigs)
    for key, sub in zip(live, subfigs):
        if key == "a":
            draw_2a(sub, human_a, model_a)
        elif key == "b":
            joint.draw_vector_row(
                sub,
                human_b,
                x_col="intimacy_rating_update",
                y_col="effort_rating_update",
                x_label="Intimacy belief update",
                y_label="Effort belief update\n(low-risk share)",
                color_col="desire_condition",
                colors=DESIRE_COLORS,
                color_handles=panels.desire_handles("line"),
                color_title="Desire",
                letter="b",
            )
        else:
            joint.draw_dv_grid(
                sub,
                human_b,
                model_b,
                dvs=DVS_B,
                color_col="desire_condition",
                color_levels=DESIRE_LEVELS,
                colors=DESIRE_COLORS,
                letter="c",
            )

    out = savefig(fig, "study2_results")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
