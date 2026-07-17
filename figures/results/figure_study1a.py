#!/usr/bin/env python3
"""Study 1a results figure (figures/study1a_results.pdf).

Out-of-sample model predictions next to the human data: belief update in
desire per observed action x given intimacy x given effort cell, one panel per
ablation (Base | Discomfort-only | Full) plus the Humans panel with 95%
subject-cluster bootstrap CIs. Model bars are the LOSO-CV per-cell delta_desire
means (cv_preds_summary.json). Python port of the R qmd's cv-overlay-bars
figure (analysis/food-inv-desire-analysis.qmd).

Usage:
    uv run python figures/results/figure_study1a.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import INTIMACY_COLORS, INTIMACY_LEVELS, apply_style, savefig  # noqa: E402

import _data as data  # noqa: E402
import _panels as panels  # noqa: E402

SLUG = "food_inv_desire"
CELL_COLS = ["action_label", "intimacy_condition", "effort_condition"]


def build_cells():
    """(human cells with CIs, model cells) aggregated to the condition grid,
    either side None when its inputs are missing."""
    trials = data.load_trials(SLUG)
    human = None
    if trials is not None:
        trials = trials.assign(action_label=data.action_label_col(trials))
        human = data.bootstrap_cell_means(
            trials,
            ["response_update"],
            CELL_COLS,
            seed=data.seed_for(f"figures:{SLUG}"),
        )
        print(
            f"[{SLUG}] humans: {trials['subject_id'].nunique()} subjects, {len(human)} cells"
        )

    preds = data.load_cv_preds(SLUG)
    model = None
    if preds is not None:
        preds = preds.assign(action_label=data.action_label_col(preds))
        model = preds.groupby(["model", *CELL_COLS], as_index=False)[
            "delta_desire"
        ].mean()
        data.warn_if_stale(SLUG, trials, data.load_comparison(SLUG))
    return human, model


def main():
    apply_style("si")
    human, model = build_cells()
    if human is None and model is None:
        print(f"[{SLUG}] nothing to draw yet")
        return

    panel_keys = []
    if model is not None:
        panel_keys += data.MODEL_ORDER
    if human is not None:
        panel_keys += ["humans"]

    fig, axes = plt.subplots(
        1, len(panel_keys), figsize=(6.5, 2.6), sharey=True, constrained_layout=True
    )
    axes = [axes] if len(panel_keys) == 1 else list(axes)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.5)
        ax.margins(y=0.06)

    for ax, key in zip(axes, panel_keys):
        if key == "humans":
            cells, value_col, ci, stub = human, "response_update", True, False
        else:
            cells = model[model["model"] == key]
            value_col, ci, stub = "delta_desire", False, key == "discomfort_only"
        panels.grouped_bars(
            ax,
            cells,
            value_col=value_col,
            fill_col="intimacy_condition",
            fill_levels=INTIMACY_LEVELS,
            fill_colors=INTIMACY_COLORS,
            ci=ci,
            zero_stub=stub,
        )
        ax.set_title(data.PANEL_LABELS[key])
    axes[0].set_ylabel("Belief update (desire)")

    fig.legend(
        handles=panels.intimacy_handles(),
        loc="outside lower left",
        ncol=2,
        title="Relationship",
        alignment="left",
        borderaxespad=0.2,
    )
    fig.legend(
        handles=panels.effort_hatch_handles(),
        loc="outside lower right",
        ncol=1,
        title="Effort of low-risk share",
        alignment="left",
        borderaxespad=0.2,
    )
    out = savefig(fig, "study1a_results")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
