#!/usr/bin/env python3
"""Render the two consolidated SI diagnostics used by the journal manuscript.

The action-set figure combines set size, composition under relationship and
desire manipulations, and the vanilla-versus-social comparison-set check. The
variability figure combines the all-study run-spread summary with representative
predictive densities. The detailed standalone figures remain available for
diagnostic use, but the journal manuscript includes only these summaries.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "model" / "cv"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_alternatives import (  # noqa: E402
    _composition_prop,
    _draw_base_vs_full,
    _draw_composition_row,
)
from plot_si_validation import (  # noqa: E402
    RUN_SPREAD_COLORS,
    extract_observed,
    load_runs,
)
from run_delta_io import (  # noqa: E402
    WORLD_STATE_DV,
    RunDeltasUnavailable,
    load_per_run_deltas,
)

from plot_style import (  # noqa: E402
    ACTION_LABELS,
    ALT_GREY,
    OBSERVED_ACTIONS,
    STUDY_LABELS,
    apply_style,
    panel_label_at,
    savefig,
)
from study_registry import SLUGS, slugs_given, studies  # noqa: E402
from utils import get_project_root  # noqa: E402

SAVE_KW = {"png": True}

# Panel letters and the bold section headings are placed in figure coordinates
# rather than as offsets from each panel's axes, because the panels here differ
# a lot in width and in how much room their y-axis furniture takes: an offset in
# axes fractions puts every letter somewhere else on the page.
LETTER_OFFSET = 0.048  # figure-x from a panel's leftmost ink to its letter
SECTION_KW = {"fontsize": 10.5, "fontweight": "bold", "ha": "left", "va": "top"}


def _ink_left(fig, axes):
    """Leftmost figure-x drawn by `axes`, tick labels and axis labels included."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    return min(ax.get_tightbbox(renderer).transformed(inverse).x0 for ax in axes)


def _align_ink_left(fig, axes, target, right):
    """Stretch a block of axes leftward so its leftmost ink sits at `target`.

    Panels that share a gridspec column share their axes boxes, not their
    appearance: the run-spread panel spends a fifth of the page on study-by-DV
    tick labels while the density grid below it only writes "Density", so equal
    boxes leave the grid looking indented. Aligning the drawn extent instead --
    holding the right edge fixed and moving the left one -- is what makes the
    two blocks read as one column.
    """
    # `original=True` throughout: an axes with a fixed aspect (the heatmap)
    # reports a box shrunk to fit that aspect, and writing the shrunken box back
    # would freeze it at its old size instead of letting it refit the new one.
    old_left = min(ax.get_position(original=True).x0 for ax in axes)
    shift = target - _ink_left(fig, axes)
    scale = (right - old_left - shift) / (right - old_left)
    for ax in axes:
        box = ax.get_position(original=True)
        x0 = right - (right - box.x0) * scale
        x1 = right - (right - box.x1) * scale
        ax.set_position([x0, box.y0, x1 - x0, box.height])


def _panel_header(fig, letter, axes, top, title=None, letter_x=None):
    """Write a panel letter, and any bold section heading, above `axes`.

    `top` is the shared figure-y of the row the panel sits in, so that panels
    whose axes end at different heights still get their letters on one line.
    """
    if letter_x is None:
        letter_x = _ink_left(fig, axes) - LETTER_OFFSET
    panel_label_at(fig, letter, letter_x, top)
    if title is not None:
        fig.text(min(ax.get_position().x0 for ax in axes), top, title, **SECTION_KW)


def _runs_with_unique_scenarios(study_runs):
    """Concatenate study runs without merging same-named food scenarios."""
    frames = []
    for study, runs in study_runs:
        frame = runs.copy()
        frame["scenario_label"] = study + "__" + frame["scenario_label"].astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _draw_size_heatmap(ax, runs_by_study):
    sizes = np.arange(8)
    matrix = []
    labels = []
    for study in SLUGS:
        if study not in runs_by_study:
            continue
        counts = runs_by_study[study]["actions"].apply(len).sub(1)
        pct = counts.value_counts(normalize=True).mul(100)
        matrix.append([pct.get(size, 0.0) for size in sizes])
        labels.append(STUDY_LABELS[study].replace("Study ", ""))
    matrix = np.asarray(matrix)
    image = ax.imshow(matrix, cmap="Greys", vmin=0, vmax=max(50, matrix.max()))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if value >= 1:
                color = "white" if value > 0.55 * image.get_clim()[1] else "#333333"
                ax.text(
                    col,
                    row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color=color,
                )
    ax.set_xticks(sizes)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Number of generated alternatives")
    ax.set_ylabel("Study")
    ax.set_title("Comparison-set size (% of sets)")
    # imshow fixes the aspect, so the heatmap can end up smaller than the box it
    # is given; anchoring it north-west keeps it flush with the panel below and
    # its title level with panel b's heading, instead of letting the shrunken
    # box float in the middle of the cell.
    ax.set_anchor("NW")


def _draw_composition_family(fig, slot, study_slugs, runs_by_study):
    family = [
        (study, runs_by_study[study]) for study in study_slugs if study in runs_by_study
    ]
    combined = _runs_with_unique_scenarios(family)
    main_col, main_name, main_levels, prop = _composition_prop(combined)
    grid = slot.subgridspec(1, 2, wspace=0.30)
    ax_obs = fig.add_subplot(grid[0, 0])
    ax_cond = fig.add_subplot(grid[0, 1], sharey=ax_obs)
    _draw_composition_row(
        ax_obs,
        ax_cond,
        prop,
        main_col,
        main_name,
        main_levels,
        legend=True,
        show_xlabel=True,
    )
    ax_obs.set_ylabel("Proportion nearest\neach action")
    # a size below the bold panel heading above them: these two name the halves
    # of one panel, and at the default title size they run into each other.
    ax_obs.set_title("By observed action", fontsize=9)
    ax_cond.set_title("By manipulated condition", fontsize=9)
    ax_cond.tick_params(labelleft=False)
    ax_obs.set_ylim(0, 0.56)
    ax_cond.set_ylim(0, 0.56)
    if main_col == "intimacy_condition":
        ax_cond.set_xticklabels(
            ["Max.\nformal", "Some.\nformal", "Some.\nintimate", "Max.\nintimate"],
            fontsize=6.8,
        )
    return ax_obs, ax_cond


def _draw_fairness(fig, slot, given_relationship, runs_by_study):
    grid = slot.subgridspec(len(given_relationship), 1, hspace=0.12)
    axes = []
    for index, study in enumerate(given_relationship):
        ax = fig.add_subplot(grid[index, 0], sharex=axes[0] if axes else None)
        directory = get_project_root() / "model" / "outputs" / "lm" / study
        base_runs = pd.read_json(directory / "lm_runs_base.jsonl", lines=True)
        _draw_base_vs_full(
            ax,
            base_runs,
            runs_by_study[study],
            ylabel=False,
        )
        ax.text(
            0.01,
            0.80,
            STUDY_LABELS[study],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
        )
        ax.tick_params(axis="both", labelsize=6.5)
        if index < len(given_relationship) - 1:
            ax.tick_params(labelbottom=False)
        axes.append(ax)
    axes[-1].set_xticklabels(
        [
            "Max.\nformal",
            "Some.\nformal",
            "Some.\nintimate",
            "Max.\nintimate",
            "Between\n(ref.)",
        ],
        fontsize=6.5,
    )
    axes[1].set_ylabel("Feature-distribution distance (lower = closer)", fontsize=8)
    return axes


def figure_action_sets(runs_by_study):
    given_relationship = [
        study for study in slugs_given("intimacy_condition") if study in runs_by_study
    ]
    given_desire = [
        study for study in slugs_given("desire_condition") if study in runs_by_study
    ]

    # Laid out in inches, since what matters here is the shape of each panel
    # rather than how the page divides up. The top row is as tall as the
    # heatmap's fixed aspect makes it; the bottom row's panels are narrow --
    # two three-point series each -- so they get a squarer box, which keeps the
    # composition lines from stretching into near-vertical strokes. The gap
    # between the rows carries the top row's x-axis and the bottom row's
    # headings; the bottom margin carries the shared legend.
    width, height = 7.6, 6.33
    rows_in, gap_in, top_in = (2.18, 1.45), 1.15, 0.42
    fig = plt.figure(figsize=(width, height))
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=rows_in,
        hspace=gap_in / np.mean(rows_in),
        wspace=0.30,
        left=0.16,
        right=0.98,
        top=1 - top_in / height,
        bottom=(height - top_in - sum(rows_in) - gap_in) / height,
    )
    ax_size = fig.add_subplot(outer[0, 0])
    _draw_size_heatmap(ax_size, runs_by_study)
    fair_axes = _draw_fairness(fig, outer[0, 1], given_relationship, runs_by_study)
    rel_axes = _draw_composition_family(
        fig, outer[1, 0], given_relationship, runs_by_study
    )
    desire_axes = _draw_composition_family(
        fig, outer[1, 1], given_desire, runs_by_study
    )

    # Equal gridspec boxes do not make a flush column: the composition panels
    # spend a two-line y-axis label on the left, so they reach a fifth of an inch
    # further out than the panels above them, and the letters -- which have to
    # clear both -- ended up stranded from the top row. Pull the top row out to
    # the same ink edge instead. The top row's height (rows_in) is what the
    # heatmap's fixed aspect needs at that wider box, so it still fills the cell.
    for top_axes, bottom_axes, cell in (
        ([ax_size], rel_axes, outer[0, 0]),
        (fair_axes, desire_axes, outer[0, 1]),
    ):
        _align_ink_left(
            fig, top_axes, _ink_left(fig, bottom_axes), cell.get_position(fig).x1
        )

    # One letter column per gridspec column, one letter row per gridspec row,
    # taken from the cells rather than the axes: the heatmap's box is shorter
    # than its cell (fixed aspect) and the fairness stack is three axes deep.
    gap = 0.021  # letter width plus a hair of clearance, in figure-x
    top_row = outer[0, 0].get_position(fig).y1 + 0.055
    bottom_row = outer[1, 0].get_position(fig).y1 + 0.075
    left_x = _ink_left(fig, [ax_size, *rel_axes]) - gap
    right_x = _ink_left(fig, [*fair_axes, *desire_axes]) - gap
    _panel_header(fig, "a", [ax_size], top_row, letter_x=left_x)
    _panel_header(
        fig,
        "b",
        fair_axes,
        top_row,
        title="Relationship-free versus conditioned sets",
        letter_x=right_x,
    )
    _panel_header(
        fig,
        "c",
        rel_axes,
        bottom_row,
        title="Relationship manipulations",
        letter_x=left_x,
    )
    _panel_header(
        fig,
        "d",
        desire_axes,
        bottom_row,
        title="Desire manipulations",
        letter_x=right_x,
    )

    handles, labels = rel_axes[0].get_legend_handles_labels()
    for ax in (*rel_axes, *desire_axes):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.025),
        frameon=False,
        fontsize=8.5,
    )
    return savefig(fig, "si_lm_action_sets_combined", **SAVE_KW)


def _run_spread_rows():
    outputs = get_project_root() / "model" / "outputs"
    rows = []
    for study in studies():
        try:
            per_run, sigma = load_per_run_deltas(outputs / study.slug)
        except RunDeltasUnavailable as error:
            print(f"{study.slug}: {error} -- skipped")
            continue
        for dv in study.dvs:
            if dv.delta_col not in per_run:
                continue
            rows.append(
                {
                    "study": STUDY_LABELS[study.slug].replace("Study ", ""),
                    "dv": dv.label,
                    "ratios": per_run[dv.delta_col].std(axis=1) / sigma,
                    "is_world": dv.name == WORLD_STATE_DV,
                }
            )
    return rows


def _draw_run_spread(ax, rows):
    rng = np.random.default_rng(0)
    ax.axvline(1.0, color=ALT_GREY, ls="--", lw=1.0, zorder=1)
    for index, row in enumerate(rows):
        y = len(rows) - 1 - index
        color = RUN_SPREAD_COLORS[row["is_world"]]
        values = row["ratios"]
        ax.scatter(
            values,
            y + rng.uniform(-0.17, 0.17, values.size),
            s=3.5,
            color=color,
            alpha=0.35,
            lw=0,
            zorder=3,
        )
        lo, hi = np.percentile(values, [10, 90])
        ax.plot([lo, hi], [y, y], color=color, lw=1.2, zorder=4)
        median = np.median(values)
        ax.plot([median, median], [y - 0.25, y + 0.25], color=color, lw=2, zorder=5)
    ax.set_yticks(
        range(len(rows)),
        [f"{row['study']}  {row['dv'].lower()}" for row in reversed(rows)],
        fontsize=8.5,
    )
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, max(1.05, max(row["ratios"].max() for row in rows) * 1.04))
    ax.set_xlabel("Within-cell SD of per-run predictions, relative to fitted $\\sigma$")
    ax.set_title("Run-to-run prediction spread")
    ax.legend(
        handles=[
            Line2D(
                [], [], color=RUN_SPREAD_COLORS[False], lw=2, label="continuous latent"
            ),
            Line2D(
                [],
                [],
                color=RUN_SPREAD_COLORS[True],
                lw=2,
                label="two-state physical environment",
            ),
        ],
        loc="lower right",
        fontsize=8.5,
    )


def _mixture_inputs():
    output = get_project_root() / "model" / "outputs" / "food_inv_desire"
    with open(output / "cv_preds_summary.json") as stream:
        rows = [row for row in json.load(stream) if row["model"] == "full"]
    with open(output / "fit_results.json") as stream:
        fits = json.load(stream)
    sigma = float(next(row for row in fits if row["model"] == "full")["param_sigma"])
    trials = pd.read_csv(
        get_project_root() / "data" / "food_inv_desire" / "main_trials.csv"
    )
    wide = trials.pivot_table(
        index=[
            "subject_id",
            "scenario_label",
            "action_condition",
            "effort_condition",
            "intimacy_condition",
        ],
        columns="stage",
        values="response",
    ).reset_index()
    wide["update"] = wide["posterior"] - wide["prior"]
    ordered = sorted(rows, key=lambda row: row["delta_desire"])
    picks = [
        ordered[int(quantile * (len(ordered) - 1))]
        for quantile in (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)
    ]
    return picks, wide, sigma


def _draw_mixture_panels(fig, slot):
    picks, observed, sigma = _mixture_inputs()
    grid = slot.subgridspec(2, 3, hspace=0.55, wspace=0.23)
    axes = np.empty((2, 3), dtype=object)
    domain = np.linspace(-1, 1, 401)
    for index, prediction in enumerate(picks):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(grid[row, col], sharex=axes[0, 0] if index else None)
        axes[row, col] = ax
        action = OBSERVED_ACTIONS[prediction["action"]]
        human = observed[
            (observed["scenario_label"] == prediction["scenario_label"])
            & (observed["action_condition"] == action)
            & (observed["effort_condition"] == prediction["effort_condition"])
            & (observed["intimacy_condition"] == prediction["intimacy_condition"])
        ]["update"].to_numpy()
        deltas = np.asarray(prediction["delta_desire_runs"])
        density = np.mean(
            np.exp(-((domain[None, :] - deltas[:, None]) ** 2) / (2 * sigma**2))
            / (sigma * np.sqrt(2 * np.pi)),
            axis=0,
        )
        ax.hist(
            human,
            bins=np.arange(-1, 1.01, 0.125),
            density=True,
            color="#DDDDDD",
            edgecolor="white",
            lw=0.5,
        )
        ax.plot(domain, density, color="#333333", lw=1.3)
        ax.vlines(deltas, 0, 0.16, color="#777777", lw=0.5, alpha=0.7)
        ax.set_title(
            f"{prediction['scenario_label']} -- {ACTION_LABELS[action].lower()}",
            fontsize=8,
        )
        ax.set_xlim(-1, 1)
        if row == 1:
            ax.set_xlabel("Belief update", fontsize=8.5)
        if col == 0:
            ax.set_ylabel("Density", fontsize=8.5)
        else:
            ax.tick_params(labelleft=False)
    return list(axes.ravel())


def figure_variability():
    rows = _run_spread_rows()
    if not rows:
        raise RuntimeError("No per-run predictions are available")
    fig = plt.figure(figsize=(7.0, 7.0))
    right = 0.98
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.05, 1.0],
        hspace=0.40,
        # Room on the left for panel a's study-by-DV tick labels, which are far
        # wider than anything the density grid below writes; the grid is pulled
        # out to match once both are drawn.
        left=0.30,
        right=right,
        top=0.94,
        bottom=0.075,
    )
    ax = fig.add_subplot(outer[0, 0])
    _draw_run_spread(ax, rows)
    mixture_axes = _draw_mixture_panels(fig, outer[1, 0])
    _align_ink_left(fig, mixture_axes, _ink_left(fig, [ax]), right)

    _panel_header(fig, "a", [ax], outer[0, 0].get_position(fig).y1 + 0.045)
    _panel_header(
        fig,
        "b",
        mixture_axes,
        max(ax.get_position().y1 for ax in mixture_axes) + 0.055,
        title="Representative predictive densities",
    )
    return savefig(fig, "si_lm_variability_checks", **SAVE_KW)


def main():
    apply_style("si")
    runs_by_study = {}
    for study in SLUGS:
        runs = load_runs(study)
        if runs is None:
            print(f"{study}: no LM runs found -- skipped")
            continue
        runs_by_study[study] = runs
        # Validate the same observed-action extraction used by the source figures.
        extract_observed(runs)
    print(f"wrote {figure_action_sets(runs_by_study)}")
    print(f"wrote {figure_variability()}")


if __name__ == "__main__":
    main()
