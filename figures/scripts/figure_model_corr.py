#!/usr/bin/env python3
"""Model-vs-human correlation scatters, per study, with the ablation contrast.

One figure per study group per granularity (6 total):
    model_corr_by_condition_study{1,2,3}
    model_corr_by_scenario_study{1,2,3}

Each figure is a 3x3 grid: rows = sub-study a, sub-study b, and the study
aggregate (pooling both); columns = Base | Discomfort-only | Full. Each panel
scatters the model's out-of-sample LOSO-CV predicted belief update (x) against
the mean human belief update (y), one point per cell, colored by inferred DV,
with the Pearson r per DV (and a pooled "All" r where a panel spans DVs). A DV an
ablation cannot infer collapses to a vertical stripe at x=0 and is labelled
r = n/a — that is the visible ablation contrast.

Granularities:
    condition : one point per condition, averaged over the 16 scenarios
                (condition_cols(slug) = display action + given conditions).
    scenario  : one point per scenario x condition (STUDY_SPECS[slug]["keys"] —
                the cells behind the paper's secondary correlations).

Usage:
    uv run python figures/scripts/figure_model_corr.py
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import DV_MARKERS, apply_style, savefig  # noqa: E402

import _data as data  # noqa: E402
import _panels as panels  # noqa: E402

# Same construct colors as figure_model_scatter.py (dark anchor per DV). Used by
# the per-study grids; the poster aggregate encodes DV by MARKER SHAPE instead.
DV_COLORS = {"desire": "#7A4A5A", "effort": "#4A7A4A", "intimacy": "#274D77"}
DV_LABELS = {"desire": "Desire", "effort": "Effort", "intimacy": "Intimacy"}
DV_LEGEND_ORDER = ["desire", "intimacy", "effort"]
# Aggregate poster figure: DV -> marker shape (from plot_style, shared with the
# points panels), one point color for all.
AGG_POINT_COLOR = "#333333"

# (figure stem, study title, [(slug, paper label)]); columns are the ablations.
STUDY_GROUPS = [
    ("study1", "Study 1", [("food_inv_desire", "1a"), ("food_inv_joint_de", "1b")]),
    ("study2", "Study 2", [("food_inv_intimacy", "2a"), ("food_inv_joint_ie", "2b")]),
    (
        "study3",
        "Study 3",
        [("nonfood_inv_joint_de", "3a"), ("nonfood_inv_joint_ie", "3b")],
    ),
]


def _cell_points(slug, level):
    """{model: [(dv, model_arr, human_arr), ...]} at `level` ('scenario' or
    'condition'). Empty dict if the study's inputs are missing."""
    trials = data.load_trials(slug)
    preds = data.load_cv_preds(slug)
    if trials is None or preds is None:
        return {}
    data.warn_if_stale(slug, trials, data.load_comparison(slug))
    spec = data.STUDY_SPECS[slug]
    if level == "scenario":
        keys = spec["keys"]
    else:  # condition-level: drop scenario, average over the 16 scenarios
        keys = data.condition_cols(slug)
        preds = preds.copy()
        preds["action_label"] = data.action_label_col(preds)
        trials = trials.copy()
        trials["action_label"] = data.action_label_col(trials)
    out = {}
    for model in data.MODEL_ORDER:
        pm = preds[preds["model"] == model]
        groups = []
        for update_col, delta_col, dv in spec["dvs"]:
            human = trials.groupby(keys, as_index=False)[update_col].mean()
            modeld = pm.groupby(keys, as_index=False)[delta_col].mean()
            merged = modeld.merge(human, on=keys, how="inner")
            groups.append(
                (dv, merged[delta_col].to_numpy(), merged[update_col].to_numpy())
            )
        out[model] = groups
    return out


def _draw_panel(ax, groups, lim, rng):
    """groups: list of (dv, x, y). Scatter (colored by DV) + identity + r text.
    A DV whose model predictions have no spread is labelled r = n/a (the flat
    ablation case)."""
    by_dv = OrderedDict()
    for dv, x, y in groups:
        by_dv.setdefault(dv, [[], []])
        by_dv[dv][0].append(x)
        by_dv[dv][1].append(y)

    r_lines, all_x, all_y = [], [], []
    for dv, (xs, ys) in by_dv.items():
        x, y = np.concatenate(xs), np.concatenate(ys)
        all_x.append(x)
        all_y.append(y)
        order = rng.permutation(len(x))
        ax.scatter(
            x[order],
            y[order],
            s=7,
            color=DV_COLORS[dv],
            alpha=0.6,
            linewidths=0,
            zorder=2,
        )
        if len(x) > 2 and np.std(x) > 1e-6 and np.std(y) > 1e-6:
            r = f"$r\\!=\\!{np.corrcoef(x, y)[0, 1]:.2f}$"
        else:
            r = "$r=$ n/a"
        r_lines.append(f"{DV_LABELS[dv]} {r}")
    all_x, all_y = np.concatenate(all_x), np.concatenate(all_y)
    if len(by_dv) > 1 and np.std(all_x) > 1e-6:
        r_lines.append(f"All $r\\!=\\!{np.corrcoef(all_x, all_y)[0, 1]:.2f}$")

    ax.plot(lim, lim, **panels.IDENTITY_LINE)
    ax.axhline(0, color="0.85", lw=0.5, zorder=1)
    ax.axvline(0, color="0.85", lw=0.5, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.text(
        0.05,
        0.97,
        "\n".join(r_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.4,
        color="0.15",
        linespacing=1.35,
    )
    ax.text(
        0.97,
        0.05,
        f"$n={len(all_x)}$",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=5.8,
        color="0.5",
    )


def build_study_figure(stem, study_name, members, level):
    figname = f"model_corr_by_{level}_{stem}"
    rng = np.random.default_rng(data.seed_for(f"figures:{figname}"))
    per_slug = {slug: _cell_points(slug, level) for slug, _ in members}
    if not any(per_slug.values()):
        print(f"[{figname}] nothing to draw yet")
        return

    # rows: (row label, {model: groups}) — sub-study a, sub-study b, aggregate.
    rows = [
        (members[0][1], per_slug[members[0][0]]),
        (members[1][1], per_slug[members[1][0]]),
    ]
    agg = {
        m: [g for slug, _ in members for g in per_slug[slug].get(m, [])]
        for m in data.MODEL_ORDER
    }
    rows.append((study_name, agg))

    vals = [
        v
        for _, md in rows
        for m in data.MODEL_ORDER
        for _, x, y in md.get(m, [])
        for v in (x, y)
    ]
    lim_hi = np.nanmax(np.abs(np.concatenate(vals))) * 1.08
    lim = (-lim_hi, lim_hi)

    fig, axes = plt.subplots(
        3, 3, figsize=(6.4, 6.9), sharex=True, sharey=True, constrained_layout=True
    )
    for i, (row_label, md) in enumerate(rows):
        for j, model in enumerate(data.MODEL_ORDER):
            ax = axes[i][j]
            _draw_panel(ax, md.get(model, []), lim, rng)
            if i == 0:
                ax.set_title(data.MODEL_LABELS[model], fontsize=9, pad=3)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        axes[i][0].set_ylabel(row_label, fontsize=10, fontweight="bold")
    # x-label on the bottom-centre panel (inside the grid) so it doesn't collide
    # with the below-plot legend the way a fig.supxlabel does.
    axes[2][1].set_xlabel("Model predicted belief update", fontsize=8.5)
    fig.supylabel("Human belief update", fontsize=8.5)

    present = {
        dv for _, md in rows for m in data.MODEL_ORDER for dv, _, _ in md.get(m, [])
    }
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=5,
            color=DV_COLORS[dv],
            label=DV_LABELS[dv],
        )
        for dv in DV_LEGEND_ORDER
        if dv in present
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        columnspacing=1.6,
        handletextpad=0.4,
    )
    out = savefig(fig, figname)
    print(f"wrote {out}")


def _agg_points():
    """{model: [(dv, x, y, y_lo, y_hi), ...]} pooling all six experiments at
    condition level (averaged over the 16 scenarios). y_lo/y_hi are the human
    condition mean's 95% subject-cluster bootstrap CI (resampling that study's
    participants); the model x is the out-of-sample CV delta averaged over
    scenarios (a point estimate, no error bar)."""
    out = {m: [] for m in data.MODEL_ORDER}
    for _stem, _name, members in STUDY_GROUPS:
        for slug, _paper in members:
            trials = data.load_trials(slug)
            preds = data.load_cv_preds(slug)
            if trials is None or preds is None:
                continue
            data.warn_if_stale(slug, trials, data.load_comparison(slug))
            spec = data.STUDY_SPECS[slug]
            keys = data.condition_cols(slug)
            trials = trials.copy()
            trials["action_label"] = data.action_label_col(trials)
            preds = preds.copy()
            preds["action_label"] = data.action_label_col(preds)
            rng = np.random.default_rng(data.seed_for(f"figures:agg:{slug}"))
            for update_col, delta_col, dv in spec["dvs"]:
                boot = data.bootstrap_cell_means(trials, [update_col], keys, rng=rng)
                for model in data.MODEL_ORDER:
                    pm = (
                        preds[preds["model"] == model]
                        .groupby(keys, as_index=False)[delta_col]
                        .mean()
                    )
                    merged = pm.merge(boot, on=keys, how="inner")
                    out[model].append(
                        (
                            dv,
                            merged[delta_col].to_numpy(),
                            merged[update_col].to_numpy(),
                            merged[f"{update_col}_ci_lower"].to_numpy(),
                            merged[f"{update_col}_ci_upper"].to_numpy(),
                        )
                    )
    return out


def _draw_agg_panel(ax, groups, lim):
    """groups: list of (dv, x, y, y_lo, y_hi). DV is encoded by MARKER SHAPE (one
    point color for all); vertical human-CI error bars sit behind the points; the
    panel shows a single pooled Pearson r over all points. Sized for a poster."""
    by_dv = OrderedDict()
    for dv, x, y, ylo, yhi in groups:
        by_dv.setdefault(dv, []).append((x, y, ylo, yhi))

    all_x, all_y = [], []
    for dv, parts in by_dv.items():
        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        ylo = np.concatenate([p[2] for p in parts])
        yhi = np.concatenate([p[3] for p in parts])
        all_x.append(x)
        all_y.append(y)
        yerr = np.clip(np.vstack([y - ylo, yhi - y]), 0, None)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="0.55",
            elinewidth=1.6,
            alpha=0.7,
            zorder=1,
        )
        ax.scatter(
            x,
            y,
            s=45,
            marker=DV_MARKERS[dv],
            color=AGG_POINT_COLOR,
            alpha=0.45,
            linewidths=0,
            zorder=3,
        )
    all_x, all_y = np.concatenate(all_x), np.concatenate(all_y)
    r = np.corrcoef(all_x, all_y)[0, 1]

    ax.plot(lim, lim, **panels.IDENTITY_LINE)
    ax.axhline(0, color="0.85", lw=0.7, zorder=0)
    ax.axvline(0, color="0.85", lw=0.7, zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=13)
    ax.text(
        0.05,
        0.95,
        f"$r = {r:.2f}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=19,
        color="0.1",
    )


def build_aggregate_figure():
    """Single 3-panel (Base | Discomfort-only | Full) scatter pooling all six
    experiments — one point per (experiment x condition x DV) — with human
    bootstrap-CI error bars."""
    figname = "model_corr_all_conditions"
    agg = _agg_points()
    vals = np.concatenate(
        [
            arr
            for m in data.MODEL_ORDER
            for _dv, x, y, ylo, yhi in agg[m]
            for arr in (x, y, ylo, yhi)
        ]
    )
    lim_hi = np.nanmax(np.abs(vals)) * 1.05
    lim = (-lim_hi, lim_hi)

    fig, axes = plt.subplots(
        1, 3, figsize=(10, 4.2), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, model in zip(axes, data.MODEL_ORDER):
        _draw_agg_panel(ax, agg[model], lim)
        ax.set_title(data.MODEL_LABELS[model], fontsize=19)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    axes[0].set_ylabel("Human belief update", fontsize=16)
    axes[1].set_xlabel("Model predicted belief update (out-of-sample)", fontsize=16)

    present = {dv for m in data.MODEL_ORDER for dv, *_ in agg[m]}
    # "Target of inference" rides along as a marker-less first entry so the
    # label and the three shape keys all sit on ONE horizontal row.
    handles = [
        Line2D([], [], linestyle="none", marker="None", label="Target of inference")
    ]
    handles += [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=DV_MARKERS[dv],
            markersize=11,
            color=AGG_POINT_COLOR,
            label=DV_LABELS[dv],
        )
        for dv in DV_LEGEND_ORDER
        if dv in present
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        columnspacing=1.6,
        handletextpad=0.5,
        handlelength=1.2,
        fontsize=15,
    )
    out = savefig(fig, figname)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--figures",
        choices=["all", "per-study", "aggregate"],
        default="all",
        help="Which figures to render (default: all).",
    )
    args = ap.parse_args()
    apply_style("si")
    if args.figures in ("all", "aggregate"):
        build_aggregate_figure()
    if args.figures in ("all", "per-study"):
        for level in ("condition", "scenario"):
            for stem, study_name, members in STUDY_GROUPS:
                build_study_figure(stem, study_name, members, level)


if __name__ == "__main__":
    main()
