#!/usr/bin/env python3
"""Model-vs-human scatter figures (figures/model_scatter_study{1,2,3}.pdf).

One figure per study group, three panels (Base | Discomfort-only | Full).
Each point is one scenario x condition cell: the model's out-of-sample
LOSO-CV predicted belief update (x) against the mean human belief update (y).
These are exactly the cells behind the paper's secondary correlations
(model/cv/model_comparison.py), and each panel is annotated with the reported
r per component from cv_model_comparison.json. Components (sub-study x
inferred variable) are distinguished by color (inferred variable) and marker
(sub-study); a study group renders whichever components have both data and CV
predictions.

Usage:
    uv run python figures/results/figure_model_scatter.py
"""

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import apply_style, savefig  # noqa: E402

import _data as data  # noqa: E402
import _panels as panels  # noqa: E402

# One color per inferred variable (the darker anchor of each construct's
# condition palette); sub-studies within a group differ by marker and shade
# (the joint 'b' study renders lighter) so same-DV components stay separable.
DV_COLORS = {"desire": "#7A4A5A", "effort": "#4A7A4A", "intimacy": "#274D77"}
SUBSTUDY_MARKERS = {"a": "o", "b": "^"}
SUBSTUDY_LIGHTEN = {"a": 0.0, "b": 0.45}


def _lighten(color, f):
    """Blend a hex color toward white by fraction f."""
    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(rgb + (1 - rgb) * f)


# (figure name, [(slug, sub-study letter, paper label)])
STUDY_GROUPS = [
    (
        "model_scatter_study1",
        [("food_inv_desire", "a", "1a"), ("food_inv_joint_de", "b", "1b")],
    ),
    (
        "model_scatter_study2",
        [("food_inv_intimacy", "a", "2a"), ("food_inv_joint_ie", "b", "2b")],
    ),
    (
        "model_scatter_study3",
        [("nonfood_inv_joint_de", "a", "3a"), ("nonfood_inv_joint_ie", "b", "3b")],
    ),
]


def build_components(members):
    """Cell-level merged (model, human) values per component of a study group:
    one tidy frame per (slug, dv) with columns model / model_value /
    human_value, plus the component's reported r per model."""
    components = []
    for slug, sub, paper in members:
        trials = data.load_trials(slug)
        preds = data.load_cv_preds(slug)
        if trials is None or preds is None:
            continue
        comparison = data.load_comparison(slug)
        data.warn_if_stale(slug, trials, comparison)
        spec = data.STUDY_SPECS[slug]
        for update_col, delta_col, dv in spec["dvs"]:
            cell_mean = trials.groupby(spec["keys"], as_index=False)[update_col].mean()
            merged = preds.merge(cell_mean, on=spec["keys"], how="inner")
            # every human cell should find its prediction in every model
            # variant; a shortfall means stale CV outputs or a label mismatch
            # (model_comparison.py raises on the same condition)
            expected = len(cell_mean) * preds["model"].nunique()
            if len(merged) != expected:
                print(
                    f"[{slug}] WARNING: {expected - len(merged)} of {expected} "
                    f"(cell x model) pairs have no match for {dv} — the plotted "
                    f"points won't line up with the reported r; re-run "
                    f"`make cv-{slug}` and `make model-comparison`."
                )
            frame = merged.rename(
                columns={delta_col: "model_value", update_col: "human_value"}
            )[["model", "model_value", "human_value"]]
            r_by_model = {
                m: data.correlation_for(comparison, m, dv) for m in data.MODEL_ORDER
            }
            components.append(
                {
                    "label": f"{paper} {dv}",
                    "dv": dv,
                    "marker": SUBSTUDY_MARKERS[sub],
                    "color": _lighten(DV_COLORS[dv], SUBSTUDY_LIGHTEN[sub]),
                    "frame": frame,
                    "r": r_by_model,
                }
            )
            print(f"[{slug}] scatter component {paper} {dv}: {len(frame)} rows")
    return components


def draw_group(figname, components):
    rng = np.random.default_rng(data.seed_for(f"figures:{figname}"))
    all_vals = np.concatenate(
        [
            c["frame"][["model_value", "human_value"]].to_numpy().ravel()
            for c in components
        ]
    )
    lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
    pad = 0.07 * (hi - lo)
    lim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(
        1, 3, figsize=(6.5, 2.75), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, model in zip(axes, data.MODEL_ORDER):
        r_lines = []
        for comp in components:
            sub = comp["frame"][comp["frame"]["model"] == model]
            order = rng.permutation(len(sub))
            ax.scatter(
                sub["model_value"].to_numpy()[order],
                sub["human_value"].to_numpy()[order],
                s=9,
                marker=comp["marker"],
                color=comp["color"],
                alpha=0.55,
                linewidths=0.3,
                edgecolors="white",
                zorder=2,
            )
            # r comes from cv_model_comparison.json so the figure shows the
            # paper's numbers; without that file, only the near-constant case
            # gets a label (n/a) — run `make model-comparison` for the rest
            corr = comp["r"].get(model)
            if corr is not None:
                r_lines.append(f"{comp['label']}: r = {corr['r']:.2f}")
            elif sub["model_value"].std() < 1e-3:
                r_lines.append(f"{comp['label']}: r = n/a")
        ax.plot(lim, lim, **panels.IDENTITY_LINE)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(data.MODEL_LABELS[model])
        if r_lines:
            ax.text(
                0.04,
                0.97,
                "\n".join(r_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.8,
                color="0.2",
                linespacing=1.35,
            )
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    axes[0].set_ylabel("Human belief update")
    axes[1].set_xlabel("Model predicted belief update (out-of-sample)")

    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=comp["marker"],
            markersize=5,
            color=comp["color"],
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=comp["label"],
        )
        for comp in components
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        columnspacing=1.4,
        handletextpad=0.4,
    )
    out = savefig(fig, figname)
    print(f"wrote {out}")


def main():
    apply_style("si")
    for figname, members in STUDY_GROUPS:
        components = build_components(members)
        if not components:
            print(f"[{figname}] nothing to draw yet")
            continue
        draw_group(figname, components)


if __name__ == "__main__":
    main()
