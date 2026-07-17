#!/usr/bin/env python3
"""Held-out log-likelihood model-comparison figure (figures/model_ll_comparison.pdf).

Forest plot of the paper's primary statistic: the full model's improvement in
per-trial held-out log-likelihood over each ablation (LOSO CV), with 95%
participant-bootstrap CIs, one row per study (cv_model_comparison.json,
written by `make model-comparison`). Positive values favor the full model.
The two contrasts sit in separate panels because their scales differ by an
order of magnitude (the discomfort-only ablation is far worse than base in
the desire studies).

Studies without CV outputs are skipped; the figure grows as they land.

Usage:
    uv run python figures/results/figure_ll_comparison.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import STUDY_LABELS, apply_style, savefig  # noqa: E402

import _data as data  # noqa: E402

# ASCII hyphen, not U+2212: Arial Nova lacks the minus glyph in titles
CONTRASTS = [
    ("full_minus_base", "Full - Base"),
    ("full_minus_discomfort_only", "Full - Discomfort-only"),
]
POINT_COLOR = "0.15"


def main():
    apply_style("si")
    rows = []
    for slug, label in STUDY_LABELS.items():
        comparison = data.load_comparison(slug)
        if comparison is None:
            continue
        data.warn_if_stale(slug, data.load_trials(slug), comparison)
        for entry in comparison["primary"]:
            rows.append(
                {
                    "study": label.replace("Study ", ""),
                    "comparison": entry["comparison"],
                    "diff": entry["mean_per_trial_ll_diff"],
                    "lo": entry["ci_95"][0],
                    "hi": entry["ci_95"][1],
                }
            )
    if not rows:
        print("[model_ll_comparison] nothing to draw yet")
        return

    studies = list(dict.fromkeys(r["study"] for r in rows))
    ypos = {s: len(studies) - 1 - i for i, s in enumerate(studies)}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(5.6, 0.55 * len(studies) + 1.0),
        sharey=True,
        constrained_layout=True,
    )
    for ax, (key, title) in zip(axes, CONTRASTS):
        ax.axvline(0, color="0.75", linestyle=(0, (4, 3)), linewidth=0.8, zorder=0)
        for r in rows:
            if r["comparison"] != key:
                continue
            y = ypos[r["study"]]
            ax.plot(
                [r["lo"], r["hi"]],
                [y, y],
                color=POINT_COLOR,
                linewidth=1.1,
                solid_capstyle="butt",
                zorder=2,
            )
            ax.plot(
                r["diff"],
                y,
                "o",
                color=POINT_COLOR,
                markersize=4.5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=3,
            )
        ax.set_title(title)
        ax.set_yticks([ypos[s] for s in studies], studies)
        ax.set_ylim(-0.6, len(studies) - 0.4)
        ax.margins(x=0.12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    fig.supxlabel(
        "Held-out log-likelihood difference per trial (95% CI)",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    axes[0].set_ylabel("Study")

    out = savefig(fig, "model_ll_comparison")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
