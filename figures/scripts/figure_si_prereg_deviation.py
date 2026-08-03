#!/usr/bin/env python3
"""What the comparison-set reweighting bought, per study, out of sample.

The reweighting is a deviation from the preregistered model specification, so the
paper reports the preregistered model (eta = 0) beside the reported one. The
numbers are in tab:prereg-deviation; this is the same six differences drawn with
their confidence intervals, because the point of the comparison is which studies
the deviation actually helps, and three of the six differences are small enough
that a column of signed numbers reads as "all positive" when it is not.

Each row is one study's paired difference in per-trial held-out log-likelihood,
reported minus preregistered, with a 95% subject-cluster bootstrap CI over the
same participants. Positive favours the reweighting. The fitted eta is printed
beside each row: it is what makes the pattern legible, since the reweighting
generalizes in the studies where eta fits large and not in the two where it fits
near zero. Study 1a has no eta at all -- the scope rule grants it no reweighting
-- so its two models are the same fit and its difference is exactly zero.

Reads each study's `alt/compare_uniform-noreweight_vs_reported.json`, written by
`model_comparison.py --compare-configs`, so nothing here recomputes a statistic
the paper states elsewhere.

Usage:
    uv run python figures/scripts/figure_si_prereg_deviation.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import ALT_GREY, apply_style, savefig  # noqa: E402
from study_registry import studies  # noqa: E402

import _panels as panels  # noqa: E402

COMPARE_NAME = "compare_uniform-noreweight_vs_reported.json"
POINT_COLOR = "#333333"
#: Rows whose CI straddles zero are drawn lighter: the deviation neither helped
#: nor hurt them detectably, which is a third outcome the eye should not have to
#: read off overlapping interval ends.
TIE_COLOR = ALT_GREY


def load_rows():
    """One row per study with its paired difference, CI and fitted eta. Studies
    without the comparison are skipped with a message rather than dropped
    silently."""
    root = Path(__file__).resolve().parent.parent.parent / "model" / "outputs"
    rows = []
    for st in studies():
        path = root / st.slug / "alt" / COMPARE_NAME
        if not path.exists():
            print(
                f"{st.slug}: no {COMPARE_NAME} — run bin/prereg-eta0.sh, then "
                f"`model_comparison.py --study {st.slug} --compare-configs "
                f"uniform-noreweight reported`"
            )
            continue
        entry = next(
            (
                e
                for e in json.loads(path.read_text())["per_variant"]
                if e["variant"] == "full"
            ),
            None,
        )
        if entry is None:
            print(f"{st.slug}: {COMPARE_NAME} has no `full` entry — skipped")
            continue
        fits = json.loads((root / st.slug / "fit_results.json").read_text())
        full = next(r for r in fits if r["model"] == "full")
        rows.append(
            {
                "label": st.paper_label.replace("Study ", ""),
                "diff": entry["mean_per_trial_ll_diff"],
                "ci": entry["ci_95"],
                # Absent eta is a fitted zero, not missing data: the scope rule
                # grants this study no reweighting, so eta does not exist.
                "eta": full.get("param_eta"),
            }
        )
    return rows


def build(figname="si_prereg_deviation"):
    rows = load_rows()
    if not rows:
        print("skipping prereg-deviation figure: no comparisons available")
        return None

    fig, ax = plt.subplots(figsize=(5.0, 0.42 * len(rows) + 1.2))
    ax.axvline(0.0, **panels.ZERO_LINE)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i  # paper order, top to bottom
        lo, hi = r["ci"]
        color = POINT_COLOR if (lo > 0 or hi < 0) else TIE_COLOR
        ax.plot([lo, hi], [y, y], color=color, lw=1.4, solid_capstyle="butt", zorder=3)
        ax.plot(
            r["diff"],
            y,
            "o",
            ms=4.6,
            color=color,
            mec="white",
            mew=0.5,
            zorder=4,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in reversed(rows)])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(
        "Held-out log-likelihood per trial,\nreported $-$ preregistered "
        "($\\eta = 0$) model"
    )

    # eta beside each row, in a right-hand column outside the data area, so the
    # explanatory variable sits next to the effect without competing with it.
    trans = ax.get_yaxis_transform()
    ax.text(1.03, 1.0, "$\\eta$", transform=ax.transAxes, fontsize=9, ha="left")
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        eta = r["eta"]
        ax.text(
            1.03,
            y,
            "--" if eta is None else f"{eta:.2f}",
            transform=trans,
            fontsize=9,
            va="center",
            ha="left",
            color=POINT_COLOR if eta is not None else ALT_GREY,
            clip_on=False,
        )
    for r in rows:
        lo, hi = r["ci"]
        verdict = "reported" if lo > 0 else ("preregistered" if hi < 0 else "tie")
        print(
            f"  {r['label']}: {r['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
            f"eta={'--' if r['eta'] is None else f'{r["eta"]:.3f}'}  -> {verdict}"
        )
    fig.tight_layout()
    return savefig(fig, figname, png=False)


if __name__ == "__main__":
    apply_style("si")
    out = build()
    if out:
        print(f"wrote {out}")
