#!/usr/bin/env python3
"""Model-vs-human scatter, one row per study and one column per ablation.

A diagnostic, NOT a paper figure -- it is deliberately not in the SI and not in
the Makefile's journal sync (decided 2026-08-02). Kept because it is the only
view that shows per-study, per-latent agreement in one place, and because it is
what surfaced the bootstrap attenuation documented in `_dv_label`.

The main text carries a single pooled correlation panel, which shows that the
full model tracks the data overall but not which studies carry that agreement.
This resolves it: each panel plots one study's out-of-sample predicted belief
update against the human mean for the same cell, at the finest grain the model
predicts -- scenario x observed action x given condition -- so the scatter is
the per-scenario counterpart of the averaged panels rather than a re-plot of
them.

Points are cells, marker shape is the inferred latent (matching the results
figures), and the annotation is the Pearson r over a study's cells with a 95%
subject-cluster bootstrap CI. Human cells at this grain rest on few judgments
each, so the scatter is deliberately drawn without error bars -- per-cell
uncertainty is what the bootstrap CI on r summarizes, and 300+ error bars per
panel would obscure the cloud they describe. An ablation that cannot infer a
latent predicts a constant for it, which shows up as a vertical stripe and an
undefined r.

Usage:
    uv run python figures/scripts/figure_si_model_scatter.py [--study <slug>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import DV_MARKERS, apply_style, savefig  # noqa: E402
from study_registry import studies, study  # noqa: E402

import _agg as agg  # noqa: E402
import _data as data  # noqa: E402
import _panels as panels  # noqa: E402

N_BOOT = 1000
POINT_COLOR = "#333333"
PANEL_IN = 1.75  # side of one square panel, inches


def study_cells(slug):
    """(cells, boots, preds) on the scenario x condition grid, or None when the
    study's data or CV predictions aren't there yet."""
    trials = data.load_trials(slug)
    preds = data.load_cv_preds(slug)
    if trials is None or preds is None:
        return None
    data.warn_if_stale(slug, trials, data.load_comparison(slug))
    trials = trials.assign(action_label=data.action_label_col(trials))
    preds = preds.assign(action_label=data.action_label_col(preds))
    keys = ["scenario_label", *data.condition_cols(slug)]
    cells, boots = data.bootstrap_cell_means(
        trials,
        [u for u, _d, _dv in data.STUDY_SPECS[slug]["dvs"]],
        keys,
        n_boot=N_BOOT,
        seed=data.seed_for(f"figures:si_model_scatter:{slug}"),
        return_boots=True,
    )
    return cells, boots, preds, keys


def panel_series(slug, cells, boots, preds, keys, model):
    """[(dv, x, y, ys_boot), ...] for one study x model panel, one entry per DV.

    The merge runs predictions-into-cells so that x, y and the resampled y share
    one row order; reversing it silently mispairs them (see `_agg.agg_points`).
    """
    series = []
    sub = preds[preds["model"] == model]
    if sub.empty:
        return series
    merged = sub.merge(cells.reset_index(), on=keys, how="inner")
    if merged.empty:
        return series
    order = merged["index"].to_numpy()
    for update_col, delta_col, dv in data.STUDY_SPECS[slug]["dvs"]:
        if delta_col not in merged.columns:
            continue
        series.append(
            (
                dv,
                merged[delta_col].to_numpy(),
                merged[update_col].to_numpy(),
                boots[update_col][:, order],
            )
        )
    return series


def _dv_label(dv, x, y, ys, *, note=None):
    """One annotation line: "desire  r = 0.83", or a statement that the variant
    predicts no variation in this latent.

    A variant lacking the term a latent needs predicts the same update for every
    cell, and a correlation against a constant is undefined rather than zero --
    printing "nan" would read as a failed computation instead of the structural
    fact it is.

    Deliberately no confidence interval. This still calls the subject-cluster
    `corr_with_ci`, which resamples participants with replacement, so a resampled
    cell mean rests on fewer unique participants than the observed one and is
    noisier; noise in the human means attenuates r against a fixed x, putting the
    whole bootstrap distribution BELOW the observed r. These scenario-level cells
    hold a handful of judgments each, so the percentile interval ends up excluding
    the estimate it is meant to bracket, for 15 of the 24 identified correlations
    here. The intervals are printed to stdout rather than published as if valid.

    The 2026-08-03 audit found the same bias at condition level too -- smaller, but
    still enough to make the pooled panel's interval exclude its own r -- so the
    condition-level panels now bootstrap the plotted points instead
    (`_agg.corr_with_pair_ci`). That construction is unbiased here and would work
    at this grain as well; it is not used because a per-panel interval was never
    the point of this diagnostic.
    """
    r, lo, hi = agg.corr_with_ci(x, y, ys)
    if note is not None and np.isfinite(r):
        note.append((dv, r, lo, hi, len(x)))
    if not np.isfinite(r):
        return f"{dv}  predicts no variation"
    return f"{dv}  $r = {r:.2f}$"


def draw_panel(ax, series, lim, note=None):
    """One study x model panel: the DV clouds, reference geometry, and r."""
    ax.plot(lim, lim, **panels.IDENTITY_LINE)
    ax.axhline(0, **panels.ZERO_LINE)
    ax.axvline(0, **panels.ZERO_LINE)
    for dv, x, y, _ys in series:
        ax.scatter(
            x,
            y,
            s=5.5,
            marker=DV_MARKERS[dv],
            color=POINT_COLOR,
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )
    if series:
        # One r per DV rather than one per panel: a study's latents are what the
        # ablations differ on, and pooling them would average a latent the
        # variant cannot infer together with one it can.
        lines = [_dv_label(dv, x, y, ys, note=note) for dv, x, y, ys in series]
        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.0,
            color="0.1",
            linespacing=1.3,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.2),
            zorder=6,
        )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6.5)


def build(slugs):
    rows = []
    for slug in slugs:
        got = study_cells(slug)
        if got is None:
            continue
        cells, boots, preds, keys = got
        by_model = {
            m: panel_series(slug, cells, boots, preds, keys, m)
            for m in data.MODEL_ORDER
        }
        rows.append((slug, by_model))
    if not rows:
        print("no study has both data and CV predictions — nothing to draw")
        return None

    # One symmetric limit for every panel: the studies are on a common belief
    # update scale, and a per-panel limit would make the clouds look alike when
    # their spreads differ.
    lim = 1.05 * max(
        float(np.nanmax(np.abs(np.concatenate([s[i] for s in series]))))
        for _slug, by_model in rows
        for series in by_model.values()
        if series
        for i in (1, 2)
    )
    lim = (-lim, lim)

    ncols = len(data.MODEL_ORDER)
    fig, axes = plt.subplots(
        len(rows),
        ncols,
        figsize=(PANEL_IN * ncols + 0.85, PANEL_IN * len(rows) + 0.75),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    outside = 0
    for i, (slug, by_model) in enumerate(rows):
        for j, model in enumerate(data.MODEL_ORDER):
            ax = axes[i][j]
            note = []
            draw_panel(ax, by_model[model], lim, note=note)
            for dv, r, lo, hi, n in note:
                bad = not (lo <= r <= hi)
                outside += bad
                print(
                    f"  {study(slug).paper_label:>9s} {model:<16s} {dv:<9s} "
                    f"n={n:4d} r={r:6.3f}  bootstrap [{lo:6.3f}, {hi:6.3f}]"
                    f"{'  (excludes r -- see _dv_label)' if bad else ''}"
                )
            if i == 0:
                ax.set_title(data.MODEL_LABELS[model], fontsize=8.5, pad=4)
        axes[i][0].set_ylabel(study(slug).paper_label, fontsize=8.5, labelpad=2)

    if outside:
        print(
            f"  note: {outside} bootstrap interval(s) exclude their observed r "
            f"(thin scenario-level cells attenuate the resampled r); the figure "
            f"annotates r only"
        )

    dvs_present = []
    for _slug, by_model in rows:
        for series in by_model.values():
            for dv, *_rest in series:
                if dv not in dvs_present:
                    dvs_present.append(dv)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                linestyle="none",
                marker=DV_MARKERS[dv],
                markersize=4,
                color=POINT_COLOR,
                label=f"Inferred {dv}",
            )
            for dv in dvs_present
        ],
        loc="lower center",
        ncol=len(dvs_present),
        frameon=False,
        fontsize=7.5,
        handletextpad=0.3,
        columnspacing=1.4,
        borderaxespad=0.1,
    )
    fig.supxlabel("Model predicted belief update (out-of-sample)", fontsize=8.5)
    fig.supylabel("Human belief update", fontsize=8.5)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", help="Render one slug only (default: all six).")
    args = ap.parse_args()
    apply_style("si")
    slugs = [s.slug for s in studies() if not args.study or s.slug == args.study]
    fig = build(slugs)
    if fig is None:
        return
    stem = f"si_model_scatter_{args.study}" if args.study else "si_model_scatter_all"
    print(f"wrote {savefig(fig, stem)}")


if __name__ == "__main__":
    main()
