#!/usr/bin/env python3
"""Visualize the space of LM-generated alternatives.

Renders two groups of figures from the artifacts written by
generate_alternatives.py + score_merged.py + embed_alternatives.py +
project_alternatives.py:

SI figures (publication-styled via the repo-root plot_style module, written to
repo-root figures/). Each spans all six active studies (loaded from
outputs/lm/<slug>/) in one consolidated figure, except where noted:

  1. si_lm_semantic_space_example_{1a,3a} — one scenario's alternatives in the
     per-scenario UMAP embedding space (from lm_alternatives_projection.jsonl),
     colored by their LM-scored risk, with numbered cluster exemplars and their
     text listed beside the map. Two separate figures: one food-family example
     (1a, "soup") and one nonfood-family example (3a, "blanket").
  2a. si_lm_alternatives_composition_{food,nonfood} — the composition of
     alternatives (nearest observed action), one row per study; columns are by
     observed action and by the study's manipulated condition. Split into a food
     figure (1a/1b/2a/2b) and a nonfood figure (3a/3b) for legible aspect ratios.
  2b. si_lm_alternatives_set_similarity_all — the embedding similarity between
     the alternative sets of two elicitation cells, by what differs between the
     cells (runs only, condition, or observed action); one panel per study on a
     3x2 grid.
  3. si_lm_g_contrast_1a / si_lm_g_contrast_all — within-choice-set range of
     goal-satisfaction g by observed action: where the design can identify
     desire, over the whole comparison set and over the forgone alternatives
     alone. DIAGNOSTICS, not paper figures (decided 2026-08-02): the fitted
     eta = 0 for the desire question is the evidence, and these only illustrate
     the mechanism behind it. They stay because the SI quotes their numbers as
     ranges (0.0-0.2% of refusal sets and 2-22% of share sets flat over the whole
     set; 40-61% of refusal sets flat among the alternatives), so `--figures si`
     regenerates them on demand.
  4. si_lm_base_vs_full_1a_1b_3a — feature-distribution (energy) distance between
     the base ablation's relationship-free alternative sets and the
     relationship-conditioned sets, one panel per given-relationship study
     (1a/1b/3a — the only studies with a relationship-free base set).

Diagnostic figures (quick-look PNGs, written to model/outputs/lm/<slug>/figures/):

  fig1_semantic_map — global UMAP colored by scenario / by nearest observed action.
  fig2_decision_space — alternatives vs. the observed action in feature space,
     with Pareto-dominance flags.

The global UMAP (diagnostics only) runs here rather than in
embed_alternatives.py so the projection can be re-tuned without re-calling the
embedding API; it reads the persisted embeddings from lm_embeddings.npz.

Usage:
    uv run python model/lm/plot_alternatives.py --figures si

Requires (produced by the elicitation pipeline for each study):
    outputs/lm/<slug>/lm_runs.jsonl, lm_alternatives.jsonl,
    lm_embeddings.npz, lm_alternatives_semantic.jsonl,
    lm_alternatives_projection.jsonl, lm_clusters.json
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import (  # noqa: E402
    ACTION_COLORS,
    ACTION_LABELS,
    ALT_GREY,
    OBSERVED_ACTIONS,
    OBSERVED_STAR_COLOR,
    DESIRE_COLORS,
    INTIMACY_COLORS,
    INTIMACY_LABELS,
    INTIMACY_LEVELS,
    RISK_CMAP,
    SI_LARGE_RC,
    STUDY_LABELS,
    apply_style,
    panel_label,
    savefig,
)
from study_registry import SLUGS, slugs_given  # noqa: E402
from utils import get_project_root  # noqa: E402

# The six active studies, in roster order (1a, 1b, 2a, 2b, 3a, 3b). The SI
# alternatives figures show all six in one consolidated figure each.
STUDIES = list(SLUGS)  # the six active studies, in paper order
# The given-relationship studies (relationship is the manipulated axis) -- also
# the only ones with a relationship-free base set (lm_runs_base.jsonl), so the
# base-vs-full figure spans these; and the given-desire studies (desire is the
# manipulated axis). The composition figures are split along this axis so each
# has a single, consistent manipulated condition.
GIVEN_RELATIONSHIP_STUDIES = slugs_given("intimacy_condition")
GIVEN_DESIRE_STUDIES = slugs_given("desire_condition")


def _load(study):
    d = get_project_root() / "model" / "outputs" / "lm" / study
    sem = pd.read_json(d / "lm_alternatives_semantic.jsonl", lines=True)
    alts = pd.read_json(d / "lm_alternatives.jsonl", lines=True)
    runs = pd.read_json(d / "lm_runs.jsonl", lines=True)
    return d, sem, alts, runs


def _run_umap(alt_emb, obs_emb, seed):
    """Project alternatives + observed actions into one 2D layout (fit jointly so the
    observed-action anchors live in the same space as the alternatives)."""
    X = np.vstack([alt_emb, obs_emb])
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed
    )
    Y = reducer.fit_transform(X)
    return Y[: len(alt_emb)], Y[len(alt_emb) :]


# ----------------------------------------------------------------------------
# SI figure 1 — one scenario's semantic space, with the alternatives spelled out
# ----------------------------------------------------------------------------


SAVE_KW = {"png": False}


def fig_si_semantic_space(
    proj, clusters, scenario, figname="si_lm_semantic_space_example"
):
    """Per-scenario UMAP layout (precomputed by project_alternatives.py) with
    numbered cluster exemplars; the exemplar and observed-action texts are listed
    beside the map so readers can see what the alternatives actually say."""
    sub = proj[proj["scenario_label"] == scenario]
    if sub.empty:
        raise SystemExit(f"scenario {scenario!r} not found in the projection file")
    alts = sub[~sub["is_observed"]]
    observed = sub[sub["is_observed"]]
    exemplars = [c for c in clusters["clusters"] if c["scenario"] == scenario]
    exemplars = sorted(exemplars, key=lambda c: c["cluster"])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(8.6, 4.4), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )

    # alternatives colored by their LM-scored risk (run-averaged, from the
    # projection file) — ties the semantic layout to the model-relevant feature
    # rather than the circular embedding-nearest-observed-action label. Opaque points
    # + shuffled draw order + thin white edge: translucent points would darken
    # where they overlap, making dense regions read as higher risk.
    scored = alts[alts["risk"].notna()]
    unscored = alts[alts["risk"].isna()]
    if len(unscored):
        ax.scatter(
            unscored["dim1"],
            unscored["dim2"],
            s=18,
            color="#DDDDDD",
            lw=0,
            zorder=1,
        )
    sx = scored["dim1"].to_numpy()
    sy = scored["dim2"].to_numpy()
    sr = scored["risk"].to_numpy()
    order = np.random.default_rng(0).permutation(len(sx))
    sc = ax.scatter(
        sx[order],
        sy[order],
        c=sr[order],
        cmap=RISK_CMAP,
        vmin=0,
        vmax=1,
        s=18,
        lw=0,
        alpha=0.8,
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("LM-scored risk")
    cbar.set_ticks([0, 0.5, 1])
    cbar.outline.set_visible(False)
    # gray stars with text labels (the three actions are named on the map, so
    # they don't need per-action colors that would compete with the risk map)
    label_bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
    x_mid = 0.5 * (alts["dim1"].min() + alts["dim1"].max())
    for _, row in observed.iterrows():
        ax.scatter(
            row["dim1"],
            row["dim2"],
            marker="*",
            s=300,
            color=OBSERVED_STAR_COLOR,
            edgecolor="black",
            lw=0.8,
            zorder=6,
        )
        # label toward the interior so right-side stars don't run off the panel
        right = row["dim1"] > x_mid
        ax.annotate(
            ACTION_LABELS[row["observed_action"]],
            (row["dim1"], row["dim2"]),
            textcoords="offset points",
            xytext=(-9 if right else 9, 7),
            ha="right" if right else "left",
            fontsize=8,
            fontweight="bold",
            zorder=7,
            bbox=label_bbox,
        )

    # frame with a margin so the numbered callouts can be pushed into white space
    # (the margin around the cloud and the gaps between clusters) rather than
    # sitting on top of the data
    xr = (alts["dim1"].min(), alts["dim1"].max())
    yr = (alts["dim2"].min(), alts["dim2"].max())
    diag = float(np.hypot(xr[1] - xr[0], yr[1] - yr[0]))
    ax.set_xlim(xr[0] - 0.09 * (xr[1] - xr[0]), xr[1] + 0.09 * (xr[1] - xr[0]))
    ax.set_ylim(yr[0] - 0.09 * (yr[1] - yr[0]), yr[1] + 0.09 * (yr[1] - yr[0]))

    # data points (and observed stars) to steer the callouts away from
    avoid = alts[["dim1", "dim2"]].to_numpy()
    if len(observed):
        avoid = np.vstack([avoid, observed[["dim1", "dim2"]].to_numpy()])
    placed = []

    def _white_spot(px, py):
        """Nearest low-density location for a numbered callout: scan rings of
        candidate offsets and pick the one with the fewest data points (and
        already-placed labels) nearby, preferring a short leader line."""
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        inset = 0.035 * diag
        best = None
        for dist in (0.10, 0.15, 0.20, 0.26):
            for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                cx = px + dist * diag * np.cos(ang)
                cy = py + dist * diag * np.sin(ang)
                if not (xlo + inset <= cx <= xhi - inset):
                    continue
                if not (ylo + inset <= cy <= yhi - inset):
                    continue
                dens = int(
                    np.count_nonzero(np.hypot(*(avoid - [cx, cy]).T) < 0.05 * diag)
                )
                for qx, qy in placed:
                    if np.hypot(qx - cx, qy - cy) < 0.06 * diag:
                        dens += 25
                score = dens + dist * 4  # prefer emptier spots and short leaders
                if best is None or score < best[0]:
                    best = (score, cx, cy)
        if best is None:
            return px, py
        placed.append((best[1], best[2]))
        return best[1], best[2]

    circle_bbox = dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=0.8)
    numbered = []
    for i, cl in enumerate(exemplars, start=1):
        text = cl["exemplars"][0]
        hit = alts[alts["action_text"] == text]
        if hit.empty:
            print(f"note: exemplar not found in projection, skipping: {text[:50]}...")
            continue
        row = hit.iloc[0]
        cx, cy = _white_spot(float(row["dim1"]), float(row["dim2"]))
        ax.annotate(
            str(i),
            xy=(row["dim1"], row["dim2"]),
            xytext=(cx, cy),
            textcoords="data",
            fontsize=9.5,
            ha="center",
            va="center",
            bbox=circle_bbox,
            zorder=7,
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7),
        )
        numbered.append((i, text))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # text panel: observed actions, then the numbered exemplars
    ax2.set_axis_off()
    dy = 0.0355
    obs_by_action = {
        r["observed_action"]: r["action_text"] for _, r in observed.iterrows()
    }
    # pre-wrap so the block height can be measured and vertically centered
    obs_bodies = [
        textwrap.fill(f"{ACTION_LABELS[act]}: {obs_by_action.get(act, '')}", 60)
        for act in OBSERVED_ACTIONS
    ]
    numbered_wrapped = [(i, textwrap.fill(text, 62)) for i, text in numbered]
    total = dy * 1.35  # "Action conditions" header
    total += sum(dy * (b.count("\n") + 1) + dy * 0.5 for b in obs_bodies)
    total += dy * 0.5  # gap before the second section
    total += dy * 1.35  # "Example alternatives" header
    total += sum(dy * (w.count("\n") + 1) + dy * 0.5 for _, w in numbered_wrapped)
    # start lower by half the leftover space so the block sits centered
    y = 1.0 - max(0.0, (1.0 - total) / 2)

    def put_header(s):
        nonlocal y
        ax2.text(0, y, s, fontsize=8, fontweight="bold", va="top", color="#222222")
        y -= dy * 1.35

    put_header("Action conditions")
    for act, body in zip(OBSERVED_ACTIONS, obs_bodies):
        # bold action name (the map labels the stars by name, not color) then the
        # full observed-action sentence; drawn gray marker matching the map's stars
        # (not a "★" glyph, missing from Arial Nova)
        ax2.scatter(
            [0.018],
            [y - 0.012],
            marker="*",
            s=85,
            color=OBSERVED_STAR_COLOR,
            edgecolor="black",
            lw=0.5,
            zorder=5,
        )
        ax2.text(0.05, y, body, fontsize=7.2, va="top", color="#222222")
        y -= dy * (body.count("\n") + 1) + dy * 0.5
    y -= dy * 0.5
    put_header("Example alternatives")
    for i, wrapped in numbered_wrapped:
        ax2.text(
            0.012,
            y - 0.008,
            str(i),
            fontsize=6.5,
            ha="center",
            va="center",
            bbox=circle_bbox,
        )
        ax2.text(0.05, y, wrapped, fontsize=7.2, va="top", color="#222222")
        y -= dy * (wrapped.count("\n") + 1) + dy * 0.5

    # the text panel mixes text and scatter in data coords; pin the limits so
    # autoscaling from the star markers can't shift the alignment
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    return savefig(fig, figname, **SAVE_KW)


# ----------------------------------------------------------------------------
# SI figure 4 — how the generated sets vary with condition vs. observed action
# ----------------------------------------------------------------------------


def _condition_axes(alts, include_effort=True):
    """The condition columns of this study's cell grid (besides observed action),
    with their level order and display palette."""
    axes = {}
    if "intimacy_condition" in alts.columns:
        axes["intimacy_condition"] = ("intimacy", INTIMACY_LEVELS, INTIMACY_COLORS)
    if "desire_condition" in alts.columns:
        axes["desire_condition"] = ("desire", ["low", "high"], DESIRE_COLORS)
    if include_effort and "effort_condition" in alts.columns:
        axes["effort_condition"] = ("effort", ["low", "high"], None)
    return axes


def _set_similarity_by_type(alts_emb, emb, cond_cols):
    """Mean pairwise cosine similarity between the alternative sets of two
    elicitation cells, per scenario, grouped by what differs between the cells.

    Embeddings are L2-normalized, so mean pairwise cosine between two sets is
    the dot product of the set means; per-(cell, run) mean vectors make every
    comparison a cheap dot product. The same-cell baseline uses run-disjoint
    pairs only (within-run pairs measure within-set diversity, not stability).
    """
    cell_cols = ["scenario_label", "observed_action", *cond_cols]
    means, keys = [], []
    for key, rows in alts_emb.groupby([*cell_cols, "run_id"])["row"]:
        keys.append(key)
        means.append(emb[rows.to_numpy()].mean(axis=0))
    means = np.asarray(means)
    kdf = pd.DataFrame(keys, columns=[*cell_cols, "run_id"])
    kdf["idx"] = np.arange(len(kdf))

    def cell_stats(g):
        M = means[g["idx"].to_numpy()]
        K = len(M)
        S = M @ M.T
        off = (S.sum() - np.trace(S)) / (K * (K - 1)) if K > 1 else np.nan
        return pd.Series({"same_cell": off, "cbar": list(M.mean(axis=0))})

    cells = kdf.groupby(cell_cols).apply(cell_stats, include_groups=False).reset_index()
    cbar = np.asarray(cells["cbar"].tolist())
    cells["gidx"] = np.arange(len(cells))  # row index into cbar

    rows = []
    for scen, sdf in cells.groupby("scenario_label"):
        rows.append(dict(scenario=scen, type="same_cell", sim=sdf["same_cell"].mean()))
        # pairs of cells differing in exactly one grid column
        vary_cols = ["observed_action", *cond_cols]
        for vary in vary_cols:
            match = [c for c in vary_cols if c != vary]
            sims = []
            for _, pair_df in sdf.groupby(match) if match else [((), sdf)]:
                idx = pair_df["gidx"].to_numpy()
                for i in range(len(idx)):
                    for j in range(i + 1, len(idx)):
                        sims.append(cbar[idx[i]] @ cbar[idx[j]])
            rows.append(dict(scenario=scen, type=vary, sim=float(np.mean(sims))))
    return pd.DataFrame(rows)


def _boot_ci(vals, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    return np.percentile(vals[idx].mean(axis=1), [2.5, 97.5])


def _composition_prop(runs):
    """For one study's lm_runs, assign every generated alternative to the observed
    action whose (g, risk, effort) feature centroid is nearest (the space the
    planner reasons over), and return (main_col, main_name, main_levels, prop),
    where prop[(observed_action, condition_level)] is the fraction of alternatives
    nearest each observed action. The condition axis is whichever non-effort
    condition the study manipulates (relationship or desire)."""
    cond_axes = _condition_axes(runs)
    main_col, (main_name, main_levels, _) = next(
        (c, v) for c, v in cond_axes.items() if c != "effort_condition"
    )
    feats = ("g", "risk", "effort")

    def feat_vec(a):
        return None if any(a[f] is None for f in feats) else [a[f] for f in feats]

    # per-scenario feature centroid of each observed action (slot-0 action),
    # averaged over runs/conditions
    acc = {}
    for rec in runs.itertuples(index=False):
        obs0 = next((a for a in rec.actions if a["is_observed"]), None)
        v = feat_vec(obs0) if obs0 is not None else None
        if v is not None:
            acc.setdefault((rec.scenario_label, rec.observed_action), []).append(v)
    centroid = {k: np.mean(v, axis=0) for k, v in acc.items()}

    counts = {
        (obs, lvl): dict.fromkeys(OBSERVED_ACTIONS, 0)
        for obs in OBSERVED_ACTIONS
        for lvl in main_levels
    }
    for rec in runs.itertuples(index=False):
        try:
            cents = np.array(
                [centroid[(rec.scenario_label, a)] for a in OBSERVED_ACTIONS]
            )
        except KeyError:
            continue
        cell = (rec.observed_action, getattr(rec, main_col))
        if cell not in counts:
            continue
        for a in rec.actions:
            v = None if a["is_observed"] else feat_vec(a)
            if v is None:
                continue
            nearest = OBSERVED_ACTIONS[int(np.argmin(((cents - v) ** 2).sum(axis=1)))]
            counts[cell][nearest] += 1

    prop = {
        cell: {
            a: (c[a] / total if (total := sum(c.values())) else 0.0)
            for a in OBSERVED_ACTIONS
        }
        for cell, c in counts.items()
    }
    return main_col, main_name, main_levels, prop


def _draw_composition_row(
    ax_obs, ax_cond, prop, main_col, main_name, main_levels, legend, show_xlabel
):
    """Draw one study's two composition panels: (left) proportion nearest each
    action vs observed action, faint lines = condition levels; (right) vs the
    manipulated condition, faint lines = observed actions. ``show_xlabel`` adds the
    x-axis labels (only the bottom row of a figure sets them, since all rows share
    the same axes)."""

    def draw(ax, x, series_lines, color, label=None):
        for ys in series_lines:
            ax.plot(x, ys, color=color, lw=0.7, alpha=0.3, zorder=2)
        ax.plot(
            x,
            np.mean(series_lines, axis=0),
            color=color,
            lw=2.2,
            zorder=4,
            marker="o",
            ms=5,
            label=label,
        )

    xa = np.arange(len(OBSERVED_ACTIONS))
    for nc in OBSERVED_ACTIONS:
        per_level = [
            [prop[(obs, lvl)][nc] for obs in OBSERVED_ACTIONS] for lvl in main_levels
        ]
        draw(
            ax_obs,
            xa,
            per_level,
            ACTION_COLORS[nc],
            label=f"Nearest: {ACTION_LABELS[nc].lower()}" if legend else None,
        )
    ax_obs.set_xticks(xa)
    ax_obs.set_xticklabels(
        [ACTION_LABELS[a].replace(" ", "\n", 1) for a in OBSERVED_ACTIONS], fontsize=8.5
    )
    ax_obs.set_xlim(-0.25, len(OBSERVED_ACTIONS) - 0.75)
    if show_xlabel:
        ax_obs.set_xlabel("Observed action")

    xb = np.arange(len(main_levels))
    for nc in OBSERVED_ACTIONS:
        per_obs = [
            [prop[(obs, lvl)][nc] for lvl in main_levels] for obs in OBSERVED_ACTIONS
        ]
        draw(ax_cond, xb, per_obs, ACTION_COLORS[nc])
    ax_cond.set_xticks(xb)
    if main_col == "intimacy_condition":
        ax_cond.set_xticklabels(
            [INTIMACY_LABELS[lvl].replace(" ", "\n") for lvl in main_levels],
            fontsize=8,
        )
        cond_xlabel = "Relationship descriptor"
    else:
        ax_cond.set_xticklabels([lvl.capitalize() for lvl in main_levels], fontsize=8.5)
        cond_xlabel = f"{main_name.capitalize()} condition"
    ax_cond.set_xlim(-0.25, len(main_levels) - 0.75)
    if show_xlabel:
        ax_cond.set_xlabel(cond_xlabel)


def fig_si_composition(runs_by_study, figname="si_lm_alternatives_composition"):
    """Composition figure with one row per study; columns are (left) by observed
    action and (right) by the study's manipulated condition. Every alternative is
    assigned to the observed action whose (g, risk, effort) centroid is nearest.
    The left column shows the observed action dominates (its own type is least
    represented among the alternatives); the right column shows the smaller,
    systematic effect of the manipulated condition. ``runs_by_study`` is a list of
    (study label, runs)."""
    fig, axes = plt.subplots(
        len(runs_by_study),
        2,
        figsize=(6.0, 2.2 * len(runs_by_study)),
        sharey=True,
        squeeze=False,
    )
    for row, (label, runs) in enumerate(runs_by_study):
        main_col, main_name, main_levels, prop = _composition_prop(runs)
        _draw_composition_row(
            axes[row, 0],
            axes[row, 1],
            prop,
            main_col,
            main_name,
            main_levels,
            legend=(row == 0),
            show_xlabel=(row == len(runs_by_study) - 1),
        )
        axes[row, 0].set_ylabel(f"{label}\nproportion nearest\neach action")
    axes[0, 0].set_title("By observed action")
    axes[0, 1].set_title("By manipulated condition")
    for ax in axes.ravel():
        ax.set_ylim(0, 0.56)
    # one horizontal legend spanning the width, just below the panels (never over
    # data). Reserve a thin ~0.32in strip at the bottom so the legend sits close to
    # the bottom-row x-labels without a large gap.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig_h = 2.2 * len(runs_by_study)
    fig.tight_layout(rect=[0, 0.32 / fig_h, 1, 1])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.5,
    )
    return savefig(fig, figname, **SAVE_KW)


# Short x-tick labels for the (narrow, 3x2) set-similarity panels; the axis
# meaning ("what differs between the two compared cells") is in the caption and
# y-label, so single-word categories read cleanly at a larger font.
_SET_SIM_LABELS = {
    "same_cell": "Same cell\n(runs)",
    "intimacy_condition": "Intimacy",
    "desire_condition": "Desire",
    "effort_condition": "Effort",
    "observed_action": "Observed\naction",
}


def _draw_set_similarity(ax, alts, sem, alt_emb, title, ylabel, include_effort=True):
    """One study's set-similarity panel: mean pairwise cosine between the
    alternative sets of two cells, grouped by what differs between them."""
    cond_axes = _condition_axes(alts, include_effort=include_effort)
    alts_emb = alts.merge(
        sem[["scenario_label", "action_text"]].assign(row=np.arange(len(sem))),
        on=["scenario_label", "action_text"],
        how="inner",
    )
    cond_cols = list(cond_axes.keys())
    sims = _set_similarity_by_type(alts_emb, alt_emb, cond_cols)

    type_order = ["same_cell", *cond_cols, "observed_action"]
    wide = sims.pivot_table(index="scenario", columns="type", values="sim").reindex(
        columns=type_order
    )
    rng = np.random.default_rng(0)
    # a small, per-scenario x-offset kept constant across the comparison types, so
    # each scenario's four points connect into a thin line (its own trajectory)
    xoff = dict(zip(wide.index, rng.uniform(-0.09, 0.09, len(wide))))
    xs_base = np.arange(len(type_order))
    for s, row in wide.iterrows():
        xs = xs_base + xoff[s]
        ys = row[type_order].to_numpy()
        ax.plot(xs, ys, color=ALT_GREY, lw=0.5, alpha=0.4, zorder=2)
        ax.scatter(xs, ys, s=11, color=ALT_GREY, alpha=0.7, lw=0, zorder=3)
    for x, t in enumerate(type_order):
        vals = wide[t].dropna().to_numpy()
        lo, hi = _boot_ci(vals)
        m = vals.mean()
        ax.errorbar(
            x,
            m,
            yerr=[[m - lo], [hi - m]],
            fmt="o",
            color="#333333",
            ms=6,
            capsize=3,
            lw=1.2,
            zorder=5,
        )
    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels([_SET_SIM_LABELS[t] for t in type_order], fontsize=9)
    if ylabel:
        ax.set_ylabel(
            "Between-set embedding\nsimilarity (mean pairwise cosine)", fontsize=10.5
        )
    ax.set_xlim(-0.5, len(type_order) - 0.5)
    ax.set_title(title)


def fig_si_set_similarity(panels, figname="si_lm_alternatives_set_similarity"):
    """One panel per study on a 3x2 grid: embedding similarity between the
    alternative sets of two elicitation cells, grouped by what differs between
    them. Each puts the condition effect on the same scale as the run-to-run
    baseline and the observed-action effect. ``panels`` is a list of
    (study, title, alts, sem, alt_emb); the effort-differs category is shown for
    the food studies (nonfood joint elicitations omit it)."""
    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.6, 2.4 * nrows), sharey=True, squeeze=False
    )
    axflat = axes.ravel()
    for i, (study, title, alts, sem, alt_emb) in enumerate(panels):
        _draw_set_similarity(
            axflat[i],
            alts,
            sem,
            alt_emb,
            title,
            ylabel=(i % ncols == 0),
            include_effort=not study.startswith("nonfood_"),
        )
    for ax in axflat[len(panels) :]:
        ax.axis("off")
    fig.tight_layout()
    return savefig(fig, figname, **SAVE_KW)


# ----------------------------------------------------------------------------
# SI figure 4 — where the design can identify desire: within-set g contrast
# ----------------------------------------------------------------------------


def fig_si_g_contrast(runs, figname="si_lm_g_contrast"):
    """Desire enters the utility as w_v * d * g(a), so an observed choice is
    informative about desire only insofar as the actions in the choice set
    differ in goal-satisfaction g. Histograms of the within-set range of g by
    observed action: no-share sets pair the g~0 observed action with g~1
    sharing alternatives (high contrast), while share-observed sets are often
    all-high-g, so those observations mainly inform risk/intimacy, not desire."""
    recs = []
    for actions, obs in zip(runs["actions"], runs["observed_action"]):
        gs = [a["g"] for a in actions if a.get("g") is not None]
        if not gs:
            continue
        recs.append((obs, max(gs) - min(gs)))
    df = pd.DataFrame(recs, columns=["obs", "grange"])

    fig, axes = plt.subplots(3, 1, figsize=(4.8, 4.8), sharex=True, sharey=True)
    bins = np.arange(0, 1.05, 0.05)
    for ax, act in zip(axes, OBSERVED_ACTIONS):
        sub = df.loc[df["obs"] == act, "grange"]
        ax.hist(
            sub,
            bins=bins,
            weights=np.full(len(sub), 100 / len(sub)),
            color=ACTION_COLORS[act],
            edgecolor="white",
            lw=0.4,
        )
        pct0 = 100 * (sub <= 0.01).mean()
        # put the annotation on whichever side the mass isn't
        x, ha = (0.03, "left") if sub.median() > 0.5 else (0.97, "right")
        ax.annotate(
            f"Observed: {ACTION_LABELS[act].lower()}\n"
            f"{pct0:.0f}% of sets have no $g$ contrast",
            (x, 0.9),
            xycoords="axes fraction",
            ha=ha,
            va="top",
            fontsize=9.5,
        )
        ax.set_ylabel("% of sets")
    axes[-1].set_xlabel("Within-choice-set range of goal-satisfaction $g$")
    fig.tight_layout()
    return savefig(fig, figname, **SAVE_KW)


def _g_contrast_rates(runs, flat=0.01):
    """(observed action) -> (% of sets with no g contrast over the whole
    comparison set, % with none among the forgone alternatives, n sets, n sets
    dropped for having no alternatives at all).

    The two quantities answer different questions. The whole-set range says
    whether the observation itself can speak to desire: an action's own g
    prices desire directly, so a set spanning g~0 to g~1 makes the choice
    informative. The alternatives-only range says whether *reweighting* that
    set can speak to desire: the reweighting moves mass among the forgone
    actions, so when those are uniform in g no weighting of them shifts the
    desire posterior, however surprising the observation is."""
    rates = {}
    for act in OBSERVED_ACTIONS:
        full, alts, empty = [], [], 0
        for actions, obs in zip(runs["actions"], runs["observed_action"]):
            if obs != act:
                continue
            gs = [a["g"] for a in actions if a.get("g") is not None]
            if not gs:
                continue
            full.append(max(gs) - min(gs))
            ga = [
                a["g"]
                for a in actions
                if a.get("g") is not None and not a.get("is_observed")
            ]
            # A set with no alternatives has nothing to reweight and no range to
            # measure; counted separately rather than scored as "no contrast".
            if ga:
                alts.append(max(ga) - min(ga))
            else:
                empty += 1
        if not full:
            continue
        rates[act] = (
            100 * float(np.mean(np.asarray(full) <= flat)),
            100 * float(np.mean(np.asarray(alts) <= flat)) if alts else float("nan"),
            len(full),
            empty,
        )
    return rates


def fig_si_g_contrast_all(studies_data, figname="si_lm_g_contrast_all"):
    """The g-contrast diagnostic across all six studies: the share of choice
    sets carrying no goal-satisfaction contrast, by observed action, measured
    over the whole comparison set (a) and over the forgone alternatives alone
    (b). Panel (b) is the quantity the comparison-set reweighting depends on --
    where it is high, no reweighting of the alternatives can inform desire."""
    rates = {s: _g_contrast_rates(runs) for s, runs in studies_data}
    labels = [STUDY_LABELS[s] for s, _r in studies_data]
    x = np.arange(len(studies_data))
    width = 0.26

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True, sharey=True)
    panels = [
        (0, "a", "Whole comparison set"),
        (1, "b", "Forgone alternatives only"),
    ]
    for ax, (which, letter, title) in zip(axes, panels):
        for j, act in enumerate(OBSERVED_ACTIONS):
            vals = [rates[s].get(act, (np.nan,) * 4)[which] for s, _r in studies_data]
            pos = x + (j - 1) * width
            ax.bar(
                pos,
                vals,
                width,
                color=ACTION_COLORS[act],
                edgecolor="white",
                lw=0.5,
                label=ACTION_LABELS[act],
            )
            # A bar this short is a real value, not missing data -- print it, or
            # panel (a)'s refusals look like an empty slot in the group.
            for xi, v in zip(pos, vals):
                if v < 2:
                    ax.annotate(
                        f"{v:.1f}",
                        (xi, v),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color="#555555",
                        rotation=90,
                    )
        ax.set_ylabel("% of sets with\nno $g$ contrast")
        ax.set_title(title, fontsize=10, pad=3)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)
        panel_label(ax, letter, dx=-0.09)
    axes[0].legend(loc="upper left", ncol=3, fontsize=8.5, frameon=False)
    axes[-1].set_xticks(x, labels)

    for s, _r in studies_data:
        for act, (pf, pa, n, empty) in rates[s].items():
            print(
                f"g-contrast {STUDY_LABELS[s]:>9s} observed={act:<11s} "
                f"set {pf:5.1f}% flat, alternatives {pa:5.1f}% flat "
                f"(n={n}, {empty} with no alternatives)"
            )
    fig.tight_layout()
    return savefig(fig, figname, **SAVE_KW)


# ----------------------------------------------------------------------------
# SI figure 5 — the base ablation's relationship-free sets vs the conditioned sets
# ----------------------------------------------------------------------------


def _alt_feature_clouds(runs, cell_cols):
    """cell key -> (N, 3) array of (risk, effort, g) for the alternatives in that
    cell, pooled over runs. Instances are kept (not deduplicated by text): the
    per-run sets are what the planner sees, so their pooled feature distribution
    -- including how often an alternative recurs -- is the landscape we compare."""
    clouds = {}
    for key, grp in runs.groupby(cell_cols):
        pts = [
            (a["risk"], a["effort"], a["g"])
            for actions in grp["actions"]
            for a in actions
            if not a["is_observed"]
            and a["risk"] is not None
            and a["effort"] is not None
            and a["g"] is not None
        ]
        if pts:
            clouds[key] = np.asarray(pts, dtype=float)
    return clouds


def _energy_distance(X, Y):
    """Szekely-Rizzo energy distance between two point sets: 2 E|X-Y| - E|X-X'|
    - E|Y-Y'|, a proper distance between distributions (0 iff equal), computed
    here in the [0,1]^3 (risk, effort, g) feature space. The within-set terms
    exclude self-pairs so the estimate is unbiased for finite samples."""

    def within(A):
        n = len(A)
        if n < 2:
            return 0.0
        D = cdist(A, A)
        return (D.sum() - np.trace(D)) / (n * (n - 1))

    return max(0.0, 2 * cdist(X, Y).mean() - within(X) - within(Y))


def _draw_base_vs_full(ax, base_runs, runs, ylabel):
    """Draw one given-relationship study's base-vs-conditioned energy distances
    onto ax: for each (scenario, observed action, effort) cell, the energy
    distance between the base set's (risk, effort, g) cloud and each relationship
    level's conditioned cloud, plus the reference distance between conditioned
    clouds at different levels."""
    cell_cols = ["scenario_label", "observed_action", "effort_condition"]
    base_clouds = _alt_feature_clouds(base_runs, cell_cols)
    cond_clouds = _alt_feature_clouds(runs, [*cell_cols, "intimacy_condition"])

    recs = []
    for key, bcloud in base_clouds.items():
        levels = {lvl: cond_clouds.get((*key, lvl)) for lvl in INTIMACY_LEVELS}
        if any(v is None for v in levels.values()):
            continue
        for lvl in INTIMACY_LEVELS:
            recs.append(
                dict(
                    scenario=key[0],
                    comparison=lvl,
                    dist=_energy_distance(bcloud, levels[lvl]),
                )
            )
        pairs = [
            _energy_distance(levels[l1], levels[l2])
            for i, l1 in enumerate(INTIMACY_LEVELS)
            for l2 in INTIMACY_LEVELS[i + 1 :]
        ]
        recs.append(
            dict(scenario=key[0], comparison="reference", dist=float(np.mean(pairs)))
        )
    per_scen = (
        pd.DataFrame(recs)
        .groupby(["scenario", "comparison"], as_index=False)["dist"]
        .mean()
    )

    cats = [*INTIMACY_LEVELS, "reference"]
    wide = per_scen.pivot_table(
        index="scenario", columns="comparison", values="dist"
    ).reindex(columns=cats)
    rng = np.random.default_rng(0)
    # per-scenario x-offset kept constant across categories, so each scenario's
    # points connect into a thin line (its trajectory across the levels)
    xoff = dict(zip(wide.index, rng.uniform(-0.1, 0.1, len(wide))))
    xs_base = np.arange(len(cats))
    for s, row in wide.iterrows():
        xs = xs_base + xoff[s]
        ys = row[cats].to_numpy()
        ax.plot(xs, ys, color=ALT_GREY, lw=0.5, alpha=0.4, zorder=2)
        ax.scatter(xs, ys, s=11, color=ALT_GREY, alpha=0.7, lw=0, zorder=3)
    for x, cat in enumerate(cats):
        vals = wide[cat].dropna().to_numpy()
        lo, hi = _boot_ci(vals)
        m = vals.mean()
        ax.errorbar(
            x,
            m,
            yerr=[[m - lo], [hi - m]],
            fmt="o",
            color=INTIMACY_COLORS.get(cat, "#333333"),
            ms=6,
            capsize=3,
            lw=1.2,
            zorder=5,
        )
    ax.axvline(len(INTIMACY_LEVELS) - 0.5, color="0.85", lw=1.1, ls=(0, (4, 3)))
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(
        [INTIMACY_LABELS[lvl].replace(" ", "\n") for lvl in INTIMACY_LEVELS]
        + ["Between\nlevels (ref.)"],
        fontsize=8.5,
    )
    if ylabel:
        ax.set_ylabel("Feature-distribution\ndistance (lower = closer)")
    ax.set_ylim(bottom=0)


def fig_si_base_vs_full(studies_data, figname="si_lm_base_vs_full"):
    """The base ablation's alternatives are elicited without the relationship
    paragraph (lm_runs_base.jsonl). Because those features -- not the exact
    wording -- are what enters the planner, we compare the feature distributions
    of the sets rather than their text. One panel per given-relationship study
    (1a/1b/3a) that has a base set. Lower is more similar, so if base-vs-
    conditioned matches the reference (between-level) distance, the relationship-
    free set presents the same feature landscape to the planner as any
    conditioned set does to another. ``studies_data`` is a list of (study, dir,
    runs)."""
    panels = []
    for study, d, runs in studies_data:
        base_path = d / "lm_runs_base.jsonl"
        if not base_path.exists():
            print(
                f"skipping base-vs-full panel ({study}): lm_runs_base.jsonl not found"
            )
            continue
        if "intimacy_condition" not in runs.columns:
            print(f"skipping base-vs-full panel ({study}): no relationship axis")
            continue
        panels.append((study, pd.read_json(base_path, lines=True), runs))
    if not panels:
        print("skipping base-vs-full figure: no study has a base set")
        return None
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(4.8, 2.5 * len(panels)), sharex=True, squeeze=False
    )
    for i, (study, base_runs, runs) in enumerate(panels):
        ax = axes[i, 0]
        _draw_base_vs_full(ax, base_runs, runs, ylabel=True)
        ax.set_title(STUDY_LABELS[study])
    fig.supxlabel(
        "Relationship-free set vs. the relationship-conditioned set at each level",
        fontsize=12,
    )
    fig.tight_layout()
    return savefig(fig, figname, **SAVE_KW)


# ----------------------------------------------------------------------------
# Diagnostic figures (quick-look PNGs in model/outputs/lm/<slug>/figures/)
# ----------------------------------------------------------------------------


def fig_semantic_map(sem, alt_xy, obs_xy, obs_action, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.8))

    # (a) colored by scenario, with the scenario name printed at each blob centroid.
    scen = sem["scenario_label"].to_numpy()
    scens = sorted(pd.unique(scen))
    cmap = plt.get_cmap("tab20", len(scens))
    for i, s in enumerate(scens):
        m = scen == s
        ax1.scatter(alt_xy[m, 0], alt_xy[m, 1], s=4, alpha=0.45, color=cmap(i), lw=0)
        cx, cy = alt_xy[m, 0].mean(), alt_xy[m, 1].mean()
        ax1.text(cx, cy, s, fontsize=7, ha="center", va="center", weight="bold")
    ax1.set_title("(a) Colored by scenario\nthe space splits into scenario blobs")

    # (b) colored by nearest observed action, observed actions as stars.
    nc = sem["nearest_observed_action"].to_numpy()
    for a in OBSERVED_ACTIONS:
        m = nc == a
        ax2.scatter(
            alt_xy[m, 0],
            alt_xy[m, 1],
            s=4,
            alpha=0.45,
            color=ACTION_COLORS[a],
            lw=0,
            label=ACTION_LABELS[a],
        )
    for j in range(len(obs_xy)):
        ax2.scatter(
            obs_xy[j, 0],
            obs_xy[j, 1],
            marker="*",
            s=110,
            color=ACTION_COLORS[obs_action[j]],
            edgecolor="black",
            lw=0.5,
            zorder=5,
        )
    ax2.set_title(
        "(b) Colored by nearest action condition\n★ = the 16×3 action conditions"
    )
    ax2.legend(loc="best", frameon=False, markerscale=2.5, fontsize=9)

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
    fig.suptitle("Semantic map of the generated alternatives", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _decision_points(runs):
    """Explode lm_runs into one row per action with a Pareto-dominance label vs the
    observed action of the same (cell, run)."""
    rows = []
    for actions, scen, obs in zip(
        runs["actions"], runs["scenario_label"], runs["observed_action"]
    ):
        observed = next((a for a in actions if a["is_observed"]), None)
        if observed is None:
            continue
        cg, ce, cr = observed["g"], observed["effort"], observed["risk"]
        rows.append(dict(observed=obs, kind="observed", risk=cr, effort=ce, g=cg))
        for a in actions:
            if a["is_observed"]:
                continue
            gg, ee, rr = a["g"], a["effort"], a["risk"]
            kind = "alt"
            if None not in (gg, ee, rr, cg, ce, cr):
                dom_by = (
                    cg >= gg
                    and ce <= ee
                    and cr <= rr
                    and (cg > gg or ce < ee or cr < rr)
                )
                dom_obs = (
                    gg >= cg
                    and ee <= ce
                    and rr <= cr
                    and (gg > cg or ee < ce or rr < cr)
                )
                kind = "dominated" if dom_by else ("dominates" if dom_obs else "alt")
            rows.append(dict(observed=obs, kind=kind, risk=rr, effort=ee, g=gg))
    return pd.DataFrame(rows)


def fig_decision_space(runs, out):
    df = _decision_points(runs)
    pairs = [("risk", "g"), ("risk", "effort")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8), sharex=True, sharey="row")
    styles = {
        "alt": dict(color="#9AA0A6", s=5, alpha=0.18, label="Alternative"),
        "dominated": dict(
            color="#E0A458", s=6, alpha=0.5, label="Dominated by observed"
        ),
        "dominates": dict(color="#B05A5A", s=8, alpha=0.7, label="Dominates observed"),
    }
    # The LM features are discretized to a 7-point lattice (the 0-6 rating scale),
    # so points collapse onto grid intersections. Jitter (fixed seed) spreads each
    # lattice cell into a little cloud, revealing relative density while keeping the
    # lattice structure visible.
    rng = np.random.default_rng(0)

    def jit(v, scale=0.028):
        return v + rng.uniform(-scale, scale, size=len(v))

    for r, (xf, yf) in enumerate(pairs):
        for c, obs in enumerate(OBSERVED_ACTIONS):
            ax = axes[r, c]
            sub = df[df["observed"] == obs]
            for kind in ("alt", "dominated", "dominates"):
                k = sub[sub["kind"] == kind]
                ax.scatter(
                    jit(k[xf].to_numpy()), jit(k[yf].to_numpy()), lw=0, **styles[kind]
                )
            obspts = sub[sub["kind"] == "observed"]
            ax.scatter(
                jit(obspts[xf].to_numpy(), 0.018),
                jit(obspts[yf].to_numpy(), 0.018),
                marker="*",
                s=70,
                color=ACTION_COLORS[obs],
                edgecolor="black",
                lw=0.4,
                alpha=0.5,
                zorder=5,
                label="Observed action",
            )
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            if r == 0:
                ax.set_title(ACTION_LABELS[obs])
            if c == 0:
                ax.set_ylabel(yf if yf != "g" else "goal-satisfaction g")
            ax.set_xlabel(xf)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        markerscale=2,
    )
    fig.suptitle(
        "Decision space: alternatives vs. the observed action, by observed action",
        fontsize=14,
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------


# The semantic-space (UMAP) example figures: one food-family example and one
# nonfood-family example, each its own figure (rather than all six studies).
# Scenarios are chosen for a clean, filled projection: "soup" for food, and
# "blanket" for nonfood (the nonfood joint studies pool fewer alternatives per
# scenario, so a denser scenario avoids a sparse, gappy layout).
SEMANTIC_SPACE_EXAMPLES = [
    ("food_inv_desire", "soup", "si_lm_semantic_space_example_1a"),
    ("nonfood_inv_joint_de", "blanket", "si_lm_semantic_space_example_3a"),
]


def main(seed, example_scenario, figures):
    apply_style()
    loaded = {}
    for study in STUDIES:
        d = get_project_root() / "model" / "outputs" / "lm" / study
        if not (d / "lm_runs.jsonl").exists():
            print(f"{study}: no lm_runs.jsonl yet — skipped")
            continue
        _, sem, alts, runs = _load(study)
        entry = {"d": d, "sem": sem, "alts": alts, "runs": runs}
        if figures in ("si", "all", "diagnostic"):
            emb_path = d / "lm_embeddings.npz"
            if not emb_path.exists():
                raise SystemExit(
                    f"{emb_path} not found — run embed_alternatives.py (and "
                    "project_alternatives.py) for this study first"
                )
            entry["alt_emb"] = np.load(emb_path, allow_pickle=False)["alt_emb"]
        loaded[study] = entry
    studies = [s for s in STUDIES if s in loaded]

    if figures in ("si", "all"):
        print("Rendering SI figures...", flush=True)
        paths = []
        # composition: one figure per manipulation type (relationship vs desire),
        # 3 rows each, so every row in a figure shares the same right-column axis.
        # Base "si" sizes -- these many-panel grids look out of proportion with the
        # larger type.
        relationship = [s for s in GIVEN_RELATIONSHIP_STUDIES if s in loaded]
        desire = [s for s in GIVEN_DESIRE_STUDIES if s in loaded]
        if relationship:
            paths.append(
                fig_si_composition(
                    [(STUDY_LABELS[s], loaded[s]["runs"]) for s in relationship],
                    figname="si_lm_alternatives_composition_relationship",
                )
            )
        if desire:
            paths.append(
                fig_si_composition(
                    [(STUDY_LABELS[s], loaded[s]["runs"]) for s in desire],
                    figname="si_lm_alternatives_composition_desire",
                )
            )
        # These render at reduced SI widths, so draw them with the larger rc
        # profile so their type reads at the same size as Fig S1. The UMAP maps
        # (below) keep the base "si" sizes -- their hand-laid text panel is tuned
        # to those.
        with plt.rc_context(SI_LARGE_RC):
            paths.append(
                fig_si_set_similarity(
                    [
                        (
                            s,
                            STUDY_LABELS[s],
                            loaded[s]["alts"],
                            loaded[s]["sem"],
                            loaded[s]["alt_emb"],
                        )
                        for s in studies
                    ],
                    figname="si_lm_alternatives_set_similarity_all",
                )
            )
            # base-vs-full: the given-relationship studies with a base set (1a/1b/3a)
            paths.append(
                fig_si_base_vs_full(
                    [
                        (s, loaded[s]["d"], loaded[s]["runs"])
                        for s in GIVEN_RELATIONSHIP_STUDIES
                        if s in loaded
                    ],
                    figname="si_lm_base_vs_full_1a_1b_3a",
                )
            )
            # g-contrast: diagnostics only — neither figure is in the paper, but
            # the SI quotes the rates the all-six pass prints, so both stay
            # runnable. See the module docstring.
            if "food_inv_desire" in loaded:
                paths.append(
                    fig_si_g_contrast(
                        loaded["food_inv_desire"]["runs"],
                        figname="si_lm_g_contrast_1a",
                    )
                )
            paths.append(
                fig_si_g_contrast_all(
                    [(s, loaded[s]["runs"]) for s in studies],
                    figname="si_lm_g_contrast_all",
                )
            )
        # semantic-space (UMAP) examples: one food, one nonfood — separate figures,
        # kept at the base "si" sizes (their side text panel is laid out for those)
        for study, scen_default, figname in SEMANTIC_SPACE_EXAMPLES:
            if study not in loaded:
                continue
            d = loaded[study]["d"]
            proj = pd.read_json(d / "lm_alternatives_projection.jsonl", lines=True)
            with open(d / "lm_clusters.json") as f:
                clusters = json.load(f)
            scen = (
                example_scenario
                if (study == "food_inv_desire" and example_scenario)
                else scen_default
            )
            paths.append(fig_si_semantic_space(proj, clusters, scen, figname=figname))
        for path in paths:
            if path:
                print(f"wrote {path}")

    if figures in ("diagnostic", "all"):
        for study in studies:
            d = loaded[study]["d"]
            npz = np.load(d / "lm_embeddings.npz", allow_pickle=False)
            print(
                f"[{study}] UMAP on {len(npz['alt_emb'])} alternatives + "
                f"{len(npz['obs_emb'])} observed actions...",
                flush=True,
            )
            alt_xy, obs_xy = _run_umap(npz["alt_emb"], npz["obs_emb"], seed)
            fig_dir = d / "figures"
            fig_dir.mkdir(exist_ok=True)
            fig_semantic_map(
                loaded[study]["sem"],
                alt_xy,
                obs_xy,
                npz["obs_action"],
                fig_dir / "fig1_semantic_map.png",
            )
            fig_decision_space(
                loaded[study]["runs"], fig_dir / "fig2_decision_space.png"
            )
            print(f"[{study}] wrote diagnostics to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="UMAP random_state.")
    parser.add_argument(
        "--example-scenario",
        default=None,
        help="Override the food semantic-space example scenario (default soup; "
        "the nonfood example uses chapstick). Soup projects cleanly: the three "
        "observed actions land in separate, matching-colored clusters with "
        "sensible exemplars nearby.",
    )
    parser.add_argument("--figures", choices=["all", "si", "diagnostic"], default="si")
    args = parser.parse_args()
    main(args.seed, args.example_scenario, args.figures)
