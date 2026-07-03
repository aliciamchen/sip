#!/usr/bin/env python3
"""Visualize the space of LM-generated alternatives.

Renders two groups of figures from the artifacts written by
generate_alternatives.py + score_merged.py + embed_alternatives.py +
project_alternatives.py:

SI figures (publication-styled via the repo-root plot_style module, written to
repo-root figures/; rendered from --study, Study 1a by default):

  1. si_lm_semantic_space_example — one scenario's alternatives in the
     per-scenario UMAP embedding space (from lm_alternatives_projection.jsonl),
     colored by their LM-scored risk, with numbered cluster exemplars and
     their text listed beside the map.
  2a. si_lm_alternatives_composition — the composition of alternatives
     (nearest observed action) by condition, per observed action.
  2b. si_lm_alternatives_set_similarity — the embedding similarity between the
     alternative sets of two elicitation cells, by what differs between the
     cells (nothing/runs only, condition, or observed action).
  3. si_lm_g_contrast — within-choice-set range of goal-satisfaction g by
     observed action: where the design can identify desire.
  4. si_lm_base_vs_full — feature-distribution (energy) distance between the
     base ablation's relationship-free alternative sets and the
     relationship-conditioned sets (skipped if the study has no base
     elicitation).

Diagnostic figures (quick-look PNGs, written to model/outputs/lm/<slug>/figures/):

  fig1_semantic_map — global UMAP colored by scenario / by nearest observed action.
  fig2_decision_space — alternatives vs. the observed action in feature space,
     with Pareto-dominance flags.

The global UMAP (diagnostics only) runs here rather than in
embed_alternatives.py so the projection can be re-tuned without re-calling the
embedding API; it reads the persisted embeddings from lm_embeddings.npz.

Usage:
    uv run python model/lm/plot_alternatives.py --study food_inv_desire

Requires (produced by the elicitation pipeline for the study):
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
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import (  # noqa: E402
    ACTION_COLORS,
    ACTION_LABELS,
    ALT_GREY,
    OBSERVED_ACTIONS,
    OBSERVED_STAR_COLOR,
    DESIRE_COLORS,
    GOAL_CMAP,
    INTIMACY_COLORS,
    INTIMACY_LABELS,
    INTIMACY_LEVELS,
    RISK_CMAP,
    apply_style,
    panel_label,
    savefig,
)
from utils import get_project_root  # noqa: E402


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


def fig_si_semantic_space(proj, clusters, scenario):
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
    # grey stars with text labels (the three actions are named on the map, so
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

    circle_bbox = dict(boxstyle="circle,pad=0.22", fc="white", ec="black", lw=0.7)
    offset_cycle = [(9, 9), (-11, 8), (9, -9), (-11, -8), (12, 0), (0, 12)]
    numbered = []
    for i, cl in enumerate(exemplars, start=1):
        text = cl["exemplars"][0]
        hit = alts[alts["action_text"] == text]
        if hit.empty:
            print(f"note: exemplar not found in projection, skipping: {text[:50]}...")
            continue
        row = hit.iloc[0]
        ax.annotate(
            str(i),
            (row["dim1"], row["dim2"]),
            textcoords="offset points",
            xytext=offset_cycle[(i - 1) % len(offset_cycle)],
            fontsize=7,
            ha="center",
            va="center",
            bbox=circle_bbox,
            zorder=7,
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
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
        # full observed-action sentence; drawn grey marker matching the map's stars
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
    return savefig(fig, "si_lm_semantic_space_example")


# ----------------------------------------------------------------------------
# SI figure 4 — how the generated sets vary with condition vs. observed action
# ----------------------------------------------------------------------------


def _condition_axes(alts):
    """The condition columns of this study's cell grid (besides observed action),
    with their level order and display palette."""
    axes = {}
    if "intimacy_condition" in alts.columns:
        axes["intimacy_condition"] = ("intimacy", INTIMACY_LEVELS, INTIMACY_COLORS)
    if "desire_condition" in alts.columns:
        axes["desire_condition"] = ("desire", ["low", "high"], DESIRE_COLORS)
    if "effort_condition" in alts.columns:
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


def fig_si_composition(alts, sem):
    """Composition of the generated alternatives (nearest observed action).
    Encodes the message directly: the observed action (x) moves the composition,
    while the relationship/desire condition barely does. Each bold line is one
    nearest-observed-action category's mean proportion across observed actions; the
    faint lines are the individual condition levels, whose tight bundling around
    the mean shows the composition hardly moves with the condition."""
    cond_axes = _condition_axes(alts)
    # the relational/motivational condition is the "does-it-matter" axis, shown
    # as the faint per-level lines; effort is a cell axis but not varied here
    main_col, (main_name, main_levels, _) = next(
        (c, v) for c, v in cond_axes.items() if c != "effort_condition"
    )

    merged = alts.merge(
        sem[["scenario_label", "action_text", "nearest_observed_action"]],
        on=["scenario_label", "action_text"],
        how="left",
    )
    n_missing = merged["nearest_observed_action"].isna().sum()
    if n_missing:
        print(f"note: {n_missing} alternative instances missing semantic labels")
        merged = merged.dropna(subset=["nearest_observed_action"])

    x = np.arange(len(OBSERVED_ACTIONS))  # observed-action positions

    def prop_nearest(sub):
        p = sub["nearest_observed_action"].value_counts(normalize=True)
        return {a: float(p.get(a, 0.0)) for a in OBSERVED_ACTIONS}

    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    for nc in OBSERVED_ACTIONS:
        color = ACTION_COLORS[nc]
        per_level = []
        for lvl in main_levels:
            ys = [
                prop_nearest(
                    merged[
                        (merged["observed_action"] == obs) & (merged[main_col] == lvl)
                    ]
                )[nc]
                for obs in OBSERVED_ACTIONS
            ]
            per_level.append(ys)
            ax.plot(x, ys, color=color, lw=0.7, alpha=0.3, zorder=2)
        mean_y = np.mean(per_level, axis=0)
        ax.plot(
            x,
            mean_y,
            color=color,
            lw=2.2,
            zorder=4,
            marker="o",
            ms=5,
            label=f"Nearest: {ACTION_LABELS[nc].lower()}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ACTION_LABELS[a].replace(" ", "\n", 1) for a in OBSERVED_ACTIONS], fontsize=8
    )
    ax.set_xlabel("Observed action")
    ax.set_ylabel("Proportion of generated\nalternatives nearest each action")
    ax.set_ylim(0, 0.56)
    ax.set_xlim(-0.25, 2.25)
    ax.legend(
        loc="lower left",
        ncol=1,
        fontsize=8.5,
        handlelength=1.3,
        handletextpad=0.5,
        labelspacing=0.5,
        borderaxespad=1.4,
    )
    fig.tight_layout()
    return savefig(fig, "si_lm_alternatives_composition")


def fig_si_set_similarity(alts, sem, alt_emb):
    """Embedding similarity between the alternative sets of two elicitation
    cells, grouped by what differs between the cells. Puts the condition effect
    on the same scale as the run-to-run baseline and the observed-action
    effect: the sets change no more across conditions than across runs."""
    cond_axes = _condition_axes(alts)
    alts_emb = alts.merge(
        sem[["scenario_label", "action_text"]].assign(row=np.arange(len(sem))),
        on=["scenario_label", "action_text"],
        how="inner",
    )
    cond_cols = list(cond_axes.keys())
    sims = _set_similarity_by_type(alts_emb, alt_emb, cond_cols)

    type_order = ["same_cell", *cond_cols, "observed_action"]
    type_labels = {
        "same_cell": "Same cell\n(across runs)",
        "intimacy_condition": "Across intimacy\nconditions",
        "desire_condition": "Across desire\nconditions",
        "effort_condition": "Across effort\nconditions",
        "observed_action": "Across\nobserved actions",
    }
    fig, ax = plt.subplots(figsize=(3.8, 2.1))
    rng = np.random.default_rng(0)
    for x, t in enumerate(type_order):
        vals = sims.loc[sims["type"] == t, "sim"].to_numpy()
        ax.scatter(
            x + rng.uniform(-0.09, 0.09, len(vals)),
            vals,
            s=13,
            color=ALT_GREY,
            alpha=0.7,
            lw=0,
            zorder=3,
        )
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
    ax.set_xticklabels([type_labels[t] for t in type_order], fontsize=6.5)
    ax.set_ylabel(
        "Between-set embedding\nsimilarity (mean pairwise cosine)", fontsize=8
    )
    ax.set_xlim(-0.5, len(type_order) - 0.5)
    fig.tight_layout()
    return savefig(fig, "si_lm_alternatives_set_similarity")


# ----------------------------------------------------------------------------
# SI figure 4 — where the design can identify desire: within-set g contrast
# ----------------------------------------------------------------------------


def fig_si_g_contrast(runs):
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
            fontsize=7.5,
        )
        ax.set_ylabel("% of sets")
    axes[-1].set_xlabel("Within-choice-set range of goal-satisfaction $g$")
    fig.tight_layout()
    return savefig(fig, "si_lm_g_contrast")


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


def fig_si_base_vs_full(d, runs):
    """The base ablation's alternatives are elicited without the relationship
    paragraph (lm_runs_base.jsonl). Because those features -- not the exact
    wording -- are what enters the planner, we compare the feature distributions
    of the sets rather than their text. For each (scenario, observed action,
    effort) cell, the energy distance between the base set's (risk, effort, g)
    cloud and each relationship level's conditioned cloud; the rightmost
    category is the reference distance between conditioned clouds at different
    levels. Lower is more similar, so if base-vs-conditioned matches the
    reference, the relationship-free set presents the same feature landscape to
    the planner as any conditioned set does to another."""
    base_path = d / "lm_runs_base.jsonl"
    if not base_path.exists():
        print("skipping base-vs-full figure: lm_runs_base.jsonl not found")
        return None
    if "intimacy_condition" not in runs.columns:
        print("skipping base-vs-full figure: study has no relationship axis")
        return None
    base_runs = pd.read_json(base_path, lines=True)

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
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    rng = np.random.default_rng(0)
    for x, cat in enumerate(cats):
        vals = per_scen.loc[per_scen["comparison"] == cat, "dist"].to_numpy()
        ax.scatter(
            x + rng.uniform(-0.1, 0.1, len(vals)),
            vals,
            s=12,
            color=ALT_GREY,
            alpha=0.7,
            lw=0,
            zorder=3,
        )
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
    ax.axvline(len(INTIMACY_LEVELS) - 0.5, color="#CCCCCC", lw=1.1, ls=(0, (4, 3)))
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(
        [INTIMACY_LABELS[lvl].replace(" ", "\n") for lvl in INTIMACY_LEVELS]
        + ["Conditioned vs.\nconditioned (ref.)"],
        fontsize=7.5,
    )
    ax.set_xlabel("Base set vs. the conditioned set at each relationship level")
    ax.set_ylabel(
        "Feature-distribution distance\n(energy distance; lower = more similar)"
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return savefig(fig, "si_lm_base_vs_full")


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


def main(study, seed, example_scenario, figures):
    apply_style()
    d, sem, alts, runs = _load(study)

    npz = None
    if figures in ("si", "all", "diagnostic"):
        emb_path = d / "lm_embeddings.npz"
        if not emb_path.exists():
            raise SystemExit(
                f"{emb_path} not found — run embed_alternatives.py (and "
                "project_alternatives.py) for this study first"
            )
        npz = np.load(emb_path, allow_pickle=False)

    if figures in ("si", "all"):
        print("Rendering SI figures...", flush=True)
        proj = pd.read_json(d / "lm_alternatives_projection.jsonl", lines=True)
        with open(d / "lm_clusters.json") as f:
            clusters = json.load(f)
        for path in (
            fig_si_semantic_space(proj, clusters, example_scenario),
            fig_si_composition(alts, sem),
            fig_si_set_similarity(alts, sem, npz["alt_emb"]),
            fig_si_g_contrast(runs),
            fig_si_base_vs_full(d, runs),
        ):
            if path:
                print(f"wrote {path}")

    if figures in ("diagnostic", "all"):
        print(
            f"Running UMAP on {len(npz['alt_emb'])} alternatives + "
            f"{len(npz['obs_emb'])} observed actions...",
            flush=True,
        )
        alt_xy, obs_xy = _run_umap(npz["alt_emb"], npz["obs_emb"], seed)
        fig_dir = d / "figures"
        fig_dir.mkdir(exist_ok=True)
        print("Rendering diagnostic figures...", flush=True)
        fig_semantic_map(
            sem,
            alt_xy,
            obs_xy,
            npz["obs_action"],
            fig_dir / "fig1_semantic_map.png",
        )
        fig_decision_space(runs, fig_dir / "fig2_decision_space.png")
        print(f"wrote diagnostics to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="food_inv_desire")
    parser.add_argument("--seed", type=int, default=42, help="UMAP random_state.")
    parser.add_argument(
        "--example-scenario",
        default="soup",
        help="Scenario for the choice-set and semantic-space example figures. "
        "Soup projects cleanly: the three observed actions land in separate, "
        "matching-colored clusters with sensible exemplars nearby. (Oysters' "
        "observed actions project almost on top of each other; basketball/wedding "
        "place a no-share exemplar far from the no-share star.)",
    )
    parser.add_argument("--figures", choices=["all", "si", "diagnostic"], default="si")
    args = parser.parse_args()
    main(args.study, args.seed, args.example_scenario, args.figures)
