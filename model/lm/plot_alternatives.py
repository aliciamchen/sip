#!/usr/bin/env python3
"""
Visualize the space of LM-generated alternatives (diagnostic figures, not part of
the fitting pipeline). Reads the artifacts written by embed_alternatives.py +
score_merged.py and renders three figures:

  1. Semantic map — a UMAP projection of the alternative embeddings, in two panels:
     (a) colored by scenario, which shows the space splits into ~16 scenario blobs
     (why clustering is done per scenario), and (b) colored by the nearest canonical
     action with the canonical actions overlaid as stars, which shows the action-type
     organization and where the canonicals sit in their local clouds.
  2. Decision space — the alternatives in the (risk, g) and (risk, effort) feature
     planes the model actually reasons over, faceted by observed action, with the
     observed action marked and Pareto-dominated / dominating alternatives flagged.
  3. Anchoring — the same UMAP layout, one panel per observed action, showing where
     that observed action's imagined set concentrates (does it shift toward the
     contrasting region — e.g. safer options after a high-risk share is observed?).

UMAP runs here (not in embed_alternatives.py) so the projection can be re-tuned
without re-calling the embedding API; it reads the persisted embeddings from
lm_embeddings.npz.

Usage:
    uv run python model/lm/plot_alternatives.py --study food_inv_desire

Requires (produced by the elicitation pipeline for the study):
    outputs/lm/<slug>/lm_embeddings.npz, lm_alternatives_semantic.jsonl,
    lm_alternatives.jsonl, lm_runs.jsonl
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

CANONICAL_ACTIONS = ["no_share", "low_risk_share", "high_risk_share"]
# Match the action palette + labels used in the R elicitation notebook.
ACTION_COLORS = {
    "no_share": "#7A9A4A",
    "low_risk_share": "#C9A85A",
    "high_risk_share": "#B05A5A",
}
ACTION_LABELS = {
    "no_share": "No share",
    "low_risk_share": "Low-risk share",
    "high_risk_share": "High-risk share",
}


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "font.size": 10,
        }
    )


def _load(study):
    """Load the per-study artifacts. The embeddings npz (large, UMAP-only) is loaded
    separately in the diagnostic branch so the manuscript figures don't pay for it."""
    d = get_project_root() / "model" / "outputs" / "lm" / study
    sem = pd.read_json(d / "lm_alternatives_semantic.jsonl", lines=True)
    alts = pd.read_json(d / "lm_alternatives.jsonl", lines=True)
    runs = pd.read_json(d / "lm_runs.jsonl", lines=True)
    return d, sem, alts, runs


def _ms_style():
    """Print-ready styling for the manuscript figures (serif, larger labels)."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _savefig(fig, stem):
    """Write a vector PDF (for LaTeX \\includegraphics) plus a PNG preview."""
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_umap(alt_emb, canon_emb, seed):
    """Project alternatives + canonicals into one 2D layout (fit jointly so the
    canonical anchors live in the same space as the alternatives)."""
    X = np.vstack([alt_emb, canon_emb])
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed
    )
    Y = reducer.fit_transform(X)
    return Y[: len(alt_emb)], Y[len(alt_emb) :]


def fig_semantic_map(sem, alt_xy, canon_xy, canon_action, out):
    _style()
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

    # (b) colored by nearest canonical action, canonical actions as stars.
    nc = sem["nearest_canonical"].to_numpy()
    for a in CANONICAL_ACTIONS:
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
    for j in range(len(canon_xy)):
        ax2.scatter(
            canon_xy[j, 0],
            canon_xy[j, 1],
            marker="*",
            s=110,
            color=ACTION_COLORS[canon_action[j]],
            edgecolor="black",
            lw=0.5,
            zorder=5,
        )
    ax2.set_title(
        "(b) Colored by nearest canonical action\n★ = the 16×3 canonical actions"
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
    observed (canonical) action of the same (cell, run)."""
    rows = []
    for actions, scen, obs in zip(
        runs["actions"], runs["scenario_label"], runs["observed_action"]
    ):
        canon = next((a for a in actions if a["is_canonical"]), None)
        if canon is None:
            continue
        cg, ce, cr = canon["g"], canon["effort"], canon["risk"]
        rows.append(dict(observed=obs, kind="observed", risk=cr, effort=ce, g=cg))
        for a in actions:
            if a["is_canonical"]:
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
    _style()
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
        for c, obs in enumerate(CANONICAL_ACTIONS):
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


def fig_anchoring(sem, alts, alt_xy, canon_xy, out):
    _style()
    coords = sem[["scenario_label", "action_text"]].copy()
    coords["x"], coords["y"] = alt_xy[:, 0], alt_xy[:, 1]
    merged = alts.merge(coords, on=["scenario_label", "action_text"], how="inner")

    xall, yall = alt_xy[:, 0], alt_xy[:, 1]
    extent = [xall.min(), xall.max(), yall.min(), yall.max()]
    cmap = "mako_r" if "mako_r" in plt.colormaps() else "viridis"
    subs = {obs: merged[merged["observed_action"] == obs] for obs in CANONICAL_ACTIONS}
    # Shared color scale so density is comparable across the three panels: take the
    # max per-hexbin count over all panels (render once offscreen to read the counts).
    vmax = 1
    for obs in CANONICAL_ACTIONS:
        tmp = plt.figure()
        h = tmp.gca().hexbin(subs[obs]["x"], subs[obs]["y"], gridsize=45, extent=extent)
        vmax = max(vmax, int(h.get_array().max()))
        plt.close(tmp)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    hb = None
    for c, obs in enumerate(CANONICAL_ACTIONS):
        ax = axes[c]
        ax.scatter(xall, yall, s=2, color="#E8E8E8", lw=0)  # full cloud for reference
        sub = subs[obs]
        hb = ax.hexbin(
            sub["x"],
            sub["y"],
            gridsize=45,
            extent=extent,
            cmap=cmap,
            mincnt=1,
            linewidths=0,
            vmin=1,
            vmax=vmax,
        )
        ax.scatter(
            canon_xy[:, 0],
            canon_xy[:, 1],
            marker="*",
            s=45,
            color="black",
            lw=0,
            alpha=0.6,
            zorder=5,
        )
        # nearest-canonical mix for this observed action, as a subtitle
        mix = sub.merge(
            sem[["scenario_label", "action_text", "nearest_canonical"]],
            on=["scenario_label", "action_text"],
        )["nearest_canonical"].value_counts(normalize=True)
        mixtxt = "  ".join(
            f"{ACTION_LABELS[a].split()[0].lower()}:{100 * mix.get(a, 0):.0f}%"
            for a in CANONICAL_ACTIONS
        )
        ax.set_title(
            f"Observed: {ACTION_LABELS[obs]}\nnearest-canon mix → {mixtxt}", fontsize=10
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(hb, ax=axes, shrink=0.7, label="alt density (shared scale)")
    fig.suptitle(
        "Anchoring: where each observed action's imagined set concentrates\n"
        "(grey = all alternatives; ★ = canonical actions)",
        fontsize=13,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Manuscript figures (print-ready; do not depend on UMAP / the embeddings npz)
# ----------------------------------------------------------------------------


def fig_choice_set_example(runs, scenario, out_stem):
    """Manuscript Fig. 1 — one scenario's elicited choice set in the model's feature
    space: the three canonical actions (the observed action is one of them) plus the
    LM-generated alternatives, on risk x g and risk x effort. Makes the G_LM + phi_tau
    pipeline concrete next to the example vignette. Features are averaged over runs
    per distinct action; effort over both effort conditions."""
    _ms_style()
    sub = runs[runs["scenario_label"] == scenario]
    if sub.empty:
        raise SystemExit(f"scenario {scenario!r} not found in lm_runs.jsonl")
    recs = []
    for actions, obs in zip(sub["actions"], sub["observed_action"]):
        for a in actions:
            recs.append(
                dict(
                    text=a["action_text"],
                    is_canon=a["is_canonical"],
                    obs=obs,
                    risk=a["risk"],
                    effort=a["effort"],
                    g=a["g"],
                )
            )
    df = pd.DataFrame(recs).dropna(subset=["risk", "g", "effort"])
    agg = df.groupby(["text", "is_canon"], as_index=False).agg(
        risk=("risk", "mean"),
        effort=("effort", "mean"),
        g=("g", "mean"),
        obs=("obs", "first"),
    )
    canon, altr = agg[agg["is_canon"]], agg[~agg["is_canon"]]

    rng = np.random.default_rng(0)
    jit = lambda v: v + rng.uniform(-0.012, 0.012, size=len(v))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax, (xf, yf, yl) in zip(
        axes, [("risk", "g", "goal-satisfaction  g"), ("risk", "effort", "effort")]
    ):
        ax.scatter(
            jit(altr[xf].to_numpy()),
            jit(altr[yf].to_numpy()),
            s=22,
            color="#9AA0A6",
            alpha=0.55,
            lw=0,
        )
        for _, row in canon.iterrows():
            act = row["obs"]
            ax.scatter(
                row[xf],
                row[yf],
                marker="*",
                s=300,
                color=ACTION_COLORS[act],
                edgecolor="black",
                lw=0.9,
                zorder=6,
            )
            ax.annotate(
                ACTION_LABELS[act],
                (row[xf], row[yf]),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=9,
                weight="bold",
            )
        ax.set_xlim(-0.05, 1.10)
        ax.set_ylim(-0.05, 1.14)
        ax.set_xlabel("risk")
        ax.set_ylabel(yl)

    # call out a few representative alternatives on the risk x g panel
    a0 = altr.sort_values("risk")
    picks = a0.iloc[[0, len(a0) // 2, len(a0) - 1]] if len(a0) >= 3 else a0
    for _, row in picks.iterrows():
        txt = (row["text"][:44] + "…") if len(row["text"]) > 45 else row["text"]
        axes[0].annotate(
            txt,
            (row["risk"], row["g"]),
            textcoords="offset points",
            xytext=(9, -11),
            fontsize=7.5,
            color="#444444",
            arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6),
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9AA0A6",
            markersize=7,
            label="LM-generated alternatives",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#888888",
            markeredgecolor="black",
            markersize=15,
            label="canonical actions",
        ),
    ]
    axes[0].legend(handles=handles, loc="lower right", frameon=False)
    fig.suptitle(
        f"Elicited choice set for an example scenario ({scenario})", fontsize=14
    )
    fig.tight_layout()
    _savefig(fig, out_stem)


def fig_feature_validation(runs, out_stem):
    """Manuscript Fig. 2 — the LM feature map recovers the intended action structure:
    risk monotone across actions, g separating no-share from sharing, and the effort
    manipulation lifting only the low-risk share. Each point is one scenario's mean;
    the dark line connects the across-scenario means."""
    _ms_style()
    recs = []
    for actions, scen, obs, eff in zip(
        runs["actions"],
        runs["scenario_label"],
        runs["observed_action"],
        runs["effort_condition"],
    ):
        c = next((a for a in actions if a["is_canonical"]), None)
        if c is None:
            continue
        recs.append(
            dict(
                scenario=scen,
                obs=obs,
                eff=eff,
                risk=c["risk"],
                g=c["g"],
                effort=c["effort"],
            )
        )
    df = pd.DataFrame(recs)
    rg = df.groupby(["scenario", "obs"], as_index=False).agg(
        risk=("risk", "mean"), g=("g", "mean")
    )
    ef = df.groupby(["scenario", "obs", "eff"], as_index=False).agg(
        effort=("effort", "mean")
    )

    order = CANONICAL_ACTIONS
    xpos = {a: i for i, a in enumerate(order)}
    rng = np.random.default_rng(0)

    def _scatter_means(ax, data, valcol, color_by_action, color=None, xoff=0.0):
        xs, ms = [], []
        for a in order:
            sd = data[data["obs"] == a]
            x = xpos[a] + xoff + rng.uniform(-0.05, 0.05, size=len(sd))
            ax.scatter(
                x,
                sd[valcol],
                s=14,
                color=(ACTION_COLORS[a] if color_by_action else color),
                alpha=0.35,
                lw=0,
            )
            xs.append(xpos[a] + xoff)
            ms.append(sd[valcol].mean())
        return xs, ms

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, valcol, title in (
        (axes[0], "risk", "Risk"),
        (axes[1], "g", "Goal-satisfaction  g"),
    ):
        xs, ms = _scatter_means(ax, rg, valcol, color_by_action=True)
        ax.plot(xs, ms, color="#333333", lw=1.4, zorder=5)
        ax.scatter(xs, ms, color="#333333", s=30, zorder=6)
        ax.set_title(title)

    eff_colors = {"low": "#B5C9A8", "high": "#4A7A4A"}
    for cond, xoff in (("low", -0.13), ("high", 0.13)):
        sd = ef[ef["eff"] == cond]
        xs, ms = _scatter_means(axes[2], sd, "effort", False, eff_colors[cond], xoff)
        axes[2].plot(
            xs,
            ms,
            color=eff_colors[cond],
            lw=1.6,
            zorder=5,
            label=f"{cond} effort condition",
        )
        axes[2].scatter(
            xs, ms, color=eff_colors[cond], edgecolor="black", lw=0.4, s=34, zorder=6
        )
    axes[2].set_title("Effort")
    axes[2].legend(frameon=False, loc="upper left")

    for ax in axes:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([ACTION_LABELS[a] for a in order], rotation=18, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.5, len(order) - 0.5)
    axes[0].set_ylabel("feature value (0–1)")
    fig.suptitle(
        "LM-elicited feature map recovers the intended action structure", fontsize=14
    )
    fig.tight_layout()
    _savefig(fig, out_stem)


def main(study, seed, example_scenario, figures):
    d, sem, alts, runs = _load(study)
    fig_dir = d / "figures"
    fig_dir.mkdir(exist_ok=True)

    if figures in ("manuscript", "all"):
        print("Rendering manuscript figures...", flush=True)
        fig_choice_set_example(
            runs, example_scenario, str(fig_dir / "ms_choice_set_example")
        )
        fig_feature_validation(runs, str(fig_dir / "ms_feature_validation"))

    if figures in ("diagnostic", "all"):
        npz = np.load(d / "lm_embeddings.npz", allow_pickle=False)
        print(
            f"Running UMAP on {len(npz['alt_emb'])} alternatives + "
            f"{len(npz['canon_emb'])} canonicals...",
            flush=True,
        )
        alt_xy, canon_xy = _run_umap(npz["alt_emb"], npz["canon_emb"], seed)
        print("Rendering diagnostic figures...", flush=True)
        fig_semantic_map(
            sem,
            alt_xy,
            canon_xy,
            npz["canon_action"],
            fig_dir / "fig1_semantic_map.png",
        )
        fig_decision_space(runs, fig_dir / "fig2_decision_space.png")
        fig_anchoring(sem, alts, alt_xy, canon_xy, fig_dir / "fig3_anchoring.png")

    print(f"\nWrote figures to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="food_inv_desire")
    parser.add_argument("--seed", type=int, default=42, help="UMAP random_state.")
    parser.add_argument(
        "--example-scenario",
        default="oysters",
        help="Scenario for the manuscript worked-example figure.",
    )
    parser.add_argument(
        "--figures", choices=["all", "manuscript", "diagnostic"], default="all"
    )
    args = parser.parse_args()
    main(args.study, args.seed, args.example_scenario, args.figures)
