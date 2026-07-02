#!/usr/bin/env python3
"""SI figures validating the LM elicitation across the four inverse studies.

Reads each study's lm_runs.jsonl (no embeddings needed) and renders three
publication figures into the repo-root figures/ directory:

  1. si_lm_feature_structure — the elicited feature map recovers the intended
     canonical-action structure in every study's elicitation: risk monotone
     across the three actions, goal-satisfaction separating no-share from the
     two sharing actions, and the physical-world manipulation lifting only the
     low-risk share's effort. Thin lines are individual scenarios.
  2. si_lm_manipulation_checks — the given-magnitude manipulations: the LM-rated
     desire scalar separates the low/high desire paragraphs (Studies 2a/2b), and
     the LM-rated intimacy magnitude increases monotonically over the four
     relationship descriptors (Studies 1a/1b).
  3. si_lm_canonical_scatter — all canonical actions in the (risk, effort)
     plane, showing the feature combinations the design targets: no-share
     low/low, low-risk share low-risk/effort-manipulated, high-risk share
     high-risk/low-effort.
  4. si_lm_run_spread — run-to-run spread of the model's predicted belief
     updates (Study 1a, full model): each elicitation run is one simulated
     observer, so the within-cell spread across the K runs is the spread of the
     mixture components, shown against the fitted response noise sigma. Unlike
     figures 1-3 this reads the model outputs (insample_preds.json, produced by
     model/inverse/predict_insample_desire.py) and is skipped with a message if
     they are missing.
  5. si_lm_choice_set_sizes — distribution of the number of alternatives per
     scored choice set (cell x run), per study.
  6. si_lm_mixture_check — predictive check of the simulated-observer mixture
     likelihood (Study 1a, full model): the K-component predictive density
     overlaid on participants' actual belief updates for six example cells.
     Reads insample_preds.json and data/food_inv_desire/main_trials.csv;
     skipped with a message if the model outputs are missing.

Usage:
    uv run python model/lm/plot_si_validation.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import (  # noqa: E402
    ACTION_COLORS,
    ACTION_LABELS,
    ALT_GREY,
    CANONICAL_ACTIONS,
    DESIRE_COLORS,
    EFFORT_COLORS,
    INTIMACY_COLORS,
    INTIMACY_LABELS,
    INTIMACY_LEVELS,
    STUDY_LABELS,
    apply_style,
    panel_label,
    savefig,
)
from utils import get_project_root  # noqa: E402

STUDIES = [
    "food_inv_desire",
    "food_inv_joint_de",
    "food_inv_intimacy",
    "food_inv_joint_ie",
]
MEAN_COLOR = "#333333"
SCENARIO_LINE = dict(color="#999999", alpha=0.4, lw=0.7, zorder=2)


def load_runs(study):
    path = get_project_root() / "model" / "outputs" / "lm" / study / "lm_runs.jsonl"
    return pd.read_json(path, lines=True)


def extract_canonical(runs):
    """One row per (run, cell) with the observed canonical action's features."""
    recs = []
    for rec in runs.itertuples(index=False):
        canon = next((a for a in rec.actions if a["is_canonical"]), None)
        if canon is None:
            continue
        row = dict(
            scenario=rec.scenario_label,
            action=rec.observed_action,
            effort_condition=rec.effort_condition,
            run_id=rec.run_id,
            risk=canon["risk"],
            effort=canon["effort"],
            g=canon["g"],
        )
        for field in ("intimacy_condition", "intimacy", "desire_condition", "desire"):
            if hasattr(rec, field):
                row[field] = getattr(rec, field)
        recs.append(row)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------- figure 1


def fig_feature_structure(canon):
    """Rows = studies, columns = risk / g / effort; thin per-scenario lines."""
    xpos = {a: i for i, a in enumerate(CANONICAL_ACTIONS)}
    fig, axes = plt.subplots(
        len(STUDIES), 3, figsize=(7.0, 8.4), sharex=True, sharey=True
    )

    def draw_lines(ax, per_scenario, mean_line, color=None, xoff=0.0, label=None):
        """per_scenario: DataFrame (scenario x action -> value); mean over top."""
        for _, srow in per_scenario.iterrows():
            ax.plot(
                [xpos[a] + xoff for a in CANONICAL_ACTIONS],
                [srow[a] for a in CANONICAL_ACTIONS],
                **{
                    **SCENARIO_LINE,
                    **({"color": color, "alpha": 0.3} if color else {}),
                },
            )
        xs = [xpos[a] + xoff for a in CANONICAL_ACTIONS]
        ax.plot(
            xs, mean_line, color=(color or MEAN_COLOR), lw=1.8, zorder=5, label=label
        )
        return xs

    for r, study in enumerate(STUDIES):
        df = canon[canon["study"] == study]
        for c, feat in enumerate(("g", "effort", "risk")):
            ax = axes[r, c]
            if feat != "effort":
                wide = (
                    df.groupby(["scenario", "action"])[feat]
                    .mean()
                    .unstack("action")[CANONICAL_ACTIONS]
                )
                draw_lines(ax, wide, wide.mean(axis=0).to_numpy())
            else:
                for cond, xoff in (("low", -0.15), ("high", 0.15)):
                    sub = df[df["effort_condition"] == cond]
                    wide = (
                        sub.groupby(["scenario", "action"])[feat]
                        .mean()
                        .unstack("action")[CANONICAL_ACTIONS]
                    )
                    draw_lines(
                        ax,
                        wide,
                        wide.mean(axis=0).to_numpy(),
                        color=EFFORT_COLORS[cond],
                        xoff=xoff,
                        label=f"{cond.capitalize()} effort condition",
                    )
                if r == 0:
                    ax.legend(loc="upper left", handlelength=1.4, fontsize=7.5)
            if r == 0:
                ax.set_title(
                    {"risk": "Risk", "g": "Goal-satisfaction $g$", "effort": "Effort"}[
                        feat
                    ]
                )
            if c == 0:
                ax.set_ylabel(f"{STUDY_LABELS[study]}\nfeature value")
    for ax in axes[-1]:
        ax.set_xticks(range(3))
        ax.set_xticklabels(
            [ACTION_LABELS[a].replace(" ", "\n", 1) for a in CANONICAL_ACTIONS],
            fontsize=8,
        )
    for ax in axes.ravel():
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.5, 2.5)
        ax.set_yticks([0, 0.5, 1])
    fig.tight_layout()
    return savefig(fig, "si_lm_feature_structure")


# ---------------------------------------------------------------- figure 2


def fig_manipulation_checks(canon):
    """(a) LM-rated intimacy by relationship descriptor (Studies 1a/1b);
    (b) LM-rated desire by desire condition (Studies 2a and 2b overlaid)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    rng = np.random.default_rng(0)

    # (a) intimacy magnitudes (Studies 1a/1b share the elicitation design)
    ax = axes[0]
    df = canon[canon["study"] == "food_inv_desire"]
    lv = df.groupby(["intimacy_condition", "run_id"], as_index=False)[
        "intimacy"
    ].first()
    spread = lv.groupby("intimacy_condition")["intimacy"].nunique()
    if (spread > 1).any():
        print("note: intimacy magnitudes vary across runs; plotting run-level points")
        for i, lvl in enumerate(INTIMACY_LEVELS):
            sub = lv[lv["intimacy_condition"] == lvl]
            ax.scatter(
                i + rng.uniform(-0.08, 0.08, len(sub)),
                sub["intimacy"],
                s=7,
                color=INTIMACY_COLORS[lvl],
                alpha=0.3,
                lw=0,
            )
    means = lv.groupby("intimacy_condition")["intimacy"].mean()[INTIMACY_LEVELS]
    ax.plot(range(4), means, color=MEAN_COLOR, lw=1.8, zorder=5)
    for i, lvl in enumerate(INTIMACY_LEVELS):
        ax.scatter(
            i,
            means[lvl],
            s=48,
            color=INTIMACY_COLORS[lvl],
            edgecolor="black",
            lw=0.5,
            zorder=6,
        )
        ax.annotate(
            f"{means[lvl]:g}",
            (i, means[lvl]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        [INTIMACY_LABELS[lvl].replace(" ", "\n") for lvl in INTIMACY_LEVELS],
        fontsize=7.5,
    )
    ax.set_xlabel("Relationship descriptor")
    ax.set_ylabel("LM-rated intimacy $I$")
    ax.set_xlim(-0.5, 3.5)
    ax.set_title("Studies 1a/1b")
    panel_label(ax, "a")

    # (b) desire by condition, Studies 2a and 2b overlaid with an x offset
    ax = axes[1]
    xpos = {"low": 0, "high": 1}
    specs = (
        ("food_inv_intimacy", -0.14, True, "-"),
        ("food_inv_joint_ie", 0.14, False, (0, (4, 2))),
    )
    for study, xoff, filled, ls in specs:
        df = canon[canon["study"] == study]
        # one point per scenario: its mean desire in each condition, so the
        # faint lines connect the actual plotted points (each line = one
        # scenario's low->high shift)
        wide = (
            df.groupby(["scenario", "desire_condition"])["desire"]
            .mean()
            .unstack("desire_condition")[["low", "high"]]
        )
        lo, hi = wide["low"].to_numpy(), wide["high"].to_numpy()
        # small jitter (shared between line endpoints and points so the lines
        # still end exactly on the points): vertical to separate scenarios on
        # the same gridded value, horizontal to avoid overplotting the column
        jl = rng.uniform(-0.012, 0.012, len(lo))
        jh = rng.uniform(-0.012, 0.012, len(hi))
        xl = xoff + rng.uniform(-0.035, 0.035, len(lo))
        xh = 1 + xoff + rng.uniform(-0.035, 0.035, len(hi))
        for i in range(len(lo)):
            ax.plot(
                [xl[i], xh[i]],
                [lo[i] + jl[i], hi[i] + jh[i]],
                color="#AAAAAA",
                alpha=0.45,
                lw=0.6,
                zorder=2,
            )
        ax.scatter(
            xl,
            lo + jl,
            s=12,
            color=DESIRE_COLORS["low"],
            alpha=0.6,
            lw=0,
            zorder=3,
        )
        ax.scatter(
            xh,
            hi + jh,
            s=12,
            color=DESIRE_COLORS["high"],
            alpha=0.6,
            lw=0,
            zorder=3,
        )
        ax.plot(
            [xoff, 1 + xoff],
            wide.mean(axis=0),
            color=MEAN_COLOR,
            lw=1.8,
            ls=ls,
            zorder=5,
            label=STUDY_LABELS[study],
        )
        for cond in ("low", "high"):
            style = (
                dict(facecolor=DESIRE_COLORS[cond], edgecolor="black", lw=0.5)
                if filled
                else dict(facecolor="white", edgecolor=DESIRE_COLORS[cond], lw=1.4)
            )
            ax.scatter(xpos[cond] + xoff, wide[cond].mean(), s=48, zorder=6, **style)
    ax.legend(loc="lower right", fontsize=7.5, handlelength=1.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Low", "High"])
    ax.set_xlabel("Desire condition")
    ax.set_xlim(-0.6, 1.6)
    ax.set_title("Studies 2a/2b")
    panel_label(ax, "b")

    for ax in axes:
        ax.set_ylim(-0.06, 1.12)
        ax.set_yticks([0, 0.5, 1])
    fig.tight_layout()
    return savefig(fig, "si_lm_manipulation_checks")


# ---------------------------------------------------------------- figure 3


def fig_canonical_scatter(canon):
    """All canonical actions in the (risk, effort) plane (Study 1a's elicitation),
    one point per scenario x action x effort condition; vertical grey segments
    connect the two effort conditions of the same scenario x action."""
    df = canon[canon["study"] == "food_inv_desire"]
    agg = df.groupby(["scenario", "action", "effort_condition"], as_index=False).agg(
        risk=("risk", "mean"), effort=("effort", "mean")
    )
    wide = agg.pivot_table(
        index=["scenario", "action"],
        columns="effort_condition",
        values=["risk", "effort"],
    )

    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    rng = np.random.default_rng(0)
    for pos in (0.5,):
        ax.axhline(pos, color="#DDDDDD", ls=(0, (4, 3)), lw=1.1, zorder=0)
        ax.axvline(pos, color="#DDDDDD", ls=(0, (4, 3)), lw=1.1, zorder=0)

    for (scenario, action), row in wide.iterrows():
        dx = rng.uniform(-0.012, 0.012)  # same x-jitter for the pair: risk is
        dy = rng.uniform(-0.008, 0.008)  # effort-marginal, so segments stay vertical
        x = {c: row[("risk", c)] + dx for c in ("low", "high")}
        y = {c: row[("effort", c)] + dy for c in ("low", "high")}
        ax.plot(
            [x["low"], x["high"]],
            [y["low"], y["high"]],
            color="#BBBBBB",
            lw=0.6,
            alpha=0.6,
            zorder=1,
        )
        for cond in ("low", "high"):
            filled = cond == "high"
            ax.scatter(
                x[cond],
                y[cond],
                s=34,
                facecolor=ACTION_COLORS[action] if filled else "white",
                edgecolor=ACTION_COLORS[action],
                lw=1.1,
                zorder=4 if filled else 3,
                alpha=0.9,
            )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ACTION_COLORS[a],
            markeredgecolor=ACTION_COLORS[a],
            markersize=7,
            label=ACTION_LABELS[a],
        )
        for a in CANONICAL_ACTIONS
    ] + [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="#777777",
            markersize=7,
            label="High effort condition",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#777777",
            markersize=7,
            label="Low effort condition",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9.5, handletextpad=0.3)
    ax.set_xlabel("Risk", fontsize=13)
    ax.set_ylabel("Effort", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_box_aspect(1)
    fig.tight_layout()
    return savefig(fig, "si_lm_canonical_scatter")


# ---------------------------------------------------------------- figure 4


def fig_run_spread():
    """Run-to-run spread of the model's predicted belief updates (Study 1a,
    full model, in-sample fitted weights). Each elicitation run is one
    simulated observer, so the per-run updates within a cell are the
    components of the mixture likelihood; the fitted response noise sigma
    gives the scale to read the spread against."""
    out_dir = get_project_root() / "model" / "outputs" / "food_inv_desire"
    preds_path = out_dir / "insample_preds.json"
    if not preds_path.exists():
        print(
            "skipping run-spread figure: insample_preds.json not found "
            "(run model/inverse/predict_insample_desire.py first)"
        )
        return None
    with open(preds_path) as f:
        rows = [r for r in json.load(f) if r["model"] == "full"]
    if "delta_pred_runs" not in rows[0]:
        print(
            "skipping run-spread figure: insample_preds.json lacks per-run "
            "deltas (re-run model/inverse/predict_insample_desire.py)"
        )
        return None
    with open(out_dir / "fit_results.json") as f:
        fits = json.load(f)
    sigma = float(next(v for v in fits if v["model"] == "full")["param_sigma"])

    means = np.array([r["delta_pred"] for r in rows])
    runs = np.array([r["delta_pred_runs"] for r in rows])  # (cells, K)
    sds = np.array([r["delta_pred_sd"] for r in rows])
    order = np.argsort(means)
    n_cells, K = runs.shape

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.7, 1]}
    )

    # (a) every cell's K per-run updates, against the fitted noise band
    ax = axes[0]
    xs = np.arange(n_cells)
    ax.fill_between(
        xs,
        means[order] - sigma,
        means[order] + sigma,
        color="#EBEBEB",
        lw=0,
        label="mean $\\pm$ fitted $\\sigma$",
    )
    ax.axhline(0, color="#CCCCCC", lw=0.7, zorder=0)
    ax.scatter(
        np.repeat(xs, K),
        runs[order].ravel(),
        s=2,
        color="#777777",
        alpha=0.25,
        lw=0,
        zorder=3,
    )
    ax.plot(
        xs, means[order], color=MEAN_COLOR, lw=1.3, zorder=5, label="mean over runs"
    )
    ax.set_xlabel(f"Condition cells, sorted by mean update ({n_cells} cells)")
    ax.set_ylabel("Predicted belief update")
    ax.set_xticks([])
    ax.legend(loc="upper left", fontsize=7.5)
    panel_label(ax, "a", dx=-0.06)

    # (b) within-cell run SD, against sigma
    ax = axes[1]
    ax.hist(sds, bins=30, color="#9AA0A6", edgecolor="white", lw=0.4)
    ax.axvline(sigma, color=MEAN_COLOR, ls="--", lw=1.3)
    ax.annotate(
        f"fitted $\\sigma$ = {sigma:.2f}",
        (sigma, ax.get_ylim()[1]),
        xytext=(-4, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.set_xlabel("Within-cell SD of update across runs")
    ax.set_ylabel("Cells")
    panel_label(ax, "b", dx=-0.18)

    print(
        f"run spread: median within-cell SD = {np.median(sds):.3f}, "
        f"fitted sigma = {sigma:.3f} (ratio {np.median(sds) / sigma:.2f})"
    )
    fig.tight_layout()
    return savefig(fig, "si_lm_run_spread")


# ---------------------------------------------------------------- figure 5


def fig_choice_set_sizes(runs_by_study):
    """Distribution of the number of LM-generated alternatives in each scored
    choice set (one set per cell x run), per study. Documents the "small,
    focused set" the generation prompt asks for — and makes visible that a
    fraction of Study 1a's sets contain no alternatives at all."""
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharex=True, sharey=True)
    xs = np.arange(0, 8)
    for ax, study in zip(axes, STUDIES):
        sizes = runs_by_study[study]["actions"].apply(len) - 1
        pct = sizes.value_counts(normalize=True).sort_index() * 100
        ax.bar(
            xs,
            [pct.get(x, 0) for x in xs],
            width=0.8,
            color=ALT_GREY,
            edgecolor="white",
            lw=0.5,
        )
        ax.set_title(STUDY_LABELS[study], fontsize=9)
        ax.set_xticks(xs[::2])
    axes[0].set_ylabel("% of choice sets")
    fig.supxlabel("Number of LM-generated alternatives in the scored set", fontsize=9)
    fig.tight_layout()
    return savefig(fig, "si_lm_choice_set_sizes")


# ---------------------------------------------------------------- figure 6


def fig_mixture_check():
    """Predictive check of the simulated-observer mixture (Study 1a, full
    model, in-sample weights): for six cells spanning the range of predicted
    updates, the K-component predictive density (1/K) sum_k N(u; delta_k,
    sigma^2) is overlaid on the distribution of participants' actual belief
    updates in that cell. Ticks at the bottom mark the K per-run delta_k."""
    out_dir = get_project_root() / "model" / "outputs" / "food_inv_desire"
    preds_path = out_dir / "insample_preds.json"
    if not preds_path.exists():
        print("skipping mixture-check figure: insample_preds.json not found")
        return None
    with open(preds_path) as f:
        rows = [r for r in json.load(f) if r["model"] == "full"]
    if "delta_pred_runs" not in rows[0]:
        print("skipping mixture-check figure: no per-run deltas in insample_preds.json")
        return None
    with open(out_dir / "fit_results.json") as f:
        fits = json.load(f)
    sigma = float(next(v for v in fits if v["model"] == "full")["param_sigma"])

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

    rows = sorted(rows, key=lambda r: r["delta_pred"])
    picks = [rows[int(q * (len(rows) - 1))] for q in (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)]

    u = np.linspace(-1, 1, 401)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), sharex=True, sharey=True)
    for ax, r in zip(axes.ravel(), picks):
        h = wide[
            (wide["scenario_label"] == r["scenario_label"])
            & (wide["action_condition"] == r["action_condition"])
            & (wide["effort_condition"] == r["effort"])
            & (wide["intimacy_condition"] == r["intimacy"])
        ]["update"].to_numpy()
        deltas = np.asarray(r["delta_pred_runs"])
        dens = np.mean(
            np.exp(-((u[None, :] - deltas[:, None]) ** 2) / (2 * sigma**2))
            / (sigma * np.sqrt(2 * np.pi)),
            axis=0,
        )
        ax.hist(
            h,
            bins=np.arange(-1, 1.01, 0.125),
            density=True,
            color="#DDDDDD",
            edgecolor="white",
            lw=0.5,
        )
        ax.plot(u, dens, color=MEAN_COLOR, lw=1.4, zorder=5)
        ax.vlines(deltas, 0, 0.16, color="#777777", lw=0.6, alpha=0.7, zorder=4)
        ax.set_title(
            f"{r['scenario_label']} - {ACTION_LABELS[r['action_condition']].lower()}\n"
            f"{r['effort']} effort, {INTIMACY_LABELS[r['intimacy']].lower()}"
            f"  (n = {len(h)})",
            fontsize=7.5,
        )
        ax.set_xlim(-1, 1)
    for ax in axes[-1]:
        ax.set_xlabel("Belief update")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    fig.tight_layout()
    return savefig(fig, "si_lm_mixture_check")


def main():
    apply_style()
    runs_by_study = {}
    frames = []
    for study in STUDIES:
        runs_by_study[study] = load_runs(study)
        df = extract_canonical(runs_by_study[study])
        df["study"] = study
        frames.append(df)
        print(f"{study}: {len(df)} canonical rows")
    canon = pd.concat(frames, ignore_index=True)
    for path in (
        fig_feature_structure(canon),
        fig_manipulation_checks(canon),
        fig_canonical_scatter(canon),
        fig_run_spread(),
        fig_choice_set_sizes(runs_by_study),
        fig_mixture_check(),
    ):
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
