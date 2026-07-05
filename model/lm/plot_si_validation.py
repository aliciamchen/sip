#!/usr/bin/env python3
"""SI figures validating the LM elicitation across the inverse studies.

Reads each study's lm_runs.jsonl (no embeddings needed) and renders the
publication figures into the repo-root figures/ directory. Studies whose
elicitation hasn't been run yet (no lm_runs.jsonl — e.g. the nonfood studies
before their LM pipeline runs) are skipped with a message, and each figure is
built from whichever studies are present:

  1. si_lm_feature_structure — the elicited feature map recovers the intended
     observed-action structure in every study's elicitation: risk monotone
     across the three actions, goal-satisfaction separating no-share from the
     two sharing actions, and the physical-world manipulation lifting only the
     low-risk share's effort. Thin lines are individual scenarios.
  2. si_lm_manipulation_checks — the given-magnitude manipulations: the LM-rated
     desire scalar separates the low/high desire paragraphs (Studies 2a/2b), and
     the LM-rated intimacy magnitude increases monotonically over the four
     relationship descriptors (Studies 1a/1b).
  3. si_lm_observed_scatter — all observed actions in the (risk, effort)
     plane, showing the feature combinations the design targets: no-share
     low/low, low-risk share low-risk/effort-manipulated, high-risk share
     high-risk/low-effort.
  4. si_lm_run_spread — run-to-run spread of the model's predicted belief
     updates (Study 1a, full model): each elicitation run is one simulated
     observer, so the within-cell spread across the K runs is the spread of the
     mixture components, shown against the fitted response noise sigma. Unlike
     figures 1-3 this reads the model's out-of-sample CV predictions
     (cv_preds_summary.json, produced by model/cv/cv_food_inv_desire.py) and is
     skipped with a message if they are missing.
  5. si_lm_choice_set_sizes — distribution of the number of alternatives per
     scored choice set (cell x run), per study.
  6. si_lm_mixture_check — predictive check of the simulated-observer mixture
     likelihood (Study 1a, full model): the K-component predictive density
     overlaid on participants' actual belief updates for six example cells.
     Reads cv_preds_summary.json and data/food_inv_desire/main_trials.csv;
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
    OBSERVED_ACTIONS,
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
    "nonfood_inv_joint_de",
    "nonfood_inv_joint_ie",
]
FOOD_STUDIES = [
    "food_inv_desire",
    "food_inv_joint_de",
    "food_inv_intimacy",
    "food_inv_joint_ie",
]
STUDY3_STUDIES = ["nonfood_inv_joint_de", "nonfood_inv_joint_ie"]
JOINT_DE_COMPARISON = ["food_inv_joint_de", "nonfood_inv_joint_de"]
JOINT_IE_COMPARISON = ["food_inv_joint_ie", "nonfood_inv_joint_ie"]

# The studies whose elicitation carries each given-magnitude scalar. The paper
# figure stays food-only; Study 3 and the joint comparisons get separate files.
FOOD_INTIMACY_PANEL_STUDIES = [
    ("food_inv_desire", "1a/1b"),
]
FOOD_DESIRE_PANEL_STUDIES = [
    ("food_inv_intimacy", "2a"),
    ("food_inv_joint_ie", "2b"),
]
STUDY3_INTIMACY_PANEL_STUDIES = [
    ("nonfood_inv_joint_de", "3a"),
]
STUDY3_DESIRE_PANEL_STUDIES = [
    ("nonfood_inv_joint_ie", "3b"),
]
JOINT_DE_INTIMACY_PANEL_STUDIES = [
    ("food_inv_joint_de", "1b"),
    ("nonfood_inv_joint_de", "3a"),
]
JOINT_IE_DESIRE_PANEL_STUDIES = [
    ("food_inv_joint_ie", "2b"),
    ("nonfood_inv_joint_ie", "3b"),
]
MEAN_COLOR = "#333333"
SCENARIO_LINE = dict(color="#999999", alpha=0.4, lw=0.7, zorder=2)
SAVE_KW = {"png": False}


def load_runs(study):
    """The study's lm_runs.jsonl as a DataFrame, or None if its elicitation
    hasn't been run yet."""
    path = get_project_root() / "model" / "outputs" / "lm" / study / "lm_runs.jsonl"
    if not path.exists():
        return None
    return pd.read_json(path, lines=True)


def extract_observed(runs):
    """One row per (run, cell) with the observed action's features."""
    recs = []
    for rec in runs.itertuples(index=False):
        observed = next((a for a in rec.actions if a["is_observed"]), None)
        if observed is None:
            continue
        row = dict(
            scenario=rec.scenario_label,
            action=rec.observed_action,
            effort_condition=rec.effort_condition,
            run_id=rec.run_id,
            risk=observed["risk"],
            effort=observed["effort"],
            g=observed["g"],
        )
        for field in ("intimacy_condition", "intimacy", "desire_condition", "desire"):
            if hasattr(rec, field):
                row[field] = getattr(rec, field)
        recs.append(row)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------- figure 1


def fig_feature_structure(
    observed,
    studies,
    figname="si_lm_feature_structure",
    show_effort_condition=True,
):
    """Rows = studies (the ones with elicitation data), columns = risk / g /
    effort; thin per-scenario lines."""
    xpos = {a: i for i, a in enumerate(OBSERVED_ACTIONS)}
    fig, axes = plt.subplots(
        len(studies), 3, figsize=(6.4, 1.4 * len(studies)), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    def draw_lines(ax, per_scenario, mean_line, color=None, xoff=0.0, label=None):
        """per_scenario: DataFrame (scenario x action -> value); mean over top."""
        for _, srow in per_scenario.iterrows():
            ax.plot(
                [xpos[a] + xoff for a in OBSERVED_ACTIONS],
                [srow[a] for a in OBSERVED_ACTIONS],
                **{
                    **SCENARIO_LINE,
                    **({"color": color, "alpha": 0.3} if color else {}),
                },
            )
        xs = [xpos[a] + xoff for a in OBSERVED_ACTIONS]
        ax.plot(
            xs, mean_line, color=(color or MEAN_COLOR), lw=1.8, zorder=5, label=label
        )
        return xs

    for r, study in enumerate(studies):
        df = observed[observed["study"] == study]
        for c, feat in enumerate(("g", "effort", "risk")):
            ax = axes[r, c]
            if feat != "effort":
                wide = (
                    df.groupby(["scenario", "action"])[feat]
                    .mean()
                    .unstack("action")[OBSERVED_ACTIONS]
                )
                draw_lines(ax, wide, wide.mean(axis=0).to_numpy())
            elif show_effort_condition:
                for cond, xoff in (("low", -0.15), ("high", 0.15)):
                    sub = df[df["effort_condition"] == cond]
                    wide = (
                        sub.groupby(["scenario", "action"])[feat]
                        .mean()
                        .unstack("action")[OBSERVED_ACTIONS]
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
            else:
                wide = (
                    df.groupby(["scenario", "action"])[feat]
                    .mean()
                    .unstack("action")[OBSERVED_ACTIONS]
                )
                draw_lines(ax, wide, wide.mean(axis=0).to_numpy())
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
            [ACTION_LABELS[a].replace(" ", "\n", 1) for a in OBSERVED_ACTIONS],
            fontsize=8,
        )
    for ax in axes.ravel():
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.5, 2.5)
        ax.set_yticks([0, 0.5, 1])
    fig.tight_layout()
    return savefig(fig, figname, **SAVE_KW)


# ---------------------------------------------------------------- figure 2


def fig_manipulation_checks(
    observed,
    intimacy_panel_studies,
    desire_panel_studies,
    figname="si_lm_manipulation_checks",
):
    """(a) LM-rated intimacy by relationship descriptor (the given-relationship
    studies); (b) LM-rated desire by desire condition (the given-desire studies
    overlaid). Each panel draws whichever of its studies have elicitation data,
    offset on x when there is more than one."""
    active_panels = [
        bool(intimacy_panel_studies),
        bool(desire_panel_studies),
    ]
    n_panels = sum(active_panels)
    if n_panels == 0:
        return None
    fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 2.9))
    axes = iter(np.atleast_1d(axes))
    rng = np.random.default_rng(0)
    present = set(observed["study"])
    linestyles = ["-", (0, (4, 2)), (0, (1, 1.4))]

    def offsets(n, span=0.14):
        return [0.0] if n == 1 else list(np.linspace(-span, span, n))

    # (a) intimacy magnitudes, one representative given-relationship study per
    # domain (1a stands in for 1a/1b, which share the elicitation design)
    if intimacy_panel_studies:
        ax = next(axes)
        panel_a = [(s, lab) for s, lab in intimacy_panel_studies if s in present]
        offs_a = offsets(len(panel_a))
        jit = 0.08 if len(panel_a) == 1 else 0.05
        for k, ((study, lab), xoff) in enumerate(zip(panel_a, offs_a)):
            df = observed[observed["study"] == study]
            lv = df.groupby(["intimacy_condition", "run_id"], as_index=False)[
                "intimacy"
            ].first()
            spread = lv.groupby("intimacy_condition")["intimacy"].nunique()
            if (spread > 1).any():
                print(
                    f"note: intimacy magnitudes vary across runs ({study}); "
                    "plotting run-level points"
                )
                for i, lvl in enumerate(INTIMACY_LEVELS):
                    sub = lv[lv["intimacy_condition"] == lvl]
                    ax.scatter(
                        i + xoff + rng.uniform(-jit, jit, len(sub)),
                        sub["intimacy"],
                        s=7,
                        color=INTIMACY_COLORS[lvl],
                        alpha=0.3,
                        lw=0,
                    )
            means = lv.groupby("intimacy_condition")["intimacy"].mean()[
                INTIMACY_LEVELS
            ]
            ax.plot(
                np.arange(4) + xoff,
                means,
                color=MEAN_COLOR,
                lw=1.8,
                ls=linestyles[k % len(linestyles)],
                zorder=5,
                label=f"Study {lab}" if len(panel_a) > 1 else None,
            )
            for i, lvl in enumerate(INTIMACY_LEVELS):
                style = (
                    dict(color=INTIMACY_COLORS[lvl], edgecolor="black", lw=0.5)
                    if k == 0
                    else dict(facecolor="white", edgecolor=INTIMACY_COLORS[lvl], lw=1.4)
                )
                ax.scatter(i + xoff, means[lvl], s=48, zorder=6, **style)
                if len(panel_a) == 1:
                    ax.annotate(
                        f"{means[lvl]:g}",
                        (i, means[lvl]),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=8,
                    )
        if len(panel_a) > 1:
            ax.legend(loc="upper left", fontsize=7.5, handlelength=1.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(
            [INTIMACY_LABELS[lvl].replace(" ", "\n") for lvl in INTIMACY_LEVELS],
            fontsize=7.5,
        )
        ax.set_xlabel("Relationship descriptor")
        ax.set_ylabel("LM-rated intimacy $I$")
        ax.set_xlim(-0.5, 3.5)
        ax.set_title("Studies " + "/".join(lab for _, lab in panel_a))
        panel_label(ax, "a")

    # (b) desire by condition, the given-desire studies overlaid with x offsets
    if desire_panel_studies:
        ax = next(axes)
        xpos = {"low": 0, "high": 1}
        panel_b = [(s, lab) for s, lab in desire_panel_studies if s in present]
        offs_b = offsets(len(panel_b))
        specs = [
            (study, xoff, k == 0, linestyles[k % len(linestyles)])
            for k, ((study, _), xoff) in enumerate(zip(panel_b, offs_b))
        ]
        for study, xoff, filled, ls in specs:
            df = observed[observed["study"] == study]
            # one point per scenario: its mean desire in each condition, so the
            # faint lines connect the actual plotted points.
            wide = (
                df.groupby(["scenario", "desire_condition"])["desire"]
                .mean()
                .unstack("desire_condition")[["low", "high"]]
            )
            lo, hi = wide["low"].to_numpy(), wide["high"].to_numpy()
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
                ax.scatter(
                    xpos[cond] + xoff, wide[cond].mean(), s=48, zorder=6, **style
                )
        if len(panel_b) > 1:
            ax.legend(loc="lower right", fontsize=7.5, handlelength=1.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low", "High"])
        ax.set_xlabel("Desire condition")
        ax.set_ylabel("LM-rated desire $d$")
        ax.set_xlim(-0.6, 1.6)
        ax.set_title("Studies " + "/".join(lab for _, lab in panel_b))
        panel_label(ax, "b" if intimacy_panel_studies else "a")

    for ax in fig.axes:
        ax.set_ylim(-0.06, 1.12)
        ax.set_yticks([0, 0.5, 1])
    fig.tight_layout()
    return savefig(fig, figname, **SAVE_KW)


# ---------------------------------------------------------------- figure 3


def fig_observed_scatter(
    observed,
    study="food_inv_desire",
    figname="si_lm_observed_scatter",
    show_effort_condition=True,
):
    """All observed actions in the (risk, effort) plane for one study's
    elicitation (Study 1a by default; the nonfood analog uses Study 3a's), one
    point per scenario x action x effort condition; vertical grey segments
    connect the two effort conditions of the same scenario x action."""
    df = observed[observed["study"] == study]
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    rng = np.random.default_rng(0)
    for pos in (0.5,):
        ax.axhline(pos, color="#DDDDDD", ls=(0, (4, 3)), lw=1.1, zorder=0)
        ax.axvline(pos, color="#DDDDDD", ls=(0, (4, 3)), lw=1.1, zorder=0)

    if show_effort_condition:
        agg = df.groupby(
            ["scenario", "action", "effort_condition"], as_index=False
        ).agg(risk=("risk", "mean"), effort=("effort", "mean"))
        wide = agg.pivot_table(
            index=["scenario", "action"],
            columns="effort_condition",
            values=["risk", "effort"],
        )
        for (scenario, action), row in wide.iterrows():
            dx = rng.uniform(-0.012, 0.012)
            dy = rng.uniform(-0.008, 0.008)
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
    else:
        agg = df.groupby(["scenario", "action"], as_index=False).agg(
            risk=("risk", "mean"), effort=("effort", "mean")
        )
        for row in agg.itertuples(index=False):
            ax.scatter(
                row.risk + rng.uniform(-0.012, 0.012),
                row.effort + rng.uniform(-0.008, 0.008),
                s=34,
                facecolor=ACTION_COLORS[row.action],
                edgecolor=ACTION_COLORS[row.action],
                lw=1.1,
                zorder=4,
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
        for a in OBSERVED_ACTIONS
    ]
    if show_effort_condition:
        handles += [
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
    return savefig(fig, figname, **SAVE_KW)


# ---------------------------------------------------------------- figure 4


def fig_run_spread(figname="si_lm_run_spread"):
    """Run-to-run spread of the model's predicted belief updates (Study 1a,
    full model, out-of-sample leave-one-scenario-out CV predictions). Each
    elicitation run is one simulated observer, so the per-run updates within a
    cell are the components of the mixture likelihood; the fitted response noise
    sigma gives the scale to read the spread against."""
    out_dir = get_project_root() / "model" / "outputs" / "food_inv_desire"
    preds_path = out_dir / "cv_preds_summary.json"
    if not preds_path.exists():
        print(
            "skipping run-spread figure: cv_preds_summary.json not found "
            "(run model/cv/cv_food_inv_desire.py first)"
        )
        return None
    with open(preds_path) as f:
        rows = [r for r in json.load(f) if r["model"] == "full"]
    if "delta_desire_runs" not in rows[0]:
        print(
            "skipping run-spread figure: cv_preds_summary.json lacks per-run "
            "deltas (re-run model/cv/cv_food_inv_desire.py)"
        )
        return None
    with open(out_dir / "fit_results.json") as f:
        fits = json.load(f)
    sigma = float(next(v for v in fits if v["model"] == "full")["param_sigma"])

    means = np.array([r["delta_desire"] for r in rows])
    runs = np.array([r["delta_desire_runs"] for r in rows])  # (cells, K)
    sds = runs.std(axis=1)
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
    return savefig(fig, figname, **SAVE_KW)


# ---------------------------------------------------------------- figure 5


def fig_choice_set_sizes(runs_by_study, figname="si_lm_choice_set_sizes"):
    """Distribution of the number of LM-generated alternatives in each scored
    choice set (one set per cell x run), per study with elicitation data.
    Documents the "small, focused set" the generation prompt asks for — and
    makes visible that a fraction of Study 1a's sets contain no alternatives
    at all."""
    studies = list(runs_by_study)
    fig, axes = plt.subplots(
        1, len(studies), figsize=(1.75 * len(studies), 1.7), sharex=True, sharey=True
    )
    xs = np.arange(0, 8)
    for ax, study in zip(np.atleast_1d(axes), studies):
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
    np.atleast_1d(axes)[0].set_ylabel("% of choice sets")
    fig.supxlabel("Number of LM-generated alternatives in the scored set", fontsize=9)
    fig.tight_layout()
    return savefig(fig, figname, **SAVE_KW)


# ---------------------------------------------------------------- figure 6


def fig_mixture_check(figname="si_lm_mixture_check"):
    """Predictive check of the simulated-observer mixture (Study 1a, full
    model, out-of-sample LOSO CV predictions): for six cells spanning the range
    of predicted updates, the K-component predictive density (1/K) sum_k N(u;
    delta_k, sigma^2) is overlaid on the distribution of participants' actual
    belief updates in that cell. Ticks at the bottom mark the K per-run delta_k."""
    out_dir = get_project_root() / "model" / "outputs" / "food_inv_desire"
    preds_path = out_dir / "cv_preds_summary.json"
    if not preds_path.exists():
        print("skipping mixture-check figure: cv_preds_summary.json not found")
        return None
    with open(preds_path) as f:
        rows = [r for r in json.load(f) if r["model"] == "full"]
    if "delta_desire_runs" not in rows[0]:
        print(
            "skipping mixture-check figure: no per-run deltas in cv_preds_summary.json"
        )
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

    rows = sorted(rows, key=lambda r: r["delta_desire"])
    picks = [rows[int(q * (len(rows) - 1))] for q in (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)]

    u = np.linspace(-1, 1, 401)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), sharex=True, sharey=True)
    for ax, r in zip(axes.ravel(), picks):
        action_condition = OBSERVED_ACTIONS[r["action"]]
        h = wide[
            (wide["scenario_label"] == r["scenario_label"])
            & (wide["action_condition"] == action_condition)
            & (wide["effort_condition"] == r["effort_condition"])
            & (wide["intimacy_condition"] == r["intimacy_condition"])
        ]["update"].to_numpy()
        deltas = np.asarray(r["delta_desire_runs"])
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
            f"{r['scenario_label']} - {ACTION_LABELS[action_condition].lower()}\n"
            f"{r['effort_condition']} effort, {INTIMACY_LABELS[r['intimacy_condition']].lower()}"
            f"  (n = {len(h)})",
            fontsize=7.5,
        )
        ax.set_xlim(-1, 1)
    for ax in axes[-1]:
        ax.set_xlabel("Belief update")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    fig.tight_layout()
    return savefig(fig, figname, **SAVE_KW)


def main():
    apply_style()
    runs_by_study = {}
    frames = []
    for study in STUDIES:
        runs = load_runs(study)
        if runs is None:
            print(f"{study}: no lm_runs.jsonl yet — skipped")
            continue
        runs_by_study[study] = runs
        df = extract_observed(runs)
        df["study"] = study
        frames.append(df)
        print(f"{study}: {len(df)} observed-action rows")
    observed = pd.concat(frames, ignore_index=True)
    studies = [s for s in FOOD_STUDIES if s in runs_by_study]
    study3 = [s for s in STUDY3_STUDIES if s in runs_by_study]
    joint_de = [s for s in JOINT_DE_COMPARISON if s in runs_by_study]
    joint_ie = [s for s in JOINT_IE_COMPARISON if s in runs_by_study]
    figures = [
        fig_feature_structure(
            observed, studies, figname="si_lm_feature_structure_1a_1b_2a_2b"
        ),
        fig_manipulation_checks(
            observed,
            FOOD_INTIMACY_PANEL_STUDIES,
            FOOD_DESIRE_PANEL_STUDIES,
            figname="si_lm_manipulation_checks_1a_1b_2a_2b",
        ),
        fig_observed_scatter(
            observed,
            study="food_inv_desire",
            figname="si_lm_observed_scatter_1a",
        ),
        fig_run_spread(figname="si_lm_run_spread_1a"),
        fig_choice_set_sizes(
            {s: runs_by_study[s] for s in studies},
            figname="si_lm_choice_set_sizes_1a_1b_2a_2b",
        ),
        fig_mixture_check(figname="si_lm_mixture_check_1a"),
    ]
    if study3:
        figures.extend(
            [
                fig_feature_structure(
                    observed,
                    study3,
                    figname="si_lm_feature_structure_3a_3b",
                    show_effort_condition=False,
                ),
                fig_manipulation_checks(
                    observed,
                    STUDY3_INTIMACY_PANEL_STUDIES,
                    STUDY3_DESIRE_PANEL_STUDIES,
                    figname="si_lm_manipulation_checks_3a_3b",
                ),
                fig_choice_set_sizes(
                    {s: runs_by_study[s] for s in study3},
                    figname="si_lm_choice_set_sizes_3a_3b",
                ),
            ]
        )
    if "nonfood_inv_joint_de" in runs_by_study:
        figures.append(
            fig_observed_scatter(
                observed,
                study="nonfood_inv_joint_de",
                figname="si_lm_observed_scatter_3a",
                show_effort_condition=False,
            )
        )
    if len(joint_de) == 2:
        figures.extend(
            [
                fig_feature_structure(
                    observed,
                    joint_de,
                    figname="si_lm_feature_structure_1b_3a",
                    show_effort_condition=False,
                ),
                fig_manipulation_checks(
                    observed,
                    JOINT_DE_INTIMACY_PANEL_STUDIES,
                    [],
                    figname="si_lm_manipulation_checks_1b_3a",
                ),
                fig_choice_set_sizes(
                    {s: runs_by_study[s] for s in joint_de},
                    figname="si_lm_choice_set_sizes_1b_3a",
                ),
            ]
        )
    if len(joint_ie) == 2:
        figures.extend(
            [
                fig_feature_structure(
                    observed,
                    joint_ie,
                    figname="si_lm_feature_structure_2b_3b",
                    show_effort_condition=False,
                ),
                fig_manipulation_checks(
                    observed,
                    [],
                    JOINT_IE_DESIRE_PANEL_STUDIES,
                    figname="si_lm_manipulation_checks_2b_3b",
                ),
                fig_choice_set_sizes(
                    {s: runs_by_study[s] for s in joint_ie},
                    figname="si_lm_choice_set_sizes_2b_3b",
                ),
            ]
        )
    for path in figures:
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
