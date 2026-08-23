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

Usage:
    uv run python figures/scripts/plot_si_validation.py
"""

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
    OBSERVED_ACTIONS,
    DESIRE_COLORS,
    EFFORT_COLORS,
    INTIMACY_COLORS,
    INTIMACY_LABELS,
    INTIMACY_LEVELS,
    SI_LARGE_RC,
    STUDY_LABELS,
    apply_style,
    panel_label,
    savefig,
)

# `study` aliased: this module reuses the bare name as its loop variable.
from study_registry import SLUGS, slugs_given  # noqa: E402
from study_registry import study as _study  # noqa: E402
from utils import get_project_root  # noqa: E402

STUDIES = list(SLUGS)  # the six active studies, in paper order
# The manipulation-check panels overlay the studies whose elicitation carries
# each given-magnitude scalar, split by domain so food and nonfood are never
# mixed in one panel: a food row (intimacy 1a/1b; desire 2a/2b) over a nonfood
# row (intimacy 3a; desire 3b). Within a domain the descriptors are shared
# (1a == 1b, 2a == 2b), so those lines coincide -- a built-in consistency check.
# Derived from the study registry rather than restating the study lists.


def _panel_studies(condition, domain):
    """(slug, short label) pairs for one manipulation-check panel: the studies
    given this condition, within one stimulus domain, in paper order."""
    return [
        (slug, _study(slug).short_label)
        for slug in slugs_given(condition)
        if _study(slug).domain == domain
    ]


# Rows of the 2x2 manipulation-check figure: (domain label, intimacy studies,
# desire studies).
MANIPULATION_ROWS = [
    (
        domain.capitalize(),
        _panel_studies("intimacy_condition", domain),
        _panel_studies("desire_condition", domain),
    )
    for domain in ("food", "nonfood")
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


_PANEL_LINESTYLES = ["-", (0, (4, 2)), (0, (1, 1.4))]


def _panel_offsets(n, span=0.14):
    return [0.0] if n == 1 else list(np.linspace(-span, span, n))


def _panel_title(domain, studies):
    labs = [lab for _, lab in studies]
    head = "Study " if len(labs) == 1 else "Studies "
    return f"{domain}: {head}" + "/".join(labs)


def _draw_intimacy_panel(ax, observed, panel_studies, present, rng, title, letter):
    """LM-rated intimacy magnitude by relationship descriptor, overlaying each
    given-relationship study that has elicitation data (offset on x when >1)."""
    panel = [(s, lab) for s, lab in panel_studies if s in present]
    if not panel:
        ax.axis("off")
        return
    offs = _panel_offsets(len(panel))
    jit = 0.08 if len(panel) == 1 else 0.05
    for k, ((study, lab), xoff) in enumerate(zip(panel, offs)):
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
        means = lv.groupby("intimacy_condition")["intimacy"].mean()[INTIMACY_LEVELS]
        ax.plot(
            np.arange(4) + xoff,
            means,
            color=MEAN_COLOR,
            lw=1.8,
            ls=_PANEL_LINESTYLES[k % len(_PANEL_LINESTYLES)],
            zorder=5,
            label=f"Study {lab}" if len(panel) > 1 else None,
        )
        for i, lvl in enumerate(INTIMACY_LEVELS):
            style = (
                dict(color=INTIMACY_COLORS[lvl], edgecolor="black", lw=0.5)
                if k == 0
                else dict(facecolor="white", edgecolor=INTIMACY_COLORS[lvl], lw=1.4)
            )
            ax.scatter(i + xoff, means[lvl], s=48, zorder=6, **style)
            if len(panel) == 1:
                ax.annotate(
                    f"{means[lvl]:g}",
                    (i, means[lvl]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=10,
                )
    if len(panel) > 1:
        ax.legend(loc="upper left", handlelength=1.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        [INTIMACY_LABELS[lvl].replace(" ", "\n") for lvl in INTIMACY_LEVELS],
        fontsize=8.5,
    )
    ax.set_xlabel("Relationship descriptor")
    ax.set_ylabel("LM-rated intimacy $I$")
    ax.set_xlim(-0.5, 3.5)
    ax.set_title(title)
    panel_label(ax, letter)


def _draw_desire_panel(ax, observed, panel_studies, present, rng, title, letter):
    """LM-rated desire by desire condition, overlaying each given-desire study
    that has elicitation data (offset on x when >1)."""
    panel = [(s, lab) for s, lab in panel_studies if s in present]
    if not panel:
        ax.axis("off")
        return
    xpos = {"low": 0, "high": 1}
    offs = _panel_offsets(len(panel))
    specs = [
        (study, xoff, k == 0, _PANEL_LINESTYLES[k % len(_PANEL_LINESTYLES)])
        for k, ((study, _), xoff) in enumerate(zip(panel, offs))
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
            ax.scatter(xpos[cond] + xoff, wide[cond].mean(), s=48, zorder=6, **style)
    if len(panel) > 1:
        ax.legend(loc="lower right", handlelength=1.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Low", "High"])
    ax.set_xlabel("Desire condition")
    ax.set_ylabel("LM-rated desire $d$")
    ax.set_xlim(-0.6, 1.6)
    ax.set_title(title)
    panel_label(ax, letter)


def fig_manipulation_checks(observed, rows, figname="si_lm_manipulation_checks"):
    """2x2 manipulation-check grid, one row per domain so food and nonfood are
    never mixed in a panel. Left column (a, c): LM-rated intimacy by relationship
    descriptor for the given-relationship studies; right column (b, d): LM-rated
    desire by desire condition for the given-desire studies. ``rows`` is a list of
    (domain label, intimacy studies, desire studies); studies within a panel are
    overlaid with x-offsets."""
    present = set(observed["study"])
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(
        len(rows), 2, figsize=(6.2, 2.7 * len(rows)), squeeze=False
    )
    letters = iter("abcdefghijkl")
    for r, (domain, intimacy_studies, desire_studies) in enumerate(rows):
        _draw_intimacy_panel(
            axes[r, 0],
            observed,
            intimacy_studies,
            present,
            rng,
            title=_panel_title(domain, intimacy_studies),
            letter=next(letters),
        )
        _draw_desire_panel(
            axes[r, 1],
            observed,
            desire_studies,
            present,
            rng,
            title=_panel_title(domain, desire_studies),
            letter=next(letters),
        )
    for ax in fig.axes:
        ax.set_ylim(-0.06, 1.12)
        ax.set_yticks([0, 0.5, 1])
    fig.tight_layout()
    return savefig(fig, figname, **SAVE_KW)


# ---------------------------------------------------------------- figure 3


def _draw_observed_scatter(ax, df, rng, show_effort_condition):
    """Draw one study's observed actions in the (risk, effort) plane onto ax.
    With show_effort_condition, each scenario x action contributes a low/high
    pair joined by a gray segment (low = open marker, high = filled); otherwise
    one filled marker per scenario x action."""
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
                    s=30,
                    facecolor=ACTION_COLORS[action] if filled else "white",
                    edgecolor=ACTION_COLORS[action],
                    lw=1.0,
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
                s=30,
                facecolor=ACTION_COLORS[row.action],
                edgecolor=ACTION_COLORS[row.action],
                lw=1.0,
                zorder=4,
                alpha=0.9,
            )


def fig_observed_scatter(
    observed,
    studies,
    figname="si_lm_observed_scatter",
    show_effort_condition=True,
):
    """The three action conditions in the (risk, effort) plane, one panel per
    study on a 3x2 grid. Each point is one scenario x action x effort condition
    (features averaged over runs and the study's other conditions), with gray
    segments joining the two effort conditions of the same scenario x action."""
    ncols = 2
    nrows = (len(studies) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4, 3.0 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    rng = np.random.default_rng(0)
    axflat = axes.ravel()
    drew_effort = False
    for ax, study in zip(axflat, studies):
        df = observed[observed["study"] == study]
        show = show_effort_condition and {"low", "high"} <= set(df["effort_condition"])
        drew_effort = drew_effort or show
        for pos in (0.5,):
            ax.axhline(pos, color="#DDDDDD", ls=(0, (4, 3)), lw=1.1, zorder=0)
            ax.axvline(pos, color="#DDDDDD", ls=(0, (4, 3)), lw=1.1, zorder=0)
        _draw_observed_scatter(ax, df, rng, show)
        ax.set_title(STUDY_LABELS[study])
        ax.set_box_aspect(1)
    for ax in axflat[len(studies) :]:
        ax.axis("off")

    action_handles = [
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
    effort_handles = []
    if drew_effort:
        effort_handles = [
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
    for ax in axes[:, 0]:
        ax.set_ylabel("Effort")
    for ax in axes[-1, :]:
        ax.set_xlabel("Risk")
    for ax in axflat:
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    # One centered legend below the panels (off the data): two stacked columns side
    # by side -- the action conditions on the left, the effort conditions on the
    # right. matplotlib fills legend cells column-major, so concatenating (actions
    # then efforts) puts each group in its own column.
    if effort_handles:
        legend_handles = action_handles + effort_handles
        ncol = 2
    else:
        legend_handles = action_handles
        ncol = 1
    fig.tight_layout(rect=[0, 0.9 / (3.0 * nrows), 1, 1])
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        columnspacing=3.0,
        handletextpad=0.5,
        labelspacing=0.5,
    )
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
    # All six studies in one consolidated figure each (roster order).
    studies = [s for s in STUDIES if s in runs_by_study]
    # The multi-panel LM-elicitation figures render at reduced SI widths, so use
    # the larger rc profile. The feature-structure grid keeps the base "si" sizes
    # (dense 6x3, already readable).
    figures = [
        fig_feature_structure(observed, studies, figname="si_lm_feature_structure_all"),
    ]
    with plt.rc_context(SI_LARGE_RC):
        figures.append(
            fig_observed_scatter(
                observed, studies, figname="si_lm_observed_scatter_all"
            )
        )
    # manipulation-checks reads slightly large in the SI_LARGE profile; nudge down.
    manip_rc = {
        **SI_LARGE_RC,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
    }
    with plt.rc_context(manip_rc):
        figures.append(
            fig_manipulation_checks(
                observed, MANIPULATION_ROWS, figname="si_lm_manipulation_checks_all"
            )
        )
    for path in figures:
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
