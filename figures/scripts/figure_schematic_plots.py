#!/usr/bin/env python3
"""Schematic-figure plot panels for Study 1a (food desire), the burrito example.

Reads the LM scores cached by `figure_lm_calls.py` and the fitted full-model
weights, recomputes the actor utility / choice / desire-posterior standalone, and
renders three vector panels for the Illustrator figure:

  1. per-action score bar charts, stacked (x = food goal / effort / risk) -> bars_all.svg
  2. total utility vs. food desire, one line per action, with a vertical  -> utility_vs_desire.svg
     dotted line at the desire that maximizes P(a_obs)
  3. posterior over food desire P(d | a_obs)                              -> posterior_desire.svg

The math mirrors `model/utility.py:get_utility_full_padded_desire` and
`model/observers.py:observer_desire_full` exactly (actor alpha = 1, uniform action
prior); we recompute rather than reuse those because the observer code is indexed
into precomputed per-scenario tables and the burrito is not a data scenario.

No API calls — fully reproducible from `figures/figure_data/figure_scores.json`.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from plot_style import (  # noqa: E402
    ACTION_COLORS as OBS_COLORS,
    INTIMACY_COLORS,
    INTIMACY_LABELS,
    INTIMACY_LEVELS,
    OTHER_ACTION_COLOR,
    apply_style,
)
from utils import get_project_root  # noqa: E402

# Manuscript aesthetic (Arial Nova, large-type Illustrator sizes, editable SVG
# text) — shared with the SI figures via plot_style.py.
apply_style("schematic")

FEATURE_ORDER = ["g", "effort", "risk"]
FEATURE_LABELS = {"g": "Goal", "effort": "Effort", "risk": "Risk"}

# The shared observed-action palette, mapped onto the burrito example's four
# actions: a_obs is the low-risk share (cut the burrito in half), a_1 the
# no-share, a_2 the high-risk share (take turns biting), and a_3 an
# extra alternative beyond the three observed.
ACTION_COLORS = {
    "a_obs": OBS_COLORS["low_risk_share"],
    "a_1": OBS_COLORS["no_share"],
    "a_2": OBS_COLORS["high_risk_share"],
    "a_3": OTHER_ACTION_COLOR,
}
ACTION_LABELS = {
    "a_obs": r"$a_\mathrm{obs}$",
    "a_1": r"$a_1$",
    "a_2": r"$a_2$",
    "a_3": r"$a_3$",
}

# Illustrative utility weights for the schematic — hand-tuned for legibility so the
# four utility lines separate cleanly, NOT the fitted values in fit_results.json.
# The features (g / risk / effort / intimacy) are still the real LM elicitation;
# only the weights are stylized. Set ILLUSTRATIVE_WEIGHTS = None to use the fit.
ILLUSTRATIVE_WEIGHTS = {"w_v": 12.0, "w_e": 3.0, "w_d": 6.5, "gamma": 1.0}

from plot_style import PANELS_SCHEMATIC  # noqa: E402

OUT_DIR = PANELS_SCHEMATIC


def _savefig(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # PDF only: Illustrator-bound figures keep editable text via pdf.fonttype 42,
    # so a parallel SVG was duplicate weight in the repo.
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_inputs():
    scores_path = get_project_root() / "figures" / "figure_data" / "figure_scores.json"
    with open(scores_path) as f:
        rec = json.load(f)
    fits_path = (
        get_project_root()
        / "model"
        / "outputs"
        / "food_inv_desire"
        / "fit_results.json"
    )
    with open(fits_path) as f:
        fits = json.load(f)
    full = next(v for v in fits if v["model"] == "full")
    return rec, full


def compute(rec, full):
    """Standalone actor utility / choice / desire posterior for the burrito example's four actions."""
    keys = [a["key"] for a in rec["actions"]]
    g = np.array([a["g"] for a in rec["actions"]])
    effort = np.array([a["effort"] for a in rec["actions"]])
    risk = np.array([a["risk"] for a in rec["actions"]])
    I = float(rec["intimacy"])

    if ILLUSTRATIVE_WEIGHTS is not None:
        w_v = ILLUSTRATIVE_WEIGHTS["w_v"]
        w_d = ILLUSTRATIVE_WEIGHTS["w_d"]
        w_e = ILLUSTRATIVE_WEIGHTS["w_e"]
        gamma = ILLUSTRATIVE_WEIGHTS["gamma"]
    else:
        w_v = full["param_w_v"]
        w_d = full["param_w_d"]
        w_e = full["param_w_e"]
        gamma = full["param_gamma"]
    alpha_obs = full["alpha_observer"]

    grid = np.arange(0, 1.01, 0.01)  # DesireLevels, 101 bins
    one_minus_I = max(1.0 - I, 1e-8)
    # U(a | d) = w_v*d*g - w_d*risk*(1-I)^gamma - w_e*effort   (actor alpha = 1)
    cost = w_d * risk * (one_minus_I**gamma) + w_e * effort  # (4,)
    U = w_v * np.outer(grid, g) - cost[None, :]  # (101, 4)

    # actor choice: softmax over actions, uniform action prior
    ex = np.exp(U - U.max(axis=1, keepdims=True))
    P_a_given_d = ex / ex.sum(axis=1, keepdims=True)  # (101, 4)
    p_aobs = P_a_given_d[:, 0]  # slot 0 = a_obs

    # posterior over desire: P(d|a_obs) ∝ P(a_obs|d)^alpha_obs * P(d), uniform P(d)
    post = p_aobs**alpha_obs
    post = post / post.sum()  # normalized over the grid
    d_star = float(grid[int(np.argmax(p_aobs))])  # desire maximizing P(a_obs) = mode

    return {
        "keys": keys,
        "g": g,
        "effort": effort,
        "risk": risk,
        "I": I,
        "grid": grid,
        "U": U,
        "p_aobs": p_aobs,
        "post": post,
        "d_star": d_star,
        "alpha_obs": alpha_obs,
    }


def plot_bars(rec, c):
    """Vertically-stacked per-action bar charts (x = food goal, effort, risk)."""
    feats_by_action = {
        a["key"]: {f: a[f] for f in FEATURE_ORDER} for a in rec["actions"]
    }
    x = np.arange(len(FEATURE_ORDER))
    labels = [FEATURE_LABELS[f] for f in FEATURE_ORDER]
    FLOOR = 0.15  # keep a zero-valued feature visible against the thicker axes

    def draw(ax, key, show_title=True, bar_width=0.6):
        vals = [max(feats_by_action[key][f], FLOOR) for f in FEATURE_ORDER]
        ax.bar(
            x, vals, color=ACTION_COLORS[key], width=bar_width
        )  # match the line-plot color
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([])  # no numerical tick labels
        ax.spines["left"].set_visible(False)  # no y axis line
        if show_title:
            ax.set_title(ACTION_LABELS[key])

    # vertically stacked; one row per action, titleless (labelled in the assembled figure)
    n = len(c["keys"])
    fig, axes = plt.subplots(
        n, 1, figsize=(2.0, 0.52 * n), sharex=True, constrained_layout=True
    )
    for ax, key in zip(axes, c["keys"]):
        # narrower figure + wider bars → less gap between Goal/Effort/Risk
        draw(ax, key, show_title=False, bar_width=0.82)
    for ax in axes[:-1]:  # keep the Goal/Effort/Risk labels on the bottom panel only
        ax.tick_params(labelbottom=False)
    _savefig(fig, "bars_all")


def plot_utility(c):
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for i, key in enumerate(c["keys"]):
        ax.plot(
            c["grid"],
            c["U"][:, i],
            color=ACTION_COLORS[key],
            lw=5.5 if key == "a_obs" else 4.0,
            label=ACTION_LABELS[key],
        )
    ax.set_xlim(0, 1)
    ax.axvline(c["d_star"], ls=":", color="#444444", lw=2.0)
    ax.set_xlabel(r"Food desire  $d$")
    ax.set_ylabel("Total utility")
    ax.set_xticks([])  # no numerical tick labels
    ax.set_yticks([])
    _savefig(fig, "utility_vs_desire")


def plot_posterior(c):
    dd = c["grid"][1] - c["grid"][0]
    density = c["post"] / dd  # normalized density over [0,1]
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    # colored with the somewhat_formal intimacy color: this base example is at the
    # somewhat_formal relationship, so it matches its curve in the by-relationship panel
    color = INTIMACY_COLORS["somewhat_formal"]
    ax.fill_between(c["grid"], density, color=color, alpha=0.18)
    ax.plot(c["grid"], density, color=color, lw=2.0)
    ax.axvline(c["d_star"], ls=":", color="#444444", lw=1.4)
    ax.set_xlabel(r"Food desire  $d$")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xticks([])  # no numerical tick labels
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)  # drop the y axis line + label
    _savefig(fig, "posterior_desire")


def load_intimacy_by_relationship():
    """The four relationship levels' LM-elicited intimacy magnitudes, keyed by
    level (max_formal → max_intimate). Read from Study 1a's own elicitation
    (`model/outputs/lm/food_inv_desire/lm_runs.jsonl`) — the same scenario-
    independent, deterministic values the fit in fit_results.json was computed
    under, so the figure and the model agree. (figure_scores.json only caches the
    single relationship the base example uses; the by-relationship panel needs
    all four.)"""
    path = (
        get_project_root()
        / "model"
        / "outputs"
        / "lm"
        / "food_inv_desire"
        / "lm_runs.jsonl"
    )
    vals = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            lvl, iv = r.get("intimacy_condition"), r.get("intimacy")
            if lvl and iv is not None and lvl not in vals:
                vals[lvl] = float(iv)
            if len(vals) == len(INTIMACY_LEVELS):
                break
    return {lvl: vals[lvl] for lvl in INTIMACY_LEVELS}


def plot_posterior_by_relationship(rec, full):
    """Posterior over food desire P(d | a_obs, I) at each of the four relationship
    levels — how relationship intimacy reshapes the desire inference.

    Same actor model as compute() (illustrative weights); only the observed
    low-risk share a_obs is conditioned on. Because a_obs has risk = 0, intimacy
    never enters its own utility — it reshapes the posterior only through the
    competing high-risk share a_2, which becomes attractive as the relationship
    warms. The upshot: at formal relationships, sharing strongly implies high
    desire (posterior climbs toward d = 1); at intimate ones it says much less
    (posterior flattens)."""
    g = np.array([a["g"] for a in rec["actions"]])
    effort = np.array([a["effort"] for a in rec["actions"]])
    risk = np.array([a["risk"] for a in rec["actions"]])
    W = ILLUSTRATIVE_WEIGHTS
    alpha_obs = full["alpha_observer"]
    grid = np.arange(0, 1.01, 0.01)
    dd = grid[1] - grid[0]
    I_by_level = load_intimacy_by_relationship()

    # wide, short axes to match the single-relationship posterior panel; the legend
    # sits outside (right) so it doesn't squish the data area.
    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    means, dens = {}, {}
    for lvl in INTIMACY_LEVELS:
        I = I_by_level[lvl]
        cost = W["w_d"] * risk * (max(1.0 - I, 1e-8) ** W["gamma"]) + W["w_e"] * effort
        U = W["w_v"] * np.outer(grid, g) - cost[None, :]
        ex = np.exp(U - U.max(axis=1, keepdims=True))
        p_aobs = (ex / ex.sum(axis=1, keepdims=True))[:, 0]  # slot 0 = a_obs
        post = p_aobs**alpha_obs
        post = post / post.sum()
        means[lvl] = float((grid * post).sum())
        dens[lvl] = post / dd  # normalized density over [0, 1]
    # the two formal levels (I = 0.0, 0.2) are nearly coincident; a small constant
    # vertical dodge (as in plot_components) separates every level. Dodging *down*
    # by level index preserves the natural plateau order (formal high → intimate
    # low). Purely cosmetic — the panel is illustrative, with no y scale.
    ymax = max(float(d.max()) for d in dens.values())
    eps = 0.05 * ymax
    for i, lvl in enumerate(INTIMACY_LEVELS):
        ax.plot(
            grid,
            dens[lvl] - i * eps,
            color=INTIMACY_COLORS[lvl],
            lw=4.0,
            solid_capstyle="round",
            label=INTIMACY_LABELS[lvl],
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Food desire  $d$")
    ax.set_xticks([])  # no numerical tick labels
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)  # drop the y axis line + label
    # legend outside on the right so it never sits on the (small, square) data area
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=13,
        handlelength=1.4,
        borderaxespad=0.0,
        frameon=False,
    )
    _savefig(fig, "posterior_desire_by_relationship")
    return means


def plot_components(rec):
    """Illustrative component-function plots: each utility term vs. the latent that
    drives it, one curve per observed action (no-share / low-risk / high-risk),
    using ILLUSTRATIVE_WEIGHTS. Coincident lines are dodged slightly so all three
    stay visible (only the action whose feature is non-zero actually varies)."""
    W = ILLUSTRATIVE_WEIGHTS
    feats = {a["key"]: a for a in rec["actions"]}
    grid = np.linspace(0.0, 1.0, 101)
    ACTS = ["a_1", "a_obs", "a_2"]  # no-share, low-risk share, high-risk share
    # a second channel (line style) so coincident curves stay distinct at any zoom
    LINESTYLES = {"a_1": ":", "a_obs": "-", "a_2": (0, (6, 3))}
    LABELS = {"a_1": "no-share", "a_obs": "low-risk", "a_2": "high-risk"}

    def panel(y_of, xlabel, ylabel, fname):
        ys = {k: np.asarray(y_of(k), dtype=float) for k in ACTS}
        ymax = max(1e-9, max(float(np.max(y)) for y in ys.values()))
        eps = 0.07 * ymax  # visibility dodge so coincident lines don't merge
        fig, ax = plt.subplots(figsize=(3.0, 2.8))
        for i, k in enumerate(ACTS):
            ax.plot(
                grid,
                ys[k] + i * eps,
                color=ACTION_COLORS[k],
                lw=4.0,
                ls=LINESTYLES[k],
                solid_capstyle="round",
                dash_capstyle="round",
            )
        ax.set_xlim(0, 1)
        y0, y1 = -0.06 * ymax, 1.12 * ymax + 2 * eps
        # direct end labels in the right margin, spread vertically so they don't collide
        gap = 0.13 * (y1 - y0)
        placed, prev = [], -1e18
        for yend, k in sorted(
            (float(ys[k][-1] + i * eps), k) for i, k in enumerate(ACTS)
        ):
            yy = max(yend, prev + gap)
            prev = yy
            placed.append((k, yy))
        y1 = max(y1, placed[-1][1] + 0.06 * (y1 - y0))  # headroom for the top label
        ax.set_ylim(y0, y1)
        for k, yy in placed:
            ax.annotate(
                LABELS[k],
                xy=(1.0, yy),
                xytext=(8, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color=ACTION_COLORS[k],
                fontsize=15,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)  # square plotting area
        _savefig(fig, fname)

    # reward = w_v · d · g(a), vs desire
    panel(
        lambda k: W["w_v"] * grid * feats[k]["g"],
        r"Food desire  $d$",
        "Reward",
        "reward_vs_desire",
    )
    # effort cost = w_e · effort(a), vs physical world state (low-risk effort swept 0→1)
    panel(
        lambda k: (
            W["w_e"]
            * (grid if k == "a_obs" else feats[k]["effort"] * np.ones_like(grid))
        ),
        "Effort of low-risk share",
        "Effort cost",
        "effort_vs_worldstate",
    )
    # discomfort = w_d · risk(a) · (1−I)^γ, vs intimacy
    panel(
        lambda k: W["w_d"] * feats[k]["risk"] * (1.0 - grid) ** W["gamma"],
        r"Relationship intimacy  $I$",
        "Discomfort cost",
        "discomfort_vs_intimacy",
    )


def main():
    rec, full = load_inputs()
    c = compute(rec, full)

    plot_bars(rec, c)
    plot_utility(c)
    plot_posterior(c)
    rel_means = plot_posterior_by_relationship(rec, full)
    plot_components(rec)

    # sanity-check print
    print(f"intimacy I = {c['I']:.3f}   alpha_obs = {c['alpha_obs']:.3f}")
    print(f"d* (argmax P(a_obs) = posterior mode) = {c['d_star']:.2f}")
    i_star = int(round(c["d_star"] * 100))
    order = np.argsort(-c["U"][i_star])
    print("utility ordering at d*:")
    for j in order:
        key = c["keys"][j]
        print(
            f"  {key:6s}  U={c['U'][i_star, j]:7.3f}  "
            f"P(a|d*)={np.exp(c['U'][i_star] - c['U'][i_star].max())[j] / np.exp(c['U'][i_star] - c['U'][i_star].max()).sum():.3f}"
        )
    print("posterior mean(d) by relationship (observed a_obs):")
    for lvl in INTIMACY_LEVELS:
        print(f"  {lvl:20s} mean(d)={rel_means[lvl]:.3f}")
    print(f"\nWrote panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
