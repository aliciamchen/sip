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
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root  # noqa: E402


def _register_arial_nova():
    """Register Arial Nova from the user's font dirs so matplotlib can use it even
    when its font cache hasn't indexed it. Returns True if the regular face was
    registered (so math can also use Arial Nova; otherwise we keep a sans fallback).
    """
    variants = (
        "ArialNova.ttf",
        "ArialNova-Bold.ttf",
        "ArialNova-Italic.ttf",
        "ArialNova-BoldItalic.ttf",
    )
    dirs = (
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
    )
    found = False
    for d in dirs:
        for v in variants:
            p = d / v
            if p.exists():
                fm.fontManager.addfont(str(p))
                if v == "ArialNova.ttf":
                    found = True
    return found


_HAS_ARIAL_NOVA = _register_arial_nova()

# ----------------------------------------------------------------------------- style
# Manuscript aesthetic (matches the R analysis font Arial Nova), editable SVG text.
_math = (
    {
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial Nova",
        "mathtext.it": "Arial Nova:italic",
        "mathtext.bf": "Arial Nova:bold",
    }
    if _HAS_ARIAL_NOVA
    else {"mathtext.fontset": "dejavusans", "mathtext.default": "it"}
)
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 2.2,  # thicker axis lines (easier to see in Illustrator)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Nova", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        # bar (action) plots set explicit x ticks → these show; the continuous plots
        # use empty tick lists → stay tickless. Tick width matches the axis line width.
        "xtick.major.size": 5.5,
        "xtick.major.width": 2.2,
        "ytick.major.size": 0,  # no y ticks anywhere
        "ytick.major.width": 2.2,
        "axes.labelpad": 12,  # gap between axis labels and the (tickless) spines
        "legend.fontsize": 17,
        "svg.fonttype": "none",  # keep text editable in Illustrator
        "pdf.fonttype": 42,
        **_math,
    }
)

FEATURE_ORDER = ["g", "effort", "risk"]
FEATURE_LABELS = {"g": "Goal", "effort": "Effort", "risk": "Risk"}

# Seaborn colorblind palette, mapped to the four actions (shared by bars + lines).
_CB = sns.color_palette("colorblind")
ACTION_COLORS = {"a_obs": _CB[0], "a_1": _CB[1], "a_2": _CB[2], "a_3": _CB[3]}
ACTION_LABELS = {
    "a_obs": r"$a_\mathrm{obs}$",
    "a_1": r"$a_1$",
    "a_2": r"$a_2$",
    "a_3": r"$a_3$",
}

OUT_DIR = get_project_root() / "figures" / "schematic_panels"


def _savefig(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
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
    """Standalone actor utility / choice / desire posterior for the 4-action set."""
    keys = [a["key"] for a in rec["actions"]]
    g = np.array([a["g"] for a in rec["actions"]])
    effort = np.array([a["effort"] for a in rec["actions"]])
    risk = np.array([a["risk"] for a in rec["actions"]])
    I = float(rec["intimacy"])

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
    ax.annotate(
        r"$\arg\max_d\, P(a_\mathrm{obs})$" + "\n" + rf"$d = {c['d_star']:.2f}$",
        xy=(c["d_star"], ax.get_ylim()[1]),  # top of the d* line
        xytext=(12, 0),  # push into the right margin, clear of the lines
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=15,
        color="#444444",
    )
    ax.set_xlabel(r"Food desire  $d$")
    ax.set_ylabel("Total utility")
    ax.set_xticks([])  # no numerical tick labels
    ax.set_yticks([])
    ax.legend(
        frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.36), handlelength=1.4
    )
    _savefig(fig, "utility_vs_desire")


def plot_posterior(c):
    dd = c["grid"][1] - c["grid"][0]
    density = c["post"] / dd  # normalized density over [0,1]
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    ax.fill_between(c["grid"], density, color="tab:blue", alpha=0.18)
    ax.plot(c["grid"], density, color="tab:blue", lw=2.0)
    ax.axvline(c["d_star"], ls=":", color="#444444", lw=1.4)
    ax.set_xlabel(r"Food desire  $d$")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xticks([])  # no numerical tick labels
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)  # drop the y axis line + label
    _savefig(fig, "posterior_desire")


def main():
    rec, full = load_inputs()
    c = compute(rec, full)

    plot_bars(rec, c)
    plot_utility(c)
    plot_posterior(c)

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
    print(f"\nWrote panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
