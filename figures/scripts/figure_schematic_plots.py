#!/usr/bin/env python3
"""Schematic-figure plot panels for Study 1a (food desire), the burrito example.

Reads the LM scores cached in `figures/figure_data/figure_scores.json` and the
fitted full-model weights, recomputes the actor utility / choice /
desire-posterior standalone, and renders the vector panels for the Illustrator
figures:

  bars_all                          per-action feature bars (goal / effort / risk)
  utility_vs_desire                 total utility vs. desire over the example's four
                                    actions, marked at the desire maximizing P(a_obs)
  posterior_desire                  posterior over desire, P(d | a_obs)
  posterior_desire_by_relationship  the same posterior at each of the four
                                    relationship levels
  reward_vs_desire                  each utility term against the latent that drives
  effort_vs_worldstate              it, one curve per observed action
  discomfort_vs_intimacy

  best_action_by_latents_utensils_far    which action maximizes total utility over
  best_action_by_latents_utensils_near   (desire x intimacy)
  posterior_by_latents_utensils_far      the HDR_MASS posterior region over those
  posterior_by_latents_utensils_near     same two latents, one per action that
                                         could have been observed

The last four come as a far/near pair because the physical world state is binary and
enters only through the low-risk share's effort; see plot_latent_space.

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
# The features (g / risk / effort / intimacy) are the real LM elicitation, with one
# exception: figure_scores.json caches only the high-effort world state, so the
# "utensils near" panels in plot_latent_space stipulate a low-risk effort of 0 rather
# than reading the low-effort condition's elicited value. That world-state contrast
# is therefore stylized too. Set ILLUSTRATIVE_WEIGHTS = None to use the fit.
ILLUSTRATIVE_WEIGHTS = {"w_v": 12.0, "w_e": 3.0, "w_d": 6.5, "gamma": 1.0}

# The three observed actions the experiments present, in the order the panels draw
# them, with the second encoding channel (line style) that keeps coincident curves
# distinct at any zoom, and the short in-panel labels. Shared by every panel that
# draws all three, so the row reads as one system. This order is cosmetic — nothing
# should depend on it for meaning (the forward panel resolves utility ties explicitly
# rather than inheriting this list's order).
OBSERVED_ACTS = ["a_1", "a_obs", "a_2"]  # no-share, low-risk share, high-risk share
ACT_LINESTYLES = {"a_1": ":", "a_obs": "-", "a_2": (0, (6, 3))}
ACT_SHORT_LABELS = {"a_1": "no-share", "a_obs": "low-risk", "a_2": "high-risk"}

# Posterior mass enclosed by the credible region the inverse latent-space panel draws.
HDR_MASS = 0.50


def resolve_weights(full):
    """The utility weights the panels draw with: the illustrative set when one is
    configured, else the fitted full-model values. Every panel goes through here, so
    the `ILLUSTRATIVE_WEIGHTS = None` toggle documented above actually works."""
    if ILLUSTRATIVE_WEIGHTS is not None:
        return dict(ILLUSTRATIVE_WEIGHTS)
    return {k: full[f"param_{k}"] for k in ("w_v", "w_e", "w_d", "gamma")}


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

    W = resolve_weights(full)
    w_v, w_d, w_e, gamma = W["w_v"], W["w_d"], W["w_e"], W["gamma"]
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

    # vertically stacked; one row per action, titleless (labeled in the assembled figure)
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
    W = resolve_weights(full)
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


def plot_latent_space(rec, full):
    """The forward/inverse pair over the two continuous latents, on shared axes
    (x = food desire, y = relationship intimacy).

      best_action_by_latents_utensils_{far,near}  which of the three observed actions
                                                  maximizes total utility at each
                                                  (d, I), as filled regions.
      posterior_by_latents_utensils_{far,near}    the HDR_MASS highest-density region
                                                  of P(d, I | a_obs) for each action
                                                  that could have been observed, under
                                                  an untempered observer (see
                                                  posteriors()).

    Each is rendered for both settings of the binary physical world state, which is
    the third latent and enters only through the low-risk share's effort. The near
    setting stipulates effort 0 rather than reading an elicited value; see the
    ILLUSTRATIVE_WEIGHTS comment.

    Together they state the model in both directions: the actor's choice partitions
    the latent space, and observing an action therefore localizes the latents to a
    region of it. The inverse panel is why the two are worth showing as a pair — each
    action's region has a different *shape*, which is the diagnosticity the studies
    turn on: no-share is a vertical band (it pins desire and leaves intimacy open),
    the high-risk share is a horizontal band (the reverse), and only the low-risk
    share constrains both.

    Each panel is conditioned on one setting of the world state rather than general,
    so the caption needs to say which. "Far" is the example's own value (its low-risk
    share has effort 0.67); "near" sets that effort to 0, the low-risk share being
    free. The contrast is the point, in both directions:

      forward — with utensils near, the low-risk share dominates the whole space (it
                matches the high-risk share's goal-satisfaction at no cost and beats
                no-share for any d > 0), so the three-way trade-off exists only when
                sharing the low-risk way costs something. That panel's two degenerate
                edges (ties with no-share along d = 0 and with the high-risk share
                along I = 1) are measure-zero and not drawn.
      inverse — a free action is weaker evidence: with utensils near, the low-risk
                share's region grows (it is consistent with more of the latent space)
                and its posterior mean desire falls back toward the 0.5 prior. The
                regions stay essentially disjoint in both world states; what changes
                is how much space each claims, not how much they overlap. main()
                prints the means and areas -- read them from a run rather than from
                this docstring, since they move with the weights and HDR_MASS.

    Only the three observed actions enter the softmax here, matching the component
    panels. The example's fourth LM alternative (a_3, a partial-goal action) is left
    out: including it would add a fourth region and shift the posteriors, since the
    softmax denominator is what the posterior normalizes against.
    """
    W = resolve_weights(full)
    feats = {a["key"]: a for a in rec["actions"]}
    # note: `full["alpha_observer"]` is deliberately not read here -- see posteriors()

    n = 401  # fine enough that the contoured boundaries read as smooth
    d = np.linspace(0.0, 1.0, n)
    I = np.linspace(0.0, 1.0, n)
    D, II = np.meshgrid(d, I, indexing="ij")

    # same 1 - I floor the other two utility computations in this file apply, so a
    # non-positive gamma from a refit can't send the I = 1 row to inf
    one_minus_I = np.maximum(1.0 - II, 1e-8)

    def utilities(effort_low_risk):
        """U(a | d, I) per observed action. The world state (`e_physical` in the
        manuscript; the `effort_condition` column in `model/tables.py`) enters only
        through the low-risk share's effort."""
        return np.stack(
            [
                W["w_v"] * D * feats[k]["g"]
                - W["w_e"] * (effort_low_risk if k == "a_obs" else feats[k]["effort"])
                - W["w_d"] * feats[k]["risk"] * one_minus_I ** W["gamma"]
                for k in OBSERVED_ACTS
            ]
        )

    def posteriors(U_state):
        """The joint posterior over both latents for each action that could have been
        observed: actor choice (alpha_actor = 1), then P(d, I | a) ∝ P(a | d, I) under
        uniform priors over d and I.

        The observer is deliberately untempered (alpha_obs = 1), so this is the raw
        likelihood of the observed action over the latent space, normalized. It does
        NOT use the fitted `alpha_observer`: that parameter belongs to Study 1a's
        observer, which is *given* intimacy and infers desire alone, so borrowing it
        for a joint inference over both latents would import a temperature fitted for
        a different question. No observer in model/observers.py performs this joint
        inference -- the panel generalizes the model rather than depicting a fit."""
        P = np.exp(U_state - U_state.max(axis=0, keepdims=True))
        P /= P.sum(axis=0, keepdims=True)
        return P / P.sum(axis=(1, 2), keepdims=True)

    def frame(ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"Food desire  $d$")
        ax.set_ylabel(r"Relationship intimacy  $I$")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)  # square, matching the component panels

    def label_regions(ax, positions, masks):
        """Place each region's name inside it. The positions are hand-tuned per panel,
        so assert they still land in the region they name -- otherwise a change to the
        weights, the elicited features or HDR_MASS silently mislabels a published
        figure instead of failing here."""
        for k, (x, y, rot) in positions.items():
            i, j = int(round(x * (n - 1))), int(round(y * (n - 1)))
            if not masks[k][i, j]:
                raise AssertionError(
                    f"label for {k!r} at (d={x}, I={y}) is outside its own region; "
                    "retune the position in STATES"
                )
            ax.text(
                x,
                y,
                ACT_SHORT_LABELS[k],
                color=ACTION_COLORS[k],
                fontsize=15,
                ha="center",
                va="center",
                rotation=rot,
            )

    # --- forward: the argmax partition, one panel per world state -----------------
    def forward_panel(U_state, positions, fname):
        # Ties are resolved explicitly rather than by np.argmax's first-index rule,
        # which would inherit its meaning from the cosmetic order of OBSERVED_ACTS.
        # The near world has 801 exactly-tied cells (the whole d = 0 column, where
        # no-share and a free low-risk share both score 0, and the whole I = 1 row,
        # where discomfort vanishes). Those edges are measure-zero, so no action
        # claims them and nothing is drawn there.
        top = U_state.max(axis=0)
        n_at_top = (U_state >= top - 1e-12).sum(axis=0)
        best = np.argmax(U_state, axis=0)
        masks = {k: (best == i) & (n_at_top == 1) for i, k in enumerate(OBSERVED_ACTS)}
        fig, ax = plt.subplots(figsize=(3.0, 2.8))
        for k in OBSERVED_ACTS:
            if not masks[k].any():
                continue
            # fills only, no per-region outline: neighbouring regions share every
            # boundary, so outlining each one draws each boundary twice in two colors
            # and styles. The change in fill color is the boundary.
            ax.contourf(
                D,
                II,
                masks[k].astype(float),
                levels=[0.5, 1.5],
                colors=[ACTION_COLORS[k]],
                alpha=0.30,
            )
        frame(ax)
        label_regions(ax, positions, masks)
        _savefig(fig, fname)

    # --- inverse: the HDR_MASS posterior region for each observable action ---------
    def inverse_panel(post_state, positions, fname):
        fig, ax = plt.subplots(figsize=(3.0, 2.8))
        masks, areas = {}, {}
        for i, k in enumerate(OBSERVED_ACTS):
            p = post_state[i]
            # density threshold enclosing HDR_MASS of the posterior. Clamped strictly
            # below the peak: if one cell ever carried >= HDR_MASS on its own, the
            # threshold would equal p.max() and contourf rejects equal levels.
            flat = np.sort(p.ravel())[::-1]
            idx = min(int(np.searchsorted(np.cumsum(flat), HDR_MASS)), flat.size - 1)
            peak = float(p.max())
            level = min(float(flat[idx]), float(np.nextafter(peak, -np.inf)))
            masks[k] = p >= level
            areas[k] = float(masks[k].mean())
            ax.contourf(
                D,
                II,
                p,
                levels=[level, peak],
                colors=[ACTION_COLORS[k]],
                alpha=0.30,
            )
            ax.contour(
                D,
                II,
                p,
                levels=[level],
                colors=[ACTION_COLORS[k]],
                linewidths=3.2,
                linestyles=[ACT_LINESTYLES[k]],
            )
        frame(ax)
        label_regions(ax, positions, masks)
        _savefig(fig, fname)
        return areas

    # Both world states, named symmetrically. Label positions are per panel because
    # the regions move: with utensils far the no-share region is a narrow left strip
    # (its label runs up the strip); with them near the low-risk share takes the whole
    # forward panel, so it gets a single centered label.
    STATES = [
        (
            "utensils_far",
            feats["a_obs"]["effort"],  # the example's own world state
            {
                "a_1": (0.085, 0.30, 90),
                "a_obs": (0.58, 0.33, 0),
                "a_2": (0.62, 0.86, 0),
            },
            {
                "a_1": (0.062, 0.24, 90),
                "a_obs": (0.62, 0.24, 0),
                "a_2": (0.55, 0.88, 0),
            },
        ),
        (
            "utensils_near",
            0.0,  # utensils to hand: the low-risk share is free
            {"a_obs": (0.5, 0.5, 0)},
            {
                # the near-world no-share strip is narrower than the far one, so its
                # label sits lower, where the strip is widest
                "a_1": (0.048, 0.28, 90),
                "a_obs": (0.60, 0.30, 0),
                "a_2": (0.58, 0.93, 0),
            },
        ),
    ]
    # Everything the docstring claims about these panels is printed by main() rather
    # than restated in prose, so a change to the weights, the elicited features or
    # HDR_MASS shows up in a normal run instead of silently falsifying a comment.
    stats = {}
    for name, effort, fwd_pos, inv_pos in STATES:
        U_state = utilities(effort)
        post_state = posteriors(U_state)
        forward_panel(U_state, fwd_pos, f"best_action_by_latents_{name}")
        areas = inverse_panel(post_state, inv_pos, f"posterior_by_latents_{name}")
        # Posterior means: the share actions barely move desire off its uniform-prior
        # mean of 0.5 (the under-identification of desire from share actions -- every
        # share has g ≈ 1, so no contrast), and with utensils near, where the low-risk
        # share is free, they barely move it at all. The region area is the other half
        # of that story: weaker evidence claims *more* of the latent space.
        stats[name] = {
            k: (
                float((D * post_state[i]).sum()),
                float((II * post_state[i]).sum()),
                areas[k],
            )
            for i, k in enumerate(OBSERVED_ACTS)
        }
    return stats


def plot_joint_desire_worldstate(rec, full):
    """The joint posterior over desire and the physical world state, as a density.

      joint_density_utensils_{far,near}

    This is the Studies 1b / 3a inference: the relationship is given (the example's
    own value) and the observer recovers desire together with the world state. That
    world state is binary in the model itself -- `observers.py` has the actor choosing
    `effort_condition in EffortConditions`, and `tables.py` defines that as exactly
    {low, high}, against a 101-point desire grid -- so the joint is a density in
    desire with two components, not a 2D surface. Drawing it as a surface would invent
    resolution the model does not have.

    Each panel holds one setting of the world state and overlays all three actions.
    The joint is split into the two things a reader can use separately:

        curve  = P(desire | action, world state), renormalized to unit area
        number in the key = P(that world state | action)

    So a curve is a *conditional*, not a marginal and not a raw joint slice: within a
    panel it answers "given the knives were there, how much did they want the food?",
    and the key answers "how sure are we the knives were there?". An earlier version
    drew the unnormalized joint, where the area under a curve was P(world state) --
    correct, but it forced a shared y scale that squashed one panel and made the
    quantity readable only by comparing areas across panels. Multiplying a curve by
    its key number recovers that joint; adding the two panels' curves for one action
    recovers the marginal over desire.
    """
    W = resolve_weights(full)
    feats = {a["key"]: a for a in rec["actions"]}
    I_given = float(rec["intimacy"])

    n = 401
    d = np.linspace(0.0, 1.0, n)
    one_minus_I = max(1.0 - I_given, 1e-8)
    # panel name -> the low-risk share's effort in that world state
    WORLDS = [("utensils_far", feats["a_obs"]["effort"]), ("utensils_near", 0.0)]

    U = np.stack(
        [
            np.stack(
                [
                    W["w_v"] * d * feats[k]["g"]
                    - W["w_e"] * (eff if k == "a_obs" else feats[k]["effort"])
                    - W["w_d"] * feats[k]["risk"] * one_minus_I ** W["gamma"]
                    for _, eff in WORLDS
                ],
                axis=-1,
            )
            for k in OBSERVED_ACTS
        ]
    )
    P = np.exp(U - U.max(axis=0, keepdims=True))
    P /= P.sum(axis=0, keepdims=True)
    post = P / P.sum(axis=(1, 2), keepdims=True)  # joint over (desire, world state)
    # The joint is split into the two things a reader can actually use: a *shape* per
    # panel and a *number* per panel. Each curve is renormalized to unit area, making
    # it the conditional P(d | action, world state) -- so no panel's peak squashes
    # another's, and the panels are comparable without a shared scale. The world-state
    # belief that the areas used to carry moves into the corner key as a number.
    p_state = post.sum(axis=1)  # P(world state | action), shape (action, state)
    cond = post / post.sum(axis=1, keepdims=True) * (n - 1)  # per unit desire
    ymax = float(cond.max()) * 1.12

    masses = {}
    for w, (wname, _) in enumerate(WORLDS):
        fig, ax = plt.subplots(figsize=(3.0, 2.8))
        for i, k in enumerate(OBSERVED_ACTS):
            y = cond[i][:, w]
            ax.fill_between(d, y, color=ACTION_COLORS[k], alpha=0.26, lw=0)
            ax.plot(
                d,
                y,
                color=ACTION_COLORS[k],
                lw=3.0,
                ls=ACT_LINESTYLES[k],
                solid_capstyle="round",
                dash_capstyle="round",
            )
            masses.setdefault(k, {})[wname] = float(p_state[i, w])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)  # no scale to read; areas carry the mass
        ax.set_xlabel(r"Food desire  $d$")
        ax.set_ylabel("Density")
        ax.set_box_aspect(1)  # square, matching the panels this pair replaces
        # A stacked key in the top-right corner rather than direct in-plot labels: the
        # three densities cross and saturate, so anything placed on a curve lands on
        # another one (no-share in particular peaks against the left spine, leaving it
        # nowhere to sit). The corner is empty in both world states. Each entry carries
        # that action's P(this world state), which the unit-area curves no longer show.
        for row, k in enumerate(OBSERVED_ACTS):
            i = OBSERVED_ACTS.index(k)
            ax.text(
                0.97,
                0.95 - row * 0.11,
                f"{ACT_SHORT_LABELS[k]}  {p_state[i, w]:.2f}",
                transform=ax.transAxes,
                color=ACTION_COLORS[k],
                fontsize=13,
                ha="right",
                va="top",
            )
        _savefig(fig, f"joint_density_{wname}")
    return masses


def plot_components(rec, full):
    """Illustrative component-function plots: each utility term vs. the latent that
    drives it, one curve per observed action (no-share / low-risk / high-risk),
    using ILLUSTRATIVE_WEIGHTS. Coincident lines are dodged slightly so all three
    stay visible (only the action whose feature is non-zero actually varies)."""
    W = resolve_weights(full)
    feats = {a["key"]: a for a in rec["actions"]}
    grid = np.linspace(0.0, 1.0, 101)
    ACTS, LINESTYLES, LABELS = OBSERVED_ACTS, ACT_LINESTYLES, ACT_SHORT_LABELS

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
    # There is deliberately no summed total-utility panel here: total utility depends
    # on all three latents at once, so plotting it against any single one implies a
    # function it doesn't have. plot_latent_space() shows the sum honestly instead, by
    # spending both axes on latents.


def main():
    rec, full = load_inputs()
    c = compute(rec, full)

    plot_bars(rec, c)
    plot_utility(c)
    plot_posterior(c)
    rel_means = plot_posterior_by_relationship(rec, full)
    plot_components(rec, full)
    latent_means = plot_latent_space(rec, full)
    world_masses = plot_joint_desire_worldstate(rec, full)

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
    # joint posterior over both latents, one row per action that could be observed.
    # Under uniform priors both latents start at mean 0.5, so the distance from 0.5 is
    # how much that action tells you about each one.
    print(
        f"joint posterior by observed action "
        f"(mean d, mean I, {HDR_MASS:.0%} region area):"
    )
    for state, per_action in latent_means.items():
        print(f"  {state}:")
        for k, (md, mi, area) in per_action.items():
            print(
                f"    {ACT_SHORT_LABELS[k]:10s} d={md:.3f}  I={mi:.3f}  area={area:.3f}"
            )
    # the world-state split is the only thing the two density panels encode, so
    # print it rather than describing it in a docstring
    print("P(world state | observed action), joint desire x world state:")
    for k, per_world in world_masses.items():
        parts = "  ".join(f"{w}={m:.3f}" for w, m in per_world.items())
        print(f"  {ACT_SHORT_LABELS[k]:10s} {parts}")
    print(f"\nWrote panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
