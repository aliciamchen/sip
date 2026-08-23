#!/usr/bin/env python3
"""Descriptive statistics of the elicited comparison sets.

The SI states several properties of what the LM elicitation produced -- how many
alternatives a set holds, how often a set carries no goal-satisfaction contrast,
how much an action's effort moves between the two physical states. Those numbers
were read off the SI figures by hand until 2026-08-16; they live here so the
prose, the figures and the macros all come from one computation.

Everything reads `outputs/lm/<slug>/lm_runs.jsonl` -- one record per (run, cell)
holding that run's scored actions, slot 0 being the observed one. Stdlib +
pandas/numpy only, so a figure script can import it without pulling in JAX.

Usage:
    uv run python model/lm/set_diagnostics.py            # print the summary
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from study_registry import STUDIES, SLUGS  # noqa: E402
from utils import get_project_root  # noqa: E402

#: A set counts as carrying no contrast in a feature when its range is at or
#: below this. The LM rates features on a 0-6 integer scale rescaled to [0, 1],
#: so the smallest real difference is 1/6 ~ 0.167 and anything under 0.01 is an
#: exactly-equal set, not a nearly-equal one. Named rather than inlined because
#: the SI quotes the resulting rates and a reader may want to check the rule.
FLAT_TOL = 0.01

#: Observed-action conditions, in the order the design orders them.
OBSERVED_ACTIONS = ("no_share", "low_risk_share", "high_risk_share")

#: The refusal condition, whose sets the SI singles out: its own g sits near 0
#: while its alternatives sit near 1, which is what makes the whole-set and
#: alternatives-only rates come apart.
REFUSAL = "no_share"


def load_runs(slug, filename="lm_runs.jsonl"):
    """One study's scored elicitation runs, or None if it has not been run.

    The observed-action labels are checked against `OBSERVED_ACTIONS` rather than
    assumed: every statistic here groups by that column, so a relabeled
    condition would silently drop its sets from the rates instead of failing.
    """
    path = get_project_root() / "model" / "outputs" / "lm" / slug / filename
    if not path.exists():
        return None
    runs = pd.read_json(path, lines=True)
    found = set(runs["observed_action"].unique())
    if found != set(OBSERVED_ACTIONS):
        raise ValueError(
            f"{slug}: observed actions in {filename} are {sorted(found)}, not "
            f"{sorted(OBSERVED_ACTIONS)} -- update OBSERVED_ACTIONS here (and "
            "plot_style.OBSERVED_ACTIONS, which orders the same conditions for "
            "the figures) rather than letting a renamed condition go uncounted."
        )
    return runs


def g_contrast_rates(runs, flat=FLAT_TOL):
    """{observed action: (whole-set %, alternatives-only %, n sets, n empty)}.

    The two percentages answer different questions. The whole-set rate says
    whether the observation itself can speak to desire: an action's own g prices
    desire directly (the reward term is `w_v * d * g`), so a set spanning g ~ 0
    to g ~ 1 makes the choice informative. The alternatives-only rate says
    whether *reweighting* that set can speak to desire: the reweighting moves
    mass among the forgone actions, so when those are uniform in g no weighting
    of them shifts the desire posterior, however surprising the observation is.

    A set with no alternatives at all has nothing to reweight and no range to
    measure, so it is counted separately rather than scored as "no contrast".
    That count is deliberately NOT the same as "no alternative carries a g": a
    set can hold alternatives whose g scoring returned null, and folding those in
    would drop real sets from the denominator of a rate the SI quotes while
    reporting them as alternative-free. Both are excluded from the rate; only the
    genuinely alternative-free ones are counted as such.
    """
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
            forgone = [a for a in actions if not a.get("is_observed")]
            ga = [a["g"] for a in forgone if a.get("g") is not None]
            if ga:
                alts.append(max(ga) - min(ga))
            elif not forgone:
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


def alt_set_sizes(runs):
    """The per-set count of generated alternatives, one entry per (run, cell)."""
    return np.asarray(
        [
            sum(1 for a in actions if not a.get("is_observed"))
            for actions in runs["actions"]
        ]
    )


def _effort_cell_keys(runs):
    """The columns identifying a cell apart from the physical state, so the two
    states of the same cell can be paired."""
    return [
        c
        for c in (
            "run_id",
            "scenario_label",
            "observed_action",
            "intimacy_condition",
            "desire_condition",
        )
        if c in runs.columns
    ]


def effort_swings(slug, runs):
    """({observed action: mean swing}, max swing over alternatives), or
    (None, None) where the physical state is given rather than inferred.

    An action's "swing" is |effort(high) - effort(low)| for the same action in
    the same cell and run. It is what the comparison-set reweighting scores an
    alternative on for a physical-world question (`v(a)` in the SI), and the
    reason the reweighting is needed: where the OBSERVED action barely swings,
    the observation prices the physical state weakly, so what it implies about
    the state has to come from the alternatives instead.

    Actions are paired across the two states by slot. That is only sound where
    the state is INFERRED: those studies generate one alternative list per cell
    and score it under both states, so a slot names the same action in both. In
    the given-state studies (1a, 2a) the state is part of the generation cell, so
    the two lists are different actions and slot 3 of one is unrelated to slot 3
    of the other -- hence the guard rather than a silently wrong pairing.
    """
    if "effort_condition" in STUDIES[slug].given_conditions:
        return None, None
    if "effort_condition" not in runs.columns:
        return None, None
    keys = _effort_cell_keys(runs)
    rows = []
    for actions, *cell, state in zip(
        runs["actions"], *[runs[k] for k in keys], runs["effort_condition"]
    ):
        for a in actions:
            if a.get("effort") is None:
                continue
            rows.append(
                (*cell, a["slot"], state, a["effort"], bool(a.get("is_observed")))
            )
    df = pd.DataFrame(rows, columns=[*keys, "slot", "state", "effort", "is_observed"])
    wide = df.pivot_table(
        index=[*keys, "slot", "is_observed"], columns="state", values="effort"
    )
    if not {"low", "high"} <= set(wide.columns):
        return None, None
    swing = (wide["high"] - wide["low"]).abs().rename("swing").reset_index()
    obs = swing[swing["is_observed"]]
    by_action = {
        act: float(obs.loc[obs["observed_action"] == act, "swing"].mean())
        for act in OBSERVED_ACTIONS
        if (obs["observed_action"] == act).any()
    }
    alt_max = float(swing.loc[~swing["is_observed"], "swing"].max())
    return by_action, alt_max


def summarize(slugs=None):
    """Per-study diagnostics, for the macros and the console summary."""
    out = {}
    for slug in slugs or SLUGS:
        runs = load_runs(slug)
        if runs is None:
            continue
        sizes = alt_set_sizes(runs)
        by_action, alt_max = effort_swings(slug, runs)
        entry = {
            "g_contrast": g_contrast_rates(runs),
            "n_alts_median": float(np.median(sizes)),
            "n_alts_q1": float(np.percentile(sizes, 25)),
            "n_alts_q3": float(np.percentile(sizes, 75)),
            "infers_effort": by_action is not None,
        }
        if by_action is not None:
            # Per observed action, because the SI's claim is specifically about
            # the refusal and the high-risk share: the low-risk share is the
            # action the physical-state paragraph is written to describe, so it
            # swings by design and averaging it in would hide the contrast.
            entry["effort_swing_observed"] = by_action
            entry["effort_swing_alt_max"] = alt_max
        out[slug] = entry
    return out


def main():
    summary = summarize()
    for slug, entry in summary.items():
        label = STUDIES[slug].short_label
        print(f"\n=== {label} ({slug}) ===")
        print(
            f"  alternatives per set: median {entry['n_alts_median']:.0f} "
            f"(IQR {entry['n_alts_q1']:.0f}-{entry['n_alts_q3']:.0f})"
        )
        for act, (whole, alts, n, empty) in entry["g_contrast"].items():
            print(
                f"  {act:16s} no g contrast: whole set {whole:5.1f}%, "
                f"alternatives only {alts:5.1f}%  (n = {n}, {empty} with no alts)"
            )
        if entry["infers_effort"]:
            swings = "  ".join(
                f"{act} {v:.3f}" for act, v in entry["effort_swing_observed"].items()
            )
            print(
                f"  effort swing (observed): {swings}   "
                f"| alternatives max {entry['effort_swing_alt_max']:.3f}"
            )


if __name__ == "__main__":
    main()
