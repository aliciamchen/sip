#!/usr/bin/env python3
"""Non-food human belief updates split by sharing domain, one panel per domain.

The Study 3 results panels average over all 16 non-food scenarios, which pools
three quite different forms of interpersonal vulnerability. This script draws the
HUMANS column of those panels three times instead of once -- once per domain
(bodily access / shared physical exposure / private access, the
`scenario_type` tag in `experiments/scenarios_nonfood.csv`) -- so a reader can
see whether the averaged pattern holds within each domain or is carried by one
of them.

No model columns: this is the data behind the Study 3 averages, not a fit check.
Everything else is the results panels' encoding, drawn through `_points.py` at
`figure_paper_panels.py`'s Illustrator-bound style, and the three panels share
one y axis so the domains are directly comparable. Legends are the ones already
written to `figures/panels/legends/` -- placed by hand like every other panel
here.

    figures/panels/results/panel_study3a_domains.pdf
    figures/panels/results/panel_study3b_domains.pdf

Usage:
    uv run python figures/scripts/figure_nonfood_domains.py [--study <slug>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_style import PANELS_RESULTS, apply_style, savefig  # noqa: E402
from study_registry import studies  # noqa: E402
from utils import get_project_root  # noqa: E402

import _data as data  # noqa: E402
import _points as points  # noqa: E402

# Style, canvas margins and rc overrides come from the four-column results panels
# rather than being restated, so this figure cannot drift from them: a change to
# the panel aesthetics shows up here too.
from figure_paper_panels import PANEL_MARGINS, PANEL_RC, STYLE  # noqa: E402

OUT_DIR = PANELS_RESULTS

# The `scenario_type` tags of scenarios_nonfood.csv, in the order the manuscript
# introduces them (Study 3's opening section), titled by the tag itself rather
# than by the manuscript's longer names for the three domains ("bodily access",
# "shared physical exposure", "private access") -- one word per panel, with the
# caption left to say what each domain covers.
DOMAIN_ORDER = ["substance", "space", "privacy"]
DOMAIN_LABELS = {
    "substance": "Substance",
    "space": "Space",
    "privacy": "Privacy",
}

# Number of columns PANEL_MARGINS was tuned against (Base | Discomfort-only |
# Full | Humans). The margins are figure fractions, so a row with a different
# column count has to rescale the horizontal pair to keep the same axes box.
REF_NCOLS = 4


def margins_for(ncols):
    """PANEL_MARGINS rescaled so `ncols` panels leave the same INCH margins as
    the four-column rows -- otherwise a narrower canvas shrinks the reserve and
    the y-axis label runs off the left edge."""
    scale = REF_NCOLS / ncols
    return {
        **PANEL_MARGINS,
        "left": PANEL_MARGINS["left"] * scale,
        "right": 1 - (1 - PANEL_MARGINS["right"]) * scale,
    }


def domain_of_scenario():
    """{scenario_label: scenario_type} from the non-food stimulus CSV."""
    csv = get_project_root() / "experiments" / "scenarios_nonfood.csv"
    scenarios = pd.read_csv(csv)
    return dict(zip(scenarios["scenario_label"], scenarios["scenario_type"]))


def build_domain_cells(slug):
    """Human cell means with bootstrap CIs on the domain x condition grid, or
    None when the study has no data yet."""
    trials = data.load_trials(slug)
    if trials is None:
        return None
    domains = domain_of_scenario()
    missing = set(trials["scenario_label"]) - domains.keys()
    if missing:
        raise ValueError(
            f"[{slug}] scenario(s) {sorted(missing)} are in the data but not in "
            "experiments/scenarios_nonfood.csv, so their domain is unknown"
        )
    trials = trials.assign(
        action_label=data.action_label_col(trials),
        scenario_type=trials["scenario_label"].map(domains),
    )
    print(
        f"[{slug}] humans: {trials['subject_id'].nunique()} subjects, "
        + ", ".join(
            f"{d}: {trials.loc[trials['scenario_type'] == d, 'scenario_label'].nunique()} scenarios"
            for d in DOMAIN_ORDER
        )
    )
    return data.bootstrap_cell_means(
        trials,
        [h for h, _d, _l in data.dvs_display(slug)],
        ["scenario_type", *data.condition_cols(slug)],
        seed=data.seed_for(f"figures:nonfood_domains:{slug}"),
    )


def symmetric_limit(slug, cells, pad=1.12):
    """Symmetric y limit covering every point and CI end across all domains, so
    the three panels can share one axis."""
    return pad * float(
        max(
            cells[c].abs().max()
            for h, _d, _l in data.dvs_display(slug)
            for c in (h, f"{h}_ci_lower", f"{h}_ci_upper")
        )
    )


def draw_panel(slug, stem):
    """One study's three-domain row of human panels, on its own artboard."""
    cells = build_domain_cells(slug)
    if cells is None:
        print(f"[{slug}] nothing to draw yet — skipped")
        return False
    # No CV staleness check, unlike the results panels: nothing here is drawn
    # from a model output, so the only vintage that matters is the data CSV's.
    fcol, flevels, fcolors, _title = points.fill_spec(slug)
    dvs = data.dvs_display(slug)
    value_cols = [(h, m) for (h, _d, _l), m in zip(dvs, points.markers_for(slug))]
    lim = symmetric_limit(slug, cells)

    fig, axes = plt.subplots(
        1,
        len(DOMAIN_ORDER),
        figsize=(STYLE.panel_w * len(DOMAIN_ORDER), STYLE.panel_h),
        sharey=True,
    )
    fig.subplots_adjust(**margins_for(len(DOMAIN_ORDER)))
    for ax, domain in zip(axes, DOMAIN_ORDER):
        points.draw_points(
            ax,
            cells[cells["scenario_type"] == domain],
            value_cols=value_cols,
            color_col=fcol,
            fill_levels=flevels,
            colors=fcolors,
            lim=lim,
            ci=True,
            style=STYLE,
            style_col=points.style_col(slug),
        )
        ax.set_title(DOMAIN_LABELS[domain], fontsize=STYLE.title_fs, y=STYLE.title_y)
        ax.tick_params(
            axis="y",
            labelsize=STYLE.tick_fs,
            length=STYLE.ytick_len,
            width=STYLE.ytick_w,
        )
    axes[0].set_ylabel(points.ylabel_for(slug), fontsize=STYLE.label_fs)
    fig.supxlabel(points.X_AXIS_LABEL, fontsize=STYLE.label_fs, y=0.015)
    # Full canvas, like the other Illustrator panels, so the axes boxes land in
    # the same place on every page and can be stacked by page origin.
    savefig(fig, stem, out_dir=OUT_DIR, tight=False)
    print(f"wrote {OUT_DIR.parent.name}/{OUT_DIR.name}/{stem}.pdf")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", help="Render one slug only (default: both non-food).")
    args = ap.parse_args()
    apply_style("si")
    drawn = 0
    with plt.rc_context(PANEL_RC):
        for s in studies():
            if s.domain != "nonfood" or (args.study and s.slug != args.study):
                continue
            drawn += draw_panel(s.slug, f"panel_study{s.short_label}_domains")
    print(f"\n{drawn} domain panel(s) in {OUT_DIR}")


if __name__ == "__main__":
    main()
