#!/usr/bin/env python3
"""Gated-model figures: three columns Full | Full + gate | Humans.

Same points-by-action design as `figure_poster_points.py` (x = observed action,
marker shape = inferred target, color = the given relationship/desire condition,
filled/open = the given effort condition in the single-DV studies), so the gated
predictions are read exactly the way the canonical ones are. This deliberately
reuses that script's renderer rather than restyling: the only changes are which
prediction sets become panels and what they are labelled.

The gated predictions are the in-sample forward predictions at each study's
fitted gated parameters (`notes/candidate-sweep-2026-07-28/gated_preds_<slug>.json`,
written by `model/diagnostics/gated.py`), whereas the Full column is the
committed LEAVE-ONE-SCENARIO-OUT prediction. That asymmetry favours the gate
slightly and is called out in the figure caption text below — Gate B previously
established that in-sample and LOSO predictions agree closely at these sample
sizes (r >= 0.995), so the comparison is still informative, but it is not
like-for-like. Held-out likelihood comparisons live in `gated_cv_<slug>.json`.

Usage:
    uv run python figures/scripts/figure_gated_points.py [--study <slug>]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent.parent))
sys.path.insert(0, str(_here))

import _data as data  # noqa: E402
import figure_poster_points as poster  # noqa: E402
from plot_style import apply_style  # noqa: E402

SWEEP = _here.parent.parent / "notes" / "candidate-sweep-2026-07-28"

# Panels: the committed canonical full fit, the gated fit, then humans.
GATED_ORDER = ["full", "gated"]
GATED_LABELS = {"full": "Full", "gated": "Full + gate"}


def load_gated_preds(slug):
    path = SWEEP / f"gated_preds_{slug}.json"
    if not path.exists():
        return None
    df = pd.DataFrame(json.loads(path.read_text()))
    return df.assign(model="gated")


# Bind the original before the patch below: `poster.data` is this same module
# object, so a patched `data.load_cv_preds` that called `data.load_cv_preds`
# would recurse into itself.
_load_cv_preds = data.load_cv_preds


def combined_preds(slug):
    """Canonical LOSO preds (full only) + the gated forward preds."""
    canon = _load_cv_preds(slug)
    gated = load_gated_preds(slug)
    if canon is None and gated is None:
        return None
    frames = []
    if canon is not None:
        frames.append(canon[canon["model"] == "full"])
    if gated is not None:
        frames.append(gated)
    return pd.concat(frames, ignore_index=True) if frames else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", help="Render one slug only (default: all six).")
    args = ap.parse_args()
    apply_style("si")

    # Swap the panel set and the prediction source, then let the poster
    # renderer do the drawing so the two figure families stay visually identical.
    data.MODEL_ORDER = GATED_ORDER
    data.MODEL_LABELS = GATED_LABELS
    data.PANEL_ORDER = [*GATED_ORDER, "humans"]
    data.PANEL_LABELS = {**GATED_LABELS, "humans": "Humans"}
    poster.data.load_cv_preds = combined_preds

    drawn = []
    for slug, paper, stem in poster.STUDIES:
        if args.study and slug != args.study:
            continue
        if load_gated_preds(slug) is None:
            print(f"[{slug}] no gated preds yet — skipped")
            continue
        poster.build_study(slug, paper, f"gated_{stem}")
        drawn.append(slug)
    print(f"\ndrew {len(drawn)} study figure(s): {', '.join(drawn)}")
    print(
        "NOTE: the Full column is leave-one-scenario-out; the Full + gate column "
        "is in-sample forward prediction at the fitted gated parameters."
    )


if __name__ == "__main__":
    main()
