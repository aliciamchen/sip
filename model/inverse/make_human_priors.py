#!/usr/bin/env python3
"""Build lm_priors_human.jsonl (K=1) from the human PRIOR-stage ratings -- the
ceiling check for LM prior quality (diagnostic only, never a paper config).

Usage: uv run python model/inverse/make_human_priors.py --study food_inv_joint_de
Then:  fit/cv with --priors informative --priors-file lm_priors_human.jsonl

The human prior-stage ratings in data/<slug>/main_trials_long.csv are already
normalized to [0, 1], so the per-cell means pass straight through (no rescaling)
and satisfy load_lm_priors' [0, 1] validation.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

# study -> (rating columns per elicited quantity, prior-cell condition columns in
# the RAW long CSV's naming: intimacy / desire / effort). The quantity keys must
# match tables._PRIOR_STUDY_SPEC so load_lm_priors picks them up.
_SPEC = {
    "food_inv_desire": ({"prior_desire": "response"}, ["effort", "intimacy"]),
    "food_inv_joint_de": (
        {"prior_desire": "desire_rating", "prior_effort_high": "effort_rating"},
        ["intimacy"],
    ),
    "food_inv_intimacy": ({"prior_intimacy": "intimacy_rating"}, ["desire", "effort"]),
    "food_inv_joint_ie": (
        {"prior_intimacy": "intimacy_rating", "prior_effort_high": "effort_rating"},
        ["desire"],
    ),
}
_RENAME = {
    "intimacy": "intimacy_condition",
    "desire": "desire_condition",
    "effort": "effort_condition",
}


def build_human_prior_rows(df, study):
    """Group the prior-stage human ratings into per-cell K=1 prior records.

    `df` is a raw main_trials_long DataFrame (both stages); this filters to the
    prior stage, averages each elicited quantity over scenario x prior-visible
    conditions, and returns a list of JSONL-ready dicts keyed exactly like
    load_lm_priors expects (run_id=0, scenario_label, renamed condition columns,
    and one field per quantity). Ratings are already in [0, 1], so they are not
    rescaled.
    """
    cols, conds = _SPEC[study]
    pri = df[df["stage"] == "prior"]
    if pri.empty:
        raise ValueError(f"{study}: no stage=='prior' rows in main_trials_long.csv")
    rating_cols = list(dict.fromkeys(cols.values()))
    grouped = pri.groupby(["scenario_label", *conds], as_index=False)[
        rating_cols
    ].mean()
    rows = []
    for _, r in grouped.iterrows():
        rec = {"run_id": 0, "scenario_label": r["scenario_label"]}
        for c in conds:
            rec[_RENAME[c]] = r[c]
        for q, col in cols.items():
            rec[q] = round(float(r[col]), 6)
        rows.append(rec)
    return rows


def main(study):
    df = pd.read_csv(_root / "data" / study / "main_trials_long.csv")
    rows = build_human_prior_rows(df, study)
    out = _root / "model" / "outputs" / "lm" / study / "lm_priors_human.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(rows)} records to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True, choices=tuple(_SPEC))
    main(p.parse_args().study)
