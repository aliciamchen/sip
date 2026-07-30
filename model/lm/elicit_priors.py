#!/usr/bin/env python3
"""Standalone prior-scalar elicitation for the informative-prior configs.

For each (scenario x prior-visible conditions) cell of a study, and each of
K_RUNS runs, the LM answers the study's PRIOR-stage questions (mirroring the
human screens exactly; see prompts.py PRIOR_*). Writes
outputs/lm/<slug>/lm_priors{_base}.jsonl — one record per (run, cell) with the
elicited scalars rescaled to [0, 1] — decoupled from the alternatives pipeline
so priors pair with any alternatives vintage by run index.

This stage feeds the informative-prior configs, which were evaluated and not
adopted as the reported model (see model/inverse/_priors.py).

Usage:
    uv run python model/lm/elicit_priors.py --study food_inv_joint_de
    K_RUNS=1 uv run python model/lm/elicit_priors.py --study food_inv_joint_de  # smoke
    uv run python model/lm/elicit_priors.py --study food_inv_desire --base
    uv run python model/lm/elicit_priors.py --study food_inv_joint_de --dry-run
"""

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from together import Together

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _features_dispatcher import (
    numeric_desire_schema,
    numeric_effort_prior_schema,
    numeric_intimacy_schema,
    parse_desire_response,
    parse_effort_prior_response,
    parse_intimacy_response,
)
from client import (
    get_ratings_concurrent,
    load_api_key,
    write_jsonl_atomic,
    write_run_manifest,
)
from prompts import (
    PRIOR_DESIRE_SYSTEM_PROMPT,
    PRIOR_EFFORT_SYSTEM_PROMPT,
    PRIOR_INTIMACY_SYSTEM_PROMPT,
    RELATIONSHIP_DESCRIPTORS,
    prior_desire_user_prompt,
    prior_effort_user_prompt,
    prior_intimacy_user_prompt,
)

K_RUNS = int(os.environ.get("K_RUNS", "20"))
PRIOR_WORKERS = int(os.environ.get("PRIOR_WORKERS", "16"))
TEMPERATURE = 0.2  # matches feature scoring

INTIMACY_LEVELS = ["max_formal", "somewhat_formal", "somewhat_intimate", "max_intimate"]
LEVELS = ["low", "high"]

# Per-study prior-cell grid: which condition columns the participant sees
# before the prior rating, and which quantities that study elicits. Must stay
# consistent with tables._PRIOR_STUDY_SPEC (the loader).
_STUDY_CONFIG = {
    "food_inv_desire": {
        "scenarios": "scenarios.csv",
        "conds": ("effort_condition", "intimacy_condition"),
        "quantities": ("prior_desire",),
    },
    "food_inv_joint_de": {
        "scenarios": "scenarios.csv",
        "conds": ("intimacy_condition",),
        "quantities": ("prior_desire", "prior_effort_high"),
    },
    "food_inv_intimacy": {
        "scenarios": "scenarios.csv",
        "conds": ("desire_condition", "effort_condition"),
        "quantities": ("prior_intimacy",),
    },
    "food_inv_joint_ie": {
        "scenarios": "scenarios.csv",
        "conds": ("desire_condition",),
        "quantities": ("prior_intimacy", "prior_effort_high"),
    },
    "nonfood_inv_joint_de": {
        "scenarios": "scenarios_nonfood.csv",
        "conds": ("intimacy_condition",),
        "quantities": ("prior_desire", "prior_effort_high"),
    },
    "nonfood_inv_joint_ie": {
        "scenarios": "scenarios_nonfood.csv",
        "conds": ("desire_condition",),
        "quantities": ("prior_intimacy", "prior_effort_high"),
    },
}
_COND_LEVELS = {
    "effort_condition": LEVELS,
    "desire_condition": LEVELS,
    "intimacy_condition": INTIMACY_LEVELS,
}
_QUANTITY_CALL = {
    "prior_desire": (
        PRIOR_DESIRE_SYSTEM_PROMPT,
        numeric_desire_schema,
        parse_desire_response,
    ),
    "prior_effort_high": (
        PRIOR_EFFORT_SYSTEM_PROMPT,
        numeric_effort_prior_schema,
        parse_effort_prior_response,
    ),
    "prior_intimacy": (
        PRIOR_INTIMACY_SYSTEM_PROMPT,
        numeric_intimacy_schema,
        parse_intimacy_response,
    ),
}


def _condition_texts(row, cell, conds):
    """The given-condition paragraphs the participant sees at the prior stage,
    rendered in the order the experiment shows them. The relationship sentence
    uses the same wording as alternatives_user_prompt; desire/effort use the
    scenario CSV's raw paragraphs. Effort paragraphs are EXCLUDED for studies
    that infer effort (they are the prior-effort question's endpoints, not
    shown context)."""
    texts = []
    if "intimacy_condition" in conds:
        texts.append(
            "The two people are in a relationship they would describe as "
            f"{RELATIONSHIP_DESCRIPTORS[cell['intimacy_condition']]}."
        )
    if "desire_condition" in conds:
        texts.append(
            row["desire_low" if cell["desire_condition"] == "low" else "desire_high"]
        )
    if "effort_condition" in conds:
        texts.append(
            row[
                "low_risk_share_effort_low"
                if cell["effort_condition"] == "low"
                else "low_risk_share_effort_high"
            ]
        )
    return tuple(texts)


def _build_prior_cells(study, base=False):
    cfg = _STUDY_CONFIG[study]
    conds = tuple(c for c in cfg["conds"] if not (base and c == "intimacy_condition"))
    csv = (
        Path(__file__).resolve().parent.parent.parent / "experiments" / cfg["scenarios"]
    )
    scenarios = pd.read_csv(csv)
    cells = []
    for _, row in scenarios.iterrows():
        combos = [{}]
        for c in conds:
            combos = [{**combo, c: lv} for combo in combos for lv in _COND_LEVELS[c]]
        for combo in combos:
            cells.append(
                {
                    "scenario_label": row["scenario_label"],
                    **combo,
                    "vignette": row["vignette"],
                    "desire_object": row["desire_object"],
                    "effort_low_text": row["low_risk_share_effort_low"],
                    "effort_high_text": row["low_risk_share_effort_high"],
                    "condition_texts": _condition_texts(row, combo, conds),
                    "quantities": cfg["quantities"],
                }
            )
    return cells


def _seed_for(cell, quantity, run_id):
    key = "|".join(
        [cell["scenario_label"]]
        + [cell[c] for c in sorted(cell) if c.endswith("_condition")]
        + [quantity, str(run_id)]
    ).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little") & 0x7FFFFFFF


def _user_prompt_for(quantity, cell):
    if quantity == "prior_desire":
        return prior_desire_user_prompt(
            cell["vignette"], cell["desire_object"], cell["condition_texts"]
        )
    if quantity == "prior_effort_high":
        return prior_effort_user_prompt(
            cell["vignette"],
            cell["effort_low_text"],
            cell["effort_high_text"],
            cell["condition_texts"],
        )
    return prior_intimacy_user_prompt(cell["vignette"], cell["condition_texts"])


def _elicit_one(client, cell, quantity, run_id):
    system_prompt, schema_fn, parse_fn = _QUANTITY_CALL[quantity]
    ratings, _ = get_ratings_concurrent(
        client,
        system_prompt,
        _user_prompt_for(quantity, cell),
        parse_fn,
        num_runs=1,
        max_tokens=60,
        temperature=TEMPERATURE,
        response_format=schema_fn(),
        label=f"{cell['scenario_label']}|{quantity}|run{run_id}",
        seed=_seed_for(cell, quantity, run_id),
    )
    if not ratings:
        raise RuntimeError(
            f"prior elicitation failed after retries: "
            f"{cell['scenario_label']} {quantity} run {run_id}"
        )
    return float(ratings[0]) / 100.0


def main(study, base=False, dry_run=False):
    cells = _build_prior_cells(study, base=base)
    jobs = [
        (ci, cell, q, k)
        for ci, cell in enumerate(cells)
        for q in cell["quantities"]
        for k in range(K_RUNS)
    ]
    out_name = f"lm_priors{'_base' if base else ''}.jsonl"
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "lm" / study
    out_path = out_dir / out_name
    print(
        f"{study}{' (base)' if base else ''}: {len(cells)} cells x "
        f"{len(cells[0]['quantities'])} quantities x K={K_RUNS} = {len(jobs)} calls "
        f"-> {out_path}"
    )
    if dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Together(api_key=load_api_key())
    results = {}
    with ThreadPoolExecutor(max_workers=PRIOR_WORKERS) as ex:
        futs = {
            ex.submit(_elicit_one, client, cell, q, k): (ci, q, k)
            for ci, cell, q, k in jobs
        }
        for i, fu in enumerate(as_completed(futs), 1):
            ci, q, k = futs[fu]
            results[(ci, q, k)] = fu.result()
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} calls done")
    rows = []
    for ci, cell in enumerate(cells):
        for k in range(K_RUNS):
            rec = {
                "run_id": k,
                "scenario_label": cell["scenario_label"],
                **{c: cell[c] for c in cell if c.endswith("_condition")},
            }
            for q in cell["quantities"]:
                rec[q] = results[(ci, q, k)]
            rows.append(rec)
    write_jsonl_atomic(out_path, rows)
    write_run_manifest(
        out_path, stage="priors", study=study, extra={"base": base, "k_runs": K_RUNS}
    )
    print(f"wrote {len(rows)} records to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", required=True, choices=tuple(_STUDY_CONFIG))
    p.add_argument("--base", action="store_true")
    p.add_argument(
        "--dry-run", action="store_true", help="print the call count and exit"
    )
    a = p.parse_args()
    if a.base and "intimacy_condition" not in _STUDY_CONFIG[a.study]["conds"]:
        p.error("--base applies only to the given-relationship studies (1a/1b/3a)")
    main(a.study, base=a.base, dry_run=a.dry_run)
