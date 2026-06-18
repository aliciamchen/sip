#!/usr/bin/env python3
"""
Generate LM counterfactual alternative actions for the 3-action inverse experiments.

The 3-act observers (Studies 1a, 1b, 2a, 2b) softmax over a per-cell action
space {observed_action} ∪ generated_alts rather than the fixed 3-action set.
Alternatives are conditioned on what the human participant sees in the trial —
i.e. observer-visible variables only (`feedback_llm_as_participant.md`).

Per-study conditioning tuples (only the variables the observer actually sees,
besides scenario + observed_action), and the resulting cell count:
    food_inv_desire   (1a): (effort, intimacy)  → 16 × 3 × 2 × 4 = 384 cells
    food_inv_joint_de (1b): (intimacy,)         → 16 × 3 × 4     = 192 cells
    food_inv_intimacy (2a): (desire, effort)    → 16 × 3 × 2 × 2 = 192 cells
    food_inv_joint_ie (2b): (desire,)           → 16 × 3 × 2     =  96 cells

One LM elicitation per cell (parse-retries up to MAX_PARSE_RETRIES); each call
returns a JSON array of variable-length alternatives.

Output (one folder per study slug):
    --study food_inv_desire  →  model/outputs/lm/food_inv_desire/lm_alternatives.csv

Usage:
    uv run python model/lm/generate_alternatives.py --study food_inv_desire

Requires:
    - TOGETHER_API_KEY environment variable or in .env file
    - `together` Python package
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from together import Together

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _alternatives_dispatcher import (
    CHECKPOINT_EVERY,
    MAX_CELL_WORKERS,
    elicit_alternatives,
)
from client import MODEL_ID, load_api_key
from prompts import alternatives_user_prompt


# The K-run simulated-observer pipeline: for each (scenario × condition) cell we
# repeat the full elicitation K_RUNS times (each run = its own alternatives set +
# feature scores), and the run-to-run spread becomes part of the model's
# predicted distribution of human responses. Cross-run *diversity* is the point,
# so generation uses a nonzero ALT_T (higher than the legacy single-run 0.2,
# which was tuned to *suppress* phrasing variation for cross-cell dedup). Dedup
# stays WITHIN a run. Both are env-overridable so the Makefile can tune them.
N_RUNS_ALT = int(os.environ.get("K_RUNS", "20"))
ALT_GEN_TEMPERATURE = float(os.environ.get("ALT_T", "0.7"))


# Per-study conditioning. `show` lists which condition paragraphs the observer
# (and hence the LM) sees on each trial — only the observer-visible variables,
# so the alternative set does not leak the latent being inferred. `cell_cols`
# are the resulting cell-key columns (besides scenario_label + observed_action)
# written to the output CSV; the downstream merged-scoring + padded-table loader
# key on these.
#   1a desire    — visible: effort, intimacy        (infers desire)
#   1b joint_de  — visible: intimacy                (infers desire + effort)
#   2a intimacy  — visible: desire, effort          (infers intimacy)
#   2b joint_ie  — visible: desire                  (infers intimacy + effort)
_STUDY_CONFIG = {
    "food_inv_desire": {
        "scenarios": "scenarios.csv",
        "output": "lm_alternatives.csv",
        "show": ("effort", "intimacy"),
        "cell_cols": ("effort_condition", "intimacy_condition"),
    },
    "food_inv_joint_de": {
        "scenarios": "scenarios.csv",
        "output": "lm_alternatives.csv",
        "show": ("intimacy",),
        "cell_cols": ("intimacy_condition",),
    },
    "food_inv_intimacy": {
        "scenarios": "scenarios.csv",
        "output": "lm_alternatives.csv",
        "show": ("desire", "effort"),
        "cell_cols": ("desire_condition", "effort_condition"),
    },
    "food_inv_joint_ie": {
        "scenarios": "scenarios.csv",
        "output": "lm_alternatives.csv",
        "show": ("desire",),
        "cell_cols": ("desire_condition",),
    },
}

ACTION_COLS = ["no_share", "low_risk_share", "high_risk_share"]
DESIRE_LEVELS = ["low", "high"]
EFFORT_LEVELS = ["low", "high"]
# Intimacy is a purely verbal manipulation: levels are identified by slug
# (ascending, formal -> intimate), never by a numeric code.
INTIMACY_LEVELS = ["max_formal", "neither", "somewhat_intimate", "max_intimate"]


def load_scenarios(study):
    cfg = _STUDY_CONFIG[study]
    scenarios_path = get_project_root() / "experiments" / cfg["scenarios"]
    return pd.read_csv(scenarios_path)


def _cell_key(cell, cell_cols, run_id=None):
    """Tuple key for resume-dedup, normalized like the output CSV row. Includes
    run_id so a resumed job skips completed (cell, run) units, not whole cells."""
    key = [cell["scenario_label"], cell["observed_action"]]
    for col in cell_cols:
        key.append(cell[col])
    if run_id is not None:
        key.append(int(run_id))
    return tuple(key)


def _run_seed(cell, cell_cols, run_id):
    """Deterministic per-(cell, run) seed so reruns reproduce."""
    return (hash(_cell_key(cell, cell_cols)) ^ (int(run_id) * 0x9E3779B1)) & 0x7FFFFFFF


def _build_cells(scenarios_df, cfg):
    """Enumerate cells for one study, iterating scenario × observed_action over
    only the observer-visible conditioning axes (per cfg['show']). Returns dicts
    with scenario_label, observed_action, the study's cell_cols, and user_prompt
    (built with only the visible condition paragraphs)."""
    show = cfg["show"]
    desire_levels = DESIRE_LEVELS if "desire" in show else [None]
    effort_levels = EFFORT_LEVELS if "effort" in show else [None]
    intimacy_levels = INTIMACY_LEVELS if "intimacy" in show else [None]

    cells = []
    for _, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        vignette = row["vignette"]
        for observed_col in ACTION_COLS:
            observed_action_text = row[observed_col]
            for desire in desire_levels:
                for effort in effort_levels:
                    for intimacy in intimacy_levels:
                        prompt_kwargs = {}
                        cell = {
                            "scenario_label": scenario,
                            "observed_action": observed_col,
                        }
                        if desire is not None:
                            prompt_kwargs["desire_text"] = row[f"desire_{desire}"]
                            cell["desire_condition"] = desire
                        if effort is not None:
                            prompt_kwargs["effort_text"] = row[
                                f"low_risk_share_effort_{effort}"
                            ]
                            cell["effort_condition"] = effort
                        if intimacy is not None:
                            prompt_kwargs["intimacy_level"] = intimacy
                            cell["intimacy_condition"] = intimacy
                        cell["user_prompt"] = alternatives_user_prompt(
                            vignette, observed_action_text, **prompt_kwargs
                        )
                        cells.append(cell)
    return cells


def main(study):
    if study not in _STUDY_CONFIG:
        raise SystemExit(
            f"Unknown study: {study!r}. Currently supported: "
            f"{sorted(_STUDY_CONFIG.keys())}"
        )
    cfg = _STUDY_CONFIG[study]
    api_key = load_api_key()

    print(f"Loading scenarios (study={study})...", flush=True)
    scenarios_df = load_scenarios(study)
    print(f"Loaded {len(scenarios_df)} scenarios", flush=True)

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    output_dir = get_project_root() / "model" / "outputs" / "lm" / study
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / cfg["output"]

    cell_cols = cfg["cell_cols"]

    # Resume: skip (cell, run) units already in the output CSV.
    done_units = set()
    results = []
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if "run_id" in existing.columns:
            done_units = set(
                _cell_key(r, cell_cols, r["run_id"]) for _, r in existing.iterrows()
            )
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with {len(done_units)} (cell, run) "
            f"units already elicited — resuming.",
            flush=True,
        )

    # Build work list as (cell, run) units, dropping done ones.
    all_cells = _build_cells(scenarios_df, cfg)
    pending = [
        (c, run)
        for c in all_cells
        for run in range(N_RUNS_ALT)
        if _cell_key(c, cell_cols, run) not in done_units
    ]
    total = len(all_cells) * N_RUNS_ALT
    print(
        f"\n{len(pending)} (cell, run) units to elicit at T={ALT_GEN_TEMPERATURE} "
        f"(K={N_RUNS_ALT} runs/cell; total expected: {total}; "
        f"{len(done_units)} already done).",
        flush=True,
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_CELL_WORKERS) as ex:
        future_to_unit = {
            ex.submit(
                elicit_alternatives,
                client,
                c["user_prompt"],
                ALT_GEN_TEMPERATURE,
                _run_seed(c, cell_cols, run),
            ): (c, run)
            for c, run in pending
        }
        for fut in as_completed(future_to_unit):
            cell, run = future_to_unit[fut]
            alts = fut.result()
            completed += 1
            cond_str = " | ".join(f"{c}={cell[c]}" for c in cell_cols)
            print(
                f"[{completed}/{len(pending)}] run {run} | {cell['scenario_label']} | "
                f"observed={cell['observed_action']} | {cond_str} | "
                f"elicited {len(alts)}",
                flush=True,
            )
            for alt_idx, alt in enumerate(alts):
                row = {
                    "scenario_label": cell["scenario_label"],
                    "observed_action": cell["observed_action"],
                }
                for col in cell_cols:
                    row[col] = cell[col]
                row["run_id"] = run
                row["alt_idx"] = alt_idx
                row["action_text"] = alt["action"]
                row["is_share"] = alt["is_share"]
                results.append(row)
            if completed % CHECKPOINT_EVERY == 0:
                pd.DataFrame(results).to_csv(output_path, index=False)
                print(
                    f"  checkpoint written ({len(results)} rows total)",
                    flush=True,
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} alternatives to {output_path}", flush=True)

    print("\n=== Summary ===")
    per_unit = results_df.groupby(
        ["scenario_label", "observed_action", *cell_cols, "run_id"]
    ).size()
    print(f"Total (cell, run) units: {len(per_unit)} (expected {total})")
    print(
        f"Alternatives per (cell, run) — min: {per_unit.min()}, max: {per_unit.max()}, "
        f"mean: {per_unit.mean():.1f}, median: {per_unit.median():.0f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=tuple(_STUDY_CONFIG.keys()),
        default="food_inv_desire",
        help="Which 3-act inverse experiment to elicit alternatives for.",
    )
    args = parser.parse_args()
    main(args.study)
