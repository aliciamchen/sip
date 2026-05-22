#!/usr/bin/env python3
"""
Generate LM counterfactual alternative actions for the 3-action inverse experiments.

The 3-act observers (Studies 2, 3a, 3b, 4a, 4b) softmax over a per-cell action
space {observed_action} ∪ generated_alts rather than the fixed 3-action set.
Alternatives are conditioned on what the human participant sees in the trial —
i.e. observer-visible variables only (`feedback_llm_as_participant.md`).

Per-study conditioning tuples (only the variables the observer actually sees,
besides scenario + observed_action):
    food_inv_desire_3act   (Study 3b): (effort_condition, intimacy_condition)

Other 4 studies (Study 2, 3a, 4a, 4b) follow in a future rollout.

For Study 3b: 16 scenarios × 3 observed actions × 2 effort × 4 intimacy = 384
cells. One LM elicitation per cell (parse-retries up to MAX_PARSE_RETRIES);
each call returns a JSON array of variable-length alternatives.

Output:
    --study food_inv_desire_3act  →  model/outputs/lm/lm_alternatives_food_inv_desire_3act.csv

Usage:
    uv run python model/lm/generate_alternatives_3act.py --study food_inv_desire_3act

Requires:
    - TOGETHER_API_KEY environment variable or in .env file
    - `together` Python package
"""

import argparse
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
from prompts import alternatives_user_prompt_3act


# Temperature for alternative generation in the 3-act pipeline. The legacy
# noalt pipeline (`_alternatives_dispatcher.TEMPERATURE = 1.0`) uses higher T
# to encourage diverse phrasing, but in the 3-act setting this produces many
# semantically-equivalent alternatives with different surface forms (e.g.
# "cut the hot dog in half with a knife" vs "use a plastic knife to slice
# the hot dog"), which case-insensitive dedup at the scoring stage doesn't
# catch. T=0.2 tightens phrasing variability so dedup catches more of the
# semantic overlap, keeping per-scenario unique-alt counts in a range the
# rating prompt can compare reliably.
ALT_GEN_TEMPERATURE = 0.2


# Per-study conditioning: tuple of (column_in_scenarios_csv, level_values).
# Each entry defines an axis the alternative-generation loop iterates over,
# alongside scenario_label and observed_action.
_STUDY_CONFIG = {
    "food_inv_desire_3act": {
        "scenarios": "scenarios_3act.csv",
        "output": "lm_alternatives_food_inv_desire_3act.csv",
        # Observer-visible variables besides scenario + observed_action.
        "conditioning_axes": ("effort_condition", "intimacy_condition"),
        "effort_levels": ["low", "high"],
        "intimacy_levels": [0, 50, 75, 100],
    },
}

ACTION_COLS_3ACT = ["action_0", "action_1", "action_2"]


def load_scenarios(study):
    cfg = _STUDY_CONFIG[study]
    scenarios_path = get_project_root() / "experiments" / cfg["scenarios"]
    return pd.read_csv(scenarios_path)


def _build_3b_cells(scenarios_df, cfg):
    """Enumerate (scenario, observed_action, effort, intimacy) cells for Study 3b
    and build the user prompt for each.

    Returns a list of dicts: {scenario_label, observed_action, effort_condition,
    intimacy_condition, user_prompt}.
    """
    cells = []
    for _, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        vignette = row["vignette"]
        for observed_col in ACTION_COLS_3ACT:
            observed_action_text = row[observed_col]
            for effort in cfg["effort_levels"]:
                effort_text = row[f"effort_{effort}"]
                for intimacy in cfg["intimacy_levels"]:
                    user_prompt = alternatives_user_prompt_3act(
                        vignette,
                        observed_action_text,
                        effort_text=effort_text,
                        intimacy_level=intimacy,
                    )
                    cells.append(
                        {
                            "scenario_label": scenario,
                            "observed_action": observed_col,
                            "effort_condition": effort,
                            "intimacy_condition": intimacy,
                            "user_prompt": user_prompt,
                        }
                    )
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

    output_dir = get_project_root() / "model" / "outputs" / "lm"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / cfg["output"]

    # Resume: skip cells already in the output CSV.
    done_cells = set()
    results = []
    if output_path.exists():
        existing = pd.read_csv(output_path)
        done_cells = set(
            (
                r["scenario_label"],
                r["observed_action"],
                r["effort_condition"],
                int(r["intimacy_condition"]),
            )
            for _, r in existing.iterrows()
        )
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with {len(done_cells)} cells "
            f"already elicited — resuming.",
            flush=True,
        )

    # Build work list, drop done cells.
    if study == "food_inv_desire_3act":
        all_cells = _build_3b_cells(scenarios_df, cfg)
    else:  # pragma: no cover — guarded above
        raise NotImplementedError(study)
    pending = [
        c
        for c in all_cells
        if (
            c["scenario_label"],
            c["observed_action"],
            c["effort_condition"],
            int(c["intimacy_condition"]),
        )
        not in done_cells
    ]
    total = len(all_cells)
    print(
        f"\n{len(pending)} cells to elicit "
        f"(total expected: {total}; {len(done_cells)} already done).",
        flush=True,
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_CELL_WORKERS) as ex:
        future_to_cell = {
            ex.submit(
                elicit_alternatives, client, c["user_prompt"], ALT_GEN_TEMPERATURE
            ): c
            for c in pending
        }
        for fut in as_completed(future_to_cell):
            cell = future_to_cell[fut]
            alts = fut.result()
            completed += 1
            print(
                f"[{completed}/{len(pending)}] {cell['scenario_label']} | "
                f"observed={cell['observed_action']} | "
                f"effort={cell['effort_condition']} | "
                f"intimacy={cell['intimacy_condition']} | "
                f"elicited {len(alts)}",
                flush=True,
            )
            for alt_idx, alt in enumerate(alts):
                results.append(
                    {
                        "scenario_label": cell["scenario_label"],
                        "observed_action": cell["observed_action"],
                        "effort_condition": cell["effort_condition"],
                        "intimacy_condition": cell["intimacy_condition"],
                        "alt_idx": alt_idx,
                        "action_text": alt["action"],
                        "is_share": alt["is_share"],
                    }
                )
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
    per_cell = results_df.groupby(
        ["scenario_label", "observed_action", "effort_condition", "intimacy_condition"]
    ).size()
    print(f"Total cells: {len(per_cell)} (expected {total})")
    print(
        f"Alternatives per cell — min: {per_cell.min()}, max: {per_cell.max()}, "
        f"mean: {per_cell.mean():.1f}, median: {per_cell.median():.0f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=tuple(_STUDY_CONFIG.keys()),
        default="food_inv_desire_3act",
        help=(
            "Which 3-act inverse experiment to elicit alternatives for. "
            "Currently only Study 3b (food_inv_desire_3act) is implemented; "
            "the other 4 studies will be added in a follow-up rollout."
        ),
    )
    args = parser.parse_args()
    main(args.study)
