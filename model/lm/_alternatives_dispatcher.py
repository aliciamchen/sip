#!/usr/bin/env python3
"""
Generate LM counterfactual alternative action sets for the no-alternatives-shown
inverse-planning variants.

Two conditioning modes are supported via `--conditioning`:

  * `motivation` (default): conditions alternatives on (observed_action,
    motivation). Used by `food_inv-intimacy_desire_noalt` (observer sees motivation,
    infers intimacy). 16 scenarios × 4 actions × 2 motivations = 128 cells.

  * `relationship`: conditions alternatives on (observed_action,
    relationship_condition). Used by `food_inv-desire_intimacy_noalt` (observer sees
    relationship, infers motivation). 16 scenarios × 4 actions × 4 relationship
    levels = 256 cells.

The general design rule: alternatives are conditioned on what the observer
observes (not on the latent), so the LM-elicited counterfactual action space
matches what a human observer would entertain.

For each cell, prompt Llama-3.3-70B-Instruct-Turbo to list the set of plausible
alternative actions the actor could have taken instead of the observed action.
The LM decides set size; no fixed quota. Each alternative is tagged with a
binary is_share flag so the V/access/effort scoring downstream can be applied.

Output:
    --conditioning motivation, --domain food   → model/outputs/lm_alternatives_food_inv-intimacy_desire_noalt.csv
    --conditioning motivation, --domain nonfood → model/outputs/lm_alternatives_nonfood.csv
    --conditioning relationship, --domain food → model/outputs/lm_alternatives_food_inv-desire_intimacy_noalt.csv
    --conditioning relationship, --domain nonfood → model/outputs/lm_alternatives_relationship_nonfood.csv

Usage:
    uv run python model/lm/generate_alternatives_motivation.py
    uv run python model/lm/generate_alternatives_relationship.py

Requires:
    - TOGETHER_API_KEY environment variable or in .env file
    - `together` Python package
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from together import Together

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

# Shared LM-call infrastructure (key loading, JSON helpers, retries).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import (
    MAX_RETRIES,
    MODEL_ID,
    alternatives_array_schema,
    find_json_array,
    load_api_key,
)

# Built once and reused — schema construction is pure.
_ALTERNATIVES_RESPONSE_FORMAT = alternatives_array_schema()

TEMPERATURE = 1.0
MAX_TOKENS = 800
MAX_PARSE_RETRIES = 5
ACTION_COLS = ["action_0", "action_1", "action_2", "action_3"]
MOTIVATIONS = ["low", "high"]
RELATIONSHIPS = [0, 50, 75, 100]

# How many cells to elicit concurrently. One LM call per cell (with up to
# MAX_PARSE_RETRIES parse-retries inside), so this is the per-call concurrency.
MAX_CELL_WORKERS = 8

# How often to flush partial results to disk while the thread pool runs.
CHECKPOINT_EVERY = 16

# Per-(domain, conditioning) input CSV and output CSV. All runs use the general
# LM prompt set; the --domain flag selects which scenario CSV (and output
# filenames) to use; --conditioning selects which axis the alternatives are
# split along.
_DOMAIN_PATHS = {
    ("food", "motivation"):    {"scenarios": "scenarios.csv",         "output": "lm_alternatives_food_inv-intimacy_desire_noalt.csv"},
    ("nonfood", "motivation"): {"scenarios": "scenarios_nonfood.csv", "output": "lm_alternatives_nonfood.csv"},
    ("food", "relationship"):    {"scenarios": "scenarios.csv",         "output": "lm_alternatives_food_inv-desire_intimacy_noalt.csv"},
    ("nonfood", "relationship"): {"scenarios": "scenarios_nonfood.csv", "output": "lm_alternatives_relationship_nonfood.csv"},
}


from prompts import ALTERNATIVES_SYSTEM_PROMPT
from prompts import alternatives_user_prompt as format_motivation_user_prompt
from prompts import alternatives_user_prompt_relationship as format_relationship_user_prompt


def parse_alternatives(response_text):
    js = find_json_array(response_text)
    if js is None:
        return None
    try:
        arr = json.loads(js)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Failed to parse JSON: {e}")
        return None
    if not isinstance(arr, list):
        return None
    out = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        is_share = item.get("is_share")
        if not isinstance(action, str) or is_share not in (0, 1, True, False):
            continue
        out.append({"action": action.strip(), "is_share": int(bool(is_share))})
    return out if out else None


def _dedup_alternatives(alts):
    seen = set()
    out = []
    for a in alts:
        key = a["action"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def elicit_alternatives(client, user_prompt):
    """Elicit alternatives for one cell. Up to MAX_PARSE_RETRIES tries to land
    a parseable response; transient errors inside each call are retried by the
    SDK via ``max_retries=MAX_RETRIES``. Returns [] when all parse retries are
    exhausted (rather than raising) so a thread-pool batch can continue."""
    messages = [
        {"role": "system", "content": ALTERNATIVES_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    retrying_client = client.with_options(max_retries=MAX_RETRIES)
    for attempt in range(MAX_PARSE_RETRIES):
        try:
            response = retrying_client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                response_format=_ALTERNATIVES_RESPONSE_FORMAT,
            )
            parsed = parse_alternatives(response.choices[0].message.content)
            if parsed:
                return _dedup_alternatives(parsed)
        except Exception as e:
            print(f"  Attempt {attempt + 1} error: {e}", flush=True)
    print(
        "  All parse retries exhausted; returning empty alternative set for this cell.",
        flush=True,
    )
    return []


def load_scenarios(domain, conditioning):
    scenarios_path = (
        get_project_root() / "experiments" / _DOMAIN_PATHS[(domain, conditioning)]["scenarios"]
    )
    return pd.read_csv(scenarios_path)


def main(domain, conditioning):
    api_key = load_api_key()

    print(f"Loading scenarios (domain={domain}, conditioning={conditioning})...", flush=True)
    scenarios_df = load_scenarios(domain, conditioning)
    print(f"Loaded {len(scenarios_df)} scenarios", flush=True)

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    if conditioning == "motivation":
        levels = MOTIVATIONS
        level_label = "motivation"
    else:
        levels = RELATIONSHIPS
        level_label = "relationship_condition"

    output_dir = get_project_root() / "model" / "outputs" / "lm"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / _DOMAIN_PATHS[(domain, conditioning)]["output"]

    # Resume: skip cells whose (scenario, observed_action, level) tuple already
    # appears in the output CSV.
    results = []
    done_cells = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        done_cells = set(
            (r["scenario_label"], r["observed_action"], r[level_label])
            for _, r in existing.iterrows()
        )
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with {len(done_cells)} cells "
            f"already elicited — resuming.",
            flush=True,
        )

    # Build the full work list, then drop already-done cells.
    all_cells = []
    for _, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        vignette = row["vignette"]
        for observed_col in ACTION_COLS:
            observed_action_text = row[observed_col]
            for level in levels:
                if (scenario, observed_col, level) in done_cells:
                    continue
                if conditioning == "motivation":
                    reward_text = row[f"reward_{level}"]
                    user_prompt = format_motivation_user_prompt(
                        vignette, reward_text, observed_action_text
                    )
                else:
                    user_prompt = format_relationship_user_prompt(
                        vignette, level, observed_action_text
                    )
                all_cells.append(
                    {
                        "scenario_label": scenario,
                        "observed_action": observed_col,
                        "level": level,
                        "user_prompt": user_prompt,
                    }
                )
    total_cells = len(scenarios_df) * len(ACTION_COLS) * len(levels)
    print(
        f"\n{len(all_cells)} cells to elicit "
        f"(total expected: {total_cells}; {len(done_cells)} already done).",
        flush=True,
    )

    # Thread across cells. Each future returns (cell_meta, list-of-alts).
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_CELL_WORKERS) as ex:
        future_to_cell = {
            ex.submit(elicit_alternatives, client, c["user_prompt"]): c
            for c in all_cells
        }
        for fut in as_completed(future_to_cell):
            cell = future_to_cell[fut]
            alts = fut.result()
            completed += 1
            print(
                f"[{completed}/{len(all_cells)}] {cell['scenario_label']} | "
                f"observed={cell['observed_action']} | "
                f"{level_label}={cell['level']} | elicited {len(alts)}",
                flush=True,
            )
            for alt_idx, alt in enumerate(alts):
                results.append(
                    {
                        "scenario_label": cell["scenario_label"],
                        "observed_action": cell["observed_action"],
                        level_label: cell["level"],
                        "alt_idx": alt_idx,
                        "action_text": alt["action"],
                        "is_share": alt["is_share"],
                    }
                )
            if completed % CHECKPOINT_EVERY == 0:
                pd.DataFrame(results).to_csv(output_path, index=False)
                print(f"  checkpoint written ({len(results)} rows total)", flush=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} alternatives to {output_path}", flush=True)

    print("\n=== Summary ===")
    per_cell = results_df.groupby(
        ["scenario_label", "observed_action", level_label]
    ).size()
    print(f"Total cells: {len(per_cell)} (expected {total_cells})")
    print(
        f"Alternatives per cell — min: {per_cell.min()}, max: {per_cell.max()}, "
        f"mean: {per_cell.mean():.1f}, median: {per_cell.median():.0f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=("food", "nonfood"),
        default="food",
        help=(
            "Which scenario set to elicit alternatives for. 'food' (default) "
            "uses scenarios.csv; 'nonfood' uses scenarios_nonfood.csv."
        ),
    )
    parser.add_argument(
        "--conditioning",
        choices=("motivation", "relationship"),
        default="motivation",
        help=(
            "Which axis to condition alternatives on. 'motivation' (default) "
            "is used by food_inv-intimacy_desire_noalt (observer sees motivation, "
            "infers intimacy); 'relationship' is used by food_inv-desire_intimacy_noalt "
            "(observer sees relationship, infers motivation)."
        ),
    )
    args = parser.parse_args()
    main(args.domain, args.conditioning)
