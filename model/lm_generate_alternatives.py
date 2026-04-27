#!/usr/bin/env python3
"""
Generate LM counterfactual alternative action sets for the no-alternatives-shown
inverse-planning variants.

Two conditioning modes are supported via `--conditioning`:

  * `motivation` (default): conditions alternatives on (observed_action,
    motivation). Used by `inv_plan_intimacy_noalt` (observer sees motivation,
    infers intimacy). 16 scenarios × 4 actions × 2 motivations = 128 cells.

  * `relationship`: conditions alternatives on (observed_action,
    relationship_condition). Used by `inv_plan_desire_noalt` (observer sees
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
    --conditioning motivation, --domain food   → model/outputs/lm_alternatives.csv
    --conditioning motivation, --domain nonfood → model/outputs/lm_alternatives_nonfood.csv
    --conditioning relationship, --domain food → model/outputs/lm_alternatives_relationship.csv
    --conditioning relationship, --domain nonfood → model/outputs/lm_alternatives_relationship_nonfood.csv

Usage:
    uv run python model/lm_generate_alternatives.py
    uv run python model/lm_generate_alternatives.py --conditioning relationship

Requires:
    - TOGETHER_API_KEY environment variable or in .env file
    - `together` Python package
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from together import Together

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TEMPERATURE = 1.0
MAX_TOKENS = 800
MAX_PARSE_RETRIES = 5
ACTION_COLS = ["action_0", "action_1", "action_2", "action_3"]
MOTIVATIONS = ["low", "high"]
RELATIONSHIPS = [0, 50, 75, 100]

# Per-(domain, conditioning) input CSV and output CSV. All runs use the general
# LM prompt set; the --domain flag selects which scenario CSV (and output
# filenames) to use; --conditioning selects which axis the alternatives are
# split along.
_DOMAIN_PATHS = {
    ("food", "motivation"):    {"scenarios": "scenarios.csv",         "output": "lm_alternatives.csv"},
    ("nonfood", "motivation"): {"scenarios": "scenarios_nonfood.csv", "output": "lm_alternatives_nonfood.csv"},
    ("food", "relationship"):    {"scenarios": "scenarios.csv",         "output": "lm_alternatives_relationship.csv"},
    ("nonfood", "relationship"): {"scenarios": "scenarios_nonfood.csv", "output": "lm_alternatives_relationship_nonfood.csv"},
}


from lm_prompts import ALTERNATIVES_SYSTEM_PROMPT
from lm_prompts import alternatives_user_prompt as format_motivation_user_prompt
from lm_prompts import alternatives_user_prompt_relationship as format_relationship_user_prompt


_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _extract_json_array(text):
    if text is None:
        return None
    match = _JSON_ARRAY_RE.search(text)
    return match.group(0) if match else None


def parse_alternatives(response_text):
    js = _extract_json_array(response_text)
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
    messages = [
        {"role": "system", "content": ALTERNATIVES_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    for attempt in range(MAX_PARSE_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            parsed = parse_alternatives(response.choices[0].message.content)
            if parsed:
                return _dedup_alternatives(parsed)
        except Exception as e:
            print(f"  Attempt {attempt + 1} error: {e}")
        time.sleep(0.5)
    print(
        "  All parse retries exhausted; returning empty alternative set for this cell."
    )
    return []


def _load_api_key():
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        env_path = get_project_root() / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("TOGETHER_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip('"').strip("'")
                        break
    if not api_key:
        print("Error: set TOGETHER_API_KEY in env or .env")
        sys.exit(1)
    return api_key


def load_scenarios(domain, conditioning):
    scenarios_path = (
        get_project_root() / "experiments" / _DOMAIN_PATHS[(domain, conditioning)]["scenarios"]
    )
    return pd.read_csv(scenarios_path)


def main(domain, conditioning):
    api_key = _load_api_key()

    print(f"Loading scenarios (domain={domain}, conditioning={conditioning})...")
    scenarios_df = load_scenarios(domain, conditioning)
    print(f"Loaded {len(scenarios_df)} scenarios")

    print(f"\nInitializing Together AI client for {MODEL_ID}...")
    client = Together(api_key=api_key)

    results = []
    if conditioning == "motivation":
        levels = MOTIVATIONS
        level_label = "motivation"
    else:
        levels = RELATIONSHIPS
        level_label = "relationship_condition"
    total_cells = len(scenarios_df) * len(ACTION_COLS) * len(levels)
    cell_idx = 0

    for _, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        vignette = row["vignette"]
        for observed_col in ACTION_COLS:
            observed_action_text = row[observed_col]
            for level in levels:
                cell_idx += 1
                if conditioning == "motivation":
                    reward_text = row[f"reward_{level}"]
                    user_prompt = format_motivation_user_prompt(
                        vignette, reward_text, observed_action_text
                    )
                else:
                    user_prompt = format_relationship_user_prompt(
                        vignette, level, observed_action_text
                    )
                print(
                    f"\n[{cell_idx}/{total_cells}] {scenario} | observed={observed_col} | "
                    f"{level_label}={level}"
                )
                alts = elicit_alternatives(client, user_prompt)
                print(f"  Elicited {len(alts)} alternatives")
                for alt_idx, alt in enumerate(alts):
                    record = {
                        "scenario_label": scenario,
                        "observed_action": observed_col,
                        level_label: level,
                        "alt_idx": alt_idx,
                        "action_text": alt["action"],
                        "is_share": alt["is_share"],
                    }
                    results.append(record)

    results_df = pd.DataFrame(results)
    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / _DOMAIN_PATHS[(domain, conditioning)]["output"]
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} alternatives to {output_path}")

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
            "is used by inv_plan_intimacy_noalt (observer sees motivation, "
            "infers intimacy); 'relationship' is used by inv_plan_desire_noalt "
            "(observer sees relationship, infers motivation)."
        ),
    )
    args = parser.parse_args()
    main(args.domain, args.conditioning)
