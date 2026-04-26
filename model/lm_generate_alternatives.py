#!/usr/bin/env python3
"""
Generate LM counterfactual alternative action sets for the no-alternatives-shown
inverse-planning variant (`inv_plan_intimacy_noalt`).

For each (scenario, observed_action, motivation) cell — 16 scenarios × 4
canonical observed actions × 2 motivation levels = 128 cells — prompt
Llama-3.3-70B-Instruct-Turbo to list the set of plausible alternative actions
the actor could have taken instead of the observed action. The LM decides set
size; no fixed quota. Each alternative is tagged with a binary is_share flag so
the stipulated goal-satisfaction gate V can be applied to arbitrary actions.

Alternatives are conditioned on (observed_action, motivation) because that's
how a human observer constructs counterfactuals — "given they did this under
high/low motivation, what else could they have done?" Motivation is stipulated
in the vignette and is observable to the participant, so conditioning on it
doesn't leak information about the latent (intimacy).

Output:
    --domain food (default) → model/outputs/lm_alternatives.csv
    --domain nonfood        → model/outputs/lm_alternatives_nonfood.csv

Both runs use the domain-general LM prompt set.

Usage:
    uv run python model/lm_generate_alternatives.py
    uv run python model/lm_generate_alternatives.py --domain nonfood

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

# Per-domain input CSV and output CSV. All runs use the general LM prompt set;
# the --domain flag selects which scenario CSV (and output filenames) to use.
_DOMAIN_PATHS = {
    "food":    {"scenarios": "scenarios.csv",         "output": "lm_alternatives.csv"},
    "nonfood": {"scenarios": "scenarios_nonfood.csv", "output": "lm_alternatives_nonfood.csv"},
}


from lm_prompts import alternatives_system_prompt
from lm_prompts import alternatives_user_prompt as format_user_prompt


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


def elicit_alternatives(client, vignette, reward_text, observed_action_text):
    messages = [
        {"role": "system", "content": alternatives_system_prompt()},
        {
            "role": "user",
            "content": format_user_prompt(vignette, reward_text, observed_action_text),
        },
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


def load_scenarios(domain):
    scenarios_path = get_project_root() / "experiments" / _DOMAIN_PATHS[domain]["scenarios"]
    return pd.read_csv(scenarios_path)


def main(domain):
    api_key = _load_api_key()

    print(f"Loading scenarios (domain={domain})...")
    scenarios_df = load_scenarios(domain)
    print(f"Loaded {len(scenarios_df)} scenarios")

    print(f"\nInitializing Together AI client for {MODEL_ID}...")
    client = Together(api_key=api_key)

    results = []
    total_cells = len(scenarios_df) * len(ACTION_COLS) * len(MOTIVATIONS)
    cell_idx = 0

    for _, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        vignette = row["vignette"]
        for observed_col in ACTION_COLS:
            observed_action_text = row[observed_col]
            for motivation in MOTIVATIONS:
                cell_idx += 1
                reward_text = row[f"reward_{motivation}"]
                print(
                    f"\n[{cell_idx}/{total_cells}] {scenario} | observed={observed_col} | motivation={motivation}"
                )
                alts = elicit_alternatives(
                    client, vignette, reward_text, observed_action_text
                )
                print(f"  Elicited {len(alts)} alternatives")
                for alt_idx, alt in enumerate(alts):
                    results.append(
                        {
                            "scenario_label": scenario,
                            "observed_action": observed_col,
                            "motivation": motivation,
                            "alt_idx": alt_idx,
                            "action_text": alt["action"],
                            "is_share": alt["is_share"],
                        }
                    )

    results_df = pd.DataFrame(results)
    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / _DOMAIN_PATHS[domain]["output"]
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} alternatives to {output_path}")

    print("\n=== Summary ===")
    per_cell = results_df.groupby(
        ["scenario_label", "observed_action", "motivation"]
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
            "uses scenarios.csv → lm_alternatives.csv; 'nonfood' uses "
            "scenarios_nonfood.csv → lm_alternatives_nonfood.csv. Both use "
            "the domain-general LM prompt set."
        ),
    )
    args = parser.parse_args()
    main(args.domain)
