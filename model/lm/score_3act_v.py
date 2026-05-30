#!/usr/bin/env python3
"""
Score signed-valence V for the 3-action canonical set (Studies 2, 3a, 3b,
4a, 4b).

For each (scenario, action, motivation) in experiments/scenarios.csv,
the LM rates how strongly each action serves the actor's motivational state
on a signed -3..+3 scale, normalized to [-1, +1]. V is independent of
effort_condition (V is about reward fulfillment, not effort), so this script
produces a table of shape 16 × 3 × 2 (96 rows).

Output: model/outputs/lm/lm_scenario_v_3act.csv (or _nonfood with --domain).

10 runs per (scenario, motivation), aggregated to mean/std. Resumes per
scenario if the output CSV already exists.

Requires TOGETHER_API_KEY in env or .env.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from together import Together

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import (
    MODEL_ID,
    aggregate_action_ratings,
    find_json,
    get_ratings_concurrent,
    load_api_key,
    numeric_action_schema,
    strip_leading_plus,
)
from _features_dispatcher import normalize_v
from prompts import system_prompt as build_system_prompt
from prompts import user_prompt as build_user_prompt


N_ACTIONS = 3

_DOMAIN_FILES = {
    "food": ("scenarios.csv", "lm_scenario_v_3act.csv"),
    "nonfood": ("scenarios_nonfood_3act.csv", "lm_scenario_v_3act_nonfood.csv"),
}


def load_scenarios(domain):
    scenarios_path = get_project_root() / "experiments" / _DOMAIN_FILES[domain][0]
    return pd.read_csv(scenarios_path)


def _action_texts_3(row):
    return [row[c] for c in ("no_share", "low_risk_share", "high_risk_share")]


def format_v_prompt(row, motivation):
    state = row[f"desire_{motivation}"]
    return build_user_prompt("v", row["vignette"], _action_texts_3(row), state=state)


def parse_action_response(response_text):
    if response_text is None:
        return None
    js = find_json(response_text)
    if js is None:
        return None
    js = strip_leading_plus(js)
    try:
        ratings = json.loads(js)
        expected = {f"action_{i}" for i in range(N_ACTIONS)}
        if expected.issubset(ratings.keys()):
            return {k: float(ratings[k]) for k in expected}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Failed to parse JSON: {e}")
    return None


def main(domain="food"):
    api_key = load_api_key()

    print(f"Loading 3-action scenarios (domain={domain})...")
    scenarios_df = load_scenarios(domain)
    print(f"Loaded {len(scenarios_df)} scenarios")

    print(f"\nInitializing Together AI client for {MODEL_ID}...")
    client = Together(api_key=api_key)

    output_dir = get_project_root() / "model" / "outputs" / "lm"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / _DOMAIN_FILES[domain][1]

    results = []
    already_done = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_done = set(existing["scenario_label"].unique())
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with "
            f"{len(already_done)} scenarios already scored — resuming.",
            flush=True,
        )

    for idx, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        if scenario in already_done:
            print(
                f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} — already scored, skipping.",
                flush=True,
            )
            continue

        print(f"\nProcessing {idx + 1}/{len(scenarios_df)}: {scenario}", flush=True)

        for motivation in ("low", "high"):
            print(
                f"  Getting V ratings (motivation={motivation}, concurrent, structured)...",
                flush=True,
            )
            v_ratings, n_failures = get_ratings_concurrent(
                client,
                build_system_prompt("v", n_actions=N_ACTIONS),
                format_v_prompt(row, motivation),
                parse_action_response,
                response_format=numeric_action_schema(N_ACTIONS),
                label=f"{scenario}/V[{motivation}]",
            )
            v_agg = aggregate_action_ratings(v_ratings, n_actions=N_ACTIONS)

            for action in range(N_ACTIONS):
                key = f"action_{action}"
                v_mean, v_std = v_agg[key]
                results.append(
                    {
                        "scenario_label": scenario,
                        "action": action,
                        "motivation": motivation,
                        "v_raw": v_mean,
                        "v_raw_std": v_std,
                        "v": normalize_v(v_mean) if not np.isnan(v_mean) else np.nan,
                        "n_runs": len(v_ratings),
                        "n_failures": n_failures,
                    }
                )

            v_str = [f"{v_agg[f'action_{i}'][0]:+.1f}" for i in range(N_ACTIONS)]
            print(f"  V {motivation} (raw): {v_str}", flush=True)

        pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\nSaved results to {output_path}", flush=True)

    print("\n=== Summary ===")
    results_df = pd.DataFrame(results)
    print(
        f"Total rows: {len(results_df)} (expected 96 = 16 scenarios × 3 actions × 2 motivations)"
    )
    for mot in ("low", "high"):
        sub = results_df[results_df["motivation"] == mot]
        print(
            f"\nV (normalized, motivation={mot}, target [-1, +1]):"
            f"\n  Mean: {sub['v'].mean():+.2f}, Std: {sub['v'].std():.2f}"
            f"\n  Range: [{sub['v'].min():+.2f}, {sub['v'].max():+.2f}]"
        )
    print("\nDone!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=("food", "nonfood"), default="food")
    args = p.parse_args()
    main(args.domain)
