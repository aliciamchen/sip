#!/usr/bin/env python3
"""
Generate LLM-derived scenario-specific parameters for the access-based models.

Uses Together AI's Llama-3.3-70B-Instruct-Turbo to estimate, for each of the 16
scenarios in experiments/scenarios.csv:

- access(a): physical / informational / spatial exposure per action  (0-6 -> [0, 2])
- effort(a): physical / logistical cost per action                   (0-6 -> [0, 1])

Reward is NOT elicited from the LLM — it's stipulated in `model/utility.py`
as a binary goal-satisfaction gate (V=1 iff the action satisfies the active
goal: sharing under HIGH motivation, not-sharing under LOW motivation).

10 runs per parameter-type per scenario, aggregated to mean/std.

Usage:
    uv run python model/lm_scenario_params.py                          # food scenarios canonical-4 (default)
    uv run python model/lm_scenario_params.py --score-alternatives     # features for LM-generated food alternatives
    uv run python model/lm_scenario_params.py --domain nonfood                       # nonfood scenarios canonical-4
    uv run python model/lm_scenario_params.py --domain nonfood --score-alternatives  # nonfood alternatives features

Requires:
    - TOGETHER_API_KEY environment variable or in .env file
    - `together` Python package (add to pyproject.toml)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from together import Together

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

# Shared LM-call infrastructure (concurrency, retries, JSON helpers, key loading).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_client import (
    MODEL_ID,
    aggregate_action_ratings,
    find_json,
    get_ratings_concurrent,
    load_api_key,
    numeric_action_schema,
    strip_leading_plus,
)

# Per-domain input CSV and output CSVs. All runs use the general LM prompt set;
# the --domain flag selects which scenario CSV (and output filenames) to use.
_DOMAIN_PATHS = {
    "food": {
        "scenarios":                          "scenarios.csv",
        "params_output":                      "lm_scenario_params.csv",
        "v_output":                           "lm_scenario_v.csv",
        "alternatives_input":                 "lm_alternatives.csv",
        "alternatives_output":                "lm_alternatives_features.csv",
        "alternatives_v_output":              "lm_alternatives_v.csv",
        "alternatives_rel_input":             "lm_alternatives_relationship.csv",
        "alternatives_rel_output":            "lm_alternatives_relationship_features.csv",
        "alternatives_rel_v_output":          "lm_alternatives_relationship_v.csv",
    },
    "nonfood": {
        "scenarios":                          "scenarios_nonfood.csv",
        "params_output":                      "lm_scenario_params_nonfood.csv",
        "v_output":                           "lm_scenario_v_nonfood.csv",
        "alternatives_input":                 "lm_alternatives_nonfood.csv",
        "alternatives_output":                "lm_alternatives_features_nonfood.csv",
        "alternatives_v_output":              "lm_alternatives_v_nonfood.csv",
        "alternatives_rel_input":             "lm_alternatives_relationship_nonfood.csv",
        "alternatives_rel_output":            "lm_alternatives_relationship_features_nonfood.csv",
        "alternatives_rel_v_output":          "lm_alternatives_relationship_v_nonfood.csv",
    },
}


# ==============================================================================
# Prompt builders (centralized in model/lm_prompts.py)
# ==============================================================================

# Imported with aliases so the function names don't collide with the
# `system_prompt` / `user_prompt` parameters of get_ratings below.
from lm_prompts import system_prompt as build_system_prompt
from lm_prompts import user_prompt as build_user_prompt


# Reward is stipulated in model/utility.py as a binary goal-satisfaction
# gate (V=1 iff the action satisfies the active goal: sharing under HIGH
# motivation, not-sharing under LOW motivation). Not elicited from the LLM.


# ==============================================================================
# Scenario loading and prompt formatting
# ==============================================================================


def load_scenarios(domain="food"):
    scenarios_path = get_project_root() / "experiments" / _DOMAIN_PATHS[domain]["scenarios"]
    return pd.read_csv(scenarios_path)


def _action_texts_4(row):
    return [row[f"action_{i}"] for i in range(4)]


def format_access_prompt(row):
    return build_user_prompt("access", row["vignette"], _action_texts_4(row))


def format_effort_prompt(row):
    return build_user_prompt("effort", row["vignette"], _action_texts_4(row))


def format_v_prompt(row, motivation):
    """Build the V prompt for a (scenario, motivation) pair.

    motivation: 'low' or 'high'. Selects which reward_* paragraph is used
    as the actor's state.
    """
    state_col = f"reward_{motivation}"
    return build_user_prompt(
        "v", row["vignette"], _action_texts_4(row), state=row[state_col],
    )


# ==============================================================================
# Parsers
# ==============================================================================


def parse_action_response(response_text):
    """Parse JSON ratings with action_0..action_3 keys.

    Tolerates a quirk of the V prompt's signed -3..+3 scale: some LM outputs
    use a leading `+` sign (e.g. `"action_1": +3`), which is invalid JSON. We
    strip leading `+` from numeric values before parsing."""
    if response_text is None:
        return None
    js = find_json(response_text)
    if js is None:
        return None
    js = strip_leading_plus(js)
    try:
        ratings = json.loads(js)
        expected = {"action_0", "action_1", "action_2", "action_3"}
        if expected.issubset(ratings.keys()):
            return {k: float(ratings[k]) for k in expected}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Failed to parse JSON: {e}")
    return None


# ==============================================================================
# Normalization (0-6 LLM scale -> model-native scales)
# ==============================================================================


def normalize_access(value, target_max=2.0):
    """0-6 -> [0, target_max]. Matches the [0, 2] range of the fixed access vector."""
    return value * (target_max / 6.0)


def normalize_effort(value, target_max=1.0):
    return value * (target_max / 6.0)


def normalize_v(value):
    """[-3, +3] -> [-1, +1]. Signed valence rating."""
    return value / 3.0


# ==============================================================================
# Main
# ==============================================================================


def main(domain="food"):
    api_key = load_api_key()

    print(f"Loading scenarios (domain={domain})...")
    scenarios_df = load_scenarios(domain)
    print(f"Loaded {len(scenarios_df)} scenarios")

    print(f"\nInitializing Together AI client for {MODEL_ID}...")
    client = Together(api_key=api_key)

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / _DOMAIN_PATHS[domain]["params_output"]

    # Resume: pick up scenarios already in the output CSV (mirrors the pattern
    # used in score_alternatives_main).
    results = []
    already_done_scenarios = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_done_scenarios = set(existing["scenario_label"].unique())
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with "
            f"{len(already_done_scenarios)} scenarios already scored — resuming.",
            flush=True,
        )

    for idx, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        if scenario in already_done_scenarios:
            print(f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} — already scored, skipping.", flush=True)
            continue

        print(f"\nProcessing {idx + 1}/{len(scenarios_df)}: {scenario}", flush=True)

        print("  Getting access ratings (concurrent, structured)...", flush=True)
        access_ratings, access_failures = get_ratings_concurrent(
            client,
            build_system_prompt("access", n_actions=4),
            format_access_prompt(row),
            parse_action_response,
            response_format=numeric_action_schema(4),
            label=f"{scenario}/access",
        )
        access_agg = aggregate_action_ratings(access_ratings, n_actions=4)

        print("  Getting effort ratings (concurrent, structured)...", flush=True)
        effort_ratings, effort_failures = get_ratings_concurrent(
            client,
            build_system_prompt("effort", n_actions=4),
            format_effort_prompt(row),
            parse_action_response,
            response_format=numeric_action_schema(4),
            label=f"{scenario}/effort",
        )
        effort_agg = aggregate_action_ratings(effort_ratings, n_actions=4)

        for action in range(4):
            key = f"action_{action}"
            a_mean, a_std = access_agg[key]
            e_mean, e_std = effort_agg[key]
            results.append(
                {
                    "scenario_label": scenario,
                    "action": action,
                    "access_raw": a_mean,
                    "access_raw_std": a_std,
                    "effort_raw": e_mean,
                    "effort_raw_std": e_std,
                    "access": normalize_access(a_mean) if not np.isnan(a_mean) else np.nan,
                    "effort": normalize_effort(e_mean) if not np.isnan(e_mean) else np.nan,
                    "n_runs_access": len(access_ratings),
                    "n_runs_effort": len(effort_ratings),
                    "n_failures_access": access_failures,
                    "n_failures_effort": effort_failures,
                }
            )

        acc_str = [f"{access_agg[f'action_{i}'][0]:.1f}" for i in range(4)]
        eff_str = [f"{effort_agg[f'action_{i}'][0]:.1f}" for i in range(4)]
        print(f"  Access (raw): {acc_str}", flush=True)
        print(f"  Effort (raw): {eff_str}", flush=True)

        # Checkpoint after each scenario.
        pd.DataFrame(results).to_csv(output_path, index=False)

    results_df = pd.DataFrame(results)
    print(f"\nSaved results to {output_path}")

    print("\n=== Summary ===")
    print(f"Total rows: {len(results_df)}")
    for col, target in [("access", "[0, 2]"), ("effort", "[0, 1]")]:
        print(
            f"\n{col.capitalize()} (normalized, target {target}):"
            f"\n  Mean: {results_df[col].mean():.2f}, Std: {results_df[col].std():.2f}"
            f"\n  Range: [{results_df[col].min():.2f}, {results_df[col].max():.2f}]"
        )

    print("\nDone!")


# ==============================================================================
# Variable-length alternative scoring (for the no-alternatives-shown variant)
# ==============================================================================


def format_access_prompt_variable(vignette, action_texts):
    return build_user_prompt("access", vignette, action_texts)


def format_effort_prompt_variable(vignette, action_texts):
    return build_user_prompt("effort", vignette, action_texts)


def format_v_prompt_variable(vignette, state, action_texts):
    return build_user_prompt("v", vignette, action_texts, state=state)


# Variable-length system prompts are domain-specific; constructed at runtime
# inside score_alternatives_main once the --domain flag is known. The two
# constants below remain (food-domain) for any external caller that may have
# imported them — they are not used internally.
VARIABLE_ACCESS_SYSTEM_PROMPT = build_system_prompt("access", n_actions=None)
VARIABLE_EFFORT_SYSTEM_PROMPT = build_system_prompt("effort", n_actions=None)


def parse_action_response_variable(response_text, n_actions):
    """Parse JSON with action_0..action_{n-1} keys."""
    if response_text is None:
        return None
    js = find_json(response_text)
    if js is None:
        return None
    js = strip_leading_plus(js)
    try:
        ratings = json.loads(js)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Failed to parse JSON: {e}")
        return None
    out = {}
    for i in range(n_actions):
        key = f"action_{i}"
        if key not in ratings:
            return None
        try:
            out[key] = float(ratings[key])
        except (TypeError, ValueError):
            return None
    return out


def _max_tokens_for(n_actions):
    """Token budget that scales with the number of actions in a variable-length call."""
    return max(200, 40 * n_actions)


def score_alternatives_main(domain="food"):
    """Score access/effort for LM-generated alternatives, batched by scenario.

    Within each scenario, unique action strings (case-insensitive) are scored
    once via a single access-prompt + effort-prompt (each with NUM_RUNS runs),
    then features are broadcast back to every (observed_action, motivation,
    alt_idx) row that used that action text. This is ~8x fewer API calls than
    per-cell batching. Features depend on the full per-scenario action set
    rather than the cell-level subset, which is more internally consistent.

    Checkpoint behavior: after each scenario finishes, the accumulated results
    are flushed to the per-domain alternatives-features CSV. If that file
    already exists on startup, scenarios already present are skipped — so the
    script resumes from where it left off.
    """
    api_key = load_api_key()

    access_system_prompt = build_system_prompt("access", n_actions=None)
    effort_system_prompt = build_system_prompt("effort", n_actions=None)

    print(f"Loading scenarios and LM alternatives (domain={domain})...", flush=True)
    scenarios_df = load_scenarios(domain)
    alt_path = get_project_root() / "model" / "outputs" / _DOMAIN_PATHS[domain]["alternatives_input"]
    if not alt_path.exists():
        print(f"Error: {alt_path} not found. Run lm_generate_alternatives.py first.", flush=True)
        sys.exit(1)
    alt_df = pd.read_csv(alt_path)
    alt_df["action_norm"] = alt_df["action_text"].str.lower().str.strip()
    n_cells = alt_df.groupby(["scenario_label", "observed_action", "motivation"]).ngroups
    print(
        f"Loaded {len(alt_df)} alternatives across {n_cells} cells "
        f"({alt_df['action_norm'].nunique()} unique action strings)",
        flush=True,
    )

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / _DOMAIN_PATHS[domain]["alternatives_output"]

    # Resume: load any scenarios already written in a prior run
    results = []
    already_done_scenarios = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_done_scenarios = set(existing["scenario_label"].unique())
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with "
            f"{len(already_done_scenarios)} scenarios already scored — resuming.",
            flush=True,
        )

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    scenario_lookup = scenarios_df.set_index("scenario_label")["vignette"].to_dict()

    scenarios_in_data = sorted(alt_df["scenario_label"].unique())
    for sc_idx, scenario in enumerate(scenarios_in_data, start=1):
        if scenario in already_done_scenarios:
            print(f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} — already scored, skipping.", flush=True)
            continue

        sc_group = alt_df[alt_df["scenario_label"] == scenario]
        unique_df = sc_group.drop_duplicates("action_norm").reset_index(drop=True)
        unique_actions = unique_df["action_text"].tolist()
        unique_norms = unique_df["action_norm"].tolist()
        n_unique = len(unique_actions)
        vignette = scenario_lookup[scenario]

        print(
            f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} | "
            f"{len(sc_group)} rows | {n_unique} unique actions",
            flush=True,
        )

        if n_unique == 0:
            continue

        print("  scoring access (concurrent, structured)...", flush=True)
        access_ratings, access_failures = get_ratings_concurrent(
            client,
            access_system_prompt,
            format_access_prompt_variable(vignette, unique_actions),
            lambda t: parse_action_response_variable(t, n_unique),
            max_tokens=_max_tokens_for(n_unique),
            response_format=numeric_action_schema(n_unique),
            label=f"{scenario}/alts/access",
        )
        access_agg = aggregate_action_ratings(access_ratings, n_unique)

        print("  scoring effort (concurrent, structured)...", flush=True)
        effort_ratings, effort_failures = get_ratings_concurrent(
            client,
            effort_system_prompt,
            format_effort_prompt_variable(vignette, unique_actions),
            lambda t: parse_action_response_variable(t, n_unique),
            max_tokens=_max_tokens_for(n_unique),
            response_format=numeric_action_schema(n_unique),
            label=f"{scenario}/alts/effort",
        )
        effort_agg = aggregate_action_ratings(effort_ratings, n_unique)

        # Build lookup from action_norm → (access, effort, ...)
        feats_by_norm = {}
        for i, norm in enumerate(unique_norms):
            key = f"action_{i}"
            a_mean, a_std = access_agg[key]
            e_mean, e_std = effort_agg[key]
            feats_by_norm[norm] = {
                "access_raw": a_mean,
                "access_raw_std": a_std,
                "effort_raw": e_mean,
                "effort_raw_std": e_std,
                "access": normalize_access(a_mean) if not np.isnan(a_mean) else np.nan,
                "effort": normalize_effort(e_mean) if not np.isnan(e_mean) else np.nan,
                "n_runs_access": len(access_ratings),
                "n_runs_effort": len(effort_ratings),
                "n_failures_access": access_failures,
                "n_failures_effort": effort_failures,
            }

        # Emit one row per original (scenario, observed, motivation, alt_idx)
        new_rows = 0
        for _, row in sc_group.iterrows():
            f = feats_by_norm.get(row["action_norm"])
            if f is None:
                continue
            results.append({
                "scenario_label": scenario,
                "observed_action": row["observed_action"],
                "motivation": row["motivation"],
                "alt_idx": int(row["alt_idx"]),
                **f,
            })
            new_rows += 1

        # Checkpoint: flush accumulated results to disk after each scenario
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"  +{new_rows} rows | checkpoint written ({len(results)} total)", flush=True)

    print(f"\nFinal: saved {len(results)} alternative-feature rows to {output_path}", flush=True)


def score_v_alternatives_main(domain="food"):
    """Score signed-valence V for LM-generated alternatives, per scenario.

    For each scenario, dedupes alternative-action texts (case-insensitive),
    then asks the LM for V under each motivation state (`reward_high`,
    `reward_low`). 10 runs per (scenario, motivation_state). Output schema:
        scenario_label, observed_action, motivation, alt_idx, motivation_query,
        v_raw, v_raw_std, v, n_runs

    `motivation` is the motivation context the alternative was generated under
    (matches lm_alternatives.csv); `motivation_query` is the state we ask V
    against. We compute V for both motivation_query values for every alt so
    the desire-noalt observer (which infers motivation as latent) has both.

    Checkpoints to disk after each scenario; resumes from existing file.
    """
    api_key = load_api_key()

    v_system_prompt = build_system_prompt("v", n_actions=None)

    print(f"Loading scenarios and LM alternatives (domain={domain})...", flush=True)
    scenarios_df = load_scenarios(domain)
    alt_path = get_project_root() / "model" / "outputs" / _DOMAIN_PATHS[domain]["alternatives_input"]
    if not alt_path.exists():
        print(f"Error: {alt_path} not found. Run lm_generate_alternatives.py first.", flush=True)
        sys.exit(1)
    alt_df = pd.read_csv(alt_path)
    alt_df["action_norm"] = alt_df["action_text"].str.lower().str.strip()

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / _DOMAIN_PATHS[domain]["alternatives_v_output"]

    results = []
    already_done_scenarios = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_done_scenarios = set(existing["scenario_label"].unique())
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with "
            f"{len(already_done_scenarios)} scenarios already scored — resuming.",
            flush=True,
        )

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    scenario_lookup = scenarios_df.set_index("scenario_label").to_dict("index")

    scenarios_in_data = sorted(alt_df["scenario_label"].unique())
    for sc_idx, scenario in enumerate(scenarios_in_data, start=1):
        if scenario in already_done_scenarios:
            print(f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} — already scored, skipping.", flush=True)
            continue

        sc_group = alt_df[alt_df["scenario_label"] == scenario]
        unique_df = sc_group.drop_duplicates("action_norm").reset_index(drop=True)
        unique_actions = unique_df["action_text"].tolist()
        unique_norms = unique_df["action_norm"].tolist()
        n_unique = len(unique_actions)
        sc_meta = scenario_lookup[scenario]
        vignette = sc_meta["vignette"]

        print(
            f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} | "
            f"{len(sc_group)} rows | {n_unique} unique actions",
            flush=True,
        )
        if n_unique == 0:
            continue

        v_by_norm_by_query = {}
        for motivation_query in ("low", "high"):
            print(f"  scoring V (motivation_query={motivation_query}, concurrent, structured)...", flush=True)
            state = sc_meta[f"reward_{motivation_query}"]
            ratings, n_failures = get_ratings_concurrent(
                client,
                v_system_prompt,
                format_v_prompt_variable(vignette, state, unique_actions),
                lambda t: parse_action_response_variable(t, n_unique),
                max_tokens=_max_tokens_for(n_unique),
                response_format=numeric_action_schema(n_unique),
                label=f"{scenario}/alts/V[{motivation_query}]",
            )
            agg = aggregate_action_ratings(ratings, n_unique)
            for i, norm in enumerate(unique_norms):
                key = f"action_{i}"
                v_mean, v_std = agg[key]
                v_by_norm_by_query.setdefault(norm, {})[motivation_query] = {
                    "v_raw": v_mean,
                    "v_raw_std": v_std,
                    "v": normalize_v(v_mean) if not np.isnan(v_mean) else np.nan,
                    "n_runs": len(ratings),
                    "n_failures": n_failures,
                }

        # Emit one row per (scenario, observed, motivation, alt_idx, motivation_query).
        new_rows = 0
        for _, row in sc_group.iterrows():
            for motivation_query in ("low", "high"):
                f = v_by_norm_by_query.get(row["action_norm"], {}).get(motivation_query)
                if f is None:
                    continue
                results.append({
                    "scenario_label": scenario,
                    "observed_action": row["observed_action"],
                    "motivation": row["motivation"],
                    "alt_idx": int(row["alt_idx"]),
                    "motivation_query": motivation_query,
                    **f,
                })
                new_rows += 1

        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"  +{new_rows} rows | checkpoint written ({len(results)} total)", flush=True)

    print(f"\nFinal: saved {len(results)} alternative-V rows to {output_path}", flush=True)


def score_alternatives_relationship_main(domain="food"):
    """Score access/effort for relationship-conditioned LM-generated alternatives.

    Mirrors score_alternatives_main but reads from lm_alternatives_relationship.csv
    (keyed by relationship_condition instead of motivation) and writes to
    lm_alternatives_relationship_features.csv. Dedupe is per scenario across all
    (observed_action, relationship_condition, alt_idx) cells, identical to the
    motivation-keyed pass — access and effort are properties of the action and
    don't depend on the conditioning axis.
    """
    api_key = load_api_key()

    access_system_prompt = build_system_prompt("access", n_actions=None)
    effort_system_prompt = build_system_prompt("effort", n_actions=None)

    print(f"Loading scenarios and relationship-conditioned LM alternatives (domain={domain})...", flush=True)
    scenarios_df = load_scenarios(domain)
    alt_path = (
        get_project_root() / "model" / "outputs" / _DOMAIN_PATHS[domain]["alternatives_rel_input"]
    )
    if not alt_path.exists():
        print(
            f"Error: {alt_path} not found. Run "
            "lm_generate_alternatives.py --conditioning relationship first.",
            flush=True,
        )
        sys.exit(1)
    alt_df = pd.read_csv(alt_path)
    alt_df["action_norm"] = alt_df["action_text"].str.lower().str.strip()
    n_cells = alt_df.groupby(["scenario_label", "observed_action", "relationship_condition"]).ngroups
    print(
        f"Loaded {len(alt_df)} alternatives across {n_cells} cells "
        f"({alt_df['action_norm'].nunique()} unique action strings)",
        flush=True,
    )

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / _DOMAIN_PATHS[domain]["alternatives_rel_output"]

    results = []
    already_done_scenarios = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_done_scenarios = set(existing["scenario_label"].unique())
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with "
            f"{len(already_done_scenarios)} scenarios already scored — resuming.",
            flush=True,
        )

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    scenario_lookup = scenarios_df.set_index("scenario_label")["vignette"].to_dict()

    scenarios_in_data = sorted(alt_df["scenario_label"].unique())
    for sc_idx, scenario in enumerate(scenarios_in_data, start=1):
        if scenario in already_done_scenarios:
            print(f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} — already scored, skipping.", flush=True)
            continue

        sc_group = alt_df[alt_df["scenario_label"] == scenario]
        unique_df = sc_group.drop_duplicates("action_norm").reset_index(drop=True)
        unique_actions = unique_df["action_text"].tolist()
        unique_norms = unique_df["action_norm"].tolist()
        n_unique = len(unique_actions)
        vignette = scenario_lookup[scenario]

        print(
            f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} | "
            f"{len(sc_group)} rows | {n_unique} unique actions",
            flush=True,
        )
        if n_unique == 0:
            continue

        print("  scoring access (concurrent, structured)...", flush=True)
        access_ratings, access_failures = get_ratings_concurrent(
            client,
            access_system_prompt,
            format_access_prompt_variable(vignette, unique_actions),
            lambda t: parse_action_response_variable(t, n_unique),
            max_tokens=_max_tokens_for(n_unique),
            response_format=numeric_action_schema(n_unique),
            label=f"{scenario}/alts_rel/access",
        )
        access_agg = aggregate_action_ratings(access_ratings, n_unique)

        print("  scoring effort (concurrent, structured)...", flush=True)
        effort_ratings, effort_failures = get_ratings_concurrent(
            client,
            effort_system_prompt,
            format_effort_prompt_variable(vignette, unique_actions),
            lambda t: parse_action_response_variable(t, n_unique),
            max_tokens=_max_tokens_for(n_unique),
            response_format=numeric_action_schema(n_unique),
            label=f"{scenario}/alts_rel/effort",
        )
        effort_agg = aggregate_action_ratings(effort_ratings, n_unique)

        feats_by_norm = {}
        for i, norm in enumerate(unique_norms):
            key = f"action_{i}"
            a_mean, a_std = access_agg[key]
            e_mean, e_std = effort_agg[key]
            feats_by_norm[norm] = {
                "access_raw": a_mean,
                "access_raw_std": a_std,
                "effort_raw": e_mean,
                "effort_raw_std": e_std,
                "access": normalize_access(a_mean) if not np.isnan(a_mean) else np.nan,
                "effort": normalize_effort(e_mean) if not np.isnan(e_mean) else np.nan,
                "n_runs_access": len(access_ratings),
                "n_runs_effort": len(effort_ratings),
                "n_failures_access": access_failures,
                "n_failures_effort": effort_failures,
            }

        new_rows = 0
        for _, row in sc_group.iterrows():
            f = feats_by_norm.get(row["action_norm"])
            if f is None:
                continue
            results.append({
                "scenario_label": scenario,
                "observed_action": row["observed_action"],
                "relationship_condition": int(row["relationship_condition"]),
                "alt_idx": int(row["alt_idx"]),
                **f,
            })
            new_rows += 1

        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"  +{new_rows} rows | checkpoint written ({len(results)} total)", flush=True)

    print(f"\nFinal: saved {len(results)} alternative-feature rows to {output_path}", flush=True)


def score_v_alternatives_relationship_main(domain="food"):
    """Score signed-valence V for relationship-conditioned LM-generated alternatives.

    Mirrors score_v_alternatives_main but reads from lm_alternatives_relationship.csv
    (keyed by relationship_condition) and writes to lm_alternatives_relationship_v.csv.
    Each unique action is still scored under both motivation_query ∈ {low, high}
    because V depends on motivation regardless of the conditioning axis. Output:
        scenario_label, observed_action, relationship_condition, alt_idx,
        motivation_query, v_raw, v_raw_std, v, n_runs
    """
    api_key = load_api_key()

    v_system_prompt = build_system_prompt("v", n_actions=None)

    print(f"Loading scenarios and relationship-conditioned LM alternatives (domain={domain})...", flush=True)
    scenarios_df = load_scenarios(domain)
    alt_path = (
        get_project_root() / "model" / "outputs" / _DOMAIN_PATHS[domain]["alternatives_rel_input"]
    )
    if not alt_path.exists():
        print(
            f"Error: {alt_path} not found. Run "
            "lm_generate_alternatives.py --conditioning relationship first.",
            flush=True,
        )
        sys.exit(1)
    alt_df = pd.read_csv(alt_path)
    alt_df["action_norm"] = alt_df["action_text"].str.lower().str.strip()

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / _DOMAIN_PATHS[domain]["alternatives_rel_v_output"]

    results = []
    already_done_scenarios = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_done_scenarios = set(existing["scenario_label"].unique())
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with "
            f"{len(already_done_scenarios)} scenarios already scored — resuming.",
            flush=True,
        )

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    scenario_lookup = scenarios_df.set_index("scenario_label").to_dict("index")

    scenarios_in_data = sorted(alt_df["scenario_label"].unique())
    for sc_idx, scenario in enumerate(scenarios_in_data, start=1):
        if scenario in already_done_scenarios:
            print(f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} — already scored, skipping.", flush=True)
            continue

        sc_group = alt_df[alt_df["scenario_label"] == scenario]
        unique_df = sc_group.drop_duplicates("action_norm").reset_index(drop=True)
        unique_actions = unique_df["action_text"].tolist()
        unique_norms = unique_df["action_norm"].tolist()
        n_unique = len(unique_actions)
        sc_meta = scenario_lookup[scenario]
        vignette = sc_meta["vignette"]

        print(
            f"\n[{sc_idx}/{len(scenarios_in_data)}] {scenario} | "
            f"{len(sc_group)} rows | {n_unique} unique actions",
            flush=True,
        )
        if n_unique == 0:
            continue

        v_by_norm_by_query = {}
        for motivation_query in ("low", "high"):
            print(f"  scoring V (motivation_query={motivation_query}, concurrent, structured)...", flush=True)
            state = sc_meta[f"reward_{motivation_query}"]
            ratings, n_failures = get_ratings_concurrent(
                client,
                v_system_prompt,
                format_v_prompt_variable(vignette, state, unique_actions),
                lambda t: parse_action_response_variable(t, n_unique),
                max_tokens=_max_tokens_for(n_unique),
                response_format=numeric_action_schema(n_unique),
                label=f"{scenario}/alts_rel/V[{motivation_query}]",
            )
            agg = aggregate_action_ratings(ratings, n_unique)
            for i, norm in enumerate(unique_norms):
                key = f"action_{i}"
                v_mean, v_std = agg[key]
                v_by_norm_by_query.setdefault(norm, {})[motivation_query] = {
                    "v_raw": v_mean,
                    "v_raw_std": v_std,
                    "v": normalize_v(v_mean) if not np.isnan(v_mean) else np.nan,
                    "n_runs": len(ratings),
                    "n_failures": n_failures,
                }

        new_rows = 0
        for _, row in sc_group.iterrows():
            for motivation_query in ("low", "high"):
                f = v_by_norm_by_query.get(row["action_norm"], {}).get(motivation_query)
                if f is None:
                    continue
                results.append({
                    "scenario_label": scenario,
                    "observed_action": row["observed_action"],
                    "relationship_condition": int(row["relationship_condition"]),
                    "alt_idx": int(row["alt_idx"]),
                    "motivation_query": motivation_query,
                    **f,
                })
                new_rows += 1

        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"  +{new_rows} rows | checkpoint written ({len(results)} total)", flush=True)

    print(f"\nFinal: saved {len(results)} alternative-V rows to {output_path}", flush=True)


def score_v_main(domain="food"):
    """Generate signed-valence (V) ratings for each (scenario, action, motivation).

    Two LM passes per scenario — once with the scenario's reward_high paragraph
    as the actor's state, once with reward_low. Same 10-run averaging as
    access/effort. Output schema: scenario_label, action, motivation, v_raw,
    v_raw_std, v (normalized to [-1,+1]), n_runs, n_failures.
    """
    api_key = load_api_key()

    print(f"Loading scenarios (domain={domain})...")
    scenarios_df = load_scenarios(domain)
    print(f"Loaded {len(scenarios_df)} scenarios")

    print(f"\nInitializing Together AI client for {MODEL_ID}...")
    client = Together(api_key=api_key)

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / _DOMAIN_PATHS[domain]["v_output"]

    # Resume from existing CSV if present.
    results = []
    already_done_scenarios = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_done_scenarios = set(existing["scenario_label"].unique())
        results = existing.to_dict("records")
        print(
            f"Found existing {output_path.name} with "
            f"{len(already_done_scenarios)} scenarios already scored — resuming.",
            flush=True,
        )

    for idx, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        if scenario in already_done_scenarios:
            print(f"\n[{idx + 1}/{len(scenarios_df)}] {scenario} — already scored, skipping.", flush=True)
            continue

        print(f"\nProcessing {idx + 1}/{len(scenarios_df)}: {scenario}", flush=True)

        for motivation in ("low", "high"):
            print(f"  Getting V ratings (motivation={motivation}, concurrent, structured)...", flush=True)
            v_ratings, n_failures = get_ratings_concurrent(
                client,
                build_system_prompt("v", n_actions=4),
                format_v_prompt(row, motivation),
                parse_action_response,
                response_format=numeric_action_schema(4),
                label=f"{scenario}/V[{motivation}]",
            )
            v_agg = aggregate_action_ratings(v_ratings, n_actions=4)

            for action in range(4):
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

            v_str = [f"{v_agg[f'action_{i}'][0]:+.1f}" for i in range(4)]
            print(f"  V {motivation} (raw): {v_str}", flush=True)

        # Checkpoint after each scenario
        pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\nSaved results to {output_path}", flush=True)

    print("\n=== Summary ===")
    results_df = pd.DataFrame(results)
    print(f"Total rows: {len(results_df)}")
    for mot in ("low", "high"):
        sub = results_df[results_df["motivation"] == mot]
        print(
            f"\nV (normalized, motivation={mot}, target [-1, +1]):"
            f"\n  Mean: {sub['v'].mean():+.2f}, Std: {sub['v'].std():.2f}"
            f"\n  Range: [{sub['v'].min():+.2f}, {sub['v'].max():+.2f}]"
        )
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-alternatives",
        action="store_true",
        help="Score access/effort for LM-generated alternatives in lm_alternatives.csv",
    )
    parser.add_argument(
        "--feature",
        choices=(
            "access_effort",
            "v",
            "v_alternatives",
            "access_effort_alternatives_relationship",
            "v_alternatives_relationship",
        ),
        default="access_effort",
        help=(
            "Which feature(s) to elicit. 'access_effort' (default) generates "
            "the canonical access+effort tables. 'v' generates signed-valence "
            "ratings per (scenario, action, motivation). 'v_alternatives' "
            "generates V for motivation-conditioned LM-generated alternatives. "
            "'access_effort_alternatives_relationship' and "
            "'v_alternatives_relationship' do the equivalent for "
            "relationship-conditioned alternatives (requires "
            "lm_alternatives_relationship.csv). The feature flags are mutually "
            "exclusive with --score-alternatives."
        ),
    )
    parser.add_argument(
        "--domain",
        choices=("food", "nonfood"),
        default="food",
        help=(
            "Which scenario set to score. 'food' (default) uses scenarios.csv "
            "and writes to lm_scenario_params{,_alternatives_features}.csv; "
            "'nonfood' uses scenarios_nonfood.csv and writes to "
            "lm_scenario_params_nonfood{,_alternatives_features_nonfood}.csv."
        ),
    )
    args = parser.parse_args()
    if args.score_alternatives and args.feature != "access_effort":
        parser.error(f"--score-alternatives is incompatible with --feature {args.feature}")
    if args.score_alternatives:
        score_alternatives_main(args.domain)
    elif args.feature == "v":
        score_v_main(args.domain)
    elif args.feature == "v_alternatives":
        score_v_alternatives_main(args.domain)
    elif args.feature == "access_effort_alternatives_relationship":
        score_alternatives_relationship_main(args.domain)
    elif args.feature == "v_alternatives_relationship":
        score_v_alternatives_relationship_main(args.domain)
    else:
        main(args.domain)
