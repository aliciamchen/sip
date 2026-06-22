#!/usr/bin/env python3
"""
Per-run scoring of the K-run simulated-observer elicitation for the 3-action
inverse studies (1a food_inv_desire, 1b food_inv_joint_de, 2a food_inv_intimacy,
2b food_inv_joint_ie). Pick the study with --study.

This is the second step of the two-step elicitation. The first step
(generate_alternatives.py) elicited, for each (scenario × condition) cell, K
independent alternative sets — one per elicitation run, tagged with `run_id`.
Here, for each (scenario, run) we build a unified action list — the 3 canonical
actions from `scenarios.csv` plus that run's unique alternatives — and have the
LM rate it ONCE on risk, effort, and goal-satisfaction g (slot 0 = the observed
canonical action, slots 1..k = the run's alternatives, all on one comparative
scale). There is NO inner rating-averaging: each run is a single scoring pass, so
the run-to-run spread of both alternatives AND feature scores becomes part of the
model's predicted distribution (the simulated-observer mixture).

Design choices carried over from the single-run pipeline:
  1. Canonical + alts scored together (one comparative reference frame).
  2. Risk is effort-marginal (vignette only, no effort paragraph), broadcast
     across effort_condition.
  3. Effort is effort-conditional; g is desire-free. Neither shows intimacy.
  4. Actions are presented to the LM in a per-call randomized order (the merged
     list is otherwise always no_share / low_risk_share / high_risk_share /
     alts...), and the ratings are mapped back to canonical order before storage,
     so the canonical actions don't sit in fixed slots and LLM position/primacy
     bias can't systematically favor the observed action (slot 0 downstream). The
     order is deterministic given (scenario, run, feature) — see `_perm_for`.

Given-magnitude scalars are scored PER RUN (folded into each lm_runs.jsonl
record, alongside the action features) rather than once run-independently — so the
run-to-run spread of the given magnitudes joins the same simulated-observer
mixture as the alternatives and the feature scores:
  - given-desire studies (2a, 2b): a per-(scenario, desire_condition) desire
    scalar, scored inside each (scenario, run) unit → each record's `desire`.
  - given-relationship studies (1a, 1b): a per-level intimacy scalar, rated from
    the DE-ANCHORED relationship descriptors (rating the anchored descriptor would
    be circular). Scenario-independent → 4 values scored once per run and reused
    across that run's scenarios → each record's `intimacy`.

Output (one folder per study, outputs/lm/<slug>/):
  - lm_runs.jsonl — one record per (run_id, cell), each carrying the run's scored
    actions (slot 0 canonical + slots 1..k alternatives) and the run's given
    magnitude (`desire` for 2a/2b, `intimacy` for 1a/1b). Consumed by the run-axis
    table loaders in model/tables.py.

Usage:
    uv run python model/lm/score_merged.py --study food_inv_desire
    # K runs / temperature / concurrency via env: K_RUNS, ALT_T, --scenario-workers

Requires:
    - TOGETHER_API_KEY in env or .env
    - outputs/lm/<slug>/lm_alternatives.jsonl produced by
      generate_alternatives.py --study <slug> (carrying a run_id field)
"""

import argparse
import hashlib
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from together import Together

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _features_dispatcher import (
    _max_tokens_for,
    format_effort_prompt_variable,
    format_g_prompt_variable,
    format_risk_prompt_variable,
    normalize_effort,
    normalize_g,
    normalize_risk,
    numeric_desire_schema,
    numeric_intimacy_schema,
    parse_action_response_variable,
    parse_desire_response,
    parse_intimacy_response,
)
from client import (
    MODEL_ID,
    TEMPERATURE,
    aggregate_action_ratings,
    get_ratings_concurrent,
    load_api_key,
    numeric_action_schema,
    write_run_manifest,
)
from prompts import (
    DESIRE_SYSTEM_PROMPT,
    INTIMACY_SYSTEM_PROMPT,
    RELATIONSHIP_DESCRIPTORS,
    desire_user_prompt,
    relationship_user_prompt,
)
from prompts import system_prompt as build_system_prompt

N_ACTIONS = 3
CANONICAL_ACTIONS = ["no_share", "low_risk_share", "high_risk_share"]
EFFORT_CONDITIONS = ["low", "high"]
DESIRES = ["low", "high"]
INTIMACY_LEVELS = ["max_formal", "somewhat_formal", "somewhat_intimate", "max_intimate"]
# Levels for each generation-cell condition column, used to enumerate cells.
_LEVELS = {
    "desire_condition": DESIRES,
    "effort_condition": EFFORT_CONDITIONS,
    "intimacy_condition": INTIMACY_LEVELS,
}

SCENARIO_WORKERS = 4

# Per-study config. `cell_cols` are the generation-cell key columns in the
# alternatives JSONL (besides scenario_label + observed_action + run_id).
# `effort_inferred` flags studies whose observer infers effort: their generation
# cell does NOT include effort, so each cell emits a record for BOTH effort
# conditions (effort is a feature axis; risk/g repeat). `desire_given` /
# `relationship_given` select which per-run given-magnitude scalar each record
# carries (`desire` for 2a/2b, `intimacy` for 1a/1b).
_STUDY_CONFIG = {
    "food_inv_desire": {
        "cell_cols": ("effort_condition", "intimacy_condition"),
        "effort_inferred": False,
        "desire_given": False,
        "relationship_given": True,
    },
    "food_inv_joint_de": {
        "cell_cols": ("intimacy_condition",),
        "effort_inferred": True,
        "desire_given": False,
        "relationship_given": True,
    },
    "food_inv_intimacy": {
        "cell_cols": ("desire_condition", "effort_condition"),
        "effort_inferred": False,
        "desire_given": True,
        "relationship_given": False,
    },
    "food_inv_joint_ie": {
        "cell_cols": ("desire_condition",),
        "effort_inferred": True,
        "desire_given": True,
        "relationship_given": False,
    },
}

# Base-model override (the `--base` mode). Scores the relationship-free alternative
# set written by `generate_alternatives.py --base`. Feature scoring (risk/effort/g)
# is already relationship-free, so the only changes are: read/write the base files,
# drop intimacy_condition from the cell grid, and skip the per-run intimacy scalar
# (the base utility has no intimacy term). Mirrors _BASE_OVERRIDE in
# generate_alternatives.py; given-relationship studies only.
_BASE_OVERRIDE = {
    "food_inv_desire": {
        "cell_cols": ("effort_condition",),
        "relationship_given": False,
        "alternatives": "lm_alternatives_base.jsonl",
        "runs": "lm_runs_base.jsonl",
    },
    # 1b later: "food_inv_joint_de": {"cell_cols": (), "relationship_given": False,
    #     "alternatives": "lm_alternatives_base.jsonl", "runs": "lm_runs_base.jsonl"},
}


def _norm(text):
    return text.lower().strip()


def _build_merged_actions(scenario_row, alt_rows_for_run):
    """Unified action list for one (scenario, run): the 3 canonical actions
    followed by the run's unique alternative texts (deduped case-insensitively
    and excluding any alt matching a canonical text). Returns
    (merged_action_texts, canonical_norms, alt_norms_in_order)."""
    canonical_actions = [scenario_row[c] for c in CANONICAL_ACTIONS]
    canonical_norms = [_norm(a) for a in canonical_actions]
    canonical_norm_set = set(canonical_norms)

    alt_norms_in_order, alt_texts_unique, seen = [], [], set()
    for _, r in alt_rows_for_run.iterrows():
        norm = _norm(r["action_text"])
        if norm in canonical_norm_set or norm in seen:
            continue
        seen.add(norm)
        alt_norms_in_order.append(norm)
        alt_texts_unique.append(r["action_text"])

    merged = canonical_actions + alt_texts_unique
    return merged, canonical_norms, alt_norms_in_order


def _score_one_call(client, system_prompt, user_prompt, n_actions, label, seed=None):
    """Single LM scoring pass (num_runs=1, no inner averaging) returning per-action
    ratings. The K elicitation runs are the variation axis, not repeated calls.
    ``seed`` pins the call for reproducible re-scoring."""
    ratings, n_failures = get_ratings_concurrent(
        client,
        system_prompt,
        user_prompt,
        lambda t: parse_action_response_variable(t, n_actions),
        num_runs=1,
        max_tokens=_max_tokens_for(n_actions),
        response_format=numeric_action_schema(n_actions),
        label=label,
        seed=seed,
    )
    agg = aggregate_action_ratings(ratings, n_actions)
    return agg, n_failures


def _seed_for(scenario, run_id, tag):
    """Stable 64-bit seed for one scoring call, from (scenario, run_id, feature
    tag). SHA-256 (not Python's salted ``hash``) so it's reproducible across
    reruns/processes; varying ``tag`` per feature decorrelates uses."""
    key = f"{scenario}|{int(run_id)}|{tag}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _lm_seed_for(scenario, run_id, tag):
    """The same stable seed masked to a non-negative 31-bit int for the Together
    ``seed`` parameter, so re-running scoring reproduces the LM's feature ratings
    (best-effort per the docs) while each (scenario, run, feature) stays distinct."""
    return _seed_for(scenario, run_id, tag) & 0x7FFFFFFF


def _perm_for(scenario, run_id, tag, n):
    """Deterministic permutation of ``range(n)`` for one scoring call, seeded
    from (scenario, run_id, feature tag).

    The presented action order is randomized per call so the canonical actions
    don't always occupy the same slots (the merged list is otherwise always
    no_share / low_risk_share / high_risk_share / alternatives...). LLM raters
    have position/primacy biases, so a fixed order would bias the canonical
    actions' features — and the fit/CV slice slot 0 (the observed action), so any
    such bias would land squarely on the modeled quantity. Seeding from a stable
    SHA-256 hash (not Python's salted ``hash``) keeps the order reproducible
    across reruns/processes; varying ``tag`` per feature (risk / effort_low /
    effort_high / g) decorrelates positions across features so no action sits in
    a fixed slot across all of them."""
    return np.random.default_rng(_seed_for(scenario, run_id, tag)).permutation(n)


def _score_feature_shuffled(
    client,
    system_prompt,
    build_user_prompt,
    merged,
    all_norms,
    normalize_fn,
    scenario,
    run_id,
    tag,
):
    """Score one feature on the merged action list, presenting the actions to the
    LM in a per-call randomized order (see ``_perm_for``) and mapping the ratings
    back to the canonical (``all_norms``) order.

    ``build_user_prompt`` takes the ordered list of action texts to present and
    returns the user prompt. Returns dict[norm] -> normalized [0,1] value (NaN if
    the call failed for that action)."""
    n = len(merged)
    perm = _perm_for(scenario, run_id, tag, n)
    presented = [merged[p] for p in perm]
    agg, _ = _score_one_call(
        client,
        system_prompt,
        build_user_prompt(presented),
        n,
        label=f"{scenario}/{tag}",
        seed=_lm_seed_for(scenario, run_id, tag),
    )
    out = {}
    for presented_slot, orig in enumerate(perm):
        v = agg[f"action_{presented_slot}"][0]
        out[all_norms[orig]] = normalize_fn(v) if not np.isnan(v) else np.nan
    return out


def _score_actions(client, scenario_row, alt_rows_for_run, system_prompts, run_id):
    """Score risk / effort / g on one (scenario, run)'s merged action list.

    Each feature is scored in a single LM pass, with the actions presented to the
    LM in a randomized order (deterministic given scenario/run/feature; see
    ``_perm_for``) and the ratings mapped back to the canonical order, so the
    canonical actions are not always shown in the same slots.

    Returns {merged_actions, canonical_norms, alt_norms_in_order, risk, effort, g}
    where risk/g are dict[norm] -> normalized [0,1] value (single value per
    action) and effort is dict[(effort_cond, norm)] -> normalized [0,1]."""
    scenario = scenario_row["scenario_label"]
    vignette = scenario_row["vignette"]
    merged, canonical_norms, alt_norms = _build_merged_actions(
        scenario_row, alt_rows_for_run
    )
    all_norms = canonical_norms + alt_norms

    # risk: one prompt, vignette only (effort-marginal).
    risk = _score_feature_shuffled(
        client,
        system_prompts["risk"],
        lambda acts: format_risk_prompt_variable(vignette, acts),
        merged,
        all_norms,
        normalize_risk,
        scenario,
        run_id,
        "risk",
    )

    # effort: one prompt per effort_condition (effort paragraph appended).
    effort = {}
    for ec in EFFORT_CONDITIONS:
        vignette_eff = f"{vignette} {scenario_row[f'low_risk_share_effort_{ec}']}"
        eff_by_norm = _score_feature_shuffled(
            client,
            system_prompts["effort"],
            lambda acts, v=vignette_eff: format_effort_prompt_variable(v, acts),
            merged,
            all_norms,
            normalize_effort,
            scenario,
            run_id,
            f"effort_{ec}",
        )
        for norm, val in eff_by_norm.items():
            effort[(ec, norm)] = val

    # g: one prompt, desire-free goal-satisfaction.
    g = _score_feature_shuffled(
        client,
        system_prompts["g"],
        lambda acts: format_g_prompt_variable(
            vignette, acts, scenario_row["desire_object"]
        ),
        merged,
        all_norms,
        normalize_g,
        scenario,
        run_id,
        "g",
    )

    return {
        "merged_actions": merged,
        "canonical_norms": canonical_norms,
        "alt_norms_in_order": alt_norms,
        "risk": risk,
        "effort": effort,
        "g": g,
    }


def _build_run_records(
    study,
    scenario_row,
    run_id,
    run_alt_rows,
    scored,
    cfg,
    given_desire=None,
    given_relationship=None,
):
    """Assemble the per-(run, cell) JSONL records for one (scenario, run).

    Enumerates the full generation cell grid (observed_action × generation
    condition levels) so every cell gets a record with its canonical slot 0,
    even cells whose run produced zero alternatives. For effort-inferred studies
    each cell emits one record per effort_condition (the effort feature axis).

    `given_desire` (dict desire_condition -> scalar, this scenario's run) and
    `given_relationship` (dict level -> scalar, this run) are the per-run given
    magnitudes, folded into each record by its condition: `desire` for 2a/2b,
    `intimacy` for 1a/1b.
    """
    scenario = scenario_row["scenario_label"]
    risk, effort, g = scored["risk"], scored["effort"], scored["g"]
    cell_cols = list(cfg["cell_cols"])
    level_lists = [_LEVELS[c] for c in cell_cols]

    records = []
    for observed_action in CANONICAL_ACTIONS:
        obs_norm = _norm(scenario_row[observed_action])
        for cond_values in itertools.product(*level_lists) if level_lists else [()]:
            cond = dict(zip(cell_cols, cond_values))
            # The run's alternatives for this exact cell, in alt_idx order.
            mask = run_alt_rows["observed_action"] == observed_action
            for c, v in cond.items():
                mask = mask & (run_alt_rows[c] == v)
            cell_alts = run_alt_rows[mask].sort_values("alt_idx")

            record_effort_conds = (
                EFFORT_CONDITIONS
                if cfg["effort_inferred"]
                else [cond["effort_condition"]]
            )
            for ec in record_effort_conds:
                actions = [
                    {
                        "slot": 0,
                        "is_canonical": True,
                        "action_text": scenario_row[observed_action],
                        "risk": _f(risk.get(obs_norm)),
                        "effort": _f(effort.get((ec, obs_norm))),
                        "g": _f(g.get(obs_norm)),
                    }
                ]
                for _, alt in cell_alts.iterrows():
                    a_norm = _norm(alt["action_text"])
                    actions.append(
                        {
                            "slot": int(alt["alt_idx"]) + 1,
                            "alt_idx": int(alt["alt_idx"]),
                            "is_canonical": False,
                            "action_text": alt["action_text"],
                            "risk": _f(risk.get(a_norm)),
                            "effort": _f(effort.get((ec, a_norm))),
                            "g": _f(g.get(a_norm)),
                        }
                    )
                record = {
                    "run_id": int(run_id),
                    "scenario_label": scenario,
                    "observed_action": observed_action,
                }
                record.update(cond)
                record["effort_condition"] = ec  # loader keys canon on this
                # Per-run given magnitude, folded in by the cell's condition.
                if given_desire is not None:
                    record["desire"] = given_desire.get(cond["desire_condition"])
                if given_relationship is not None:
                    record["intimacy"] = given_relationship.get(
                        cond["intimacy_condition"]
                    )
                record["actions"] = actions
                records.append(record)
    return records


def _f(x):
    """JSON-safe float (NaN -> None)."""
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else x


def _rate_desire_for_scenario(client, scenario_row, run_id):
    """Per-desire_condition desire scalar in [0, 1] for one scenario, scored in
    one run's pass (folded into that (scenario, run)'s records). The K runs are
    the variation axis, so this is a single scoring pass per (scenario, condition,
    run) at the elicitation temperature; ``run_id`` seeds it for reproducibility."""
    scenario = scenario_row["scenario_label"]
    out = {}
    for dc in DESIRES:
        ratings, _ = get_ratings_concurrent(
            client,
            DESIRE_SYSTEM_PROMPT,
            desire_user_prompt(
                scenario_row["vignette"],
                scenario_row[f"desire_{dc}"],
                scenario_row["desire_object"],
            ),
            parse_desire_response,
            num_runs=1,
            max_tokens=64,
            response_format=numeric_desire_schema(),
            label=f"{scenario}/desire[{dc}]",
            seed=_lm_seed_for(scenario, run_id, f"desire_{dc}"),
        )
        out[dc] = (float(ratings[0]) / 100.0) if ratings else None
    return out


def _rate_relationship_values(client, run_id):
    """Per-level intimacy scalar in [0, 1] from the de-anchored relationship
    descriptors (scenario-independent → 4 values). Single scoring pass per level,
    per run (the K runs are the variation axis); ``run_id`` seeds it so re-runs
    reproduce."""
    out = {}
    for level in INTIMACY_LEVELS:
        ratings, _ = get_ratings_concurrent(
            client,
            INTIMACY_SYSTEM_PROMPT,
            relationship_user_prompt(RELATIONSHIP_DESCRIPTORS[level]),
            parse_intimacy_response,
            num_runs=1,
            max_tokens=64,
            response_format=numeric_intimacy_schema(),
            label=f"relationship[{level}]",
            seed=_lm_seed_for("__relationship__", run_id, f"intimacy_{level}"),
        )
        out[level] = (float(ratings[0]) / 100.0) if ratings else None
    return out


def main(study, scenario_workers=SCENARIO_WORKERS, base=False):
    if study not in _STUDY_CONFIG:
        raise SystemExit(
            f"Unknown study: {study!r}. Supported: {sorted(_STUDY_CONFIG.keys())}"
        )
    cfg = dict(_STUDY_CONFIG[study])
    if base:
        if study not in _BASE_OVERRIDE:
            raise SystemExit(
                f"--base is only defined for {sorted(_BASE_OVERRIDE)}; "
                f"{study!r} has no relationship paragraph."
            )
        cfg.update(_BASE_OVERRIDE[study])
    api_key = load_api_key()

    scenarios_path = get_project_root() / "experiments" / "scenarios.csv"
    study_dir = get_project_root() / "model" / "outputs" / "lm" / study
    study_dir.mkdir(parents=True, exist_ok=True)
    alts_path = study_dir / cfg.get("alternatives", "lm_alternatives.jsonl")
    runs_path = study_dir / cfg.get("runs", "lm_runs.jsonl")
    if not alts_path.exists():
        raise SystemExit(
            f"Alternatives JSONL not found at {alts_path}. Run "
            f"model/lm/generate_alternatives.py --study {study}"
            f"{' --base' if base else ''} first."
        )

    scenarios_df = pd.read_csv(scenarios_path).set_index("scenario_label", drop=False)
    alts_df = pd.read_json(alts_path, lines=True)
    if "run_id" not in alts_df.columns:
        raise SystemExit(
            f"{alts_path} has no run_id field — re-run generate_alternatives.py "
            "with the K-run pipeline first."
        )

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    system_prompts = {
        "risk": build_system_prompt("risk"),
        "effort": build_system_prompt("effort"),
        "g": build_system_prompt("g"),
    }

    # Resume: skip (scenario, run) units already written to lm_runs.jsonl.
    done_units = set()
    existing_records = []
    if runs_path.exists():
        with open(runs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing_records.append(rec)
                done_units.add((rec["scenario_label"], int(rec["run_id"])))
        print(
            f"Found {len(done_units)} (scenario, run) units already scored — resuming.",
            flush=True,
        )

    run_ids = sorted(alts_df["run_id"].dropna().unique().astype(int))

    # Per-run given magnitudes (folded into each record). Relationship intimacy
    # (1a/1b) is scenario-independent, so it's scored ONCE per run and reused
    # across that run's scenario units; on resume, runs that already have records
    # are reconstructed from them (any completed (scenario, run) unit covers all 4
    # levels) so they aren't re-scored with different values. Desire (2a/2b) is
    # per (scenario, desire_condition) and is scored inside each unit below.
    relationship_by_run = {}
    if cfg["relationship_given"]:
        for rec in existing_records:
            if rec.get("intimacy") is not None:
                relationship_by_run.setdefault(int(rec["run_id"]), {})[
                    rec["intimacy_condition"]
                ] = float(rec["intimacy"])
        for rid in run_ids:
            have = relationship_by_run.get(int(rid), {})
            if not all(lvl in have for lvl in INTIMACY_LEVELS):
                print(
                    f"Rating per-level relationship intimacy (de-anchored), run {rid}...",
                    flush=True,
                )
                relationship_by_run[int(rid)] = _rate_relationship_values(client, rid)

    units = [
        (s, r)
        for s in scenarios_df.index
        for r in run_ids
        if (s, int(r)) not in done_units
    ]
    print(
        f"\n{len(units)} (scenario, run) units to score "
        f"(scenarios={len(scenarios_df)}, runs={len(run_ids)}; "
        f"{scenario_workers} concurrent).",
        flush=True,
    )

    def _process_unit(scenario, run_id):
        scenario_row = scenarios_df.loc[scenario]
        run_alt_rows = alts_df[
            (alts_df["scenario_label"] == scenario) & (alts_df["run_id"] == run_id)
        ]
        scored = _score_actions(
            client, scenario_row, run_alt_rows, system_prompts, run_id
        )
        given_desire = (
            _rate_desire_for_scenario(client, scenario_row, run_id)
            if cfg["desire_given"]
            else None
        )
        given_relationship = (
            relationship_by_run.get(int(run_id)) if cfg["relationship_given"] else None
        )
        return _build_run_records(
            study,
            scenario_row,
            run_id,
            run_alt_rows,
            scored,
            cfg,
            given_desire=given_desire,
            given_relationship=given_relationship,
        )

    all_records = list(existing_records)
    done_count = 0
    with ThreadPoolExecutor(max_workers=max(1, scenario_workers)) as ex:
        futures = {ex.submit(_process_unit, s, r): (s, r) for s, r in units}
        for fut in as_completed(futures):
            s, r = futures[fut]
            all_records.extend(fut.result())
            done_count += 1
            # Checkpoint: rewrite the full JSONL (append-only set of records).
            with open(runs_path, "w") as f:
                for rec in all_records:
                    f.write(json.dumps(rec) + "\n")
            print(
                f"  [{done_count}/{len(units)}] {s} / run {r} done — checkpointed",
                flush=True,
            )

    manifest_path = write_run_manifest(
        runs_path,
        stage="score_merged",
        study=study,
        extra={
            "k_runs": len(run_ids),
            "score_temperature": TEMPERATURE,
            "n_scenarios": len(scenarios_df),
            "n_records": len(all_records),
        },
    )

    print("\n=== Done ===")
    print(f"  {runs_path.name}  ({len(all_records)} records)")
    print(f"  {manifest_path.name}  (provenance manifest)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=tuple(_STUDY_CONFIG.keys()),
        default="food_inv_desire",
    )
    parser.add_argument(
        "--scenario-workers",
        type=int,
        default=SCENARIO_WORKERS,
        help="How many (scenario, run) units to score concurrently.",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Base-model mode: score the relationship-free alternative set "
        "(lm_alternatives_base.jsonl) into lm_runs_base.jsonl; skips the per-run "
        "intimacy scalar. Given-relationship studies only.",
    )
    args = parser.parse_args()
    main(args.study, scenario_workers=args.scenario_workers, base=args.base)
