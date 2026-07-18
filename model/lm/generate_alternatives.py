#!/usr/bin/env python3
"""
Generate LM counterfactual alternative actions for the 3-action inverse experiments.

The 3-act observers (Studies 1a, 1b, 2a, 2b on the food set; 3a, 3b on the
nonfood set) softmax over a per-cell action space {observed_action} ∪
generated_alts rather than the fixed 3-action set.
Alternatives are conditioned on what the human participant sees in the trial —
i.e. observer-visible variables only (`feedback_llm_as_participant.md`).

Per-study conditioning tuples (only the variables the observer actually sees,
besides scenario + observed_action), and the resulting cell count:
    food_inv_desire      (1a): (effort, intimacy)  → 16 × 3 × 2 × 4 = 384 cells
    food_inv_joint_de    (1b): (intimacy,)         → 16 × 3 × 4     = 192 cells
    food_inv_intimacy    (2a): (desire, effort)    → 16 × 3 × 2 × 2 = 192 cells
    food_inv_joint_ie    (2b): (desire,)           → 16 × 3 × 2     =  96 cells
    nonfood_inv_joint_de (3a): (intimacy,)         → 16 × 3 × 4     = 192 cells
    nonfood_inv_joint_ie (3b): (desire,)           → 16 × 3 × 2     =  96 cells

One LM elicitation per cell (parse-retries up to MAX_PARSE_RETRIES); each call
returns a JSON array of variable-length alternatives.

Output (one folder per study slug):
    --study food_inv_desire  →  model/outputs/lm/food_inv_desire/lm_alternatives.jsonl
        (one JSON record per generated alternative — the stage-1 texts; the
        feature scores are added by score_merged.py into lm_runs.jsonl)

Usage:
    uv run python model/lm/generate_alternatives.py --study food_inv_desire

Requires:
    - TOGETHER_API_KEY environment variable or in .env file
    - `together` Python package
"""

import argparse
import hashlib
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
from client import (
    MODEL_ID,
    guard_resume_prompt_mismatch,
    load_api_key,
    read_jsonl_checked,
    write_jsonl_atomic,
    write_run_manifest,
)
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
# written to the output JSONL; the downstream merged-scoring + padded-table loader
# key on these.
#   1a desire            — visible: effort, intimacy  (infers desire)
#   1b joint_de          — visible: intimacy          (infers desire + effort)
#   2a intimacy          — visible: desire, effort    (infers intimacy)
#   2b joint_ie          — visible: desire            (infers intimacy + effort)
#   3a nonfood joint_de  — visible: intimacy          (infers desire + effort)
#   3b nonfood joint_ie  — visible: desire            (infers intimacy + effort)
_STUDY_CONFIG = {
    "food_inv_desire": {
        "scenarios": "scenarios.csv",
        "output": "lm_alternatives.jsonl",
        "show": ("effort", "intimacy"),
        "cell_cols": ("effort_condition", "intimacy_condition"),
    },
    "food_inv_joint_de": {
        "scenarios": "scenarios.csv",
        "output": "lm_alternatives.jsonl",
        "show": ("intimacy",),
        "cell_cols": ("intimacy_condition",),
    },
    "food_inv_intimacy": {
        "scenarios": "scenarios.csv",
        "output": "lm_alternatives.jsonl",
        "show": ("desire", "effort"),
        "cell_cols": ("desire_condition", "effort_condition"),
    },
    "food_inv_joint_ie": {
        "scenarios": "scenarios.csv",
        "output": "lm_alternatives.jsonl",
        "show": ("desire",),
        "cell_cols": ("desire_condition",),
    },
    # Study 3 (nonfood stimulus set): 3a mirrors 1b (relationship given, infers
    # desire + effort), 3b mirrors 2b (desire given, infers intimacy + effort).
    "nonfood_inv_joint_de": {
        "scenarios": "scenarios_nonfood.csv",
        "output": "lm_alternatives.jsonl",
        "show": ("intimacy",),
        "cell_cols": ("intimacy_condition",),
    },
    "nonfood_inv_joint_ie": {
        "scenarios": "scenarios_nonfood.csv",
        "output": "lm_alternatives.jsonl",
        "show": ("desire",),
        "cell_cols": ("desire_condition",),
    },
}

# Base-model override (the `--base` mode). The `base` ablation has no intimacy
# term, so its alternative set should NOT be conditioned on the relationship — we
# drop intimacy from `show`/`cell_cols` and write to a separate file that coexists
# with the relationship-conditioned `lm_alternatives.jsonl` used by full/
# discomfort_only. Only the given-relationship studies (1a/1b) have a relationship
# paragraph to drop; 2a/2b infer intimacy and never show one, so they're excluded.
#   1a desire   — (effort, intimacy) -> (effort,)   = 16×3×2   = 96 cells
_BASE_OVERRIDE = {
    "food_inv_desire": {
        "output": "lm_alternatives_base.jsonl",
        "show": ("effort",),
        "cell_cols": ("effort_condition",),
    },
    # 1b: effort is inferred, so base generation shows neither effort nor
    # intimacy — 16 scenarios × 3 observed actions = 48 cells.
    "food_inv_joint_de": {
        "output": "lm_alternatives_base.jsonl",
        "show": (),
        "cell_cols": (),
    },
    # 3a mirrors 1b on the nonfood set — 48 cells.
    "nonfood_inv_joint_de": {
        "output": "lm_alternatives_base.jsonl",
        "show": (),
        "cell_cols": (),
    },
}


def _output_path_for(study, base):
    """Alternatives output path for one (study, base). Importable and API-free so
    the compliance test can pin the naming."""
    cfg = dict(_STUDY_CONFIG[study])
    if base:
        cfg.update(_BASE_OVERRIDE[study])
    output_dir = get_project_root() / "model" / "outputs" / "lm" / study
    return output_dir / cfg["output"]


def _latent_awareness_kwargs(study, row):
    """Per-study epistemic kwargs threaded into every cell's user prompt: make the
    LM aware of exactly the latent(s) the study infers, so the generated set spans
    their range (mirroring the participant, who has seen the DV questions and so
    knows which quantities the trial leaves open). Always applied. Keyed on the
    base study slug only, so a study's `--base` variant gets the same kwargs as
    its standard run.

      - effort-inferred (1b/2b/3a/3b): the two effort paragraphs as hypotheses
      - desire-inferred (1a/1b/3a):    the desire object flagged unknown-magnitude
      - intimacy-inferred (2a/2b/3b):  the relationship flagged unknown
    """
    kwargs = {}
    if study in (
        "food_inv_joint_de",
        "nonfood_inv_joint_de",
        "food_inv_joint_ie",
        "nonfood_inv_joint_ie",
    ):
        kwargs["effort_hypotheses"] = (
            row["low_risk_share_effort_low"],
            row["low_risk_share_effort_high"],
        )
    if study in ("food_inv_desire", "food_inv_joint_de", "nonfood_inv_joint_de"):
        kwargs["unknown_desire_object"] = row["desire_object"]
    if study in ("food_inv_intimacy", "food_inv_joint_ie", "nonfood_inv_joint_ie"):
        kwargs["unknown_intimacy"] = True
    return kwargs


ACTION_COLS = ["no_share", "low_risk_share", "high_risk_share"]
DESIRE_LEVELS = ["low", "high"]
EFFORT_LEVELS = ["low", "high"]
# Intimacy is a purely verbal manipulation: levels are identified by slug
# (ascending, formal -> intimate), never by a numeric code.
INTIMACY_LEVELS = ["max_formal", "somewhat_formal", "somewhat_intimate", "max_intimate"]


def load_scenarios(study):
    cfg = _STUDY_CONFIG[study]
    scenarios_path = get_project_root() / "experiments" / cfg["scenarios"]
    return pd.read_csv(scenarios_path)


def _empty_units_path(output_path):
    """Sidecar path recording (cell, run) units whose elicitation VALIDLY
    returned zero alternatives (lm_alternatives.jsonl ->
    lm_alternatives.empty_units.jsonl). Such units write no rows to the main
    JSONL, so without this record they'd be indistinguishable from
    never-attempted and re-elicited (re-paid) on every resume. Format: one JSON
    object per line with the unit key fields {scenario_label, observed_action,
    *cell_cols, run_id}. Resume unions these with the row-derived done units;
    no consumer of lm_alternatives.jsonl reads this file."""
    output_path = Path(output_path)
    return output_path.with_name(output_path.stem + ".empty_units.jsonl")


def _sorted_rows(rows, cell_cols):
    """Stable row order (cell key, run, alt index) so writes are
    byte-reproducible regardless of thread-completion order — score_merged.py
    derives each run's merged action order from row order, and no consumer
    depends on arrival order beyond per-cell grouping."""

    def key(r):
        return (
            r["scenario_label"],
            r["observed_action"],
            *(r[c] for c in cell_cols),
            int(r.get("run_id", -1)),
            int(r.get("alt_idx", -1)),
        )

    return sorted(rows, key=key)


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
    """Deterministic per-(cell, run) seed so reruns reproduce. Hashes the cell
    key with SHA-256 (Python's builtin ``hash`` is salted per process, so it
    would give every fresh invocation — including resumes — different seeds),
    masked to the non-negative 31-bit range the Together ``seed`` param takes.
    Mirrors ``_seed_for`` in score_merged.py."""
    key = "|".join(str(part) for part in _cell_key(cell, cell_cols, run_id)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & 0x7FFFFFFF


def _build_cells(scenarios_df, cfg, study):
    """Enumerate cells for one study, iterating scenario × observed_action over
    only the observer-visible conditioning axes (per cfg['show']). Returns dicts
    with scenario_label, observed_action, the study's cell_cols, and user_prompt
    (built with only the visible condition paragraphs). The per-study
    latent-awareness blocks (constant within a study) are threaded into every
    cell's user prompt via `_latent_awareness_kwargs`."""
    show = cfg["show"]
    desire_levels = DESIRE_LEVELS if "desire" in show else [None]
    effort_levels = EFFORT_LEVELS if "effort" in show else [None]
    intimacy_levels = INTIMACY_LEVELS if "intimacy" in show else [None]

    cells = []
    for _, row in scenarios_df.iterrows():
        scenario = row["scenario_label"]
        vignette = row["vignette"]
        latent_kwargs = _latent_awareness_kwargs(study, row)
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
                            vignette,
                            observed_action_text,
                            **prompt_kwargs,
                            **latent_kwargs,
                        )
                        cells.append(cell)
    return cells


def main(study, base=False):
    if study not in _STUDY_CONFIG:
        raise SystemExit(
            f"Unknown study: {study!r}. Currently supported: "
            f"{sorted(_STUDY_CONFIG.keys())}"
        )
    cfg = dict(_STUDY_CONFIG[study])
    if base:
        if study not in _BASE_OVERRIDE:
            raise SystemExit(
                f"--base is only defined for {sorted(_BASE_OVERRIDE)} (the studies "
                f"with a relationship paragraph to drop); {study!r} has none."
            )
        cfg.update(_BASE_OVERRIDE[study])
    api_key = load_api_key()

    print(f"Loading scenarios (study={study})...", flush=True)
    scenarios_df = load_scenarios(study)
    print(f"Loaded {len(scenarios_df)} scenarios", flush=True)

    print(f"\nInitializing Together AI client for {MODEL_ID}...", flush=True)
    client = Together(api_key=api_key)

    output_path = _output_path_for(study, base)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cell_cols = cfg["cell_cols"]

    # Resume: skip (cell, run) units already elicited. Done units are the union
    # of units with rows in the output JSONL and units in the empty-units
    # sidecar (see _empty_units_path — a valid zero-alternative elicitation
    # writes no rows, so it is tracked there instead). Refuses to resume across
    # a prompts.py edit (see guard_resume_prompt_mismatch).
    empty_units_path = _empty_units_path(output_path)
    done_units = set()
    results = []
    empty_units = []
    if output_path.exists():
        guard_resume_prompt_mismatch(output_path)
        results = read_jsonl_checked(output_path)
        if results and "run_id" in results[0]:
            done_units = set(_cell_key(r, cell_cols, r["run_id"]) for r in results)
    if empty_units_path.exists():
        empty_units = read_jsonl_checked(empty_units_path)
        done_units |= set(_cell_key(u, cell_cols, u["run_id"]) for u in empty_units)
    if output_path.exists():
        print(
            f"Found existing {output_path.name} with {len(done_units)} (cell, run) "
            f"units already elicited ({len(empty_units)} recorded as empty) — "
            "resuming.",
            flush=True,
        )

    # Build work list as (cell, run) units, dropping done ones.
    all_cells = _build_cells(scenarios_df, cfg, study)
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

    def _checkpoint():
        write_jsonl_atomic(output_path, _sorted_rows(results, cell_cols))
        write_jsonl_atomic(empty_units_path, _sorted_rows(empty_units, cell_cols))

    completed = 0
    n_new_units = 0
    n_failed_units = 0
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
            unit_str = (
                f"run {run} | {cell['scenario_label']} | "
                f"observed={cell['observed_action']} | {cond_str}"
            )
            if alts is None:
                # All parse retries exhausted: leave the unit pending so a
                # re-invocation retries it (see elicit_alternatives).
                n_failed_units += 1
                print(
                    f"[{completed}/{len(pending)}] {unit_str} | FAILED — left "
                    "pending for a future invocation",
                    flush=True,
                )
                continue
            n_new_units += 1
            print(
                f"[{completed}/{len(pending)}] {unit_str} | elicited {len(alts)}",
                flush=True,
            )
            if not alts:
                # Valid empty elicitation — durable via the sidecar, so resume
                # doesn't re-pay for it.
                unit = {
                    "scenario_label": cell["scenario_label"],
                    "observed_action": cell["observed_action"],
                }
                for col in cell_cols:
                    unit[col] = cell[col]
                unit["run_id"] = run
                empty_units.append(unit)
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
                results.append(row)
            if completed % CHECKPOINT_EVERY == 0:
                _checkpoint()
                print(
                    f"  checkpoint written ({len(results)} rows total)",
                    flush=True,
                )

    _checkpoint()
    print(
        f"\nSaved {len(results)} alternatives to {output_path} "
        f"({len(empty_units)} empty units in {empty_units_path.name})",
        flush=True,
    )
    if n_failed_units:
        print(
            f"{n_failed_units} units failed all parse retries — re-run to retry them.",
            flush=True,
        )

    # The manifest describes how the data were produced, so only (re)write it
    # when this invocation actually elicited something — a no-op resume must
    # not stamp the current prompts/git hashes over data produced earlier.
    if n_new_units:
        manifest_path = write_run_manifest(
            output_path,
            stage="generate_alternatives",
            study=study,
            extra={
                "k_runs": N_RUNS_ALT,
                "gen_temperature": ALT_GEN_TEMPERATURE,
                "n_cells": len(all_cells),
                "n_alternatives": len(results),
                "n_empty_units": len(empty_units),
            },
        )
        print(f"Wrote provenance manifest to {manifest_path}", flush=True)
    else:
        print("No new units elicited — manifest left unchanged.", flush=True)

    print("\n=== Summary ===")
    if results:
        results_df = pd.DataFrame(results)
        per_unit = results_df.groupby(
            ["scenario_label", "observed_action", *cell_cols, "run_id"]
        ).size()
        print(
            f"Total (cell, run) units: {len(per_unit) + len(empty_units)} "
            f"(expected {total}; {len(empty_units)} empty)"
        )
        print(
            f"Alternatives per non-empty (cell, run) — min: {per_unit.min()}, "
            f"max: {per_unit.max()}, mean: {per_unit.mean():.1f}, "
            f"median: {per_unit.median():.0f}"
        )
    else:
        print(
            f"Total (cell, run) units: {len(empty_units)} (expected {total}; all empty)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=tuple(_STUDY_CONFIG.keys()),
        default="food_inv_desire",
        help="Which 3-act inverse experiment to elicit alternatives for.",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Base-model mode: elicit alternatives WITHOUT the relationship "
        "paragraph (given-relationship studies only), writing a separate "
        "lm_alternatives_base.jsonl.",
    )
    args = parser.parse_args()
    main(args.study, base=args.base)
