#!/usr/bin/env python3
"""One-shot backfill of the empty-units sidecar for a COMPLETED elicitation.

Historically, a (cell, run) unit whose elicitation returned zero alternatives
wrote no rows to lm_alternatives.jsonl, so on resume it was indistinguishable
from a never-attempted unit and was re-elicited (re-paid) on every invocation.
generate_alternatives.py now records such units in an empty-units sidecar (see
``_empty_units_path`` there for the format); this script backfills that sidecar
for data elicited before the sidecar existed.

VALIDITY ASSUMPTION — read before running: marking grid-missing units as empty
is only valid because a *completed* elicitation pass attempted every (cell,
run) unit in the study's grid. New manifests state this explicitly with
``status: "complete"``; legacy manifests were written only at the end of a
pass. Therefore, any unit absent from the JSONL was attempted and yielded zero
alternatives rather than never being attempted. That "zero" covers both a
valid empty response and a unit that exhausted its parse retries — the two
were indistinguishable in the old pipeline, and score_merged.py already scored
both as observed-action-only cells, so marking them empty preserves the
historical semantics. The script therefore refuses to run unless the manifest
identifies a completed pass AND its recorded generation-prompt fingerprint
matches the current prompt (a prompt edit would make "attempted" no longer mean
"attempted with these prompts").

Usage (never runs any LM call; purely local):
    uv run python model/lm/backfill_empty_units.py --study food_inv_desire
    uv run python model/lm/backfill_empty_units.py --study food_inv_desire --base
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import (
    _manifest_prompt_hashes,
    manifest_prompt_matches,
    read_jsonl_checked,
    read_run_manifest,
    write_jsonl_atomic,
)
from generate_alternatives import (
    _BASE_OVERRIDE,
    _STUDY_CONFIG,
    _build_cells,
    _cell_key,
    _empty_units_path,
    _sorted_rows,
    load_scenarios,
)

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root


def _manifest_is_completed(manifest):
    """Legacy manifests were end-only; new checkpoint manifests are explicit."""
    return manifest.get("status") in (None, "complete")


def main(study, base=False):
    if study not in _STUDY_CONFIG:
        raise SystemExit(
            f"Unknown study: {study!r}. Supported: {sorted(_STUDY_CONFIG.keys())}"
        )
    cfg = dict(_STUDY_CONFIG[study])
    if base:
        if study not in _BASE_OVERRIDE:
            raise SystemExit(
                f"--base is only defined for {sorted(_BASE_OVERRIDE)}; "
                f"{study!r} has no base elicitation."
            )
        cfg.update(_BASE_OVERRIDE[study])
    cell_cols = cfg["cell_cols"]

    output_path = (
        get_project_root() / "model" / "outputs" / "lm" / study / cfg["output"]
    )
    if not output_path.exists():
        raise SystemExit(f"{output_path} not found — nothing to backfill.")

    # The validity gate: only a completed pass under the current prompts
    # justifies reading absence-from-grid as empty.
    manifest = read_run_manifest(output_path)
    if manifest is None:
        raise SystemExit(
            f"No run manifest next to {output_path.name} — cannot confirm the "
            "elicitation pass completed, so grid-missing units cannot be "
            "assumed empty. Re-run generate_alternatives.py to completion "
            "instead."
        )
    if not _manifest_is_completed(manifest):
        raise SystemExit(
            f"The run manifest for {output_path.name} has "
            f"status={manifest.get('status')!r}, so the elicitation grid may be "
            "incomplete. Resume generation instead of backfilling."
        )
    if not manifest_prompt_matches(manifest):
        recorded, current, field = _manifest_prompt_hashes(manifest)
        raise SystemExit(
            f"The generation prompt has changed since {output_path.name} was "
            f"elicited (manifest {field}={recorded}, current={current}) — "
            "grid-missing units may simply never have been attempted under "
            "the current prompt. Refusing to backfill."
        )
    k_runs = manifest.get("k_runs")
    if not k_runs:
        raise SystemExit(
            f"Manifest for {output_path.name} has no k_runs — cannot "
            "enumerate the (cell, run) grid."
        )

    rows = read_jsonl_checked(output_path)
    if not rows or "run_id" not in rows[0]:
        raise SystemExit(
            f"{output_path} has no K-run rows (missing run_id) — refusing to "
            "backfill a pre-K-run file."
        )
    present = set(_cell_key(r, cell_cols, r["run_id"]) for r in rows)

    empty_units_path = _empty_units_path(output_path)
    empty_units = (
        read_jsonl_checked(empty_units_path) if empty_units_path.exists() else []
    )
    present |= set(_cell_key(u, cell_cols, u["run_id"]) for u in empty_units)

    all_cells = _build_cells(load_scenarios(study), cfg)
    n_new = 0
    for cell in all_cells:
        for run in range(int(k_runs)):
            if _cell_key(cell, cell_cols, run) in present:
                continue
            unit = {
                "scenario_label": cell["scenario_label"],
                "observed_action": cell["observed_action"],
            }
            for col in cell_cols:
                unit[col] = cell[col]
            unit["run_id"] = run
            empty_units.append(unit)
            n_new += 1

    total = len(all_cells) * int(k_runs)
    if n_new:
        write_jsonl_atomic(empty_units_path, _sorted_rows(empty_units, cell_cols))
        print(
            f"Backfilled {n_new} grid-missing (cell, run) units as empty into "
            f"{empty_units_path} ({len(empty_units)} total; grid = {total})."
        )
    else:
        print(
            f"Nothing to backfill: all {total} grid units are accounted for "
            f"({len(empty_units)} already recorded as empty)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        required=True,
        choices=tuple(_STUDY_CONFIG.keys()),
        help="Study whose COMPLETED elicitation to backfill (explicit on "
        "purpose — see the validity assumption in the module docstring).",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Backfill the base-mode files (lm_alternatives_base.jsonl).",
    )
    args = parser.parse_args()
    main(args.study, base=args.base)
