"""
Incremental fold checkpoint for the LOSO CV dispatcher.

A joint-family CV run is many hours of compute whose 48 (variant × fold)
refits previously lived only in the parent's memory until the final write —
an interrupt (Ctrl-C, crash, reboot) lost everything. This module gives
`_run_loso` a JSONL side file, `outputs/<slug>/cv_checkpoint.jsonl`, holding
one record per completed fold, appended (and fsynced) as each fold finishes,
so a rerun resumes from the completed folds instead of refitting them.

The file's first line is a header carrying a fingerprint of everything that
determines fold results: the data CSV, the LM run tables, the warm-start fit,
the refit config (patience, restarts), and the content of the model-math
source files themselves (`_CODE_FILES` — so a mid-run code edit invalidates
the checkpoint, while unrelated commits don't; hashing the git SHA instead
would discard a multi-hour checkpoint on every commit from a concurrent
session). A rerun whose fingerprint matches resumes; any mismatch discards
the checkpoint and starts fresh — a checkpoint must never splice folds from
different vintages into one output set.

Each fold record is a single JSON line written in one call and fsynced, so a
hard kill can at worst truncate the final line — `init_checkpoint` drops a
partial tail (repairing the file) and resumes from the complete records. The
final CV outputs are still written only when every fold is present, so
downstream consumers (the Makefile graph, model_comparison.py) never see a
partial set; `clear_checkpoint` removes the side file once they land.

Deliberately standalone (stdlib only, no jax/_helpers imports) so the unit
tests in test_checkpoint.py stay light.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Bump when the record format or fingerprint contents change shape — an old
# checkpoint then mismatches on version and is discarded instead of misread.
CHECKPOINT_VERSION = 1

CHECKPOINT_NAME = "cv_checkpoint.jsonl"

# The three CV output files written together per study by the dispatcher's
# _write_outputs and hashed into cv_manifest.json. Defined here (stdlib-only
# module) as the single source for both the dispatcher and model_comparison.py
# — the latter must stay importable without JAX for the figure scripts.
CV_OUTPUT_NAMES = ("cv_preds_summary.json", "cv_folds.jsonl", "cv_trial_ll.jsonl")

# The config descriptor a plain (default, reweighted) CV run fingerprints
# under — the reported model. Kept as a module constant so the no-config
# default and the checkpoint header agree on the exact dict. Renaming a field
# changes the fingerprint, so any checkpoint written before a rename is
# discarded on the next run rather than misread — the designed behavior for an
# input change.
_REPORTED_CONFIG = {
    "tag": "reported",
    "runs": "lm_runs.jsonl",
}


def _base_sibling(name):
    """The `_base` companion of an LM table vintage: `lm_runs<suffix>.jsonl` →
    `lm_runs_base<suffix>.jsonl` (the relationship-free set for the base
    ablation). Returns None when the name doesn't follow the `lm_runs...` shape
    (e.g. an absolute-path override), so the caller just hashes the single
    file."""
    prefix = "lm_runs"
    if name.startswith(prefix) and not name.startswith(prefix + "_base"):
        return name.replace(prefix, prefix + "_base", 1)
    return None


# The model-math sources whose content determines a fold refit's result: the
# fit/likelihood helpers, the memo model stack, and the dispatcher's fold
# bodies. Fingerprinting their bytes means an edit to any of them discards the
# checkpoint (correct: its folds were computed under different math), while
# orchestration-only files (cv wrappers, this module) stay out so incidental
# tweaks don't cost a resume.
_CODE_FILES = (
    "model/inverse/_helpers.py",
    "model/inverse/_reweighting.py",
    "model/observers.py",
    "model/actors.py",
    "model/utility.py",
    "model/tables.py",
    "model/cv/_inverse_dispatcher.py",
)
# Every file whose contents change a fold's numbers must be listed above —
# `_reweighting.py` (the comparison-set reweighting) sits at the likelihood
# layer, not just the model stack. A mid-run edit to an unlisted file would
# silently splice two model vintages into one CV output set, so add any future
# likelihood-layer module here at the same time it is written.


def checkpoint_path(outputs_dir):
    """The study's checkpoint file path under its outputs directory."""
    return Path(outputs_dir) / CHECKPOINT_NAME


def _sha256(path):
    """Hex SHA-256 of a file's bytes, or None when the file is missing (a
    missing warm start means CV cold-starts — still a fingerprintable state)."""
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fingerprint(
    slug, family, patience, n_restarts, config_fields=None, project_root=None
):
    """Fingerprint of everything that determines a fold refit's result.

    Content hashes (not mtimes) of the study's data CSV, its LM run tables,
    the full-data fit that warm-starts every fold, and the model-math source
    files (`_CODE_FILES`), plus the refit config. The worker count and thread
    caps are deliberately excluded: fold results are identical across
    execution layouts (verified byte-identical parallel-vs-sequential, and
    1-thread vs 4-thread workers in the interrupt/resume smoke), so a resume
    may change CV_WORKERS or CV_WORKER_THREADS freely — including retuning
    threads partway through a long run.

    `config_fields` (the RunConfig descriptor: `{"tag", "runs", "fit_dir"}`)
    ties the checkpoint to the run configuration, so the reported run and a
    variant run over the same study — a suffixed alternatives vintage, or
    `--no-reweighting` — never resume from each other's folds. It selects which
    LM run tables (the config's `runs` vintage + its `_base` sibling) and which
    fit directory (`fit_dir`) feed the hash. The default is the reported
    descriptor.
    """
    if project_root is None:
        from utils import get_project_root  # deferred: repo import, tests inject

        project_root = get_project_root()
    root = Path(project_root)
    lm_dir = root / "model" / "outputs" / "lm" / slug
    config = config_fields or _REPORTED_CONFIG

    # LM run tables: the config's alternatives vintage plus its `_base` sibling
    # (the relationship-free set for the base ablation), keyed by file name.
    runs_name = config.get("runs", _REPORTED_CONFIG["runs"])
    lm_names = [runs_name]
    if (sib := _base_sibling(runs_name)) is not None:
        lm_names.append(sib)
    lm_sha = {name: _sha256(lm_dir / name) for name in lm_names}

    # Warm-start fit from the config's own fit directory (a non-default run
    # writes outside the default outputs/<slug>/); default is the study root.
    fit_dir = config.get("fit_dir")
    warm_path = (
        Path(fit_dir) if fit_dir else root / "model" / "outputs" / slug
    ) / "fit_results.json"

    return {
        "version": CHECKPOINT_VERSION,
        "slug": slug,
        "family": family,
        "patience": int(patience),
        "n_restarts": int(n_restarts),
        "config": config,
        "data_sha256": _sha256(root / "data" / slug / "main_trials_long.csv"),
        "lm_sha256": lm_sha,
        "warm_sha256": _sha256(warm_path),
        "code_sha256": {rel: _sha256(root / rel) for rel in _CODE_FILES},
    }


def _write_header(path, fingerprint):
    """Start a fresh checkpoint: truncate to a single header line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(
            json.dumps(
                {
                    "kind": "header",
                    "fingerprint": fingerprint,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def init_checkpoint(path, fingerprint):
    """Open (or create) the checkpoint for this run and return its completed
    folds as {(variant, fold): (pred_rows, fold_row, trial_ll_rows)}.

    Leaves the file ready for `append_fold` in every case:
      - missing file → fresh header, {};
      - unreadable/corrupt header or fingerprint mismatch → discard (fresh
        header, {}), with the reason on stderr;
      - partial tail line (the previous run died mid-append) → drop it and
        rewrite the file to the complete records.
    """
    path = Path(path)
    if not path.exists():
        _write_header(path, fingerprint)
        return {}

    with open(path) as f:
        lines = f.readlines()

    def _discard(reason):
        print(
            f"CV checkpoint {path} {reason} — discarding it and starting fresh.",
            file=sys.stderr,
        )
        _write_header(path, fingerprint)
        return {}

    try:
        header = json.loads(lines[0])
    except (json.JSONDecodeError, IndexError):
        return _discard("has an unreadable header")
    if not isinstance(header, dict) or header.get("kind") != "header":
        return _discard("has an unreadable header")
    if header.get("fingerprint") != fingerprint:
        return _discard(
            "was written under different inputs, config, or model code "
            "(stale fingerprint)"
        )

    folds, n_good = {}, 1
    for line in lines[1:]:
        # Any malformed line — a partial tail from a mid-append death, or a
        # record that parses but isn't a complete fold object — ends the good
        # prefix; everything before it is kept.
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict) or rec.get("kind") != "fold":
                break
            folds[(str(rec["variant"]), int(rec["fold"]))] = (
                rec["pred_rows"],
                rec["fold_row"],
                rec["trial_ll_rows"],
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            break
        n_good += 1

    if n_good < len(lines):
        # Repair: rewrite to just the complete records so appends stay parseable.
        with open(path, "w") as f:
            f.writelines(lines[:n_good])
            f.flush()
            os.fsync(f.fileno())
    return folds


def append_fold(path, variant, fold, pred_rows, fold_row, trial_ll_rows):
    """Append one completed fold. A single write + fsync per fold (folds are
    tens of minutes of compute, so the sync cost is nothing) — a hard kill
    can only truncate the last line, which init_checkpoint repairs. Strict
    JSON (`allow_nan=False`): a non-serializable value OR a NaN/Inf fold
    result fails that fold loudly rather than silently persisting garbage —
    a NaN here means the refit diverged, and bare NaN tokens aren't valid
    JSON for non-Python readers anyway."""
    rec = {
        "kind": "fold",
        "variant": variant,
        "fold": fold,
        "pred_rows": pred_rows,
        "fold_row": fold_row,
        "trial_ll_rows": trial_ll_rows,
    }
    with open(path, "a") as f:
        f.write(json.dumps(rec, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def clear_checkpoint(path):
    """Remove the checkpoint once the final CV outputs are written (or when a
    run is deliberately reset). Missing file is fine."""
    Path(path).unlink(missing_ok=True)
