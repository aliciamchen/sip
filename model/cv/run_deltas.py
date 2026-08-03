"""Recover the per-run held-out belief updates behind an existing CV run.

The elicitation-sample mixture likelihood scores a participant's belief update
under `(1/K) Σ_k N(u | δ_k, σ²)`, so whether it behaves sensibly turns on how the
within-cell spread of the K per-run deltas `δ_k` compares to the fitted response
noise σ. σ is fitted per study, so this is a check that can fail in one study and
pass in another — not a figure to re-render once and generalize by assertion.

Only the desire study wrote `delta_desire_runs` into `cv_preds_summary.json`; the
other three families computed the per-run deltas and then averaged them away. The
fold bodies now keep them (see `PER_RUN_DELTA_KEYS`), but the *reported* CV
outputs predate that, and re-running CV to recover them would cost hours and
re-vintage every CV artifact — invalidating fits, results figures, and
`tab:fitted-params`, which are currently frozen and mutually consistent.

So this recomputes them instead, without refitting. Each fold's fitted parameters
are already persisted in `cv_folds.jsonl`, and the step from parameters to
per-cell deltas is a pure forward pass: build the observer tables at those
parameters, slice slot 0 of the held-out scenario, reduce over the latent axes.
The result is written as a **sidecar**, `cv_run_deltas.json`; nothing under
`outputs/<slug>/` is rewritten, so the CV outputs keep their vintage and their
manifest stays valid.

The correctness argument is the verification gate, not care in copying: the
recomputed per-run means must reproduce the stored `delta_*` for every cell of
every fold to within `TOL`. That is what makes the small amount of slicing logic
here safe to keep separate from the dispatcher's fold bodies — if the two ever
diverge, this script fails loudly instead of quietly reporting a different
model's spread. For the desire study the gate is stronger still: its stored
per-run arrays are compared element-wise, so the recompute is checked against K
values per cell rather than their mean.

Supports the uniform-prior path only — which is the reported configuration. A
fold fitted with an informative prior carries `param_prior_nu` and is refused
rather than silently scored under a flat prior.

Usage:
    uv run python model/cv/run_deltas.py                      # all six, `full`
    uv run python model/cv/run_deltas.py --study food_inv_joint_de
    uv run python model/cv/run_deltas.py --variant discomfort_only
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))
sys.path.insert(0, str(_project_root / "model" / "cv"))

import numpy as np  # noqa: E402

import _reweighting  # noqa: E402
from _helpers import (  # noqa: E402
    _build_observer_tables_runs,
    delta_joint,
    delta_latent,
    read_jsonl,
    write_json,
)
from _inverse_dispatcher import (  # noqa: E402
    EFFORT_PRIOR_MEAN_F,
    GRID_NP,
    INTIMACY_IDX_TO_LEVEL,
    N_ACTIONS,
    PER_RUN_DELTA_KEYS,
    PRIOR_MEAN_F,
    _FAMILIES,
    _cv_worker_init,
    _rw_cached,
    _tk_cached,
)
from _reweighting import FAMILY_BY_SLUG  # noqa: E402
from run_config import RunConfig  # noqa: E402
from run_delta_io import OUTPUT_NAME, sha256_file  # noqa: E402
from study_registry import studies  # noqa: E402
from tables import STUDY_SCENARIO_LABELS  # noqa: E402
from utils import get_project_root  # noqa: E402

#: Gate tolerance on |recomputed mean − stored delta|. Loose enough for XLA
#: reduction-order differences (this process runs a full-width CPU thread pool;
#: the CV workers ran single-threaded) and float32 accumulation, tight enough that
#: any real divergence in the prediction path fails it — the deltas themselves are
#: O(0.1), so this is a relative slack of ~1e-4.
TOL = 1e-5


def _fold_params(row, utility_names):
    """The fold's optimizer vector core `[*utility, alpha_observer, sigma]` plus
    its eta, read from a `cv_folds.jsonl` row by NAME rather than by offset."""
    if row.get("param_prior_nu") is not None:
        raise SystemExit(
            f"{row['experiment']}/{row['variant']} fold {row['fold']} was fitted "
            "with an informative prior (param_prior_nu present). This recompute "
            "implements the uniform-prior path only — the reported configuration. "
            "Scoring it flat would silently report a different model's deltas."
        )
    core = [float(row[f"param_{n}"]) for n in utility_names]
    core += [float(row["alpha_observer"]), float(row["param_sigma"])]
    return np.asarray(core, dtype=np.float32), float(row.get("param_eta", 0.0))


# ------------------------------------------------------------------------------
# Per-family cell enumeration: the held-out scenario's cells, each as the key
# columns that identify it in cv_preds_summary.json plus the per-run deltas.
#
# Cells are matched to the stored rows BY KEY, never by position, so this cannot
# drift out of sync with the order the fold bodies happen to loop in. The delta
# names and their order come from PER_RUN_DELTA_KEYS.
# ------------------------------------------------------------------------------


def _cells_desire(tables, fold):
    for a in range(N_ACTIONS):
        for rel in range(4):
            for e in (0, 1):
                keys = {
                    "action": a,
                    "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel],
                    "effort_condition": "low" if e == 0 else "high",
                }
                post = tables[:, 0, fold, a, e, rel, :]  # (K, 101)
                yield keys, (delta_latent(post, GRID_NP, PRIOR_MEAN_F),)


def _cells_intimacy(tables, fold):
    for a in range(N_ACTIONS):
        for r in (0, 1):
            for e in (0, 1):
                keys = {
                    "action": a,
                    "desire_condition": "low" if r == 0 else "high",
                    "effort_condition": "low" if e == 0 else "high",
                }
                post = tables[:, 0, fold, a, r, e, :]  # (K, 101)
                yield keys, (delta_latent(post, GRID_NP, PRIOR_MEAN_F),)


def _cells_joint_de(tables, fold):
    for a in range(N_ACTIONS):
        for rel in range(4):
            keys = {
                "action": a,
                "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel],
            }
            joint = tables[:, 0, fold, a, rel, :, :]  # (K, 101, 2)
            yield keys, delta_joint(joint, GRID_NP, PRIOR_MEAN_F, EFFORT_PRIOR_MEAN_F)


def _cells_joint_ie(tables, fold):
    for a in range(N_ACTIONS):
        for r in (0, 1):
            keys = {"action": a, "desire_condition": "low" if r == 0 else "high"}
            joint = tables[:, 0, fold, a, r, :, :]  # (K, 101, 2)
            yield keys, delta_joint(joint, GRID_NP, PRIOR_MEAN_F, EFFORT_PRIOR_MEAN_F)


_CELLS = {
    "desire": _cells_desire,
    "intimacy": _cells_intimacy,
    "joint_de": _cells_joint_de,
    "joint_ie": _cells_joint_ie,
}


def _stored_index(pred_rows, variant, key_names):
    """Stored prediction rows for one variant, keyed by (scenario, *cell keys)."""
    out = {}
    for r in pred_rows:
        if r["model"] != variant:
            continue
        out[(r["scenario_label"], *(r[k] for k in key_names))] = r
    return out


def recompute_study(slug, variant, outputs_dir):
    """Per-run held-out deltas for one (study, variant), gated against the stored
    means. Returns the sidecar dict, or None when the study's CV outputs are
    missing."""
    family = FAMILY_BY_SLUG[slug]
    fam = _FAMILIES[family]
    if variant not in fam["variants"]:
        print(f"[{slug}] no `{variant}` variant in family {family} — skipped")
        return None
    _, utility_names = fam["variants"][variant]
    delta_keys = PER_RUN_DELTA_KEYS[family]

    folds_path = outputs_dir / "cv_folds.jsonl"
    preds_path = outputs_dir / "cv_preds_summary.json"
    if not folds_path.exists() or not preds_path.exists():
        print(f"[{slug}] no CV outputs in {outputs_dir} — skipped")
        return None
    fold_rows = [r for r in read_jsonl(folds_path) if r["variant"] == variant]
    if not fold_rows:
        print(f"[{slug}] cv_folds.jsonl has no `{variant}` rows — skipped")
        return None
    with open(preds_path) as f:
        pred_rows = json.load(f)

    # The dispatcher's own table builder and reweighting config, so the only
    # thing this module reimplements is the slice-and-reduce step.
    _cv_worker_init(family, slug, arrays=None, config=RunConfig())
    tk = _tk_cached(family, slug, variant)
    rw = _rw_cached(slug, variant)
    obs_fn = fam["variants"][variant][0]

    # The key columns that identify a cell come from the family's own enumeration,
    # so index the stored rows on the first cell rather than hardcoding them here
    # (one more thing that would otherwise have to be kept in sync by hand).
    key_names, stored = None, {}
    cells_out, sigma_by_fold, worst = [], {}, 0.0
    worst_where, runs_checked, runs_worst = None, 0, 0.0
    for row in sorted(fold_rows, key=lambda r: r["fold"]):
        fold = int(row["fold"])
        scenario_label = STUDY_SCENARIO_LABELS[slug][fold]
        core, eta = _fold_params(row, utility_names)
        sigma_by_fold[scenario_label] = float(row["param_sigma"])
        tables = np.asarray(
            _build_observer_tables_runs(
                obs_fn,
                core,
                utility_names,
                _reweighting.apply(rw, tk, core, eta if rw else 0.0),
            )
        )
        for keys, deltas in _CELLS[family](tables, fold):
            if key_names is None:
                key_names = list(keys)
                stored = _stored_index(pred_rows, variant, key_names)
            rec = {"scenario_label": scenario_label, **keys}
            for name, d in zip(delta_keys, deltas):
                d = np.asarray(d)
                rec[name] = float(d.mean())
                rec[f"{name}_runs"] = [float(x) for x in d]
            s = stored.get((scenario_label, *(keys[k] for k in key_names)))
            if s is None:
                raise RuntimeError(
                    f"{slug}/{variant}: recomputed cell {rec['scenario_label']} "
                    f"{keys} has no matching row in cv_preds_summary.json — the "
                    "cell grid here disagrees with the one CV wrote."
                )
            for name in delta_keys:
                diff = abs(rec[name] - float(s[name]))
                if diff > worst:
                    worst, worst_where = diff, (scenario_label, dict(keys), name)
                # Where CV already stored the per-run values, check them
                # element-wise — a far stronger gate than matching their mean.
                if f"{name}_runs" in s:
                    rd = np.abs(
                        np.asarray(rec[f"{name}_runs"]) - np.asarray(s[f"{name}_runs"])
                    ).max()
                    runs_checked += 1
                    runs_worst = max(runs_worst, float(rd))
            cells_out.append(rec)

    if worst > TOL:
        sc, keys, name = worst_where
        raise SystemExit(
            f"{slug}/{variant}: recomputed {name} differs from the stored CV "
            f"prediction by {worst:.2e} > TOL={TOL:.0e} (worst at {sc} {keys}).\n"
            "The recompute is wrong, not the stored values — the fold parameters "
            "in cv_folds.jsonl are the ones CV predicted from. Check that this "
            "module's slicing matches the dispatcher's fold body before trusting "
            "any per-run spread computed here."
        )
    if runs_worst > TOL:
        raise SystemExit(
            f"{slug}/{variant}: recomputed per-run deltas differ from the stored "
            f"ones by {runs_worst:.2e} > TOL={TOL:.0e} across {runs_checked} "
            "cell-DV pairs, even though their means agree. Compensating errors in "
            "the per-run values — do not use this sidecar."
        )

    sidecar = {
        "experiment": slug,
        "variant": variant,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "delta_keys": list(delta_keys),
        "n_runs": len(cells_out[0][f"{delta_keys[0]}_runs"]),
        "tolerance": TOL,
        "max_abs_mean_diff": worst,
        "per_run_pairs_checked_elementwise": runs_checked,
        "max_abs_run_diff": runs_worst,
        # Ties the sidecar to the CV vintage it recomputed: if either source file
        # changes, this sidecar is stale and must be regenerated.
        "source": {
            "cv_folds.jsonl": sha256_file(folds_path),
            "cv_preds_summary.json": sha256_file(preds_path),
        },
        "sigma_by_fold": sigma_by_fold,
        "cells": cells_out,
    }
    # A study whose CV outputs ALREADY carry the per-run deltas needs no sidecar:
    # consumers read cv_preds_summary.json in preference, so writing one would
    # commit a second copy of data nothing reads. The recompute still runs for
    # such a study, because that is the only place the gate can compare per-run
    # values element-wise rather than just their means — it is the control that
    # certifies this module against the dispatcher, and it is worth paying for
    # every time even though its output is discarded.
    out_path = outputs_dir / OUTPUT_NAME
    if runs_checked:
        note = (
            "    control study: cv_preds_summary.json already carries the runs, "
            "so no sidecar written"
        )
    else:
        write_json(out_path, sidecar)
        note = f"    wrote {out_path}"
    sds = np.array([np.std(c[f"{k}_runs"]) for c in cells_out for k in delta_keys])
    sigma = float(np.mean(list(sigma_by_fold.values())))
    print(
        f"[{slug}/{variant}] {len(cells_out)} cells x K={sidecar['n_runs']}; "
        f"gate max |Δmean| = {worst:.2e}"
        + (
            f", max |Δrun| = {runs_worst:.2e} over {runs_checked} pairs"
            if runs_checked
            else ""
        )
        + f"\n    median within-cell run SD = {np.median(sds):.4f}, "
        f"mean fold sigma = {sigma:.4f} (ratio {np.median(sds) / sigma:.3f})"
        f"\n{note}"
    )
    return sidecar


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", help="one slug (default: all six)")
    ap.add_argument(
        "--variant",
        default="full",
        help="which ablation to recompute (default: full, the reported model)",
    )
    args = ap.parse_args()
    root = get_project_root() / "model" / "outputs"
    done = 0
    for st in studies():
        if args.study and st.slug != args.study:
            continue
        if recompute_study(st.slug, args.variant, root / st.slug) is not None:
            done += 1
    print(f"\n{done} study/studies recomputed and gated against their stored CV")


if __name__ == "__main__":
    main()
