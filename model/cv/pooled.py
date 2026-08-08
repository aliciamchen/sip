"""One utility across several experiments: nested pooled fits.

The reported model estimates a separate utility for each of the six
experiments. This script asks how much of that separation the data actually
require, by fitting progressively coarser groupings and cross-validating each:

    rung 1  one utility per experiment            6 utilities   (the reported model)
    rung 3  one utility per stimulus domain       2 utilities   food / nonfood
    rung 4  one utility for everything            1 utility

Each grouping nests in the one above, so the two steps ask one question each:

  rung 1 -> 3   Does a single utility serve every observer task within a domain?
                The utility describes the ACTOR; which latent the observer
                infers is a property of the observer's TASK, and the same actor
                is being reasoned about throughout. So this is close to a
                coherence requirement rather than a hypothesis -- with the
                caveat that the LM-elicited comparison sets also differ across
                experiments, so a loss here implicates either the utility or
                those alternative sets.

  rung 3 -> 4   Does a single utility serve both domains? Here "risk" means
                saliva-sharing discomfort on one side and disclosure risk on
                the other, so this is a substantive, contestable claim, and the
                interesting one. A loss is a finding, not a defect.

Rung 2 (one utility per paper Study) nests between them and is available as a
group set, but its three cells each test something different -- Studies 1 and 2
pool across DV dimensionality within one inference problem, Study 3 pools across
the inference problem -- so it is harder to read than either neighbour.

WHAT POOLS. Only the utility weights. `alpha_observer`, `sigma` and `eta` stay
per experiment; see `model/inverse/_pooled.py` for why, and the transfer
analysis (`model/cv/transfer.py`) for the measurement that motivates the split.

HOW IT RUNS, in two stages, so that nothing about scoring is reimplemented:

  1. For each fold, fit the pooled vector on the group's training trials. Fold k
     holds out scenario index k in EVERY experiment. The food and nonfood
     stimulus sets are disjoint but both have 16 scenarios in a fixed order, so
     this gives 16 folds at every rung with identical held-out trials -- which
     is what makes the rungs comparable to each other and to the reported run on
     matched trials.
  2. Score each experiment's held-out trials by handing its slice of that fold's
     pooled vector to the ordinary LOSO machinery with nothing free
     (`RunOverride` + an all-False `free_mask`). The existing, already-verified
     fold bodies do the scoring and write the standard CV output set to
     `outputs/<slug>/alt/pooled-<group>/`.

`full` only: the rung question is about the full model, and running the
ablations pooled would answer a different question at three times the cost.

EXPLORATORY. In no preregistration; this supplements the per-experiment fits
rather than replacing them.

Usage:
    uv run python model/cv/pooled.py --rung 3
    uv run python model/cv/pooled.py --group nonfood
    uv run python model/cv/pooled.py --summary-only
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))
sys.path.insert(0, str(_project_root / "model" / "cv"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _fit_dispatcher import FAMILY_BY_SLUG, fit_context  # noqa: E402
from _helpers import (  # noqa: E402
    ALPHA_OBS_SEEDS,
    read_jsonl,
    write_json,
    write_jsonl,
)
from _inverse_dispatcher import (  # noqa: E402
    N_RESTARTS_CV,
    RunOverride,
    _capped_worker_threads,
    _run_loso,
    _write_outputs,
)
from _pooled import LOSS_FACTORY, build_layout, fit_pooled, pooled_init  # noqa: E402
from model_comparison import _bootstrap_mean_by_subject  # noqa: E402
from study_registry import SLUGS, STUDIES  # noqa: E402
from utils import get_project_root  # noqa: E402

VARIANT = "full"
N_FOLDS = 16

FOOD = [s for s in SLUGS if STUDIES[s].domain == "food"]
NONFOOD = [s for s in SLUGS if STUDIES[s].domain == "nonfood"]

#: name -> (description, slugs). Rung 3 is {food, nonfood}; rung 4 is {all}.
GROUPS = {
    "food": ("one utility across the four food experiments", FOOD),
    "nonfood": ("one utility across the two nonfood experiments", NONFOOD),
    "all": ("one utility across all six experiments", list(SLUGS)),
}
RUNGS = {"3": ["food", "nonfood"], "4": ["all"]}

# A pooled worker holds every experiment in its group, so the count scales down
# as the group grows -- but measured on the nonfood group a worker is only
# ~1.1 GB for two experiments, so even the six-experiment group fits several
# times over in 48 GB and the real constraint is cores, not memory. These are
# sized for ~12 usable threads. `POOLED_WORKERS` overrides.
DEFAULT_WORKERS = {"nonfood": 6, "food": 5, "all": 5}


def outputs_dir(slug, group):
    return get_project_root() / "model" / "outputs" / slug / "alt" / f"pooled-{group}"


def group_dir(group):
    return get_project_root() / "model" / "outputs" / "pooled" / group


# ---------------------------------------------------------------------------
# Stage 1 -- the pooled fits
# ---------------------------------------------------------------------------
# Worker state: each spawn worker rebuilds the group's fit contexts once (the LM
# tables are a few MB per experiment), then runs whole folds from them.
_W = {}


def _worker_init(group):
    slugs = GROUPS[group][1]
    ctxs = {s: fit_context(s) for s in slugs}
    _W.update(group=group, slugs=slugs, ctxs=ctxs, layout=_layout(slugs, ctxs))


def _layout(slugs, ctxs):
    names = {s: tuple(ctxs[s].variants[VARIANT][1]) for s in slugs}
    if len(set(names.values())) != 1:
        # The shared block is positional, so if two families ever spelled `full`
        # with different (or differently ordered) weights, slot i would mean a
        # different parameter per experiment and the pooled fit would be quietly
        # meaningless rather than an error.
        raise ValueError(
            f"group members disagree about the `{VARIANT}` utility weights, so "
            f"they cannot share one block: {names}"
        )
    _, utility_names = ctxs[slugs[0]].variants[VARIANT]
    return build_layout(
        slugs,
        len(utility_names),
        {s: ctxs[s].reweighting(VARIANT) is not None for s in slugs},
    )


def _study_losses(slugs, ctxs, fold):
    """One loss closure per experiment, over the trials this fit trains on.

    `fold=None` trains on everything (the full-data fit); otherwise scenario
    index `fold` is held out of every experiment.
    """
    losses = []
    for slug in slugs:
        ctx = ctxs[slug]
        data = dict(ctx.data_kwargs)
        if fold is not None:
            keep = np.asarray(data["scenario_idx"]) != fold
            data = {k: np.asarray(v)[keep] for k, v in data.items()}
        obs_fn, utility_names = ctx.variants[VARIANT]
        loss, _, _ = LOSS_FACTORY[FAMILY_BY_SLUG[slug]](
            observer_fn=obs_fn,
            utility_param_names=utility_names,
            table_kwargs=ctx.table_kwargs[VARIANT],
            priors=ctx.priors[VARIANT],
            reweighting=ctx.reweighting(VARIANT),
            **data,
        )
        losses.append(loss)
    return losses


def _own_fit_vectors(slugs, ctxs):
    """Each experiment's reported full-data parameters, in fit-vector order --
    the starting point the pooled fit is seeded from."""
    out = []
    for slug in slugs:
        path = get_project_root() / "model" / "outputs" / slug / "fit_results.json"
        with open(path) as f:
            row = next(r for r in json.load(f) if r["model"] == VARIANT)
        _, utility_names = ctxs[slug].variants[VARIANT]
        vec = [float(row[f"param_{n}"]) for n in utility_names]
        vec += [float(row["alpha_observer"]), float(row["param_sigma"])]
        if ctxs[slug].reweighting(VARIANT) is not None:
            vec.append(float(row["param_eta"]))
        out.append(np.asarray(vec, dtype=float))
    return out


def _fit_fold(fold, init, n_restarts, patience):
    """One pooled fold fit, from the worker's own contexts. Top-level and
    picklable so a process pool can run folds concurrently."""
    losses = _study_losses(_W["slugs"], _W["ctxs"], fold)
    params, nll, _ = fit_pooled(
        _W["layout"],
        losses,
        init_params=init,
        n_restarts=n_restarts,
        patience=patience,
        seed_key=f"pooled|{_W['group']}|{VARIANT}|fold{fold}",
    )
    return fold, np.asarray(params, dtype=float), float(nll)


def fit_full_data(group):
    """The pooled fit on every scenario -- reported as the group's utility, and
    the warm start for the folds.

    alpha_observer's two basins are covered explicitly: the per-experiment fits
    already sit in one basin each, and two further inits put EVERY experiment's
    alpha at each `ALPHA_OBS_SEEDS` value, so the pooled fit is not held in
    whichever basin the seeds happened to start in.
    """
    slugs = GROUPS[group][1]
    ctxs = {s: fit_context(s) for s in slugs}
    layout = _layout(slugs, ctxs)
    losses = _study_losses(slugs, ctxs, fold=None)
    base = pooled_init(layout, _own_fit_vectors(slugs, ctxs))

    inits = [base]
    for seed in ALPHA_OBS_SEEDS:
        alt = base.copy()
        for start, _ in layout.blocks:
            alt[start] = float(seed)
        inits.append(alt)

    best = None
    for k, init in enumerate(inits):
        params, nll, _ = fit_pooled(
            layout,
            losses,
            init_params=init,
            n_restarts=1,
            seed_key=f"pooled|{group}|{VARIANT}|full|init{k}",
        )
        print(f"    init {k}: NLL {nll:.2f}")
        if best is None or nll < best[1]:
            best = (np.asarray(params, dtype=float), float(nll))
    return layout, best[0], best[1], ctxs


def _stage1_fingerprint(group, layout, full_params, patience):
    """What determines a pooled fold's result: the group, the vector layout, the
    warm start every fold begins from, and the refit configuration. A checkpoint
    whose fingerprint differs is from another run and is discarded rather than
    spliced."""
    return {
        "group": group,
        "slugs": list(layout.slugs),
        "variant": VARIANT,
        "n_params": layout.n_params,
        "full_data_params": [round(float(x), 10) for x in full_params],
        "n_restarts": N_RESTARTS_CV,
        "patience": int(patience),
    }


def _load_stage1_checkpoint(path, fingerprint):
    """Completed pooled folds from an interrupted run, as {fold: params}."""
    if not path.exists():
        return {}
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError):
        return {}
    if not rows or rows[0].get("fingerprint") != fingerprint:
        if rows:
            print("    checkpoint is from a different run — discarding")
        return {}
    return {
        int(r["fold"]): np.asarray(r["params"], dtype=float)
        for r in rows[1:]
        if "fold" in r
    }


def stage1(group, workers=None, patience=100):
    """Pooled fits: the full-data one, then one per fold.

    Each completed fold is appended to `pooled_checkpoint.jsonl` as it lands, so
    an interrupted run resumes instead of redoing hours of fitting; the file is
    removed once `pooled_fit.json` is written. The full-data fit is cheap
    relative to the folds and is always redone -- it is also what the checkpoint
    fingerprint is keyed on, so a resume can only reuse folds that started from
    the same warm start.
    """
    print(f"  stage 1: pooled fits ({GROUPS[group][0]})")
    layout, full_params, full_nll, ctxs = fit_full_data(group)
    names = layout.param_names(list(ctxs[layout.slugs[0]].variants[VARIANT][1]))
    print(f"    full-data NLL {full_nll:.2f}")
    for n, v in zip(names, full_params):
        print(f"      {n:<38} {v:.4f}")

    workers = workers or int(
        os.environ.get("POOLED_WORKERS") or DEFAULT_WORKERS.get(group, 4)
    )
    gd = group_dir(group)
    gd.mkdir(parents=True, exist_ok=True)
    ckpt = gd / "pooled_checkpoint.jsonl"
    fingerprint = _stage1_fingerprint(group, layout, full_params, patience)
    per_fold = _load_stage1_checkpoint(ckpt, fingerprint)
    if per_fold:
        print(f"    resuming: {len(per_fold)}/{N_FOLDS} pooled folds already done")
    else:
        write_jsonl(ckpt, [{"fingerprint": fingerprint}])

    def record(fold, params):
        per_fold[fold] = params
        with open(ckpt, "a") as f:
            f.write(
                json.dumps({"fold": int(fold), "params": [float(x) for x in params]})
                + "\n"
            )
            f.flush()
            os.fsync(f.fileno())

    jobs = [f for f in range(N_FOLDS) if f not in per_fold]
    if jobs and workers > 1:
        print(f"    {len(jobs)} pooled fold fits on {workers} workers")
        with _capped_worker_threads(1):
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=_worker_init,
                initargs=(group,),
            ) as ex:
                futs = {
                    ex.submit(_fit_fold, f, full_params, N_RESTARTS_CV, patience): f
                    for f in jobs
                }
                try:
                    for fu in as_completed(futs):
                        fold, params, nll = fu.result()
                        record(fold, params)
                        print(
                            f"      [{len(per_fold)}/{N_FOLDS}] fold {fold} "
                            f"NLL {nll:.2f}",
                            flush=True,
                        )
                except BaseException:
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
    elif jobs:
        _worker_init(group)
        for f in jobs:
            fold, params, nll = _fit_fold(f, full_params, N_RESTARTS_CV, patience)
            record(fold, params)
            print(f"      fold {fold} NLL {nll:.2f}", flush=True)

    write_json(
        gd / "pooled_fit.json",
        {
            "group": group,
            "description": GROUPS[group][0],
            "slugs": list(layout.slugs),
            "variant": VARIANT,
            "n_params": layout.n_params,
            "param_names": names,
            "full_data_nll": full_nll,
            "full_data_params": [float(x) for x in full_params],
            "per_fold_params": {
                str(f): [float(x) for x in per_fold[f]] for f in sorted(per_fold)
            },
            "n_restarts_cv": N_RESTARTS_CV,
        },
    )
    print(f"    wrote {gd / 'pooled_fit.json'}")
    # The per-fold vectors are now in pooled_fit.json, so the side file has
    # served its purpose (same lifecycle as the CV fold checkpoint).
    ckpt.unlink(missing_ok=True)
    return layout, per_fold


def stage2(group, layout, per_fold):
    """Score each experiment's held-out trials at its slice of the fold's pooled
    vector, through the ordinary LOSO machinery with nothing free."""
    print("  stage 2: scoring each experiment at its slice of the pooled fit")
    for i, slug in enumerate(layout.slugs):
        ctx = fit_context(slug)
        _, utility_names = ctx.variants[VARIANT]
        n_study = len(utility_names) + 2 + (ctx.reweighting(VARIANT) is not None)
        inits = {
            (VARIANT, f): np.asarray(layout.study_slice(per_fold[f], i))
            for f in per_fold
        }
        out_dir = outputs_dir(slug, group)
        override = RunOverride(
            variants=(VARIANT,),
            init_params=inits,
            free_mask={VARIANT: np.zeros(n_study, dtype=bool)},
            outputs_dir=out_dir,
            fingerprint={
                "pooled_group": group,
                "pooled_slugs": list(layout.slugs),
                "pooled_params": {
                    str(f): [float(x) for x in inits[(VARIANT, f)]] for f in per_fold
                },
            },
        )
        _write_outputs(
            slug,
            *_run_loso(ctx.family, slug, override=override),
            outputs_dir=out_dir,
        )


def run_group(group, workers=None):
    print("=" * 70)
    print(f"Pooled fit: {group} -- {GROUPS[group][0]}")
    print("=" * 70)
    layout, per_fold = stage1(group, workers=workers)
    stage2(group, layout, per_fold)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(n_boot=1000, seed=0):
    """Per-experiment held-out likelihood at each rung, against the reported
    per-experiment fit.

    Reported per experiment and never as the pooled total: the joint objective
    weights experiments by trial count and DV dimensionality, so only the split
    numbers show whether pooling bought one experiment's fit at another's
    expense.
    """
    rng = np.random.default_rng(seed)
    root = get_project_root() / "model" / "outputs"
    rows = []
    per_group = {}
    for group in GROUPS:
        for slug in GROUPS[group][1]:
            path = outputs_dir(slug, group) / "cv_trial_ll.jsonl"
            own_path = root / slug / "cv_trial_ll.jsonl"
            if not path.exists() or not own_path.exists():
                continue
            own = pd.DataFrame(read_jsonl(own_path))
            got = pd.DataFrame(read_jsonl(path))
            a = own[own["model"] == VARIANT]
            b = got[got["model"] == VARIANT]
            wide = a.merge(
                b, on=["subject_id", "scenario_label"], suffixes=("_own", "_pool")
            )
            if len(wide) != len(a) or len(wide) != len(b):
                raise RuntimeError(
                    f"{slug}/{group}: {len(a)} own vs {len(b)} pooled trials, "
                    f"matched {len(wide)} — mixed data vintages; re-run CV"
                )
            wide = wide.assign(
                subject_id=STUDIES[slug].short_label
                + "|"
                + wide["subject_id"].astype(str)
            )
            diff = (wide["held_out_ll_pool"] - wide["held_out_ll_own"]).to_numpy()
            per_group.setdefault(group, []).append(wide)
            boots = _bootstrap_mean_by_subject(
                diff, wide["subject_id"].to_numpy(), n_boot, rng
            )
            rows.append(
                {
                    "group": group,
                    "experiment": STUDIES[slug].short_label,
                    "slug": slug,
                    "n_trials": int(len(wide)),
                    "pooled_ll": float(wide["held_out_ll_pool"].mean()),
                    "own_ll": float(wide["held_out_ll_own"].mean()),
                    "diff": float(diff.mean()),
                    "ci_95": [
                        float(np.percentile(boots, 2.5)),
                        float(np.percentile(boots, 97.5)),
                    ],
                }
            )

    # One combined row per group: the paper quotes an interval on every other
    # number, so the aggregate needs one too. Participants are namespaced by
    # experiment so the cluster bootstrap cannot merge two experiments' ids.
    combined = []
    for group in GROUPS:
        parts = [p for p in per_group.get(group, []) if len(p)]
        if not parts:
            continue
        allw = pd.concat(parts)
        diff = (allw["held_out_ll_pool"] - allw["held_out_ll_own"]).to_numpy()
        boots = _bootstrap_mean_by_subject(
            diff, allw["subject_id"].to_numpy(), n_boot, rng
        )
        combined.append(
            {
                "group": group,
                "experiment": "combined",
                "n_trials": int(len(allw)),
                "pooled_ll": float(allw["held_out_ll_pool"].mean()),
                "own_ll": float(allw["held_out_ll_own"].mean()),
                "diff": float(diff.mean()),
                "ci_95": [
                    float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5)),
                ],
            }
        )

    out_path = root / "pooled" / "pooled_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        out_path,
        {
            "n_boot": n_boot,
            "seed": seed,
            "variant": VARIANT,
            "rows": rows,
            "combined": combined,
        },
    )
    if rows:
        print("\n=== Pooled vs per-experiment fit (per-trial held-out LL) ===")
        for group in GROUPS:
            sub = [r for r in rows if r["group"] == group]
            if not sub:
                continue
            print(f"\n  {group}: {GROUPS[group][0]}")
            for r in sub:
                lo, hi = r["ci_95"]
                print(
                    f"    {r['experiment']:<4} pooled {r['pooled_ll']:+.4f}  "
                    f"own {r['own_ll']:+.4f}  delta {r['diff']:+.4f} "
                    f"[{lo:+.4f}, {hi:+.4f}]"
                )
    print(f"\nWrote {out_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rung", choices=sorted(RUNGS), action="append")
    parser.add_argument("--group", choices=sorted(GROUPS), action="append")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="re-run groups whose outputs exist"
    )
    args = parser.parse_args()

    if not args.summary_only:
        wanted = list(args.group or [])
        for rung in args.rung or ([] if args.group else sorted(RUNGS)):
            wanted += [g for g in RUNGS[rung] if g not in wanted]
        for group in wanted:
            done = all(
                (outputs_dir(s, group) / "cv_trial_ll.jsonl").exists()
                for s in GROUPS[group][1]
            )
            if done and not args.force:
                print(f"skipping {group}: outputs already on disk")
                continue
            run_group(group, workers=args.workers)

    summarize(n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
