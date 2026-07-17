"""
Leave-one-scenario-out (LOSO) CV for the active inverse experiments
(Studies 1a, 1b, 2a, 2b on the food set; 3a, 3b on the nonfood set).

For each variant (full / discomfort_only / base) and each of the 16 scenarios,
hold the scenario out, jointly refit the actor utility weights, `alpha_observer`,
and the response-noise `sigma` on the remaining 15 scenarios, then predict the
held-out scenario from that refit.

Outputs per `outputs/<slug>/`:
  - `cv_trial_ll.jsonl` — the PRIMARY metric: one record per held-out trial with
    its held-out log-likelihood `log p(u)` under the cross-validated mixture and
    its `subject_id` (so the analysis can bootstrap the full−ablation difference
    by participant).
  - `cv_preds_summary.json` — the secondary descriptive prediction: the model
    belief update `delta_<latent>` per held-out cell × variant (for the
    condition-averaged model-vs-human correlation). The desire study additionally
    carries `delta_desire_runs` (the K per-run held-out deltas per cell), used by
    the SI run-spread and mixture-check figures.
  - `cv_folds.jsonl` — per-fold refit diagnostics (params, train/test NLL).

Each `main_*()` runs end-to-end for one experiment and is exposed through the
corresponding `cv/cv_food_inv_*.py` thin wrapper.

The experiments differ in which latent the observer infers and how many slider
responses participants give per trial:

  Study 2a (`food_inv_intimacy`)      — infer intimacy given (desire, effort)
  Study 1a (`food_inv_desire`)        — infer desire given (effort, intimacy)
  Study 1b (`food_inv_joint_de`)      — joint over (desire, effort) given intimacy
  Study 2b (`food_inv_joint_ie`)      — joint over (intimacy, effort) given desire
  Study 3a (`nonfood_inv_joint_de`)   — Study 1b's design on the nonfood set
  Study 3b (`nonfood_inv_joint_ie`)   — Study 2b's design on the nonfood set

The nonfood studies reuse the joint mains with their own slug
(`main_joint_de("nonfood_inv_joint_de")` etc.): the designs are identical, and
only the stimulus set, scenario labels, and LM-table folder differ.

All share the joint-fit logic in `model/inverse/_helpers.py` — there is no
transfer between studies.
"""

import contextlib
import functools
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    EFFORT_PRIOR_MEAN,
    GRID,
    PRIOR_MEAN,
    _build_observer_tables_runs,
    delta_joint,
    delta_latent,
    desire_table_kwargs,
    fit_desire_observer_joint,
    fit_intimacy_observer_joint,
    fit_joint_de_observer_joint,
    fit_joint_ie_observer_joint,
    intimacy_table_kwargs,
    joint_de_table_kwargs,
    joint_ie_table_kwargs,
    git_sha,
    load_desire_data,
    load_fit_results,
    load_intimacy_data,
    load_joint_de_data,
    load_joint_ie_data,
    mixture_nll_1d,
    mixture_nll_2d,
    params_dict_to_array,
    sha256_file,
    verify_fit_manifest,
    write_json,
    write_jsonl,
)
from observers import (  # noqa: E402
    VARIANTS_DESIRE,
    VARIANTS_INTIMACY,
    VARIANTS_JOINT_DE,
    VARIANTS_JOINT_IE,
)
from tables import (  # noqa: E402
    INTIMACY_CONDITIONS,
    STUDY_SCENARIO_LABELS,
    actions,
)
from utils import get_project_root  # noqa: E402


GRID_NP = np.asarray(GRID)  # 101-bin [0, 1] latent grid
PRIOR_MEAN_F = float(PRIOR_MEAN)  # model prior mean of a continuous latent (= 0.5)
EFFORT_PRIOR_MEAN_F = float(EFFORT_PRIOR_MEAN)  # 2-state effort prior mean (= 0.5)
# Map the RelationshipConditions axis index back to the verbal condition slug
# written into the prediction outputs (so they merge with the human data, which
# stores intimacy_condition as a slug — never a numeric code).
INTIMACY_IDX_TO_LEVEL = dict(enumerate(INTIMACY_CONDITIONS))
N_ACTIONS = int(len(actions))
# Restarts per fold refit. Each refit warm-starts from the full-data fit (see
# `full_fit` below) — a leave-one-scenario-out refit only perturbs it slightly —
# but the full-data fit saw the held-out scenario, so a warm start alone would
# let held-out information pick the fold's basin of attraction. The default of
# 2 therefore adds one cold (lognormal) restart per fold and keeps the better
# NLL, so every fold has an init that never saw the held-out scenario.
# Env-tunable via CV_RESTARTS (1 = warm-only, for quick smoke runs).
N_RESTARTS_CV = int(os.environ.get("CV_RESTARTS", "2"))


def _domain_for(slug):
    """LM-table domain for a study slug — routes the *_table_kwargs loaders to
    the study's stimulus set (nonfood_inv_* slugs read scenarios_nonfood.csv
    tables; everything else the food set)."""
    return "nonfood" if slug.startswith("nonfood_") else "food"


# The per-variant (observer_fn, utility_param_names) registries — one per
# observer family — are the single source of truth in observers.py, imported
# above and shared with the fit wrappers so fit and CV never disagree on which
# ablations exist or which weights each fits.
# ------------------------------------------------------------------------------
# Helpers shared across the four LOSO mains.
# ------------------------------------------------------------------------------


def _held_out_ll_1d(deltas_per_trial, u_per_trial, sigma):
    """Held-out log-likelihood log p(u) per trial under the 1-D mixture, using
    the same `mixture_nll_1d` as the fit. `deltas_per_trial` is (n_test, K),
    `u_per_trial` is (n_test,). Returns a numpy array (n_test,)."""
    lls = jax.vmap(lambda u, d: -mixture_nll_1d(u, d, sigma))(
        jnp.asarray(u_per_trial), jnp.asarray(deltas_per_trial)
    )
    return np.asarray(lls)


def _held_out_ll_2d(deltas_per_trial, u_per_trial, sigma):
    """Held-out log p(u) per trial under the bivariate mixture. `deltas_per_trial`
    is (n_test, K, 2), `u_per_trial` is (n_test, 2)."""
    lls = jax.vmap(lambda u, d: -mixture_nll_2d(u, d, sigma))(
        jnp.asarray(u_per_trial), jnp.asarray(deltas_per_trial)
    )
    return np.asarray(lls)


def _fold_row(
    slug,
    variant,
    fold,
    scenario_label,
    params_arr,
    utility_param_names,
    train_nll,
    test_nll,
    n_train,
    n_test,
):
    row = {
        "experiment": slug,
        "variant": variant,
        "fold": fold,
        "held_out_scenario": scenario_label,
        "alpha_observer": float(params_arr[-2]),
        "param_sigma": float(params_arr[-1]),
        "train_nll": float(train_nll),
        "test_nll": float(test_nll),
        "n_train": int(n_train),
        "n_test": int(n_test),
    }
    for i, name in enumerate(utility_param_names):
        row[f"param_{name}"] = float(params_arr[i])
    return row


def _load_verified_warm_start(slug):
    """Warm-start params for CV folds from the full-data fit, checking
    provenance when it's available. A missing fit is a loud warning and a cold
    start (CV still runs — it just refits each fold from a lognormal init
    instead of the full-data fit, which is slower and skips the leak-mitigation
    intent, so a fit first is recommended). A fit whose manifest is *present but
    mismatched* is a hard error via verify_fit_manifest (genuine staleness); a
    fit with no manifest warns and is used as-is."""
    fit_path = get_project_root() / "model" / "outputs" / slug / "fit_results.json"
    if not fit_path.exists():
        print(
            f"WARNING: no fit_results.json for {slug} — CV will cold-start every "
            f"fold. Run `make fit-{slug}` first for a warm start (faster, and it "
            f"avoids folds depending on an init that saw the held-out scenario).",
            file=sys.stderr,
        )
        return {}
    verify_fit_manifest(slug)
    return load_fit_results(slug)


# The three CV output files written together per study; the manifest hashes
# them so model_comparison.py can refuse stale or mixed-vintage combinations.
CV_OUTPUT_NAMES = ("cv_preds_summary.json", "cv_folds.jsonl", "cv_trial_ll.jsonl")


def _write_outputs(slug, pred_rows, fold_rows, trial_ll_rows):
    if not trial_ll_rows:
        raise RuntimeError(
            f"CV for {slug} scored no held-out trials — refusing to write "
            "empty outputs. Check the data loader and the fold train/test masks."
        )
    outputs_dir = get_project_root() / "model" / "outputs" / slug
    outputs_dir.mkdir(parents=True, exist_ok=True)
    write_json(outputs_dir / "cv_preds_summary.json", pred_rows)
    write_jsonl(outputs_dir / "cv_folds.jsonl", fold_rows)
    write_jsonl(outputs_dir / "cv_trial_ll.jsonl", trial_ll_rows)
    print(f"\nWrote {outputs_dir / 'cv_trial_ll.jsonl'} (primary metric)")
    print(f"Wrote {outputs_dir / 'cv_preds_summary.json'}")
    print(f"Wrote {outputs_dir / 'cv_folds.jsonl'}")

    # Provenance manifest: records the run's git SHA plus content hashes of the
    # three outputs and the input data, so model_comparison.py can verify it is
    # combining files from a single CV run over the current data.
    data_csv = get_project_root() / "data" / slug / "main_trials_long.csv"
    manifest = {
        "experiment": slug,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "outputs": {name: sha256_file(outputs_dir / name) for name in CV_OUTPUT_NAMES},
        "input_data": {
            "path": str(data_csv.relative_to(get_project_root())),
            "sha256": sha256_file(data_csv),
        },
    }
    write_json(outputs_dir / "cv_manifest.json", manifest)
    print(f"Wrote {outputs_dir / 'cv_manifest.json'}")

    print("\n=== Per-variant summary (held-out log-likelihood) ===")
    trial_df = pd.DataFrame(trial_ll_rows)
    folds_df = pd.DataFrame(fold_rows)
    for variant, sub in trial_df.groupby("model"):
        fsub = folds_df[folds_df["variant"] == variant]
        print(
            f"  {variant}: mean held-out LL/trial = {sub['held_out_ll'].mean():.4f} "
            f"(alpha_obs = {fsub['alpha_observer'].mean():.3f}, "
            f"sigma = {fsub['param_sigma'].mean():.3f})"
        )


def _print_fold_header(slug, variant, fold, scenario_label, n_train, n_test):
    n_folds = len(STUDY_SCENARIO_LABELS[slug])
    print(
        f"  {slug} / {variant} / fold {fold + 1}/{n_folds} "
        f"({scenario_label}): train={n_train}, test={n_test}"
    )


# ------------------------------------------------------------------------------
# Generic LOSO runner (all four observer families).
#
# The (variant × fold) refits are independent and each is deterministic given
# (variant, fold, warm start, patience) — the cold-restart RNG is seeded from
# (slug, variant, held-out scenario) — so they can run in a process pool with
# output identical to the sequential loop; only the execution overlaps. The
# per-family pieces (data-array loader, table-kwargs builder, fold body) live in
# the study sections below and are wired together in the `_FAMILIES` registry at
# the bottom of the module.
# ------------------------------------------------------------------------------

# Worker-process state: the full (numpy) data arrays are shared into each worker
# once via the pool initializer, so per-fold jobs ship only the small
# (variant, fold, warm-params) tuple. The sequential path sets it in-process.
_CV_W = {}


def _cv_worker_init(family, slug, arrays):
    _CV_W.update(family=family, slug=slug, arrays=arrays)


@functools.lru_cache(maxsize=None)
def _tk_cached(family, slug, variant):
    """Per-process cache of a variant's LM table kwargs (the lm_runs.jsonl load
    is a few MB, so build it once per worker per variant, not once per fold).
    Keyed on (family, slug) too — the joint families serve two slugs each."""
    fam = _FAMILIES[family]
    _, utility_names = fam["variants"][variant]
    return fam["table_kwargs"](variant, utility_names, slug)


def _cv_fold(variant, fold, warm, patience):
    """One leave-one-scenario-out refit + held-out scoring. Reads the shared
    data (and the observer family) from `_CV_W` and dispatches to the family's
    fold body. Top-level + picklable so a ProcessPoolExecutor can run folds
    concurrently; fully deterministic given (variant, fold, warm, patience), so
    the parallel output equals the sequential."""
    return _FAMILIES[_CV_W["family"]]["fold_impl"](variant, fold, warm, patience)


@contextlib.contextmanager
def _capped_worker_threads():
    """Cap the XLA/OpenMP thread pools of the spawn workers while the pool is
    alive. Each worker re-imports JAX and would otherwise spin up its own
    full-width XLA CPU thread pool — CV_WORKERS × cores threads oversubscribe
    the machine. Spawn children inherit os.environ at process creation (setting
    env in the pool initializer would be too late: the child imports jax while
    unpickling the module), so the caps are set in the parent right before the
    pool starts and restored right after. The parent's already-initialized JAX
    is unaffected. Explicit user-set values are respected."""
    saved = {k: os.environ.get(k) for k in ("XLA_FLAGS", "OMP_NUM_THREADS")}
    xla = os.environ.get("XLA_FLAGS", "")
    # Match on the flag NAME (not name=value), so a user-set value for either
    # flag — whatever it is — is left alone instead of being overridden by an
    # appended duplicate (XLA parses the last occurrence).
    add = [
        f
        for f in (
            "--xla_cpu_multi_thread_eigen=false",
            "intra_op_parallelism_threads=1",
        )
        if f.split("=")[0] not in xla
    ]
    if add:
        os.environ["XLA_FLAGS"] = " ".join(([xla] if xla else []) + add)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_loso(family, slug, workers=None, patience=None):
    """LOSO CV for one study. Runs the (variant × fold) refits concurrently when
    `workers` > 1 (env `CV_WORKERS`, default 1): folds are independent and each
    refit is deterministic, so the output is identical to the sequential run —
    only the execution overlaps. `patience` (env `CV_PATIENCE`, default 100)
    trims the Adam no-improvement tail of each warm-started refit."""
    workers = workers if workers is not None else int(os.environ.get("CV_WORKERS", "1"))
    patience = (
        patience if patience is not None else int(os.environ.get("CV_PATIENCE", "100"))
    )
    fam = _FAMILIES[family]
    arrays = fam["load_arrays"](slug)
    variants = fam["variants"]
    # Warm-start source: the full-data fit (refits perturb it only slightly),
    # provenance-verified against fit_manifest.json and the current data CSV.
    full_fit = _load_verified_warm_start(slug)
    warms = {
        v: (
            np.asarray(params_dict_to_array(full_fit[v], util))
            if v in full_fit
            else None
        )
        for v, (_, util) in variants.items()
    }
    # Resolve every variant's LM tables before submitting any fold refits, so a
    # missing table (e.g. an unelicited lm_runs_base.jsonl) fails up front
    # rather than after hours of fitting earlier variants (workers rebuild their
    # own per-process cache; the parent's warm-up validates the files load).
    for v in variants:
        _tk_cached(family, slug, v)
    n_folds = len(STUDY_SCENARIO_LABELS[slug])
    jobs = [(v, f) for v in variants for f in range(n_folds)]

    if workers and workers > 1:
        print(f"  parallel {family} CV: {workers} workers, patience={patience}")
        with _capped_worker_threads():
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=_cv_worker_init,
                initargs=(family, slug, arrays),
            ) as ex:
                futs = {
                    ex.submit(_cv_fold, v, f, warms[v], patience): (v, f)
                    for v, f in jobs
                }
                results = {}
                try:
                    for fu in as_completed(futs):
                        v, f = futs[fu]
                        results[(v, f)] = fu.result()
                        print(
                            f"    [{len(results)}/{len(jobs)}] {slug} / {v} / "
                            f"fold {f + 1}/{n_folds} done",
                            flush=True,
                        )
                except BaseException:
                    # A failed refit dooms the run (nothing is written from
                    # partial results), so drop the queued jobs and surface the
                    # error now rather than after the remaining folds burn
                    # hours of compute. Already-running folds still finish.
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
    else:
        _cv_worker_init(family, slug, arrays)
        results = {(v, f): _cv_fold(v, f, warms[v], patience) for v, f in jobs}

    pred_rows, fold_rows, trial_ll_rows = [], [], []
    for key in jobs:  # deterministic order — matches the sequential run
        pr, fr, tr = results[key]
        pred_rows.extend(pr)
        fold_rows.append(fr)
        trial_ll_rows.extend(tr)
    return pred_rows, fold_rows, trial_ll_rows


# ==============================================================================
# Study 2a — infer intimacy given (desire, effort)
# ==============================================================================


def _load_arrays_intimacy(slug):
    data, action, scenario_idx, desire_condition, effort_condition, response = (
        load_intimacy_data(slug)
    )
    return dict(
        action=np.asarray(action),
        scenario=np.asarray(scenario_idx),
        desire=np.asarray(desire_condition),
        effort=np.asarray(effort_condition),
        response=np.asarray(response),
        subj=np.asarray(data["subject_id"].values),
    )


def _tk_intimacy(variant, utility_names, slug):
    return intimacy_table_kwargs(utility_names)


def _fold_impl_intimacy(variant, fold, warm, patience):
    obs_fn, utility_names = VARIANTS_INTIMACY[variant]
    slug = _CV_W["slug"]
    tk = _tk_cached(_CV_W["family"], slug, variant)
    arr = _CV_W["arrays"]
    sc, act = arr["scenario"], arr["action"]
    des, eff = arr["desire"], arr["effort"]
    resp, subj = arr["response"], arr["subj"]
    scenario_label = STUDY_SCENARIO_LABELS[slug][fold]
    train_mask, test_mask = sc != fold, sc == fold
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    _print_fold_header(slug, variant, fold, scenario_label, n_train, n_test)

    params, train_nll, _ = fit_intimacy_observer_joint(
        observer_fn=obs_fn,
        utility_param_names=utility_names,
        action=jnp.asarray(act[train_mask]),
        scenario_idx=jnp.asarray(sc[train_mask]),
        desire_condition=jnp.asarray(des[train_mask]),
        effort_condition=jnp.asarray(eff[train_mask]),
        response=jnp.asarray(resp[train_mask]),
        table_kwargs=tk,
        verbose=False,
        n_restarts=N_RESTARTS_CV,
        init_params=warm,
        patience=patience,
        seed_key=f"{slug}|{variant}|{scenario_label}",
    )
    sigma = float(params[-1])
    # (run, slot, scenario, observed_action, desire, effort, intimacy_101)
    tables = np.asarray(_build_observer_tables_runs(obs_fn, params, utility_names, tk))

    # Predicted belief update δ per held-out cell (mean over runs).
    pred_rows = []
    for a_idx in range(N_ACTIONS):
        for r in (0, 1):
            for e in (0, 1):
                density_runs = tables[:, 0, fold, a_idx, r, e, :]  # (K, 101)
                deltas = delta_latent(density_runs, GRID_NP, PRIOR_MEAN_F)  # (K,)
                pred_rows.append(
                    {
                        "experiment": slug,
                        "scenario_label": scenario_label,
                        "action": a_idx,
                        "desire_condition": "low" if r == 0 else "high",
                        "effort_condition": "low" if e == 0 else "high",
                        "delta_intimacy": float(deltas.mean()),
                        "model": variant,
                    }
                )

    # Per-trial held-out log-likelihood under the mixture.
    trial_ll_rows = []
    ti = np.where(test_mask)[0]
    if len(ti):
        post = tables[:, 0, sc[ti], act[ti], des[ti], eff[ti], :]  # (K, n_test, 101)
        deltas_t = delta_latent(post, GRID_NP, PRIOR_MEAN_F).T  # (n_test, K)
        lls = _held_out_ll_1d(deltas_t, resp[ti], sigma)
        test_nll = -float(lls.sum())
        for j, i in enumerate(ti):
            trial_ll_rows.append(
                {
                    "experiment": slug,
                    "model": variant,
                    "subject_id": str(subj[i]),
                    "scenario_label": scenario_label,
                    "held_out_ll": float(lls[j]),
                }
            )
    else:
        test_nll = 0.0

    fold_row = _fold_row(
        slug,
        variant,
        fold,
        scenario_label,
        params,
        utility_names,
        train_nll,
        test_nll,
        n_train,
        n_test,
    )
    return pred_rows, fold_row, trial_ll_rows


def main_intimacy():
    slug = "food_inv_intimacy"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(slug, *_run_loso("intimacy", slug))


# ==============================================================================
# Study 1a — infer desire given (effort, intimacy)
# ==============================================================================


def _load_arrays_desire(slug):
    data, action, scenario_idx, effort_condition, relationship_condition, response = (
        load_desire_data(slug)
    )
    return dict(
        action=np.asarray(action),
        scenario=np.asarray(scenario_idx),
        effort=np.asarray(effort_condition),
        rel=np.asarray(relationship_condition),
        response=np.asarray(response),
        subj=np.asarray(data["subject_id"].values),
    )


def _tk_desire(variant, utility_names, slug):
    return desire_table_kwargs(utility_names, base=(variant == "base"))


def _fold_impl_desire(variant, fold, warm, patience):
    obs_fn, utility_names = VARIANTS_DESIRE[variant]
    slug = _CV_W["slug"]
    tk = _tk_cached(_CV_W["family"], slug, variant)
    arr = _CV_W["arrays"]
    sc, act = arr["scenario"], arr["action"]
    eff, rel = arr["effort"], arr["rel"]
    resp, subj = arr["response"], arr["subj"]
    scenario_label = STUDY_SCENARIO_LABELS[slug][fold]
    train_mask, test_mask = sc != fold, sc == fold
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    _print_fold_header(slug, variant, fold, scenario_label, n_train, n_test)

    params, train_nll, _ = fit_desire_observer_joint(
        observer_fn=obs_fn,
        utility_param_names=utility_names,
        action=jnp.asarray(act[train_mask]),
        scenario_idx=jnp.asarray(sc[train_mask]),
        effort_condition=jnp.asarray(eff[train_mask]),
        relationship_condition=jnp.asarray(rel[train_mask]),
        response=jnp.asarray(resp[train_mask]),
        table_kwargs=tk,
        verbose=False,
        n_restarts=N_RESTARTS_CV,
        init_params=warm,
        patience=patience,
        seed_key=f"{slug}|{variant}|{scenario_label}",
    )
    sigma = float(params[-1])
    # (run, slot, scenario, observed_action, effort, intimacy, desire_101)
    tables = np.asarray(_build_observer_tables_runs(obs_fn, params, utility_names, tk))

    pred_rows = []
    for a_idx in range(N_ACTIONS):
        for rel_idx in range(4):
            for e in (0, 1):
                deltas = delta_latent(
                    tables[:, 0, fold, a_idx, e, rel_idx, :], GRID_NP, PRIOR_MEAN_F
                )
                pred_rows.append(
                    {
                        "experiment": slug,
                        "scenario_label": scenario_label,
                        "action": a_idx,
                        "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel_idx],
                        "effort_condition": "low" if e == 0 else "high",
                        "delta_desire": float(deltas.mean()),
                        # Per-run (K simulated-observer) held-out deltas for this
                        # cell, kept for the SI run-spread + mixture-check figures.
                        # Out-of-sample: predicted from the fold that held this
                        # scenario out.
                        "delta_desire_runs": [float(x) for x in deltas],
                        "model": variant,
                    }
                )

    trial_ll_rows = []
    ti = np.where(test_mask)[0]
    if len(ti):
        post = tables[:, 0, sc[ti], act[ti], eff[ti], rel[ti], :]  # (K, n_test, 101)
        deltas_t = delta_latent(post, GRID_NP, PRIOR_MEAN_F).T
        lls = _held_out_ll_1d(deltas_t, resp[ti], sigma)
        test_nll = -float(lls.sum())
        for j, i in enumerate(ti):
            trial_ll_rows.append(
                {
                    "experiment": slug,
                    "model": variant,
                    "subject_id": str(subj[i]),
                    "scenario_label": scenario_label,
                    "held_out_ll": float(lls[j]),
                }
            )
    else:
        test_nll = 0.0

    fold_row = _fold_row(
        slug,
        variant,
        fold,
        scenario_label,
        params,
        utility_names,
        train_nll,
        test_nll,
        n_train,
        n_test,
    )
    return pred_rows, fold_row, trial_ll_rows


def main_desire():
    slug = "food_inv_desire"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(slug, *_run_loso("desire", slug))


# ==============================================================================
# Study 1b — joint over (desire, effort) given intimacy
# ==============================================================================


def _load_arrays_joint_de(slug):
    (
        data,
        action,
        scenario_idx,
        relationship_condition,
        response_desire,
        response_effort,
    ) = load_joint_de_data(slug)
    return dict(
        action=np.asarray(action),
        scenario=np.asarray(scenario_idx),
        rel=np.asarray(relationship_condition),
        rd=np.asarray(response_desire),
        re=np.asarray(response_effort),
        subj=np.asarray(data["subject_id"].values),
    )


def _tk_joint_de(variant, utility_names, slug):
    return joint_de_table_kwargs(
        utility_names, domain=_domain_for(slug), base=(variant == "base")
    )


def _fold_impl_joint_de(variant, fold, warm, patience):
    obs_fn, utility_names = VARIANTS_JOINT_DE[variant]
    slug = _CV_W["slug"]
    tk = _tk_cached(_CV_W["family"], slug, variant)
    arr = _CV_W["arrays"]
    sc, act = arr["scenario"], arr["action"]
    rel = arr["rel"]
    rd, re = arr["rd"], arr["re"]
    subj = arr["subj"]
    scenario_label = STUDY_SCENARIO_LABELS[slug][fold]
    train_mask, test_mask = sc != fold, sc == fold
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    _print_fold_header(slug, variant, fold, scenario_label, n_train, n_test)

    params, train_nll, _ = fit_joint_de_observer_joint(
        observer_fn=obs_fn,
        utility_param_names=utility_names,
        action=jnp.asarray(act[train_mask]),
        scenario_idx=jnp.asarray(sc[train_mask]),
        relationship_condition=jnp.asarray(rel[train_mask]),
        response_desire=jnp.asarray(rd[train_mask]),
        response_effort=jnp.asarray(re[train_mask]),
        table_kwargs=tk,
        verbose=False,
        n_restarts=N_RESTARTS_CV,
        init_params=warm,
        patience=patience,
        seed_key=f"{slug}|{variant}|{scenario_label}",
    )
    sigma = float(params[-1])
    # (run, slot, scenario, observed_action, relationship_4, desire_101, effort_2)
    tables = np.asarray(_build_observer_tables_runs(obs_fn, params, utility_names, tk))

    pred_rows = []
    for a_idx in range(N_ACTIONS):
        for rel_idx in range(4):
            joint_runs = tables[:, 0, fold, a_idx, rel_idx, :, :]  # (K,101,2)
            d_desire, d_effort = delta_joint(
                joint_runs, GRID_NP, PRIOR_MEAN_F, EFFORT_PRIOR_MEAN_F
            )  # each (K,)
            pred_rows.append(
                {
                    "experiment": slug,
                    "scenario_label": scenario_label,
                    "action": a_idx,
                    "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel_idx],
                    "delta_desire": float(d_desire.mean()),
                    "delta_effort": float(d_effort.mean()),
                    "model": variant,
                }
            )

    trial_ll_rows = []
    ti = np.where(test_mask)[0]
    if len(ti):
        joint_t = tables[:, 0, sc[ti], act[ti], rel[ti], :, :]  # (K, n_test, 101, 2)
        d_desire_t, d_effort_t = delta_joint(
            joint_t, GRID_NP, PRIOR_MEAN_F, EFFORT_PRIOR_MEAN_F
        )  # each (K, n_test)
        deltas_t = np.stack([d_desire_t, d_effort_t], axis=-1)  # (K, n_test, 2)
        deltas_t = np.transpose(deltas_t, (1, 0, 2))  # (n_test, K, 2)
        u_t = np.stack([rd[ti], re[ti]], axis=1)  # (n_test, 2)
        lls = _held_out_ll_2d(deltas_t, u_t, sigma)
        test_nll = -float(lls.sum())
        for j, i in enumerate(ti):
            trial_ll_rows.append(
                {
                    "experiment": slug,
                    "model": variant,
                    "subject_id": str(subj[i]),
                    "scenario_label": scenario_label,
                    "held_out_ll": float(lls[j]),
                }
            )
    else:
        test_nll = 0.0

    fold_row = _fold_row(
        slug,
        variant,
        fold,
        scenario_label,
        params,
        utility_names,
        train_nll,
        test_nll,
        n_train,
        n_test,
    )
    return pred_rows, fold_row, trial_ll_rows


def main_joint_de(slug="food_inv_joint_de"):
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(slug, *_run_loso("joint_de", slug))


# ==============================================================================
# Study 2b — joint over (intimacy, effort) given desire
# ==============================================================================


def _load_arrays_joint_ie(slug):
    (
        data,
        action,
        scenario_idx,
        desire_condition,
        response_intimacy,
        response_effort,
    ) = load_joint_ie_data(slug)
    return dict(
        action=np.asarray(action),
        scenario=np.asarray(scenario_idx),
        desire=np.asarray(desire_condition),
        ri=np.asarray(response_intimacy),
        re=np.asarray(response_effort),
        subj=np.asarray(data["subject_id"].values),
    )


def _tk_joint_ie(variant, utility_names, slug):
    return joint_ie_table_kwargs(utility_names, domain=_domain_for(slug))


def _fold_impl_joint_ie(variant, fold, warm, patience):
    obs_fn, utility_names = VARIANTS_JOINT_IE[variant]
    slug = _CV_W["slug"]
    tk = _tk_cached(_CV_W["family"], slug, variant)
    arr = _CV_W["arrays"]
    sc, act = arr["scenario"], arr["action"]
    des = arr["desire"]
    ri, re = arr["ri"], arr["re"]
    subj = arr["subj"]
    scenario_label = STUDY_SCENARIO_LABELS[slug][fold]
    train_mask, test_mask = sc != fold, sc == fold
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    _print_fold_header(slug, variant, fold, scenario_label, n_train, n_test)

    params, train_nll, _ = fit_joint_ie_observer_joint(
        observer_fn=obs_fn,
        utility_param_names=utility_names,
        action=jnp.asarray(act[train_mask]),
        scenario_idx=jnp.asarray(sc[train_mask]),
        desire_condition=jnp.asarray(des[train_mask]),
        response_intimacy=jnp.asarray(ri[train_mask]),
        response_effort=jnp.asarray(re[train_mask]),
        table_kwargs=tk,
        verbose=False,
        n_restarts=N_RESTARTS_CV,
        init_params=warm,
        patience=patience,
        seed_key=f"{slug}|{variant}|{scenario_label}",
    )
    sigma = float(params[-1])
    # (run, slot, scenario, observed_action, desire, intimacy_101, effort_2)
    tables = np.asarray(_build_observer_tables_runs(obs_fn, params, utility_names, tk))

    pred_rows = []
    for a_idx in range(N_ACTIONS):
        for r in (0, 1):
            joint_runs = tables[:, 0, fold, a_idx, r, :, :]  # (K,101,2)
            d_intimacy, d_effort = delta_joint(
                joint_runs, GRID_NP, PRIOR_MEAN_F, EFFORT_PRIOR_MEAN_F
            )  # each (K,)
            pred_rows.append(
                {
                    "experiment": slug,
                    "scenario_label": scenario_label,
                    "action": a_idx,
                    "desire_condition": "low" if r == 0 else "high",
                    "delta_intimacy": float(d_intimacy.mean()),
                    "delta_effort": float(d_effort.mean()),
                    "model": variant,
                }
            )

    trial_ll_rows = []
    ti = np.where(test_mask)[0]
    if len(ti):
        joint_t = tables[:, 0, sc[ti], act[ti], des[ti], :, :]  # (K, n_test, 101, 2)
        d_intimacy_t, d_effort_t = delta_joint(
            joint_t, GRID_NP, PRIOR_MEAN_F, EFFORT_PRIOR_MEAN_F
        )  # each (K, n_test)
        deltas_t = np.stack([d_intimacy_t, d_effort_t], axis=-1)  # (K,n_test,2)
        deltas_t = np.transpose(deltas_t, (1, 0, 2))  # (n_test, K, 2)
        u_t = np.stack([ri[ti], re[ti]], axis=1)  # (n_test, 2)
        lls = _held_out_ll_2d(deltas_t, u_t, sigma)
        test_nll = -float(lls.sum())
        for j, i in enumerate(ti):
            trial_ll_rows.append(
                {
                    "experiment": slug,
                    "model": variant,
                    "subject_id": str(subj[i]),
                    "scenario_label": scenario_label,
                    "held_out_ll": float(lls[j]),
                }
            )
    else:
        test_nll = 0.0

    fold_row = _fold_row(
        slug,
        variant,
        fold,
        scenario_label,
        params,
        utility_names,
        train_nll,
        test_nll,
        n_train,
        n_test,
    )
    return pred_rows, fold_row, trial_ll_rows


def main_joint_ie(slug="food_inv_joint_ie"):
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(slug, *_run_loso("joint_ie", slug))


# ------------------------------------------------------------------------------
# Family registry wiring the generic LOSO runner (`_run_loso`, `_cv_fold`,
# `_tk_cached`) to the per-family pieces defined above. Defined last so every
# fold body it references exists. The variants registries stay the single
# source of truth in observers.py.
# ------------------------------------------------------------------------------
_FAMILIES = {
    "intimacy": {
        "variants": VARIANTS_INTIMACY,
        "load_arrays": _load_arrays_intimacy,
        "table_kwargs": _tk_intimacy,
        "fold_impl": _fold_impl_intimacy,
    },
    "desire": {
        "variants": VARIANTS_DESIRE,
        "load_arrays": _load_arrays_desire,
        "table_kwargs": _tk_desire,
        "fold_impl": _fold_impl_desire,
    },
    "joint_de": {
        "variants": VARIANTS_JOINT_DE,
        "load_arrays": _load_arrays_joint_de,
        "table_kwargs": _tk_joint_de,
        "fold_impl": _fold_impl_joint_de,
    },
    "joint_ie": {
        "variants": VARIANTS_JOINT_IE,
        "load_arrays": _load_arrays_joint_ie,
        "table_kwargs": _tk_joint_ie,
        "fold_impl": _fold_impl_joint_ie,
    },
}
