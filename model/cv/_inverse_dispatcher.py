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
    condition-averaged model-vs-human correlation), plus `delta_<latent>_runs`,
    the K per-run held-out deltas the mean was taken over. Those are the
    mixture's own components, and the SI run-spread / mixture-check figures read
    them to show the within-cell run spread against the fitted sigma.
    `PER_RUN_DELTA_KEYS` names them per family. Only the desire study wrote them
    until 2026-08-03; older CV outputs for the other five carry the means only,
    and `cv/run_deltas.py` recomputes the runs for a vintage that predates this.
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

All share the joint-fit logic in `model/inverse/_helpers.py`. The reported fits
estimate every study's parameters from its own data alone. Two exploratory
analyses reuse this runner through `RunOverride` to score parameters that came
from elsewhere: `model/cv/transfer.py` (one study's parameters on another) and
`model/cv/pooled.py` (a utility shared across several studies).
"""

import contextlib
import functools
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))
sys.path.insert(0, str(_project_root / "model" / "cv"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _checkpoint import (  # noqa: E402
    append_fold,
    checkpoint_path,
    clear_checkpoint,
    init_checkpoint,
    run_fingerprint,
)
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
    load_intimacy_data,
    load_joint_de_data,
    load_joint_ie_data,
    mixture_nll_1d,
    mixture_nll_2d,
    params_dict_to_array,
    parse_run_config_args,
    sha256_file,
    verify_fit_manifest,
    write_json,
    write_jsonl,
)
import _reweighting  # noqa: E402
from _priors import (  # noqa: E402
    beta_prior_on_grid,
    build_priors_kwarg,
    priors_base_variant,
    reweight_grid,
    reweight_joint,
)
from observers import (  # noqa: E402
    VARIANTS_DESIRE,
    VARIANTS_INTIMACY,
    VARIANTS_JOINT_DE,
    VARIANTS_JOINT_IE,
)
from run_config import RunConfig  # noqa: E402
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
# The belief-update prediction each family writes per held-out cell, in the order
# its fold body computes them. `cv_preds_summary.json` carries `<key>` (the mean
# over elicitation runs) and `<key>_runs` (the K per-run values). Single source of
# truth for consumers that need to know which deltas a study has — the SI
# run-spread figure and `run_deltas.py`, which must agree with the fold bodies
# about both the names and the order.
PER_RUN_DELTA_KEYS = {
    "desire": ("delta_desire",),
    "intimacy": ("delta_intimacy",),
    "joint_de": ("delta_desire", "delta_effort"),
    "joint_ie": ("delta_intimacy", "delta_effort"),
}
# Restarts per fold refit. Each refit warm-starts from the full-data fit (see
# `full_fit` below) — a leave-one-scenario-out refit only perturbs it slightly —
# but the full-data fit saw the held-out scenario, so a warm start alone would
# let held-out information pick the fold's basin of attraction. The default of
# 2 therefore adds one cold (lognormal) restart per fold and keeps the better
# NLL, so every fold has an init that never saw the held-out scenario.
# Env-tunable via CV_RESTARTS (1 = warm-only, for quick smoke runs).
N_RESTARTS_CV = int(os.environ.get("CV_RESTARTS", "2"))
# Per-family execution defaults (env CV_WORKERS / CV_WORKER_THREADS override;
# see the `default_workers` / `worker_threads` entries in _FAMILIES below).
# All four families now share one profile: many single-threaded workers. The
# joint families briefly needed 3 × 4-thread workers when their observers
# carried ~8 GB of XLA temps each, but the fast joint observers (see
# observers.py: direct Bayesian inversion of the actor policy) brought a
# worker back to ~1.5 GB, so the memory cap is gone. The per-family keys stay
# as the tuning point if a family's profile ever diverges again. Neither knob
# changes the refit results (fold outputs are reduction-order-stable across
# thread counts; verified byte-identical in the interrupt/resume smoke) —
# they are purely execution-layout choices.
DEFAULT_WORKERS = 8
DEFAULT_WORKER_THREADS = 1


def _domain_for(slug):
    """LM-table domain for a study slug — routes the *_table_kwargs loaders to
    the study's stimulus set (nonfood_inv_* slugs read scenarios_nonfood.csv
    tables; everything else the food set)."""
    return "nonfood" if slug.startswith("nonfood_") else "food"


def _grid_prior_active(slug, config):
    """Whether this run puts an informative Beta prior on a grid latent
    (desire/intimacy) for `slug`, so the fit vector gains the fitted `prior_nu`
    at index `n_core`. False for the preregistered uniform-prior run and for
    effort-only priors (the 2-state effort prior adds no shape parameter)."""
    return any(lat in ("desire", "intimacy") for lat in config.active_latents(slug))


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
    extra_param_names=(),
):
    """One `cv_folds.jsonl` diagnostic row. The param vector layout is
    `[*utility_param_names, alpha_observer, sigma, *extra_param_names]` — the
    same names-based flattening as `restart_records_to_rows`, so an
    informative-prior fit's `prior_nu` (in `extra_param_names`) is read from its
    real index rather than mis-indexed off the vector's tail. Every param is
    written `param_<name>` except `alpha_observer`, which stays bare."""
    row = {
        "experiment": slug,
        "variant": variant,
        "fold": fold,
        "held_out_scenario": scenario_label,
        "train_nll": float(train_nll),
        "test_nll": float(test_nll),
        "n_train": int(n_train),
        "n_test": int(n_test),
    }
    names = (
        list(utility_param_names)
        + ["alpha_observer", "sigma"]
        + list(extra_param_names)
    )
    for i, name in enumerate(names):
        key = name if name == "alpha_observer" else f"param_{name}"
        row[key] = float(params_arr[i])
    return row


def _read_fit_results(fit_dir):
    """Parse `fit_results.json` from an explicit fit directory into the
    per-variant params dicts the warm start needs.

    Mirrors `_helpers.load_fit_results` but reads the RunConfig's own fit dir
    (informative/suffixed runs write outside the preregistered outputs/<slug>/) and
    additionally carries `param_prior_nu` when present, so the extended
    informative-prior warm-start vector round-trips through
    `params_dict_to_array(..., extra_param_names=("prior_nu",))`."""
    with open(Path(fit_dir) / "fit_results.json") as f:
        rows = json.load(f)
    out = {}
    for row in rows:
        variant = str(row["model"])
        params = {
            "alpha": float(row.get("param_alpha", 1.0)),
            "alpha_observer": float(row["alpha_observer"]),
        }
        if row.get("param_sigma") is not None:
            params["sigma"] = float(row["param_sigma"])
        # Every optimizer-vector member that fit_results.json stores under a
        # `param_` prefix must be listed, or the fold's warm start cannot be
        # rebuilt. `eta` was missing until 2026-07-31, which made CV raise
        # KeyError on all 12 reweighted (study, variant) pairs; the round-trip is
        # now pinned by test_fit_protocol.py so a future extra fails in the suite
        # rather than an hour into a CV run.
        for pn in ("w_v", "w_d", "w_e", "gamma", "prior_nu", "eta"):
            if row.get(f"param_{pn}") is not None:
                params[pn] = float(row[f"param_{pn}"])
        out[variant] = params
    return out


def _load_verified_warm_start(slug, fit_dir):
    """Warm-start params for CV folds from the full-data fit under THIS run's
    config (`fit_dir` = `config.outputs_dir(slug)`), checking provenance when
    it's available. A missing fit is a loud warning and a cold start (CV still
    runs — it just refits each fold from a lognormal init instead of the
    full-data fit, which is slower and skips the leak-mitigation intent, so a
    fit first is recommended). A fit whose manifest is *present but mismatched*
    is a hard error via verify_fit_manifest (genuine staleness); a fit with no
    manifest warns and is used as-is."""
    fit_dir = Path(fit_dir)
    fit_path = fit_dir / "fit_results.json"
    if not fit_path.exists():
        print(
            f"WARNING: no fit_results.json for {slug} in {fit_dir} — CV will "
            f"cold-start every fold. Run `make fit-{slug}` first for a warm start "
            f"(faster, and it avoids folds depending on an init that saw the "
            f"held-out scenario).",
            file=sys.stderr,
        )
        return {}
    verify_fit_manifest(slug, output_dir=fit_dir)
    return _read_fit_results(fit_dir)


# The three CV output files written together per study; the manifest hashes
# them so model_comparison.py can refuse stale or mixed-vintage combinations.
CV_OUTPUT_NAMES = ("cv_preds_summary.json", "cv_folds.jsonl", "cv_trial_ll.jsonl")


def _write_outputs(slug, pred_rows, fold_rows, trial_ll_rows, outputs_dir):
    if not trial_ll_rows:
        raise RuntimeError(
            f"CV for {slug} scored no held-out trials — refusing to write "
            "empty outputs. Check the data loader and the fold train/test masks."
        )
    outputs_dir = Path(outputs_dir)
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

    # The run is fully written (outputs + manifest), so the fold checkpoint
    # has served its purpose — remove it rather than leave a stale side file
    # for the next run to re-validate. Deleted last, so even a failure in the
    # cosmetic summary above can't cost a resume.
    clear_checkpoint(checkpoint_path(outputs_dir))


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


def _cv_worker_init(family, slug, arrays, config, free_mask=None):
    _CV_W.update(
        family=family, slug=slug, arrays=arrays, config=config, free_mask=free_mask
    )


def _free_mask(variant):
    """The fold refit's free-parameter mask for one variant, or None (estimate
    every parameter — the reported path). Non-None only under a `RunOverride`;
    see `fit_masked` in `model/inverse/_helpers.py`."""
    masks = _CV_W.get("free_mask")
    return None if masks is None else masks[variant]


@functools.lru_cache(maxsize=None)
def _tk_cached(family, slug, variant):
    """Per-process cache of a variant's LM table kwargs (the lm_runs.jsonl load
    is a few MB, so build it once per worker per variant, not once per fold).
    Keyed on (family, slug) too — the joint families serve two slugs each. The
    run config's alternatives vintage comes from `_CV_W["config"]`, which is one
    fixed config per process (parent and each spawn worker), so it need not be a
    cache key."""
    fam = _FAMILIES[family]
    _, utility_names = fam["variants"][variant]
    return fam["table_kwargs"](variant, utility_names, slug)


# Per-process cache of a variant's informative-prior tables (K-broadcast to the
# feature tables' run count). None in uniform mode — the preregistered path never
# reweights.
_PRIORS_CACHE = {}


def _rw_cached(slug, variant):
    """The reweighting config for one (study, variant), or None when the scope
    rule grants none (in which case the fold fit is the preregistered one and carries
    no eta). Cheap enough not to cache, but kept symmetrical with
    `_priors_cached` so the fold bodies read the same way. Utility names come
    from the family registry (as in `_tk_cached`), not the worker state, which
    carries only family/slug/arrays/config.

    `--no-reweighting` (the preregistered model) disables it for every variant;
    the flag is read from the worker's own config copy, so parallel workers and
    the sequential path agree by construction."""
    _, utility_names = _FAMILIES[_CV_W["family"]]["variants"][variant]
    return _reweighting.config_for(
        slug,
        variant,
        list(utility_names),
        enabled=not _CV_W["config"].no_reweighting,
    )


def _priors_cached(slug, variant):
    """Build (once per process, per variant) the `priors=` dict the fold
    reweights the observer posterior with, or None in uniform mode. The elicited
    prior tables carry a leading run axis of 1 when the priors were elicited in a
    single run; tile it to the alternatives tables' K so the per-run reweighting
    broadcasts (matching the fit helpers)."""
    key = (slug, variant == "base")
    if key not in _PRIORS_CACHE:
        cfg = _CV_W["config"]
        pr = build_priors_kwarg(
            slug, cfg, base=priors_base_variant(slug, variant, cfg.priors_file)
        )
        if pr is not None:
            k_tables = _tk_cached(_CV_W["family"], slug, variant)["risk_table"].shape[0]
            pr = {
                k: (
                    jnp.repeat(v, k_tables, axis=0)
                    if v is not None and v.shape[0] == 1 and k_tables > 1
                    else v
                )
                for k, v in pr.items()
            }
        _PRIORS_CACHE[key] = pr
    return _PRIORS_CACHE[key]


def _cv_fold(variant, fold, warm, patience):
    """One leave-one-scenario-out refit + held-out scoring. Reads the shared
    data (and the observer family) from `_CV_W` and dispatches to the family's
    fold body. Top-level + picklable so a ProcessPoolExecutor can run folds
    concurrently; fully deterministic given (variant, fold, warm, patience), so
    the parallel output equals the sequential."""
    return _FAMILIES[_CV_W["family"]]["fold_impl"](variant, fold, warm, patience)


@contextlib.contextmanager
def _capped_worker_threads(n_threads=1):
    """Cap the XLA/OpenMP thread pools of the spawn workers while the pool is
    alive. Each worker re-imports JAX and would otherwise spin up its own
    full-width XLA CPU thread pool — CV_WORKERS × cores threads oversubscribe
    the machine. Spawn children inherit os.environ at process creation (setting
    env in the pool initializer would be too late: the child imports jax while
    unpickling the module), so the caps are set in the parent right before the
    pool starts and restored right after. The parent's already-initialized JAX
    is unaffected. Explicit user-set values are respected.

    `n_threads` > 1 gives each worker a small multi-threaded pool instead of a
    single-threaded one — useful when a family's worker count is capped below
    the core count (as the joint families' was while their observers were
    memory-bound; no family defaults to it today). Keep workers × threads ≲
    the machine's cores."""
    saved = {k: os.environ.get(k) for k in ("XLA_FLAGS", "OMP_NUM_THREADS")}
    xla = os.environ.get("XLA_FLAGS", "")
    # Match on the flag NAME (not name=value), so a user-set value for either
    # flag — whatever it is — is left alone instead of being overridden by an
    # appended duplicate (XLA parses the last occurrence).
    add = [
        f
        for f in (
            f"--xla_cpu_multi_thread_eigen={'false' if n_threads == 1 else 'true'}",
            f"intra_op_parallelism_threads={n_threads}",
        )
        if f.split("=")[0] not in xla
    ]
    if add:
        os.environ["XLA_FLAGS"] = " ".join(([xla] if xla else []) + add)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@dataclass(frozen=True)
class RunOverride:
    """Substitutions that turn `_run_loso` into a non-reported run.

    Two users, both exploratory analyses that need a fold to start from
    parameters this study's own data did not produce and to hold some of them
    fixed: the cross-study transfer analysis (`model/cv/transfer.py`), which
    starts from a DONOR study's fit, and the pooled fits (`model/cv/pooled.py`),
    which start from a utility shared across several studies. Everything here
    defaults to None/(), and `None` for the whole override is the reported run
    — so the reported path never sees a branch it did not have before.

      variants     restrict which ablations run; () = the family's full set.
      init_params  {variant: vector} -- or {(variant, fold): vector} when the
                   start differs per fold -- replacing the warm start normally
                   read from this study's own full-data fit. The per-fold form
                   is what the pooled fits use: each fold has its own shared
                   utility, estimated on that fold's training scenarios.
      free_mask    {variant: boolean mask over that vector}; False slots stay at
                   their init value. An all-False mask estimates nothing, which
                   is the zero-free-parameter transfer arm.
      outputs_dir  where the fold checkpoint lives (the caller writes the final
                   outputs itself).
      fingerprint  extra fields folded into the checkpoint fingerprint, so a
                   transfer run can never resume from the reported run's folds
                   — or from another donor's.
    """

    variants: tuple = ()
    init_params: dict | None = None
    free_mask: dict | None = None
    outputs_dir: Path | None = None
    fingerprint: dict | None = None


def _run_loso(family, slug, workers=None, patience=None, config=None, override=None):
    """LOSO CV for one study. Runs the (variant × fold) refits concurrently when
    `workers` > 1 (env `CV_WORKERS`; default from the family registry): folds
    are independent and each refit is deterministic, so the output is identical
    to the sequential run — only the execution overlaps. Each worker's XLA pool
    gets the family's `worker_threads` (env `CV_WORKER_THREADS` overrides).
    `patience` (env `CV_PATIENCE`, default 100) trims the Adam no-improvement
    tail of each warm-started refit.

    `config` (a RunConfig; default the preregistered uniform-prior config) selects
    the alternatives vintage, whether the observer posterior is reweighted by an
    informative prior, which fit dir warm-starts the folds, and where the
    outputs land. The default keeps the preregistered path byte-identical.

    `override` (a `RunOverride`; default None) replaces the warm start, freezes
    part of the parameter vector, restricts the variant set, and redirects the
    checkpoint — everything the cross-study transfer analysis needs. None is the
    reported run.

    Every completed fold is appended to `<outputs_dir>/cv_checkpoint.jsonl`
    (fingerprint-guarded; see _checkpoint.py), and completed folds found there
    at startup are skipped — so an interrupted multi-hour run resumes instead
    of starting over. The final outputs are still written only when every fold
    is present, so consumers never see a partial set."""
    config = config or RunConfig()
    fam = _FAMILIES[family]
    # Worker count: explicit arg, then env CV_WORKERS, then the family default
    # from the registry (the joint families default lower — memory-bound).
    # An empty CV_WORKERS falls through to the family default.
    workers = (
        workers
        if workers is not None
        else int(os.environ.get("CV_WORKERS") or fam["default_workers"])
    )
    patience = (
        patience if patience is not None else int(os.environ.get("CV_PATIENCE", "100"))
    )
    worker_threads = int(os.environ.get("CV_WORKER_THREADS") or fam["worker_threads"])
    if worker_threads < 1:
        raise ValueError(
            f"CV_WORKER_THREADS must be >= 1, got {worker_threads} — XLA/OpenMP "
            f"thread pools need a positive thread count."
        )
    arrays = fam["load_arrays"](slug)
    override = override or RunOverride()
    variants = {
        v: spec
        for v, spec in fam["variants"].items()
        if not override.variants or v in override.variants
    }
    if override.variants and set(override.variants) - set(fam["variants"]):
        raise ValueError(
            f"override.variants {sorted(set(override.variants) - set(fam['variants']))} "
            f"are not {family} ablations ({sorted(fam['variants'])})"
        )
    # Populate the parent's worker state up front: the warm-up `_tk_cached`
    # calls below read the run config's alternatives suffix from `_CV_W`, and
    # the sequential path reuses this same state. Spawn workers get their own
    # copy via the pool `initargs`.
    _cv_worker_init(family, slug, arrays, config, override.free_mask)
    # The informative-prior fit appends a fitted `prior_nu` to the param vector
    # (index `n_core`), so the warm start must carry it too — otherwise a length
    # mismatch is silently mis-sliced. Uniform runs keep the bare vector.
    extra = ("prior_nu",) if _grid_prior_active(slug, config) else ()

    # The reweighted fit appends `eta` LAST (after any prior_nu), and whether a
    # variant has one is decided per variant by the scope rule — so the warm
    # start's extras are per variant, not shared.
    def _extras(variant, util):
        return extra + (
            ("eta",)
            if _reweighting.uses_reweighting(
                slug, list(util), enabled=not config.no_reweighting
            )
            else ()
        )

    # Warm-start source: the full-data fit under THIS config (refits perturb it
    # only slightly), provenance-verified against fit_manifest.json and the data.
    # A RunOverride supplies its own starting vectors instead — for the transfer
    # analysis those carry the donor study's utility weights, which no fit of
    # this study could provide.
    if override.init_params is not None:
        warms = {v: None for v in variants}  # resolved per fold below
    else:
        full_fit = _load_verified_warm_start(slug, fit_dir=config.outputs_dir(slug))
        warms = {
            v: (
                np.asarray(
                    params_dict_to_array(
                        full_fit[v], util, extra_param_names=_extras(v, util)
                    )
                )
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

    def warm_for(variant, fold):
        """This fold's starting vector. Without an override it is the study's
        full-data fit (one per variant); an override may instead key its vectors
        by (variant, fold), which is how the pooled fits hand each fold the
        shared utility estimated on that fold's training scenarios."""
        ip = override.init_params
        if ip is None:
            return warms[variant]
        return np.asarray(ip[(variant, fold)] if (variant, fold) in ip else ip[variant])

    # Resume any completed folds from an interrupted run's checkpoint. The
    # fingerprint ties them to this run's exact inputs, run config, and refit
    # config, so a resume can never splice folds from different vintages; keys
    # outside this run's job list are dropped rather than trusted.
    outputs_dir = override.outputs_dir or config.outputs_dir(slug)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_path(outputs_dir)
    fingerprint = run_fingerprint(
        slug,
        family,
        patience,
        N_RESTARTS_CV,
        config_fields={
            "tag": config.tag() if not config.is_default else "reported",
            "runs": "lm_runs.jsonl",
            "priors": (
                config.priors_filename(False)
                if config.priors_mode != "uniform"
                else None
            ),
            "fit_dir": str(config.outputs_dir(slug)),
            **(override.fingerprint or {}),
        },
    )
    results = {
        k: v for k, v in init_checkpoint(ckpt, fingerprint).items() if k in set(jobs)
    }
    pending = [j for j in jobs if j not in results]
    if results:
        print(
            f"  resuming from checkpoint: {len(results)}/{len(jobs)} "
            f"(variant × fold) refits already done"
        )

    if pending and workers > 1:
        print(
            f"  parallel {family} CV: {workers} workers × {worker_threads} "
            f"thread(s), patience={patience}"
        )
        with _capped_worker_threads(worker_threads):
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=_cv_worker_init,
                initargs=(family, slug, arrays, config, override.free_mask),
            ) as ex:
                futs = {
                    ex.submit(_cv_fold, v, f, warm_for(v, f), patience): (v, f)
                    for v, f in pending
                }
                try:
                    for fu in as_completed(futs):
                        v, f = futs[fu]
                        res = fu.result()
                        results[(v, f)] = res
                        append_fold(ckpt, v, f, *res)
                        print(
                            f"    [{len(results)}/{len(jobs)}] {slug} / {v} / "
                            f"fold {f + 1}/{n_folds} done",
                            flush=True,
                        )
                except BaseException:
                    # A failed refit dooms the run (the final outputs need every
                    # fold), so drop the queued jobs and surface the error now
                    # rather than after the remaining folds burn hours of
                    # compute. Already-running folds still finish (but only the
                    # ones already consumed above reached the checkpoint).
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
    elif pending:
        # `_CV_W` is already populated in the parent above; the sequential path
        # runs folds in-process, so no re-init is needed.
        for v, f in pending:
            res = _cv_fold(v, f, warm_for(v, f), patience)
            results[(v, f)] = res
            append_fold(ckpt, v, f, *res)

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
    priors = _priors_cached(slug, variant)  # None in uniform mode
    rw = _rw_cached(slug, variant)  # None where the scope rule grants none
    use_grid = priors is not None and priors.get("m_latent") is not None
    n_core = len(utility_names) + 2
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
        priors=priors,
        reweighting=rw,
        verbose=False,
        n_restarts=N_RESTARTS_CV,
        init_params=warm,
        patience=patience,
        seed_key=f"{slug}|{variant}|{scenario_label}",
        free_mask=_free_mask(variant),
    )
    sigma = float(params[n_core - 1])
    nu = float(params[n_core]) if use_grid else None
    # (run, slot, scenario, observed_action, desire, effort, intimacy_101)
    tables = np.asarray(
        _build_observer_tables_runs(
            obs_fn,
            params[:n_core],
            utility_names,
            _reweighting.apply(rw, tk, params[:n_core], params[-1] if rw else 0.0),
        )
    )

    # Predicted belief update δ per held-out cell (mean over runs). In
    # informative mode the intimacy posterior is reweighted by its per-cell Beta
    # prior before the mean (matching the fit's likelihood layer).
    pred_rows = []
    for a_idx in range(N_ACTIONS):
        for r in (0, 1):
            for e in (0, 1):
                density_runs = tables[:, 0, fold, a_idx, r, e, :]  # (K, 101)
                if use_grid:
                    w = beta_prior_on_grid(priors["m_latent"][:, fold, r, e], nu)
                    density_runs = np.asarray(
                        reweight_grid(jnp.asarray(density_runs), w)
                    )
                    lat_pm = np.asarray(w @ GRID_NP)
                else:
                    lat_pm = PRIOR_MEAN_F
                deltas = delta_latent(density_runs, GRID_NP, lat_pm)  # (K,)
                pred_rows.append(
                    {
                        "experiment": slug,
                        "scenario_label": scenario_label,
                        "action": a_idx,
                        "desire_condition": "low" if r == 0 else "high",
                        "effort_condition": "low" if e == 0 else "high",
                        "delta_intimacy": float(deltas.mean()),
                        # The K per-run deltas behind that mean — the mixture's
                        # components. See PER_RUN_DELTA_KEYS.
                        "delta_intimacy_runs": [float(x) for x in deltas],
                        "model": variant,
                    }
                )

    # Per-trial held-out log-likelihood under the mixture.
    trial_ll_rows = []
    ti = np.where(test_mask)[0]
    if len(ti):
        post = tables[:, 0, sc[ti], act[ti], des[ti], eff[ti], :]  # (K, n_test, 101)
        if use_grid:
            w_t = beta_prior_on_grid(
                priors["m_latent"][:, sc[ti], des[ti], eff[ti]], nu
            )
            post = np.asarray(reweight_grid(jnp.asarray(post), w_t))
            lat_pm_t = np.asarray((w_t * GRID_NP).sum(-1))  # (K, n_test)
        else:
            lat_pm_t = PRIOR_MEAN_F
        deltas_t = delta_latent(post, GRID_NP, lat_pm_t).T  # (n_test, K)
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
        extra_param_names=(("prior_nu",) if use_grid else ())
        + (("eta",) if rw else ()),
    )
    return pred_rows, fold_row, trial_ll_rows


def main_intimacy(config=None):
    config = config or parse_run_config_args()
    slug = "food_inv_intimacy"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(
        slug,
        *_run_loso("intimacy", slug, config=config),
        outputs_dir=config.outputs_dir(slug),
    )


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
    priors = _priors_cached(slug, variant)  # None in uniform mode
    rw = _rw_cached(slug, variant)  # None where the scope rule grants none
    use_grid = priors is not None and priors.get("m_latent") is not None
    n_core = len(utility_names) + 2
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
        priors=priors,
        reweighting=rw,
        verbose=False,
        n_restarts=N_RESTARTS_CV,
        init_params=warm,
        patience=patience,
        seed_key=f"{slug}|{variant}|{scenario_label}",
        free_mask=_free_mask(variant),
    )
    sigma = float(params[n_core - 1])
    nu = float(params[n_core]) if use_grid else None
    # (run, slot, scenario, observed_action, effort, intimacy, desire_101)
    tables = np.asarray(
        _build_observer_tables_runs(
            obs_fn,
            params[:n_core],
            utility_names,
            _reweighting.apply(rw, tk, params[:n_core], params[-1] if rw else 0.0),
        )
    )

    pred_rows = []
    for a_idx in range(N_ACTIONS):
        for rel_idx in range(4):
            for e in (0, 1):
                post = tables[:, 0, fold, a_idx, e, rel_idx, :]  # (K, 101)
                if use_grid:
                    w = beta_prior_on_grid(priors["m_latent"][:, fold, e, rel_idx], nu)
                    post = np.asarray(reweight_grid(jnp.asarray(post), w))
                    lat_pm = np.asarray(w @ GRID_NP)
                else:
                    lat_pm = PRIOR_MEAN_F
                deltas = delta_latent(post, GRID_NP, lat_pm)
                pred_rows.append(
                    {
                        "experiment": slug,
                        "scenario_label": scenario_label,
                        "action": a_idx,
                        "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel_idx],
                        "effort_condition": "low" if e == 0 else "high",
                        "delta_desire": float(deltas.mean()),
                        # The K per-run deltas behind that mean — the mixture's
                        # components. See PER_RUN_DELTA_KEYS.
                        "delta_desire_runs": [float(x) for x in deltas],
                        "model": variant,
                    }
                )

    trial_ll_rows = []
    ti = np.where(test_mask)[0]
    if len(ti):
        post = tables[:, 0, sc[ti], act[ti], eff[ti], rel[ti], :]  # (K, n_test, 101)
        if use_grid:
            w_t = beta_prior_on_grid(
                priors["m_latent"][:, sc[ti], eff[ti], rel[ti]], nu
            )
            post = np.asarray(reweight_grid(jnp.asarray(post), w_t))
            lat_pm_t = np.asarray((w_t * GRID_NP).sum(-1))  # (K, n_test)
        else:
            lat_pm_t = PRIOR_MEAN_F
        deltas_t = delta_latent(post, GRID_NP, lat_pm_t).T
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
        extra_param_names=(("prior_nu",) if use_grid else ())
        + (("eta",) if rw else ()),
    )
    return pred_rows, fold_row, trial_ll_rows


def main_desire(config=None):
    config = config or parse_run_config_args()
    slug = "food_inv_desire"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(
        slug,
        *_run_loso("desire", slug, config=config),
        outputs_dir=config.outputs_dir(slug),
    )


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
    priors = _priors_cached(slug, variant)  # None in uniform mode
    rw = _rw_cached(slug, variant)  # None where the scope rule grants none
    use_grid = priors is not None and priors.get("m_latent") is not None
    use_eff = priors is not None and priors.get("p_effort") is not None
    n_core = len(utility_names) + 2
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
        priors=priors,
        reweighting=rw,
        verbose=False,
        n_restarts=N_RESTARTS_CV,
        init_params=warm,
        patience=patience,
        seed_key=f"{slug}|{variant}|{scenario_label}",
        free_mask=_free_mask(variant),
    )
    sigma = float(params[n_core - 1])
    nu = float(params[n_core]) if use_grid else None
    # (run, slot, scenario, observed_action, relationship_4, desire_101, effort_2)
    tables = np.asarray(
        _build_observer_tables_runs(
            obs_fn,
            params[:n_core],
            utility_names,
            _reweighting.apply(rw, tk, params[:n_core], params[-1] if rw else 0.0),
        )
    )

    # In informative mode the (desire, effort) joint is reweighted by the
    # per-cell desire Beta prior and/or the elicited P(effort=high) before the
    # marginal means, matching the fit's likelihood layer; the uniform-prior path
    # passes None/None (reweight_joint returns the joint unchanged).
    pred_rows = []
    for a_idx in range(N_ACTIONS):
        for rel_idx in range(4):
            joint_runs = tables[:, 0, fold, a_idx, rel_idx, :, :]  # (K, 101, 2)
            if use_grid:
                w = beta_prior_on_grid(priors["m_latent"][:, fold, rel_idx], nu)
                lat_pm = np.asarray(w @ GRID_NP)
            else:
                w, lat_pm = None, PRIOR_MEAN_F
            if use_eff:
                p = priors["p_effort"][:, fold, rel_idx]
                eff_pm = np.asarray(p)
            else:
                p, eff_pm = None, EFFORT_PRIOR_MEAN_F
            joint_runs = np.asarray(reweight_joint(jnp.asarray(joint_runs), w, p))
            d_desire, d_effort = delta_joint(joint_runs, GRID_NP, lat_pm, eff_pm)
            pred_rows.append(
                {
                    "experiment": slug,
                    "scenario_label": scenario_label,
                    "action": a_idx,
                    "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel_idx],
                    "delta_desire": float(d_desire.mean()),
                    "delta_effort": float(d_effort.mean()),
                    # The K per-run deltas behind those means — the mixture's
                    # components. See PER_RUN_DELTA_KEYS.
                    "delta_desire_runs": [float(x) for x in d_desire],
                    "delta_effort_runs": [float(x) for x in d_effort],
                    "model": variant,
                }
            )

    trial_ll_rows = []
    ti = np.where(test_mask)[0]
    if len(ti):
        joint_t = tables[:, 0, sc[ti], act[ti], rel[ti], :, :]  # (K, n_test, 101, 2)
        if use_grid:
            w_t = beta_prior_on_grid(priors["m_latent"][:, sc[ti], rel[ti]], nu)
            lat_pm_t = np.asarray((w_t * GRID_NP).sum(-1))  # (K, n_test)
        else:
            w_t, lat_pm_t = None, PRIOR_MEAN_F
        if use_eff:
            p_t = priors["p_effort"][:, sc[ti], rel[ti]]  # (K, n_test)
            eff_pm_t = np.asarray(p_t)
        else:
            p_t, eff_pm_t = None, EFFORT_PRIOR_MEAN_F
        joint_t = np.asarray(reweight_joint(jnp.asarray(joint_t), w_t, p_t))
        d_desire_t, d_effort_t = delta_joint(joint_t, GRID_NP, lat_pm_t, eff_pm_t)
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
        extra_param_names=(("prior_nu",) if use_grid else ())
        + (("eta",) if rw else ()),
    )
    return pred_rows, fold_row, trial_ll_rows


def main_joint_de(slug="food_inv_joint_de", config=None):
    config = config or parse_run_config_args()
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(
        slug,
        *_run_loso("joint_de", slug, config=config),
        outputs_dir=config.outputs_dir(slug),
    )


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
    priors = _priors_cached(slug, variant)  # None in uniform mode
    rw = _rw_cached(slug, variant)  # None where the scope rule grants none
    use_grid = priors is not None and priors.get("m_latent") is not None
    use_eff = priors is not None and priors.get("p_effort") is not None
    n_core = len(utility_names) + 2
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
        priors=priors,
        reweighting=rw,
        verbose=False,
        n_restarts=N_RESTARTS_CV,
        init_params=warm,
        patience=patience,
        seed_key=f"{slug}|{variant}|{scenario_label}",
        free_mask=_free_mask(variant),
    )
    sigma = float(params[n_core - 1])
    nu = float(params[n_core]) if use_grid else None
    # (run, slot, scenario, observed_action, desire, intimacy_101, effort_2)
    tables = np.asarray(
        _build_observer_tables_runs(
            obs_fn,
            params[:n_core],
            utility_names,
            _reweighting.apply(rw, tk, params[:n_core], params[-1] if rw else 0.0),
        )
    )

    # In informative mode the (intimacy, effort) joint is reweighted by the
    # per-cell intimacy Beta prior and/or the elicited P(effort=high) before the
    # marginal means; the uniform-prior path passes None/None (joint unchanged).
    pred_rows = []
    for a_idx in range(N_ACTIONS):
        for r in (0, 1):
            joint_runs = tables[:, 0, fold, a_idx, r, :, :]  # (K, 101, 2)
            if use_grid:
                w = beta_prior_on_grid(priors["m_latent"][:, fold, r], nu)
                lat_pm = np.asarray(w @ GRID_NP)
            else:
                w, lat_pm = None, PRIOR_MEAN_F
            if use_eff:
                p = priors["p_effort"][:, fold, r]
                eff_pm = np.asarray(p)
            else:
                p, eff_pm = None, EFFORT_PRIOR_MEAN_F
            joint_runs = np.asarray(reweight_joint(jnp.asarray(joint_runs), w, p))
            d_intimacy, d_effort = delta_joint(joint_runs, GRID_NP, lat_pm, eff_pm)
            pred_rows.append(
                {
                    "experiment": slug,
                    "scenario_label": scenario_label,
                    "action": a_idx,
                    "desire_condition": "low" if r == 0 else "high",
                    "delta_intimacy": float(d_intimacy.mean()),
                    "delta_effort": float(d_effort.mean()),
                    # The K per-run deltas behind those means — the mixture's
                    # components. See PER_RUN_DELTA_KEYS.
                    "delta_intimacy_runs": [float(x) for x in d_intimacy],
                    "delta_effort_runs": [float(x) for x in d_effort],
                    "model": variant,
                }
            )

    trial_ll_rows = []
    ti = np.where(test_mask)[0]
    if len(ti):
        joint_t = tables[:, 0, sc[ti], act[ti], des[ti], :, :]  # (K, n_test, 101, 2)
        if use_grid:
            w_t = beta_prior_on_grid(priors["m_latent"][:, sc[ti], des[ti]], nu)
            lat_pm_t = np.asarray((w_t * GRID_NP).sum(-1))  # (K, n_test)
        else:
            w_t, lat_pm_t = None, PRIOR_MEAN_F
        if use_eff:
            p_t = priors["p_effort"][:, sc[ti], des[ti]]  # (K, n_test)
            eff_pm_t = np.asarray(p_t)
        else:
            p_t, eff_pm_t = None, EFFORT_PRIOR_MEAN_F
        joint_t = np.asarray(reweight_joint(jnp.asarray(joint_t), w_t, p_t))
        d_intimacy_t, d_effort_t = delta_joint(joint_t, GRID_NP, lat_pm_t, eff_pm_t)
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
        extra_param_names=(("prior_nu",) if use_grid else ())
        + (("eta",) if rw else ()),
    )
    return pred_rows, fold_row, trial_ll_rows


def main_joint_ie(slug="food_inv_joint_ie", config=None):
    config = config or parse_run_config_args()
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(
        slug,
        *_run_loso("joint_ie", slug, config=config),
        outputs_dir=config.outputs_dir(slug),
    )


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
        "default_workers": DEFAULT_WORKERS,
        "worker_threads": DEFAULT_WORKER_THREADS,
    },
    "desire": {
        "variants": VARIANTS_DESIRE,
        "load_arrays": _load_arrays_desire,
        "table_kwargs": _tk_desire,
        "fold_impl": _fold_impl_desire,
        "default_workers": DEFAULT_WORKERS,
        "worker_threads": DEFAULT_WORKER_THREADS,
    },
    "joint_de": {
        "variants": VARIANTS_JOINT_DE,
        "load_arrays": _load_arrays_joint_de,
        "table_kwargs": _tk_joint_de,
        "fold_impl": _fold_impl_joint_de,
        "default_workers": DEFAULT_WORKERS,
        "worker_threads": DEFAULT_WORKER_THREADS,
    },
    "joint_ie": {
        "variants": VARIANTS_JOINT_IE,
        "load_arrays": _load_arrays_joint_ie,
        "table_kwargs": _tk_joint_ie,
        "fold_impl": _fold_impl_joint_ie,
        "default_workers": DEFAULT_WORKERS,
        "worker_threads": DEFAULT_WORKER_THREADS,
    },
}
