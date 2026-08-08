"""
Shared infrastructure for the inverse-planning fit scripts.

Each experiment has its own thin fit script that imports the helpers it needs
from this module (CV reuses the same helpers via the dispatcher; there is no
separate predict step). Shared concerns:

  - Loss functions (the belief-update Gaussian-mixture NLLs mixture_nll_1d / _2d)
  - Observer fit loops (joint padded utility weights + α_observer + σ)
  - Data loaders (per experiment, returning per-trial belief updates)
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax.scipy.special import logsumexp

import _reweighting
from _priors import beta_prior_on_grid, reweight_grid, reweight_joint
from tables import (
    ACTION_LABEL_TO_IDX,
    DesireLevels,
    EFFORT_CONDITION_TO_IDX,
    INTIMACY_CONDITION_TO_IDX,
    scenario_to_idx_for_study,
)
from utils import get_project_root


def parse_run_config_args(argv=None, description=None):
    """Shared CLI for the fit wrappers and CV scripts: the run configuration.
    The default (no flags) is the reported config — uniform priors, the
    unsuffixed lm_runs.jsonl vintage, the comparison-set reweighting where its
    scope rule applies, outputs to outputs/<slug>/ — so a plain invocation stays
    byte-identical to the pre-config pipeline."""
    import argparse

    from run_config import RunConfig

    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--priors",
        default="uniform",
        help="uniform | informative | informative:<latent,...> (e.g. informative:desire)",
    )
    p.add_argument(
        "--priors-file",
        default=None,
        help="override the priors JSONL name (e.g. lm_priors_human.jsonl)",
    )
    p.add_argument(
        "--no-reweighting",
        action="store_true",
        help="fit the PREREGISTERED model: no comparison-set reweighting, no eta "
        "parameter. Outputs go to outputs/<slug>/alt/<tag>/, never over the "
        "reported ones.",
    )
    a = p.parse_args(argv)
    return RunConfig.parse(a.priors, a.priors_file, a.no_reweighting)


# Optional upper bound on the observer inverse temperature. DEFAULT: OFF.
#
# Why there is a bound to speak of at all: under the float32 power-form
# sharpening this project used previously, the likelihood was not representable
# for alpha_observer above roughly 15-20 (a diffuse latent row's entries, raised
# to that power, underflow — see observers._sharpened_posterior_logspace). Every
# fit was therefore silently confined below that ceiling. Computing the
# likelihood correctly removes the fence, and doing so reveals a second, genuine
# local optimum for Study 1b at alpha_observer ~ 30 which a held-out check shows
# GENERALIZES BETTER (+0.0047 [+0.0020, +0.0074] per trial).
#
# The bound stays off because the preregistration specified a model and maximum
# likelihood, not a ceiling on alpha_observer: reporting the actual MLE is
# fidelity to it. Imposing one would preserve an arithmetic artifact and, worse,
# handicap the ablation baselines that `full - ablation` is measured against.
#
# The reproducibility concern a bound would address — two well-separated optima,
# so the reported fit depends on where the optimizer starts — is handled instead
# by ALPHA_OBS_SEEDS below: every fit deliberately starts from both basins and
# keeps the better one, and `fit_restarts.jsonl` records what each restart found.
# The bound machinery is kept for diagnostic use — set ALPHA_OBS_MAX to a float
# to re-enable it.
ALPHA_OBS_MAX = None

# Explicit alpha_observer starting points, so a fit covers both known basins
# rather than depending on luck. Low ~ the preregistered-regime optimum; high ~
# the sharper optimum reachable since the log-space rewrite.
ALPHA_OBS_SEEDS = (3.0, 30.0)

# Restarts per full-data fit, before the ALPHA_OBS_SEEDS basin seeds are added.
# 3 gives 1 all-ones init + 2 cold lognormal draws, and the two basin seeds bring
# the total to 5 -- the same cost as fits before the basin seeds existed, while
# still covering both alpha_observer optima deterministically rather than relying
# on a random draw landing in the better one. Raise for a more thorough sweep of
# the gamma power law's local minima; the basin coverage is independent of it.
# CV fold refits are unaffected: they pass n_restarts=CV_RESTARTS explicitly and
# warm-start from the full-data fit, which has already explored both basins.
N_RESTARTS_FIT = int(os.environ.get("FIT_RESTARTS", "3"))


def param_upper_bounds(n_params, alpha_obs_index, alpha_max=None):
    """Elementwise upper bounds for the fit's parameter vector, or None when no
    bound is active (the default — see ALPHA_OBS_MAX). `alpha_obs_index` is
    len(utility_param_names), the slot the fit helpers place it in."""
    alpha_max = ALPHA_OBS_MAX if alpha_max is None else alpha_max
    if alpha_max is None:
        return None
    upper = np.full(n_params, np.inf)
    upper[alpha_obs_index] = float(alpha_max)
    return jnp.asarray(upper)


def _fit_with_adam(
    loss_fn,
    init_params,
    lr=0.01,
    max_steps=5000,
    verbose=True,
    label="",
    patience=100,
    tol=1e-6,
    grad_fn=None,
    upper=None,
):
    """Adam fit loop with non-negativity clipping, best-so-far tracking, and a
    patience stop.

    `upper` (optional, elementwise) caps parameters from above after each step —
    used to bound alpha_observer (see ALPHA_OBS_MAX). The init is clipped to the
    box too, so a cold restart drawn above the bound starts inside it rather
    than walking down from outside.

    Adam is not monotone even on full-batch problems, so the loop keeps the
    best (params, NLL) seen so far and stops once the best NLL hasn't improved
    by more than `tol` for `patience` consecutive steps. Returns the tracked
    best iterate, not the last one. A non-finite NLL abandons the restart
    (returned best_nll stays inf if no finite NLL was ever seen, so
    `_fit_multistart` counts it as failed rather than keeping the init).
    """
    params = jnp.array(init_params)
    if upper is not None:
        params = jnp.minimum(params, upper)
    # jit the value+grad so the whole per-step graph (K-run observer build →
    # mixture NLL → backward) compiles once and reruns fast, instead of being
    # re-dispatched eagerly every Adam step. Compile cost amortizes over the
    # fit's steps; the heavy joint observers stay compute-bound (see note).
    # `_fit_multistart` passes a pre-jitted `grad_fn` so the compilation is
    # shared across restarts instead of redone per restart.
    if grad_fn is None:
        grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    best_params = params
    best_nll = jnp.inf
    steps_without_improvement = 0
    for step in range(max_steps):
        nll, grad = grad_fn(params)  # NLL at the current params, pre-update
        if not jnp.isfinite(nll):
            # NaN/inf loss: the gradient is unusable and every later step would
            # inherit the NaN, so stop here. best_nll keeps its last finite
            # value (inf if none), flagging this restart as failed.
            if verbose:
                print(f"  {label} non-finite NLL at step {step}, abandoning restart")
            break
        if nll < best_nll - tol:
            best_nll = nll
            best_params = params
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if verbose and step % 1000 == 0:
            print(f"  Step {step}, NLL: {nll:.4f}, params: {params}")

        if steps_without_improvement >= patience:
            if verbose:
                print(f"  No improvement for {patience} steps, stopping at step {step}")
            break

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 1e-6, jnp.inf if upper is None else upper)

    best_nll = float(best_nll)
    if verbose:
        print(f"  {label} final NLL: {best_nll:.4f}, params: {best_params}")
    return best_params, best_nll


def _restart_seed(seed_key):
    """Deterministic 64-bit seed for the cold-restart RNG from a string key
    (e.g. "slug|variant|held_out_scenario"). SHA-256, mirroring `_seed_for` in
    model/lm/score_merged.py (Python's builtin `hash` is salted per process),
    so every (study, variant, fold) gets decorrelated inits while staying
    reproducible across reruns."""
    return int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little")


def _fit_multistart(
    loss_fn,
    n_params,
    n_restarts=N_RESTARTS_FIT,
    lr=0.1,
    max_steps=1000,
    verbose=True,
    label="",
    init_params=None,
    patience=100,
    seed_key=None,
    upper=None,
    alpha_obs_index=None,
):
    """Run `_fit_with_adam` from several inits and keep the best final NLL.

    `upper` is passed through to every restart (see ALPHA_OBS_MAX); cold draws
    above the bound are clipped into the box by `_fit_with_adam`.

    `alpha_obs_index` (the alpha_observer slot) adds one deliberate restart per
    ALPHA_OBS_SEEDS value, each a copy of the primary init with alpha_observer set
    to that seed. This makes a fit cover both known alpha basins reproducibly
    instead of depending on which one a random draw happens to fall into.

    Without `init_params`, inits are the canonical all-ones vector plus
    `n_restarts - 1` seeded lognormal(0, 0.5) draws (positive, centered at 1,
    deterministic via `_restart_seed(seed_key)`), guarding against local minima
    from the gamma power law. When `init_params` is given (a warm start — e.g.
    CV refits seeded from the full-data fit, which a leave-one-scenario-out
    refit only perturbs slightly), it is the first init, and only
    `n_restarts - 1` cold draws are added; with `n_restarts=1` the fit is a
    single warm start.

    Returns (best_params, best_nll, records) where `records` is one dict per
    restart {restart, init, final_params, nll} for stability auditing.
    """
    rng = np.random.default_rng(_restart_seed(seed_key) if seed_key else 0)
    inits = [
        jnp.asarray(init_params, dtype=jnp.float32)
        if init_params is not None
        else jnp.ones(n_params)
    ]
    while len(inits) < n_restarts:
        inits.append(jnp.array(rng.lognormal(mean=0.0, sigma=0.5, size=n_params)))
    # Basin seeds are EXTRA restarts, appended after the cold draws so they never
    # displace them — a CV fold's cold restart is the one init that never saw the
    # held-out scenario, and that guarantee must survive. They are added only for
    # full-data fits (`init_params is None`): a fold refit warm-starts from the
    # full-data fit, which already explored both basins, so it inherits the
    # winning one and keeps its independent cold draw rather than paying for two
    # more fits per fold.
    if alpha_obs_index is not None and init_params is None:
        for seed in ALPHA_OBS_SEEDS:
            inits.append(jnp.asarray(inits[0]).at[alpha_obs_index].set(float(seed)))

    # Compile the value+grad once here and share it across restarts (the loss
    # is identical per restart; only the init differs).
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    best_params, best_nll = None, float("inf")
    records = []
    for ri, init in enumerate(inits):
        params, nll = _fit_with_adam(
            loss_fn,
            init,
            lr=lr,
            max_steps=max_steps,
            verbose=verbose,
            label=f"{label}[restart {ri}]",
            patience=patience,
            grad_fn=grad_fn,
            upper=upper,
        )
        records.append(
            {
                "restart": ri,
                "init": np.asarray(init).tolist(),
                "final_params": np.asarray(params).tolist(),
                "nll": float(nll),
            }
        )
        if nll < best_nll:
            best_nll, best_params = float(nll), params
    if best_params is None:
        raise RuntimeError(
            f"{label or 'fit'}: no restart reached a finite NLL "
            f"({len(inits)} restart(s), seed_key={seed_key!r}) — the loss was "
            "NaN/inf from every init. Check the LM tables and belief-update "
            "data for non-finite values before re-running the fit."
        )
    return best_params, best_nll, records


def fit_masked(
    loss_fn,
    n_params,
    free_mask=None,
    init_params=None,
    alpha_obs_index=None,
    upper=None,
    **kwargs,
):
    """`_fit_multistart` restricted to a subset of the parameter vector.

    `free_mask` is a boolean array over the full vector: True slots are
    estimated, False slots stay at their `init_params` value. `None` (the
    default) means every slot is free, and the call delegates to
    `_fit_multistart` verbatim — the reported fits go through that path
    unchanged.

    A mask is only meaningful relative to a starting vector, so `init_params` is
    required whenever one is given. A mask with NO free slot skips the optimizer
    entirely and returns the init with its loss: that is the cross-study
    transfer analysis's zero-free-parameter arm, where a donor study's
    parameters are scored on a recipient study without any of them being
    estimated there.

    Restarts, bounds, and the returned records all live in the reduced space and
    are mapped back to full-length vectors before returning, so callers — and
    `restart_records_to_rows`, which flattens records by parameter name — see
    the same layout either way.
    """
    if free_mask is None:
        return _fit_multistart(
            loss_fn,
            n_params=n_params,
            init_params=init_params,
            alpha_obs_index=alpha_obs_index,
            upper=upper,
            **kwargs,
        )
    if init_params is None:
        raise ValueError(
            "fit_masked needs init_params when free_mask is given — the frozen "
            "slots have no other source of values."
        )
    free = np.asarray(free_mask, dtype=bool)
    if free.shape != (n_params,):
        raise ValueError(
            f"free_mask has shape {free.shape}, expected ({n_params},) — a mask "
            "of the wrong length would freeze or free the wrong parameters."
        )
    if len(init_params) != n_params:
        # JAX clamps out-of-bounds `.at[]` indices rather than raising, so a
        # short init would silently write a free slot's value into the last
        # position instead of failing — the same trap the fit helpers guard
        # against for a prior_nu-length mismatch.
        raise ValueError(
            f"init_params has length {len(init_params)} but this fit expects "
            f"{n_params} — a masked fit indexes the init by slot, and JAX would "
            "clamp the out-of-bounds writes instead of raising."
        )
    base = jnp.asarray(init_params, dtype=jnp.float32)
    idx = np.flatnonzero(free)

    def _to_full(vec):
        return np.asarray(base.at[jnp.asarray(idx)].set(jnp.asarray(vec))).tolist()

    if idx.size == 0:
        nll = float(loss_fn(base))
        full = np.asarray(base).tolist()
        return (
            base,
            nll,
            [{"restart": 0, "init": full, "final_params": full, "nll": nll}],
        )

    jidx = jnp.asarray(idx)

    def loss_free(theta):
        return loss_fn(base.at[jidx].set(theta))

    # alpha_observer's index within the REDUCED vector (None when it is frozen),
    # so `_fit_multistart`'s basin seeding still targets the right slot.
    free_alpha = (
        int(np.searchsorted(idx, alpha_obs_index))
        if alpha_obs_index is not None and free[alpha_obs_index]
        else None
    )
    best, nll, records = _fit_multistart(
        loss_free,
        n_params=int(idx.size),
        init_params=base[jidx],
        alpha_obs_index=free_alpha,
        upper=None if upper is None else upper[jidx],
        **kwargs,
    )
    records = [
        dict(r, init=_to_full(r["init"]), final_params=_to_full(r["final_params"]))
        for r in records
    ]
    return base.at[jidx].set(best), nll, records


def restart_records_to_rows(
    slug, variant, utility_param_names, records, extra_param_names=()
):
    """Flatten `_fit_multistart` records into rows for fit_restarts.jsonl.

    One row per restart with init_<name> / param_<name> columns (the params
    layout is [*utility_param_names, alpha_observer, sigma, *extra_param_names]).
    `extra_param_names` carries the informative-prior fit's fitted `prior_nu`
    (empty in the preregistered uniform-prior fits). Variants with different
    parameter sets just leave the other variants' columns empty.
    """
    rows = []
    names = (
        list(utility_param_names)
        + ["alpha_observer", "sigma"]
        + list(extra_param_names)
    )
    for rec in records:
        row = {
            "experiment": slug,
            "model": variant,
            "restart": rec["restart"],
            "nll": rec["nll"],
        }
        for i, name in enumerate(names):
            row[f"init_{name}"] = rec["init"][i]
            row[f"param_{name}"] = rec["final_params"][i]
        rows.append(row)
    return rows


# ==============================================================================
# Likelihood: elicitation-sample Gaussian mixture over belief updates
# ==============================================================================
# We fit the human *belief update* u = (posterior rating − prior rating) against
# the model's belief update. Each elicitation run k yields a model update
# δ_k = (posterior mean − prior mean) for the inferred latent, and a participant's
# update is scored under the K-component mixture (1/K) Σ_k N(u | δ_k, σ²); σ is a
# fitted response-noise scale shared across cells. Joint studies (1b, 2b) use a
# bivariate Gaussian per component with a single isotropic σ (covariance σ²·I₂),
# so the cross-dimension correlation comes from the spread of the runs' joint δ_k.

GRID = DesireLevels  # 101-bin [0, 1] latent grid (== IntimacyLevels)
# The model prior over each continuous latent is uniform on the grid; compute its
# mean rather than hardcoding 0.5, so it stays correct if the prior ever changes.
_UNIFORM_PRIOR = jnp.ones_like(GRID) / GRID.shape[0]
PRIOR_MEAN = jnp.dot(_UNIFORM_PRIOR, GRID)
# Effort/world-state is a 2-state latent {low=0, high=1} with a uniform prior; its
# prior mean is likewise computed (= 0.5).
_EFFORT_STATES = jnp.array([0.0, 1.0])
EFFORT_PRIOR_MEAN = jnp.dot(jnp.ones(2) / 2, _EFFORT_STATES)

_LOG_2PI = jnp.log(2.0 * jnp.pi)


def delta_latent(density, grid, prior_mean):
    """Belief-update δ for a 1-D inferred latent (desire or intimacy): the
    posterior mean minus the prior mean, reducing the last (grid) axis.

    Deliberately array-library-agnostic — the fit losses call it with jnp
    arrays and the CV scorer with numpy — so a single definition drives both.
    `density` is (..., n_grid) over `grid` (n_grid,); returns shape (...).
    """
    return density @ grid - prior_mean


def delta_joint(joint, grid, latent_prior_mean, effort_prior_mean):
    """Belief-update δ for a (latent, effort) joint posterior shaped
    (..., n_grid, 2) — last two axes are the latent grid and the two effort
    states {low, high}. Returns (latent_delta, effort_delta), each shaped like
    `joint` with those two axes removed: the latent delta marginalizes effort
    then takes the posterior mean minus prior; the effort delta is
    P(effort=HIGH) minus its prior. The negative-axis reductions make it
    identical for the fit's per-run (K, n_grid, 2) tables and the CV scorer's
    per-run and per-trial (K, n_test, n_grid, 2) tables, so fit and CV score the
    same quantity. See test_model_compliance.test_delta_helpers_*.
    """
    latent_mean = joint.sum(axis=-1) @ grid  # marginalize effort, then mean
    p_effort_high = joint[..., 1].sum(axis=-1)  # sum grid bins of the HIGH slab
    return latent_mean - latent_prior_mean, p_effort_high - effort_prior_mean


@jax.jit
def mixture_nll_1d(u, deltas, sigma):
    """−log[(1/K) Σ_k N(u | δ_k, σ²)] for a scalar belief update `u` and per-run
    model updates `deltas` (shape (K,)). Uses logsumexp for stability."""
    sigma = jnp.clip(sigma, 1e-6, jnp.inf)
    K = deltas.shape[0]
    log_components = -0.5 * (
        _LOG_2PI + 2.0 * jnp.log(sigma) + ((u - deltas) / sigma) ** 2
    )
    return -(logsumexp(log_components) - jnp.log(K))


@jax.jit
def mixture_nll_2d(u_vec, deltas, sigma):
    """Bivariate isotropic (covariance σ²·I₂) analog of `mixture_nll_1d`.

    `u_vec` is the participant's 2-D belief update (2,); `deltas` is the per-run
    model updates (K, 2). The two dimensions' correlation is carried by the
    spread of the runs' joint δ_k, not a fitted off-diagonal term.
    """
    sigma = jnp.clip(sigma, 1e-6, jnp.inf)
    K = deltas.shape[0]
    sq = jnp.sum(((u_vec[None, :] - deltas) / sigma) ** 2, axis=1)  # (K,)
    # 2-D isotropic Gaussian: log|Σ| = log(σ⁴) = 4·log σ, d = 2.
    log_components = -0.5 * (2.0 * _LOG_2PI + 4.0 * jnp.log(sigma) + sq)
    return -(logsumexp(log_components) - jnp.log(K))


# ==============================================================================
# JSON / JSONL output helpers
# ==============================================================================


def write_json(path, obj):
    """Write `obj` as pretty-printed JSON (small structured artifacts:
    fit_results, per-cell prediction summaries)."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_jsonl(path, rows):
    """Write an iterable of dicts as JSON Lines (one record per line: per-restart
    diagnostics, per-trial held-out log-likelihoods)."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    """Read a JSON Lines file into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path):
    """Hex SHA-256 of a file's bytes (fit/CV provenance manifests: written by
    the fit wrappers and the CV dispatcher, verified by model_comparison.py)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_sha():
    """Current commit SHA for provenance manifests; None if git is unavailable
    (e.g. an exported tree) — a manifest still ties its outputs together via
    their content hashes."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


# The fit output files written together per study; fit_manifest.json hashes
# them so the CV warm start and model_comparison.py can refuse a fit that is
# stale relative to the data or a mixed-vintage combination of outputs.
FIT_OUTPUT_NAMES = ("fit_results.json", "fit_restarts.jsonl")


def data_csv_path(slug):
    """The per-trial long CSV a study's fit and CV both consume."""
    return get_project_root() / "data" / slug / "main_trials_long.csv"


def write_fit_manifest(slug, output_dir, data_csv=None):
    """Provenance manifest for a fit run: the git SHA plus content hashes of
    the fit outputs and the input data CSV. Written by the fit_*.py wrappers
    right after the outputs; verified by the CV dispatcher (before
    warm-starting folds from the fit) and by model_comparison.py."""
    output_dir = Path(output_dir)
    data_csv = Path(data_csv) if data_csv is not None else data_csv_path(slug)
    manifest = {
        "experiment": slug,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "outputs": {name: sha256_file(output_dir / name) for name in FIT_OUTPUT_NAMES},
        "input_data": {
            # Relative to the project root when inside it (the normal case);
            # absolute for out-of-tree paths (e.g. test fixtures).
            "path": (
                str(data_csv.relative_to(get_project_root()))
                if data_csv.is_relative_to(get_project_root())
                else str(data_csv)
            ),
            "sha256": sha256_file(data_csv),
        },
    }
    write_json(output_dir / "fit_manifest.json", manifest)
    print(f"Wrote {output_dir / 'fit_manifest.json'}")
    return manifest


def verify_fit_manifest(slug, output_dir=None, data_csv=None):
    """Provenance check for a fit, with a deliberate asymmetry:

      - A *present* fit_manifest.json that no longer matches (the fit outputs
        were partially rewritten, or the input data CSV changed since the fit
        ran) is a hard error — that is genuine, detectable staleness.
      - A *missing* manifest only warns and proceeds. A fit produced before
        provenance tracking existed can't be verified, but refusing to use it
        would break the ability to run CV on a pre-existing fit; we surface the
        limitation rather than force a re-fit.

    Returns the manifest dict, or None when there is no manifest to check."""
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_project_root() / "model" / "outputs" / slug
    )
    data_csv = Path(data_csv) if data_csv is not None else data_csv_path(slug)
    manifest_path = output_dir / "fit_manifest.json"
    if not manifest_path.exists():
        print(
            f"WARNING: no fit_manifest.json for {slug} — this fit predates "
            f"provenance tracking, so it can't be verified against the current "
            f"data. Proceeding; re-run `make fit-{slug}` to record provenance "
            f"(recommended before trusting the final published numbers).",
            file=sys.stderr,
        )
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)
    stale = [
        name
        for name in FIT_OUTPUT_NAMES
        if sha256_file(output_dir / name) != manifest.get("outputs", {}).get(name)
    ]
    if stale:
        raise RuntimeError(
            f"fit output file(s) {stale} do not match fit_manifest.json — stale "
            f"or mixed-vintage fit outputs for {slug}; re-run `make fit-{slug}`."
        )
    if sha256_file(data_csv) != manifest.get("input_data", {}).get("sha256"):
        raise RuntimeError(
            f"{data_csv} changed since the {slug} fit ran — the fit (and any CV "
            f"warm-started from it) is stale; re-run `make fit-{slug}` and then "
            f"`make cv-{slug}`."
        )
    return manifest


# ==============================================================================
# Frozen-param loaders
# ==============================================================================


def load_fit_results(slug: str) -> dict:
    """Load per-variant {actor utility weights + alpha_observer + sigma} for a
    3-action inverse experiment.

    Reads `outputs/<slug>/fit_results.json` (written by the joint
    fit_food_inv_*.py scripts) — a list of per-variant objects. Returns a dict
    mapping variant name (e.g. 'full', 'discomfort_only', 'base') to a params
    dict `{alpha, alpha_observer, sigma, w_v?, w_d?, w_e?, gamma?}` (only the
    weights present for that variant; alpha defaults to 1.0). `sigma` is the
    response-noise scale (a likelihood param; predict need not pass it to the
    observer).
    """
    path = get_project_root() / "model" / "outputs" / slug / "fit_results.json"
    with open(path) as f:
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
        # Must list every optimizer-vector member that fit_results.json stores
        # under a `param_` prefix, including the extras (`prior_nu` for
        # informative-prior runs, `eta` for reweighted variants) -- otherwise a
        # caller round-tripping through params_dict_to_array gets a KeyError, or
        # worse a short vector. The CV dispatcher's own reader had the same
        # omission; see test_fit_protocol.py.
        for pn in ("w_v", "w_d", "w_e", "gamma", "prior_nu", "eta"):
            if row.get(f"param_{pn}") is not None:
                params[pn] = float(row[f"param_{pn}"])
        out[variant] = params
    return out


# ==============================================================================
# Data loaders, table kwargs, and joint fit helpers
# ==============================================================================
# Each active inverse study jointly fits its actor utility weights + α_observer
# from its own posterior data. The joint studies (1b, 2b) marginalize the joint
# observer table to each slider judgment and sum the two per-slider NLLs.


def _map_condition(data, column, mapping, slug):
    """Map a condition-label column to model indices, refusing to let an
    unmapped label become a silent NaN index (a stale label would otherwise
    surface layers away as an opaque indexing TypeError)."""
    mapped = data[column].map(mapping)
    if mapped.isna().any():
        bad = sorted(data.loc[mapped.isna(), column].astype(str).unique())
        raise ValueError(
            f"data/{slug}/main_trials_long.csv: column '{column}' has unmapped "
            f"label(s) {bad}; expected one of {sorted(mapping)}. Fix the labels "
            f"in the CSV or re-run the data pipeline (`make data-{slug}`)."
        )
    return mapped


def _validate_long_raw(raw, rating_cols, slug):
    """Fail fast on malformed main_trials_long.csv rows before the prior↔
    posterior pivot: duplicate (subject, scenario, stage) rows would cross-join
    in the merge, and NaN or out-of-[0, 1] ratings would silently poison the
    belief updates (all rating DVs are normalized to [0, 1] in preprocessing)."""
    stage_key = ["subject_id", "scenario_label", "stage"]
    dup = raw.duplicated(stage_key, keep=False)
    if dup.any():
        offenders = raw.loc[dup, stage_key].drop_duplicates().head(5)
        raise ValueError(
            f"data/{slug}/main_trials_long.csv has duplicate "
            f"(subject_id, scenario_label, stage) rows — the prior↔posterior "
            f"merge would cross-join them. First offenders:\n{offenders}\n"
            f"Re-run the data pipeline (`make data-{slug}`)."
        )
    for c in rating_cols:
        bad = raw[c].isna() | (raw[c] < 0) | (raw[c] > 1)
        if bad.any():
            raise ValueError(
                f"data/{slug}/main_trials_long.csv: column '{c}' has "
                f"{int(bad.sum())} value(s) that are NaN or outside [0, 1] "
                f"(ratings are normalized to [0, 1] in preprocessing). "
                f"Re-run the data pipeline (`make data-{slug}`)."
            )


def _load_long(slug):
    """Load a 3-action experiment's main_trials_long.csv as one row per trial,
    carrying belief-update columns (`<rating>_update = posterior − prior`).

    The DV is the belief update (manuscript): each trial has a `prior` and a
    `posterior` row (same `subject_id` × `scenario_label`), and we subtract them
    per rating column. Returns a DataFrame with the posterior-stage condition
    columns plus `subject_id` and `<rating>_update` for each rating column present
    (`response`, `intimacy_rating`, `desire_rating`, `effort_rating`), mapped to
    model indices: scenario_idx, action, intimacy_idx_4, desire_condition
    (0/1), effort_condition (0/1).

    Column assumptions:
      - `action_condition` like 'no_share' / 'low_risk_share' / 'high_risk_share'
      - `desire_condition` (or `desire`) in {'low', 'high'} when present
      - `effort_condition` (or `effort`) in {'low', 'high'} when present
      - `intimacy` (or `relationship_condition`) in {max_formal, somewhat_formal, somewhat_intimate, max_intimate} when present
      - `stage` in {'prior', 'posterior'}
    """
    filepath = get_project_root() / "data" / slug / "main_trials_long.csv"
    raw = pd.read_csv(filepath)

    # Pivot prior↔posterior into per-trial belief updates (mirrors the R
    # calculate_belief_update; grouped by subject × scenario).
    rating_cols = [
        c
        for c in ("response", "intimacy_rating", "desire_rating", "effort_rating")
        if c in raw.columns
    ]
    _validate_long_raw(raw, rating_cols, slug)
    key = ["subject_id", "scenario_label"]
    prior = raw[raw["stage"] == "prior"]
    post = raw[raw["stage"] == "posterior"].copy()
    data = post.merge(prior[key + rating_cols], on=key, suffixes=("", "_prior"))
    if len(data) != len(post):
        dropped = post.merge(prior[key], on=key, how="left", indicator=True)
        offenders = dropped.loc[dropped["_merge"] == "left_only", key]
        raise ValueError(
            f"data/{slug}/main_trials_long.csv: {len(offenders)} posterior "
            f"trial(s) have no matching prior row and would be silently "
            f"dropped. First offenders:\n{offenders.head(5)}\n"
            f"Re-run the data pipeline (`make data-{slug}`)."
        )
    for c in rating_cols:
        data[f"{c}_update"] = data[c] - data[f"{c}_prior"]

    data["action"] = _map_condition(data, "action_condition", ACTION_LABEL_TO_IDX, slug)
    # Scenario indices follow the study's own stimulus set (food vs. nonfood
    # labels; see STUDY_SCENARIO_LABELS in tables.py).
    data["scenario_idx"] = _map_condition(
        data, "scenario_label", scenario_to_idx_for_study(slug), slug
    )

    desire_map = {"low": 0, "high": 1}
    if "desire" in data.columns:
        data["desire_condition"] = _map_condition(data, "desire", desire_map, slug)
    elif (
        "desire_condition" in data.columns and data["desire_condition"].dtype == object
    ):
        data["desire_condition"] = _map_condition(
            data, "desire_condition", desire_map, slug
        )

    if "effort" in data.columns:
        data["effort_condition"] = _map_condition(
            data, "effort", EFFORT_CONDITION_TO_IDX, slug
        )
    elif (
        "effort_condition" in data.columns and data["effort_condition"].dtype == object
    ):
        data["effort_condition"] = _map_condition(
            data, "effort_condition", EFFORT_CONDITION_TO_IDX, slug
        )

    # Intimacy is stored as a verbal slug (no numeric code). Map it to the
    # 4-level RelationshipConditions index.
    if "intimacy" in data.columns:
        data["intimacy_idx_4"] = _map_condition(
            data, "intimacy", INTIMACY_CONDITION_TO_IDX, slug
        )
    elif "relationship_condition" in data.columns:
        data["intimacy_idx_4"] = _map_condition(
            data, "relationship_condition", INTIMACY_CONDITION_TO_IDX, slug
        )

    return data


def load_intimacy_data(slug="food_inv_intimacy"):
    """Study 2a — observer knows (desire, effort), infers intimacy. `response` is
    the intimacy belief update (posterior − prior intimacy rating)."""
    data = _load_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    desire_condition = jnp.array(data["desire_condition"].values)
    effort_condition = jnp.array(data["effort_condition"].values)
    response = jnp.array(data["intimacy_rating_update"].values)
    print(f"Loaded {len(data)} belief-update data points")
    return data, action, scenario_idx, desire_condition, effort_condition, response


def load_desire_data(slug="food_inv_desire"):
    """Study 1a — observer knows (effort, intimacy), infers desire.

    The desire DV is a continuous rating ("how much would the two people like the
    food?"); `response` is its belief update (posterior − prior, on the 0-1
    scale), scored against the model's desire belief update with the mixture
    likelihood.
    """
    data = _load_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    effort_condition = jnp.array(data["effort_condition"].values)
    relationship_condition = jnp.array(data["intimacy_idx_4"].values)
    response = jnp.array(data["response_update"].values)  # 0-1 desire belief update
    print(f"Loaded {len(data)} belief-update data points")
    return (
        data,
        action,
        scenario_idx,
        effort_condition,
        relationship_condition,
        response,
    )


def load_joint_de_data(slug="food_inv_joint_de"):
    """Study 1b (or Study 3a via slug="nonfood_inv_joint_de") — observer knows
    intimacy, jointly infers (desire, effort).

    Each posterior trial contributes two slider responses: `desire_rating` (the
    continuous desire DV) and `effort_rating` (the effort slider, "which effort
    situation is more likely"; 0 = effort_low ... 1 = effort_high). Both are
    normalized to the 0-1 scale in preprocessing.
    """
    data = _load_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    relationship_condition = jnp.array(data["intimacy_idx_4"].values)
    resp_desire = jnp.array(data["desire_rating_update"].values)  # 0-1 desire update
    resp_effort = jnp.array(data["effort_rating_update"].values)  # 0-1 effort update
    print(f"Loaded {len(data)} belief-update data points (2 slider updates each)")
    return data, action, scenario_idx, relationship_condition, resp_desire, resp_effort


def load_joint_ie_data(slug="food_inv_joint_ie"):
    """Study 2b (or Study 3b via slug="nonfood_inv_joint_ie") — observer knows
    desire, jointly infers (intimacy, effort).

    Each posterior trial contributes two slider responses; expects columns
    `intimacy_rating` and `effort_rating`, both on the 0-1 scale (normalized in
    preprocessing).
    """
    data = _load_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    desire_condition = jnp.array(data["desire_condition"].values)
    resp_intimacy = jnp.array(data["intimacy_rating_update"].values)  # 0-1 update
    resp_effort = jnp.array(data["effort_rating_update"].values)  # 0-1 effort update
    print(f"Loaded {len(data)} belief-update data points (2 slider updates each)")
    return data, action, scenario_idx, desire_condition, resp_intimacy, resp_effort


# ------------------------------------------------------------------------------
# Table-kwargs helpers for the new pipeline (load lazily — tables may be None
# during early development if the LM CSVs haven't been produced yet).
# ------------------------------------------------------------------------------


def _padded_table_kwargs(
    loader,
    utility_param_names,
    study,
    slug_hint,
    desire_loader=None,
    relationship_loader=None,
):
    """Shared assembly for the padded LM-alternatives observer kwargs. Raises
    FileNotFoundError if the study's LM tables are missing (run
    generate_alternatives.py + score_merged.py --study <slug> first).

    Which optional tables to include is derived from the variant's
    `utility_param_names`:
      - reward term present (`w_v` in names → full/base) → add the
        goal-satisfaction table `g`, and (given-desire studies 2a/2b) the
        per-condition desire scalar via `desire_loader`.
      - relational cost present (`gamma` in names → full/discomfort_only) →
        (given-relationship studies 1a/1b) add the LM-rated intimacy 4-vector via
        `relationship_loader`.
    The base variant has neither `gamma` nor the relational term, so it never gets
    `relationship_values`; discomfort_only has no reward term, so it never gets
    `g`/`desire_table` — matching the observer memo signatures.
    """
    uses_reward = "w_v" in utility_param_names
    uses_intimacy = "gamma" in utility_param_names
    padded = loader()
    if padded is None:
        raise FileNotFoundError(
            f"Padded LM tables for {study} not found. Run "
            f"`model/lm/generate_alternatives.py --study {slug_hint}` and "
            f"`model/lm/score_merged.py --study {slug_hint}` first."
        )
    kw = {
        "risk_table": padded["risk"],
        "effort_table": padded["effort"],
        "prior_table": padded["prior"],
    }
    if uses_reward:
        kw["g_padded_table"] = padded["g"]
        if desire_loader is not None:
            desire = desire_loader()
            if desire is None:
                raise FileNotFoundError(
                    f"desire scalar not found for {study} — the given-desire "
                    "studies need the per-condition desire scalar. Run "
                    f"`model/lm/score_merged.py --study {slug_hint}` first."
                )
            kw["desire_table"] = desire
    if uses_intimacy and relationship_loader is not None:
        # Per-run LM-rated intimacy magnitude per relationship level, shape (K, 4)
        # (falls back to the placeholder RELATIONSHIP_LEVEL_VALUES as K=1 until the
        # per-run `intimacy` field exists in lm_runs.jsonl).
        kw["relationship_values"] = relationship_loader()
    return kw


def desire_table_kwargs(utility_param_names, domain="food", base=False):
    """Padded LM tables for Study 1a (food_inv_desire). Observer's actor
    softmaxes over LM-generated alternatives per (scenario, observed_action,
    effort, intimacy) cell. Desire is inferred (no desire scalar); intimacy is
    given, so full/discomfort_only get the LM-rated `relationship_values`.

    `base=True` (the base ablation, which has no intimacy term) loads the
    relationship-free alternative set (`lm_runs_base.jsonl`) and broadcasts it
    across the relationship axis, so the base table — and the base model's
    predictions — are relationship-invariant. full/discomfort_only keep the
    relationship-conditioned `lm_runs.jsonl`."""
    from tables import load_lm_relationship_values, load_padded_lm_tables_desire

    if domain != "food":
        raise NotImplementedError(
            "Padded LM tables are only available for the food domain."
        )
    runs_filename = f"lm_runs{'_base' if base else ''}.jsonl"
    loader = (
        (
            lambda: load_padded_lm_tables_desire(
                runs_filename=runs_filename, broadcast_relationship=True
            )
        )
        if base
        else (lambda: load_padded_lm_tables_desire(runs_filename=runs_filename))
    )
    return _padded_table_kwargs(
        loader,
        utility_param_names,
        "Study 1a",
        "food_inv_desire" + (" --base" if base else ""),
        relationship_loader=(
            None
            if base
            else (
                lambda: load_lm_relationship_values(
                    "food_inv_desire", runs_filename=runs_filename
                )
            )
        ),
    )


def intimacy_table_kwargs(utility_param_names, domain="food"):
    """Padded LM tables for Study 2a (food_inv_intimacy). Cell grid
    (scenario, observed_action, desire, effort); infers intimacy (continuous, no
    relationship_values). Desire is given, so the per-condition desire scalar is
    loaded for full/base. 2a has no base variant."""
    from tables import (
        load_lm_scenario_desire,
        load_padded_lm_tables_intimacy,
    )

    if domain != "food":
        raise NotImplementedError(
            "Padded LM tables are only available for the food domain."
        )
    runs_filename = "lm_runs.jsonl"
    return _padded_table_kwargs(
        lambda: load_padded_lm_tables_intimacy(runs_filename=runs_filename),
        utility_param_names,
        "Study 2a",
        "food_inv_intimacy",
        desire_loader=lambda: load_lm_scenario_desire(
            "food_inv_intimacy", runs_filename=runs_filename
        ),
    )


def joint_de_table_kwargs(utility_param_names, domain="food", base=False):
    """Padded LM tables for the joint desire+effort studies: Study 1b
    (food_inv_joint_de, domain="food") and Study 3a (nonfood_inv_joint_de,
    domain="nonfood"). Cell grid (scenario, observed_action, intimacy); jointly
    infers (desire, effort). Desire is inferred (no desire scalar); intimacy is
    given, so full/discomfort_only get the LM-rated `relationship_values`.

    `base=True` (the base ablation, which has no intimacy term) loads the
    relationship-free alternative set (`lm_runs_base.jsonl`) and broadcasts it
    across the relationship axis, exactly as in `desire_table_kwargs`."""
    from tables import load_lm_relationship_values, load_padded_lm_tables_joint_de

    slug, study = {
        "food": ("food_inv_joint_de", "Study 1b"),
        "nonfood": ("nonfood_inv_joint_de", "Study 3a"),
    }[domain]
    runs_filename = f"lm_runs{'_base' if base else ''}.jsonl"
    loader = lambda: load_padded_lm_tables_joint_de(  # noqa: E731
        slug=slug,
        runs_filename=runs_filename,
        **({"broadcast_relationship": True} if base else {}),
    )
    return _padded_table_kwargs(
        loader,
        utility_param_names,
        study,
        slug + (" --base" if base else ""),
        relationship_loader=(
            None
            if base
            else (
                lambda: load_lm_relationship_values(slug, runs_filename=runs_filename)
            )
        ),
    )


def joint_ie_table_kwargs(utility_param_names, domain="food"):
    """Padded LM tables for the joint intimacy+effort studies: Study 2b
    (food_inv_joint_ie, domain="food") and Study 3b (nonfood_inv_joint_ie,
    domain="nonfood"). Cell grid (scenario, observed_action, desire); infers
    (intimacy, effort) (continuous intimacy, no relationship_values). Desire is
    given, so the per-condition desire scalar is loaded for full/base. 2b/3b have
    no base variant."""
    from tables import (
        load_lm_scenario_desire,
        load_padded_lm_tables_joint_ie,
    )

    slug, study = {
        "food": ("food_inv_joint_ie", "Study 2b"),
        "nonfood": ("nonfood_inv_joint_ie", "Study 3b"),
    }[domain]
    runs_filename = "lm_runs.jsonl"
    return _padded_table_kwargs(
        lambda: load_padded_lm_tables_joint_ie(slug=slug, runs_filename=runs_filename),
        utility_param_names,
        study,
        slug,
        desire_loader=lambda: load_lm_scenario_desire(
            slug, runs_filename=runs_filename
        ),
    )


def resolve_variant_table_kwargs(variants, table_kwargs_fn):
    """Resolve every variant's LM table kwargs up front, before any fitting
    starts. A missing LM table (e.g. an unelicited lm_runs_base.jsonl) then
    fails immediately with the loader's FileNotFoundError instead of crashing
    after hours of fitting the earlier variants. `variants` maps variant name →
    (observer_fn, utility_param_names); `table_kwargs_fn(variant_name,
    utility_param_names)` builds one variant's kwargs. Returns
    {variant_name: table_kwargs}."""
    return {
        name: table_kwargs_fn(name, utility_names)
        for name, (_, utility_names) in variants.items()
    }


# ------------------------------------------------------------------------------
# Single-target fits — reuse _fit_alpha_observer with the appropriate slicer.
# The 3-action observer tables are 5-D: (action, scenario, intimacy/rel, desire, effort).
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Joint fits (Studies 1b, 2b)
# ------------------------------------------------------------------------------


# ==============================================================================
# Joint fits — utility weights + α_observer (Studies 1a, 1b, 2a, 2b)
# ==============================================================================
# Each inverse experiment fits its own actor utility weights jointly with
# α_observer from its own posterior data. Actor weights are NOT transferred
# between studies — the studies have different data and different
# identifiability, so transferring would conflate the fits.
#
# Params layout in the returned 1-D array: [*utility_param_values, alpha_observer,
# sigma]. Caller maps utility_param_names → param_<name> columns; reads
# params[-2] as alpha_observer and params[-1] as the response-noise σ (σ is a
# likelihood param, not passed to the observer). Actor inverse-temperature is
# held at α=1 (same convention as the legacy padded joint fits).

# Padded LM feature tables that carry a leading elicitation-run axis (see
# tables.py). The given-magnitude tables (desire_table, relationship_values) are
# scored per run too, so they carry the same leading axis and are sliced per run
# alongside the features (the observer memo still sees one run's slice).
_RUN_AXIS_TABLES = (
    "risk_table",
    "effort_table",
    "g_padded_table",
    "prior_table",
    "desire_table",
    "relationship_values",
)


def params_dict_to_array(params, utility_param_names, extra_param_names=()):
    """Reconstruct the optimizer's parameter vector [*utility, alpha_observer,
    sigma, *extra] from a `load_fit_results` dict, for re-running the observer
    (e.g. the CV warm-start). sigma defaults to 1.0 if absent (it doesn't affect
    the observer build); `extra_param_names` (e.g. the informative-prior fit's
    `prior_nu`) are appended and KeyError if missing — a deliberate fail-fast on
    a warm start whose params don't carry the extended vector."""
    return jnp.array(
        [params[name] for name in utility_param_names]
        + [params["alpha_observer"], params.get("sigma", 1.0)]
        + [params[n] for n in extra_param_names]
    )


def _build_observer_tables_runs(observer_fn, params, utility_param_names, table_kwargs):
    """Run the observer once per elicitation run and stack the posteriors on a
    leading K axis → (K, *observer_dims).

    Vectorized over the K elicitation runs with `jax.vmap` (one batched observer
    eval over the run axis instead of a Python loop), keeping the run axis a
    likelihood-side construct rather than a memo dimension. K=1 (legacy single-run
    tables) reproduces the pre-mixture single-component behavior. σ (`params[-1]`)
    is a likelihood param and is *not* passed to the observer.
    """
    actor_kwargs = {"alpha": 1.0}
    for i, name in enumerate(utility_param_names):
        actor_kwargs[name] = params[i]
    alpha_observer = params[-2]
    # Tables carrying the leading run axis are mapped over (axis 0); any others
    # (none today) are broadcast unchanged.
    run_tables = {k: v for k, v in table_kwargs.items() if k in _RUN_AXIS_TABLES}
    fixed_tables = {k: v for k, v in table_kwargs.items() if k not in _RUN_AXIS_TABLES}

    def _run_one(run_slice):
        return observer_fn(
            **actor_kwargs,
            alpha_observer=alpha_observer,
            **run_slice,
            **fixed_tables,
        )

    return jax.vmap(_run_one)(run_tables)


def _intimacy_loss(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    desire_condition,
    effort_condition,
    response,
    table_kwargs,
    priors=None,
    reweighting=None,
):
    """Study 2a's mixture NLL as a function of the fit's parameter vector.

    Split out of `fit_intimacy_observer_joint` so the pooled cross-experiment fit
    (`model/inverse/_pooled.py`) can sum this study's loss with the others'
    under one shared utility. The fit helper is the only other caller; both go
    through this, so there is one definition of what a study's likelihood is.

    Returns (loss_fn, n_params, n_core), where the parameter vector is
    `[*utility, alpha_observer, sigma, *(prior_nu), *(eta)]` and `n_core` is the
    index one past sigma.
    """
    if priors is not None and all(v is None for v in priors.values()):
        priors = None
    n_core = len(utility_param_names) + 2
    use_grid = priors is not None and priors.get("m_latent") is not None
    n_params = n_core + (1 if use_grid else 0) + (1 if reweighting else 0)
    # eta is the LAST slot when the reweighting is active (after prior_nu).
    i_eta = n_params - 1 if reweighting else None

    def loss_fn(params):
        tk = _reweighting.apply(
            reweighting,
            table_kwargs,
            params[:n_core],
            params[i_eta] if reweighting else 0.0,
        )
        tables = _build_observer_tables_runs(
            observer_fn, params[:n_core], utility_param_names, tk
        )
        sigma = params[n_core - 1]
        nu = params[n_core] if use_grid else None

        def nll_trial(a, s, r, e, u):
            post_runs = tables[:, 0, s, a, r, e, :]  # (K, 101)
            if use_grid:
                w = beta_prior_on_grid(priors["m_latent"][:, s, r, e], nu)  # (K, 101)
                post_runs = reweight_grid(post_runs, w)
                prior_mean = w @ GRID  # (K,)
            else:
                prior_mean = PRIOR_MEAN
            deltas = delta_latent(post_runs, GRID, prior_mean)  # (K,)
            return mixture_nll_1d(u, deltas, sigma)

        return jnp.sum(
            jax.vmap(nll_trial)(
                action, scenario_idx, desire_condition, effort_condition, response
            )
        )

    return loss_fn, n_params, n_core


def fit_intimacy_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    desire_condition,
    effort_condition,
    response,
    table_kwargs,
    priors=None,
    reweighting=None,
    lr=0.1,
    max_steps=1000,
    verbose=True,
    n_restarts=N_RESTARTS_FIT,
    init_params=None,
    patience=100,
    seed_key=None,
    free_mask=None,
):
    """Study 2a — joint fit of utility weights + α_observer + σ on the intimacy
    belief update via the K-run Gaussian mixture.

    The stacked observer table is 7-D — (run, padded_slot, scenario,
    observed_action, desire, effort, relationship[101]). The observed action sits
    in slot 0, so each run's intimacy posterior is
    `tables[:, 0, scenario, action, desire, effort, :]`; its mean minus the prior
    mean gives that run's model update δ_k. `response` is the intimacy belief
    update u, scored under (1/K)Σ N(u | δ_k, σ²).

    `priors=None` keeps the uniform-prior path byte-identical to the
    preregistered fit. When `priors["m_latent"]` (shape (K, 16, 2, 2)) is active,
    the intimacy latent gets a per-cell discretized-Beta prior with a single
    fitted concentration `prior_nu` appended to the param vector at index
    `n_core`; the uniform fit is nested at m=0.5, nu=2.

    `free_mask` (see `fit_masked`) estimates only a subset of the vector,
    holding the rest at `init_params`; the cross-study transfer analysis uses it
    to freeze the utility weights. `None` is the ordinary fit.
    """
    loss_fn, n_params, n_core = _intimacy_loss(
        observer_fn,
        utility_param_names,
        action,
        scenario_idx,
        desire_condition,
        effort_condition,
        response,
        table_kwargs,
        priors=priors,
        reweighting=reweighting,
    )
    if init_params is not None and len(init_params) != n_params:
        raise ValueError(
            f"init_params has length {len(init_params)} but this fit expects "
            f"{n_params}: utility+alpha_observer+sigma ({n_core}) plus "
            f"{n_params - n_core} extra slot(s) for this configuration's "
            "prior_nu / eta. A warm start that doesn't match would be silently "
            "mis-sliced (JAX clamps out-of-bounds indices, so prior_nu would "
            "read sigma). Build it with params_dict_to_array(..., "
            "extra_param_names=[...]) naming the same extras."
        )

    params, nll, restarts = fit_masked(
        loss_fn,
        n_params=n_params,
        free_mask=free_mask,
        n_restarts=n_restarts,
        init_params=init_params,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="intimacy_joint",
        patience=patience,
        seed_key=seed_key,
        upper=param_upper_bounds(n_params, len(utility_param_names)),
        alpha_obs_index=len(utility_param_names),
    )
    return params, float(nll), restarts


def _desire_loss(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    effort_condition,
    relationship_condition,
    response,
    table_kwargs,
    priors=None,
    reweighting=None,
):
    """Study 1a's mixture NLL as a function of the fit's parameter vector.

    Split out of `fit_desire_observer_joint` so the pooled cross-experiment fit
    (`model/inverse/_pooled.py`) can sum this study's loss with the others'
    under one shared utility. The fit helper is the only other caller; both go
    through this, so there is one definition of what a study's likelihood is.

    Returns (loss_fn, n_params, n_core), where the parameter vector is
    `[*utility, alpha_observer, sigma, *(prior_nu), *(eta)]` and `n_core` is the
    index one past sigma.
    """
    if priors is not None and all(v is None for v in priors.values()):
        priors = None
    n_core = len(utility_param_names) + 2
    use_grid = priors is not None and priors.get("m_latent") is not None
    n_params = n_core + (1 if use_grid else 0) + (1 if reweighting else 0)
    # eta is the LAST slot when the reweighting is active (after prior_nu).
    i_eta = n_params - 1 if reweighting else None

    def loss_fn(params):
        tk = _reweighting.apply(
            reweighting,
            table_kwargs,
            params[:n_core],
            params[i_eta] if reweighting else 0.0,
        )
        tables = _build_observer_tables_runs(
            observer_fn, params[:n_core], utility_param_names, tk
        )
        sigma = params[n_core - 1]
        nu = params[n_core] if use_grid else None

        def nll_trial(a, s, e, rel, u):
            post_runs = tables[:, 0, s, a, e, rel, :]  # (K, 101)
            if use_grid:
                w = beta_prior_on_grid(priors["m_latent"][:, s, e, rel], nu)  # (K, 101)
                post_runs = reweight_grid(post_runs, w)
                prior_mean = w @ GRID  # (K,)
            else:
                prior_mean = PRIOR_MEAN
            deltas = delta_latent(post_runs, GRID, prior_mean)  # (K,)
            return mixture_nll_1d(u, deltas, sigma)

        return jnp.sum(
            jax.vmap(nll_trial)(
                action, scenario_idx, effort_condition, relationship_condition, response
            )
        )

    return loss_fn, n_params, n_core


def fit_desire_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    effort_condition,
    relationship_condition,
    response,
    table_kwargs,
    priors=None,
    reweighting=None,
    lr=0.1,
    max_steps=1000,
    verbose=True,
    n_restarts=N_RESTARTS_FIT,
    init_params=None,
    patience=100,
    seed_key=None,
    free_mask=None,
):
    """Study 1a — joint fit of utility weights + α_observer + σ on the desire
    belief update via the K-run Gaussian mixture.

    The stacked observer table is 7-D —
    `(run, padded_slot, scenario, observed_action, effort, intimacy, desire[101])`.
    The observed action lives in slot 0, so each run's desire posterior is
    `tables[:, 0, scenario, action, effort, intimacy, :]`; its mean minus the
    prior mean gives that run's δ_k. `response` is the desire belief update u,
    scored under (1/K)Σ N(u | δ_k, σ²).

    `priors=None` keeps the uniform-prior path byte-identical to the
    preregistered fit. When `priors["m_latent"]` (shape (K, 16, 2, 4)) is active,
    the desire latent gets a per-cell discretized-Beta prior with a single fitted
    concentration `prior_nu` appended to the param vector at index `n_core`; the
    uniform fit is nested at m=0.5, nu=2.

    `free_mask` (see `fit_masked`) estimates only a subset of the vector,
    holding the rest at `init_params`; the cross-study transfer analysis uses it
    to freeze the utility weights. `None` is the ordinary fit.
    """
    loss_fn, n_params, n_core = _desire_loss(
        observer_fn,
        utility_param_names,
        action,
        scenario_idx,
        effort_condition,
        relationship_condition,
        response,
        table_kwargs,
        priors=priors,
        reweighting=reweighting,
    )
    if init_params is not None and len(init_params) != n_params:
        raise ValueError(
            f"init_params has length {len(init_params)} but this fit expects "
            f"{n_params}: utility+alpha_observer+sigma ({n_core}) plus "
            f"{n_params - n_core} extra slot(s) for this configuration's "
            "prior_nu / eta. A warm start that doesn't match would be silently "
            "mis-sliced (JAX clamps out-of-bounds indices, so prior_nu would "
            "read sigma). Build it with params_dict_to_array(..., "
            "extra_param_names=[...]) naming the same extras."
        )

    params, nll, restarts = fit_masked(
        loss_fn,
        n_params=n_params,
        free_mask=free_mask,
        n_restarts=n_restarts,
        init_params=init_params,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="desire_joint",
        patience=patience,
        seed_key=seed_key,
        upper=param_upper_bounds(n_params, len(utility_param_names)),
        alpha_obs_index=len(utility_param_names),
    )
    return params, float(nll), restarts


def _joint_de_loss(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    relationship_condition,
    response_desire,
    response_effort,
    table_kwargs,
    priors=None,
    reweighting=None,
):
    """Studies 1b/3a's bivariate mixture NLL as a function of the fit's parameter vector.

    Split out of `fit_joint_de_observer_joint` so the pooled cross-experiment fit
    (`model/inverse/_pooled.py`) can sum this study's loss with the others'
    under one shared utility. The fit helper is the only other caller; both go
    through this, so there is one definition of what a study's likelihood is.

    Returns (loss_fn, n_params, n_core), where the parameter vector is
    `[*utility, alpha_observer, sigma, *(prior_nu), *(eta)]` and `n_core` is the
    index one past sigma.
    """
    if priors is not None and all(v is None for v in priors.values()):
        priors = None
    n_core = len(utility_param_names) + 2
    use_grid = priors is not None and priors.get("m_latent") is not None
    use_eff = priors is not None and priors.get("p_effort") is not None
    n_params = n_core + (1 if use_grid else 0) + (1 if reweighting else 0)
    # eta is the LAST slot when the reweighting is active (after prior_nu).
    i_eta = n_params - 1 if reweighting else None

    def loss_fn(params):
        tk = _reweighting.apply(
            reweighting,
            table_kwargs,
            params[:n_core],
            params[i_eta] if reweighting else 0.0,
        )
        tables = _build_observer_tables_runs(
            observer_fn, params[:n_core], utility_param_names, tk
        )
        sigma = params[n_core - 1]
        nu = params[n_core] if use_grid else None

        def nll_trial(a, s, rel, u_desire, u_effort):
            joint = tables[:, 0, s, a, rel, :, :]  # (K, 101, 2)
            if use_grid:
                w = beta_prior_on_grid(priors["m_latent"][:, s, rel], nu)
                lat_prior_mean = w @ GRID
            else:
                w = None
                lat_prior_mean = PRIOR_MEAN
            if use_eff:
                p = priors["p_effort"][:, s, rel]
                eff_prior_mean = p
            else:
                p = None
                eff_prior_mean = EFFORT_PRIOR_MEAN
            joint = reweight_joint(joint, w, p)
            d_desire, d_effort = delta_joint(
                joint, GRID, lat_prior_mean, eff_prior_mean
            )  # each (K,)
            deltas = jnp.stack([d_desire, d_effort], axis=1)  # (K, 2)
            return mixture_nll_2d(jnp.array([u_desire, u_effort]), deltas, sigma)

        return jnp.sum(
            jax.vmap(nll_trial)(
                action,
                scenario_idx,
                relationship_condition,
                response_desire,
                response_effort,
            )
        )

    return loss_fn, n_params, n_core


def fit_joint_de_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    relationship_condition,
    response_desire,
    response_effort,
    table_kwargs,
    priors=None,
    reweighting=None,
    lr=0.1,
    max_steps=1000,
    verbose=True,
    n_restarts=N_RESTARTS_FIT,
    init_params=None,
    patience=100,
    seed_key=None,
    free_mask=None,
):
    """Study 1b — joint fit of utility weights + α_observer + σ on the joint
    (desire, effort) belief update via the K-run bivariate Gaussian mixture
    (isotropic σ²·I₂).

    The stacked observer table is 7-D — (run, padded_slot, scenario,
    observed_action, relationship, desire[101], effort[2]). The observed action
    sits in slot 0, so each run's joint over (desire, effort) is
    `tables[:, 0, scenario, action, relationship, :, :]`. Per run we take the
    desire-marginal mean and P(effort=HIGH); each minus its prior mean gives that
    run's 2-D model update δ_k. `response_desire`/`response_effort` are the two
    belief updates, scored jointly under (1/K)Σ N(u | δ_k, σ²·I₂).

    `priors=None` keeps the uniform-prior path byte-identical to the
    preregistered fit. `priors` may carry `m_latent` (desire, shape (K, 16, 4);
    adds a fitted `prior_nu` at index `n_core`) and/or `p_effort` (the elicited
    P(effort=high), shape (K, 16, 4)); each None leaves that latent uniform. The
    uniform fit is nested at m=0.5, nu=2, p=0.5.

    `free_mask` (see `fit_masked`) estimates only a subset of the vector,
    holding the rest at `init_params`; the cross-study transfer analysis uses it
    to freeze the utility weights. `None` is the ordinary fit.
    """
    loss_fn, n_params, n_core = _joint_de_loss(
        observer_fn,
        utility_param_names,
        action,
        scenario_idx,
        relationship_condition,
        response_desire,
        response_effort,
        table_kwargs,
        priors=priors,
        reweighting=reweighting,
    )
    if init_params is not None and len(init_params) != n_params:
        raise ValueError(
            f"init_params has length {len(init_params)} but this fit expects "
            f"{n_params}: utility+alpha_observer+sigma ({n_core}) plus "
            f"{n_params - n_core} extra slot(s) for this configuration's "
            "prior_nu / eta. A warm start that doesn't match would be silently "
            "mis-sliced (JAX clamps out-of-bounds indices, so prior_nu would "
            "read sigma). Build it with params_dict_to_array(..., "
            "extra_param_names=[...]) naming the same extras."
        )

    params, nll, restarts = fit_masked(
        loss_fn,
        n_params=n_params,
        free_mask=free_mask,
        n_restarts=n_restarts,
        init_params=init_params,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="joint_de_joint",
        patience=patience,
        seed_key=seed_key,
        upper=param_upper_bounds(n_params, len(utility_param_names)),
        alpha_obs_index=len(utility_param_names),
    )
    return params, float(nll), restarts


def _joint_ie_loss(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    desire_condition,
    response_intimacy,
    response_effort,
    table_kwargs,
    priors=None,
    reweighting=None,
):
    """Studies 2b/3b's bivariate mixture NLL as a function of the fit's parameter vector.

    Split out of `fit_joint_ie_observer_joint` so the pooled cross-experiment fit
    (`model/inverse/_pooled.py`) can sum this study's loss with the others'
    under one shared utility. The fit helper is the only other caller; both go
    through this, so there is one definition of what a study's likelihood is.

    Returns (loss_fn, n_params, n_core), where the parameter vector is
    `[*utility, alpha_observer, sigma, *(prior_nu), *(eta)]` and `n_core` is the
    index one past sigma.
    """
    if priors is not None and all(v is None for v in priors.values()):
        priors = None
    n_core = len(utility_param_names) + 2
    use_grid = priors is not None and priors.get("m_latent") is not None
    use_eff = priors is not None and priors.get("p_effort") is not None
    n_params = n_core + (1 if use_grid else 0) + (1 if reweighting else 0)
    # eta is the LAST slot when the reweighting is active (after prior_nu).
    i_eta = n_params - 1 if reweighting else None

    def loss_fn(params):
        tk = _reweighting.apply(
            reweighting,
            table_kwargs,
            params[:n_core],
            params[i_eta] if reweighting else 0.0,
        )
        tables = _build_observer_tables_runs(
            observer_fn, params[:n_core], utility_param_names, tk
        )
        sigma = params[n_core - 1]
        nu = params[n_core] if use_grid else None

        def nll_trial(a, s, r, u_intimacy, u_effort):
            joint = tables[:, 0, s, a, r, :, :]  # (K, 101, 2)
            if use_grid:
                w = beta_prior_on_grid(priors["m_latent"][:, s, r], nu)
                lat_prior_mean = w @ GRID
            else:
                w = None
                lat_prior_mean = PRIOR_MEAN
            if use_eff:
                p = priors["p_effort"][:, s, r]
                eff_prior_mean = p
            else:
                p = None
                eff_prior_mean = EFFORT_PRIOR_MEAN
            joint = reweight_joint(joint, w, p)
            d_intimacy, d_effort = delta_joint(
                joint, GRID, lat_prior_mean, eff_prior_mean
            )  # each (K,)
            deltas = jnp.stack([d_intimacy, d_effort], axis=1)  # (K, 2)
            return mixture_nll_2d(jnp.array([u_intimacy, u_effort]), deltas, sigma)

        return jnp.sum(
            jax.vmap(nll_trial)(
                action,
                scenario_idx,
                desire_condition,
                response_intimacy,
                response_effort,
            )
        )

    return loss_fn, n_params, n_core


def fit_joint_ie_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    desire_condition,
    response_intimacy,
    response_effort,
    table_kwargs,
    priors=None,
    reweighting=None,
    lr=0.1,
    max_steps=1000,
    verbose=True,
    n_restarts=N_RESTARTS_FIT,
    init_params=None,
    patience=100,
    seed_key=None,
    free_mask=None,
):
    """Study 2b — joint fit of utility weights + α_observer + σ on the joint
    (intimacy, effort) belief update via the K-run bivariate Gaussian mixture
    (isotropic σ²·I₂).

    The stacked observer table is 7-D — (run, padded_slot, scenario,
    observed_action, desire, relationship[101], effort[2]). The observed action
    sits in slot 0, so each run's joint over (intimacy, effort) is
    `tables[:, 0, scenario, action, desire, :, :]`. Per run we take the
    intimacy-marginal mean and P(effort=HIGH); each minus its prior mean gives
    that run's 2-D model update δ_k. `response_intimacy`/`response_effort` are the
    two belief updates, scored jointly under (1/K)Σ N(u | δ_k, σ²·I₂).

    `priors=None` keeps the uniform-prior path byte-identical to the
    preregistered fit. `priors` may carry `m_latent` (intimacy, shape (K, 16, 2);
    adds a fitted `prior_nu` at index `n_core`) and/or `p_effort` (the elicited
    P(effort=high), shape (K, 16, 2)); each None leaves that latent uniform. The
    uniform fit is nested at m=0.5, nu=2, p=0.5.

    `free_mask` (see `fit_masked`) estimates only a subset of the vector,
    holding the rest at `init_params`; the cross-study transfer analysis uses it
    to freeze the utility weights. `None` is the ordinary fit.
    """
    loss_fn, n_params, n_core = _joint_ie_loss(
        observer_fn,
        utility_param_names,
        action,
        scenario_idx,
        desire_condition,
        response_intimacy,
        response_effort,
        table_kwargs,
        priors=priors,
        reweighting=reweighting,
    )
    if init_params is not None and len(init_params) != n_params:
        raise ValueError(
            f"init_params has length {len(init_params)} but this fit expects "
            f"{n_params}: utility+alpha_observer+sigma ({n_core}) plus "
            f"{n_params - n_core} extra slot(s) for this configuration's "
            "prior_nu / eta. A warm start that doesn't match would be silently "
            "mis-sliced (JAX clamps out-of-bounds indices, so prior_nu would "
            "read sigma). Build it with params_dict_to_array(..., "
            "extra_param_names=[...]) naming the same extras."
        )

    params, nll, restarts = fit_masked(
        loss_fn,
        n_params=n_params,
        free_mask=free_mask,
        n_restarts=n_restarts,
        init_params=init_params,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="joint_ie_joint",
        patience=patience,
        seed_key=seed_key,
        upper=param_upper_bounds(n_params, len(utility_param_names)),
        alpha_obs_index=len(utility_param_names),
    )
    return params, float(nll), restarts
