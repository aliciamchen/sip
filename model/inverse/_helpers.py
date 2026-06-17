"""
Shared infrastructure for inverse-planning fit + predict scripts.

Each experiment has its own thin fit/predict script that imports the helpers
it needs from this module. Shared concerns:

  - Loss functions (intimacy NLL, effort BCE NLL, desire NLL)
  - Observer fit loops (joint padded utility weights + α_observer)
  - Data loaders (per experiment)
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from tables import (
    ACTION_LABEL_TO_IDX,
    EFFORT_CONDITION_TO_IDX,
    INTIMACY_CONDITION_TO_IDX,
    RELATIONSHIP_LEVEL_VALUES,
    SCENARIO_TO_IDX,
)
from utils import get_project_root


def _fit_with_adam(
    loss_fn,
    init_params,
    lr=0.01,
    max_steps=5000,
    verbose=True,
    label="",
    patience=100,
    tol=1e-6,
):
    """Adam fit loop with non-negativity clipping, best-so-far tracking, and a
    patience stop.

    Adam is not monotone even on full-batch problems, so the loop keeps the
    best (params, NLL) seen so far and stops once the best NLL hasn't improved
    by more than `tol` for `patience` consecutive steps. Returns the tracked
    best iterate, not the last one.
    """
    params = jnp.array(init_params)
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    best_params = params
    best_nll = jnp.inf
    steps_without_improvement = 0
    for step in range(max_steps):
        nll, grad = grad_fn(params)  # NLL at the current params, pre-update
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
        params = jnp.clip(params, 1e-6, jnp.inf)

    best_nll = float(best_nll)
    if verbose:
        print(f"  {label} final NLL: {best_nll:.4f}, params: {best_params}")
    return best_params, best_nll


def _fit_multistart(
    loss_fn,
    n_params,
    n_restarts=5,
    lr=0.1,
    max_steps=1000,
    verbose=True,
    label="",
):
    """Run `_fit_with_adam` from several inits and keep the best final NLL.

    Inits are the canonical all-ones vector plus `n_restarts - 1` seeded
    lognormal(0, 0.5) draws (positive, centered at 1, deterministic via
    `default_rng(0)`), guarding against local minima from the gamma power law.

    Returns (best_params, best_nll, records) where `records` is one dict per
    restart {restart, init, final_params, nll} for stability auditing.
    """
    rng = np.random.default_rng(0)
    inits = [jnp.ones(n_params)]
    for _ in range(max(0, n_restarts - 1)):
        inits.append(jnp.array(rng.lognormal(mean=0.0, sigma=0.5, size=n_params)))

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
    return best_params, best_nll, records


def restart_records_to_rows(slug, variant, utility_param_names, records):
    """Flatten `_fit_multistart` records into rows for fit_restarts.csv.

    One row per restart with init_<name> / param_<name> columns (the params
    layout is [*utility_param_names, alpha_observer]). Variants with different
    parameter sets just leave the other variants' columns empty.
    """
    rows = []
    names = list(utility_param_names) + ["alpha_observer"]
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
# Loss functions
# ==============================================================================


@jax.jit
def compute_intimacy_nll(posterior, response):
    """NLL = -log(P(intimacy = response)).

    posterior: shape (101,) over the [0, 1] IntimacyLevels grid.
    response: float on the 0-1 scale (the normalized human rating); mapped onto
    the 101-bin grid as bin `round(response * 100)`.
    """
    epsilon = 1e-8
    response_idx = jnp.clip(jnp.round(response * 100).astype(int), 0, 100)
    prob = posterior[response_idx]
    return -jnp.log(jnp.clip(prob, epsilon, 1.0))


@jax.jit
def compute_effort_nll(p_high, response):
    """Binary cross-entropy NLL for the effort slider (P(effort=HIGH)).

    response is on the 0-1 scale (directly interpreted as P(high)). Used for the
    effort slider in the joint studies (1b, 2b), a continuous rating between two
    states.
    """
    epsilon = 1e-8
    p_human = response
    p_model = jnp.clip(p_high, epsilon, 1.0 - epsilon)
    return -(p_human * jnp.log(p_model) + (1 - p_human) * jnp.log(1 - p_model))


@jax.jit
def compute_desire_nll(posterior, response):
    """NLL = -log(P(desire = response)) for the continuous desire DV
    (Studies 1a, 1b).

    Desire is a continuous latent inferred over the 101-bin DesireLevels grid
    ([0, 1] = "not at all" → "extremely"). `posterior` is the (101,) posterior
    over that grid; `response` is the participant's normalized 0-1 rating ("how
    much would the two people like the food?"), mapped onto the grid as bin
    `round(response * 100)`, an exact parallel of `compute_intimacy_nll`.
    """
    epsilon = 1e-8
    response_idx = jnp.clip(jnp.round(response * 100).astype(int), 0, 100)
    prob = posterior[response_idx]
    return -jnp.log(jnp.clip(prob, epsilon, 1.0))


# ==============================================================================
# Frozen-param loaders
# ==============================================================================


def load_fit_results(slug: str) -> dict:
    """Load per-variant {actor utility weights + alpha_observer} for a 3-action
    inverse experiment.

    Reads `outputs/<slug>/fit_results.csv` (written by the joint
    fit_food_inv_*.py scripts). Returns a dict mapping variant name (e.g.
    'full', 'discomfort_only', 'base') to a kwargs dict suitable for calling
    the observer function: `{alpha, alpha_observer, w_v?, w_d?, w_e?, gamma?}`
    (only the columns present for that variant; alpha defaults to 1.0).
    """
    path = get_project_root() / "model" / "outputs" / slug / "fit_results.csv"
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        variant = str(row["model"])
        params = {
            "alpha": float(row["param_alpha"])
            if "param_alpha" in row and pd.notna(row.get("param_alpha", None))
            else 1.0,
            "alpha_observer": float(row["alpha_observer"]),
        }
        for pn in ("w_v", "w_d", "w_e", "gamma"):
            col = f"param_{pn}"
            if col in row and pd.notna(row[col]):
                params[pn] = float(row[col])
        out[variant] = params
    return out


# ==============================================================================
# Data loaders, table kwargs, and joint fit helpers
# ==============================================================================
# Each active inverse study jointly fits its actor utility weights + α_observer
# from its own posterior data. The joint studies (1b, 2b) marginalize the joint
# observer table to each slider judgment and sum the two per-slider NLLs.


def _load_long(slug):
    """Load the posterior-stage rows of a 3-action experiment's main_trials_long.csv.

    Returns the pandas DataFrame with columns: scenario_label, scenario_idx,
    action, intimacy_idx_4, intimacy_idx_101, desire_condition (0/1),
    effort_condition (0/1), response (or two responses for joint studies).

    The exact column names in incoming CSVs may need to be normalized — this
    loader assumes:
      - `action_condition` like 'no_share' / 'low_risk_share' / 'high_risk_share'
      - `desire_condition` (or `desire`) in {'low', 'high'} when present
      - `effort_condition` (or `effort`) in {'low', 'high'} when present
      - `intimacy` (or `relationship_condition`) in {max_formal, neither, somewhat_intimate, max_intimate} when present
      - `stage` filter on 'posterior'
    """
    filepath = get_project_root() / "data" / slug / "main_trials_long.csv"
    data = pd.read_csv(filepath)
    data = data[data["stage"] == "posterior"].copy()
    data["action"] = data["action_condition"].map(ACTION_LABEL_TO_IDX)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    desire_map = {"low": 0, "high": 1}
    if "desire" in data.columns:
        data["desire_condition"] = data["desire"].map(desire_map)
    elif (
        "desire_condition" in data.columns and data["desire_condition"].dtype == object
    ):
        data["desire_condition"] = data["desire_condition"].map(desire_map)

    if "effort" in data.columns:
        data["effort_condition"] = data["effort"].map(EFFORT_CONDITION_TO_IDX)
    elif (
        "effort_condition" in data.columns and data["effort_condition"].dtype == object
    ):
        data["effort_condition"] = data["effort_condition"].map(EFFORT_CONDITION_TO_IDX)

    # Intimacy is stored as a verbal slug (no numeric code). Map it to the
    # 4-level RelationshipConditions index, and to the 101-bin index of its
    # placeholder continuous magnitude (RELATIONSHIP_LEVEL_VALUES × 100).
    intimacy_map = INTIMACY_CONDITION_TO_IDX
    intimacy_bin_101 = {
        slug: int(round(float(RELATIONSHIP_LEVEL_VALUES[idx]) * 100))
        for slug, idx in INTIMACY_CONDITION_TO_IDX.items()
    }
    if "intimacy" in data.columns:
        data["intimacy_idx_4"] = data["intimacy"].map(intimacy_map)
        data["intimacy_idx_101"] = data["intimacy"].map(intimacy_bin_101)
    elif "relationship_condition" in data.columns:
        data["intimacy_idx_4"] = data["relationship_condition"].map(intimacy_map)
        data["intimacy_idx_101"] = data["relationship_condition"].map(intimacy_bin_101)

    return data


def load_intimacy_data(slug="food_inv_intimacy"):
    """Study 2a — observer knows (desire, effort), infers intimacy."""
    data = _load_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    desire_condition = jnp.array(data["desire_condition"].values)
    effort_condition = jnp.array(data["effort_condition"].values)
    response = jnp.array(data["intimacy_rating"].values)
    print(f"Loaded {len(data)} posterior data points")
    return data, action, scenario_idx, desire_condition, effort_condition, response


def load_desire_data(slug="food_inv_desire"):
    """Study 1a — observer knows (effort, intimacy), infers desire.

    The desire DV is a continuous rating ("how much would the two people like the
    food?"), normalized to the 0-1 scale in preprocessing; the `response` column
    holds the posterior 0-1 rating. The fit scores it against the observer's
    101-bin desire posterior with `compute_desire_nll`.
    """
    data = _load_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    effort_condition = jnp.array(data["effort_condition"].values)
    relationship_condition = jnp.array(data["intimacy_idx_4"].values)
    response = jnp.array(data["response"].values)  # 0-1 desire rating
    print(f"Loaded {len(data)} posterior data points")
    return (
        data,
        action,
        scenario_idx,
        effort_condition,
        relationship_condition,
        response,
    )


def load_joint_de_data(slug="food_inv_joint_de"):
    """Study 1b — observer knows intimacy, jointly infers (desire, effort).

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
    resp_desire = jnp.array(data["desire_rating"].values)  # 0-1 desire rating
    resp_effort = jnp.array(data["effort_rating"].values)  # 0-1
    print(f"Loaded {len(data)} posterior data points (with 2 slider responses each)")
    return data, action, scenario_idx, relationship_condition, resp_desire, resp_effort


def load_joint_ie_data(slug="food_inv_joint_ie"):
    """Study 2b — observer knows desire, jointly infers (intimacy, effort).

    Each posterior trial contributes two slider responses; expects columns
    `intimacy_rating` and `effort_rating`, both on the 0-1 scale (normalized in
    preprocessing).
    """
    data = _load_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    desire_condition = jnp.array(data["desire_condition"].values)
    resp_intimacy = jnp.array(data["intimacy_rating"].values)
    resp_effort = jnp.array(data["effort_rating"].values)
    print(f"Loaded {len(data)} posterior data points (with 2 slider responses each)")
    return data, action, scenario_idx, desire_condition, resp_intimacy, resp_effort


# ------------------------------------------------------------------------------
# Table-kwargs helpers for the new pipeline (load lazily — tables may be None
# during early development if the LM CSVs haven't been produced yet).
# ------------------------------------------------------------------------------


def _padded_table_kwargs(loader, uses_g, study, slug_hint, desire_loader=None):
    """Shared assembly for the padded LM-alternatives kwargs. Raises
    FileNotFoundError if the study's LM CSVs are missing (run
    generate_alternatives.py + score_merged.py --study <slug> first).

    `uses_g` adds the goal-satisfaction table for the full/base variants (the
    discomfort-only variant has no desire term). `desire_loader`, when given (the
    given-desire studies 2a/2b), additionally loads the per-condition desire
    scalar table — desire there is observer-visible context, so the actor reads
    its magnitude from the LM rather than inferring it."""
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
    if uses_g:
        kw["g_padded_table"] = padded["g"]
        if desire_loader is not None:
            desire = desire_loader()
            if desire is None:
                raise FileNotFoundError(
                    f"lm_scenario_desire.csv not found for {study} — the "
                    "given-desire studies need the per-condition desire scalar. "
                    f"Run `model/lm/score_merged.py --study {slug_hint}` first."
                )
            kw["desire_table"] = desire
    return kw


def desire_table_kwargs(uses_g, domain="food"):
    """Padded LM tables for Study 1a (food_inv_desire). Observer's actor
    softmaxes over LM-generated alternatives per (scenario, observed_action,
    effort, intimacy) cell. Desire is inferred, so no desire scalar is loaded."""
    from tables import load_padded_lm_tables_desire

    if domain != "food":
        raise NotImplementedError(
            "Padded LM tables are only available for the food domain."
        )
    return _padded_table_kwargs(
        load_padded_lm_tables_desire, uses_g, "Study 1a", "food_inv_desire"
    )


def intimacy_table_kwargs(uses_g, domain="food"):
    """Padded LM tables for Study 2a (food_inv_intimacy). Cell grid
    (scenario, observed_action, desire, effort); infers intimacy. Desire is
    given, so the per-condition desire scalar is loaded for full/base."""
    from tables import (
        load_lm_scenario_desire,
        load_padded_lm_tables_intimacy,
    )

    if domain != "food":
        raise NotImplementedError(
            "Padded LM tables are only available for the food domain."
        )
    return _padded_table_kwargs(
        load_padded_lm_tables_intimacy,
        uses_g,
        "Study 2a",
        "food_inv_intimacy",
        desire_loader=lambda: load_lm_scenario_desire("food_inv_intimacy"),
    )


def joint_de_table_kwargs(uses_g, domain="food"):
    """Padded LM tables for Study 1b (food_inv_joint_de). Cell grid
    (scenario, observed_action, intimacy); jointly infers (desire, effort).
    Desire is inferred, so no desire scalar is loaded."""
    from tables import load_padded_lm_tables_joint_de

    if domain != "food":
        raise NotImplementedError(
            "Padded LM tables are only available for the food domain."
        )
    return _padded_table_kwargs(
        load_padded_lm_tables_joint_de, uses_g, "Study 1b", "food_inv_joint_de"
    )


def joint_ie_table_kwargs(uses_g, domain="food"):
    """Padded LM tables for Study 2b (food_inv_joint_ie). Cell grid
    (scenario, observed_action, desire); infers (intimacy, effort). Desire is
    given, so the per-condition desire scalar is loaded for full/base."""
    from tables import (
        load_lm_scenario_desire,
        load_padded_lm_tables_joint_ie,
    )

    if domain != "food":
        raise NotImplementedError(
            "Padded LM tables are only available for the food domain."
        )
    return _padded_table_kwargs(
        load_padded_lm_tables_joint_ie,
        uses_g,
        "Study 2b",
        "food_inv_joint_ie",
        desire_loader=lambda: load_lm_scenario_desire("food_inv_joint_ie"),
    )


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
# from the forward-planning (Study 1a) fit — the forward and inverse tasks have
# different data and different identifiability, so transferring would conflate
# the two fits.
#
# Params layout in the returned 1-D array: [*utility_param_values, alpha_observer].
# Caller maps utility_param_names → param_<name> columns; reads params[-1] as
# alpha_observer. Actor inverse-temperature is held at α=1 (same convention as
# the legacy padded joint fits).


def _build_observer_table(observer_fn, params, utility_param_names, table_kwargs):
    """Shared closure for the 5 joint fits below."""
    actor_kwargs = {"alpha": 1.0}
    for i, name in enumerate(utility_param_names):
        actor_kwargs[name] = params[i]
    return observer_fn(**actor_kwargs, alpha_observer=params[-1], **table_kwargs)


def fit_intimacy_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    desire_condition,
    effort_condition,
    response,
    table_kwargs,
    lr=0.1,
    max_steps=1000,
    verbose=True,
    n_restarts=5,
):
    """Study 2a — joint fit of utility weights + α_observer (intimacy NLL).

    Padded observer table is 6-D — (padded_slot, scenario, observed_action,
    desire, effort, relationship[101]). The observed action sits in slot 0, so
    the per-trial intimacy posterior is `table[0, scenario, action, desire, effort, :]`.
    """

    def nll_trial(table, a, s, r, e, resp):
        post = table[0, s, a, r, e, :]
        return compute_intimacy_nll(post, resp)

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params):
        table = _build_observer_table(
            observer_fn, params, utility_param_names, table_kwargs
        )
        return jnp.sum(
            vmap_nll(
                table,
                action,
                scenario_idx,
                desire_condition,
                effort_condition,
                response,
            )
        )

    params, nll, restarts = _fit_multistart(
        loss_fn,
        n_params=len(utility_param_names) + 1,
        n_restarts=n_restarts,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="intimacy_joint",
    )
    return params, float(nll), restarts


def fit_desire_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    effort_condition,
    relationship_condition,
    response,
    table_kwargs,
    lr=0.1,
    max_steps=1000,
    verbose=True,
    n_restarts=5,
):
    """Study 1a — joint fit of utility weights + α_observer (NLL on the
    continuous 0-1 desire DV).

    With the LM-generated alternatives pipeline, the observer table is 6-D —
    `(padded_slot, scenario, observed_action, effort, intimacy, desire[101])`. The
    canonical observed action lives in slot 0 by construction, so the per-trial
    desire posterior is `table[0, scenario, action, effort, intimacy, :]` (where
    `action` is the participant-observed action index 0/1/2), scored against the
    0-1 rating with `compute_desire_nll`.
    """

    def nll_trial(table, a, s, e, rel, resp):
        # Desire posterior over the 101-bin DesireLevels grid for the observed
        # action (slot 0). Table dims: (slot, scenario, observed_action, effort,
        # relationship, desire[101]).
        post = table[0, s, a, e, rel, :]
        return compute_desire_nll(post, resp)

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params):
        table = _build_observer_table(
            observer_fn, params, utility_param_names, table_kwargs
        )
        return jnp.sum(
            vmap_nll(
                table,
                action,
                scenario_idx,
                effort_condition,
                relationship_condition,
                response,
            )
        )

    params, nll, restarts = _fit_multistart(
        loss_fn,
        n_params=len(utility_param_names) + 1,
        n_restarts=n_restarts,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="desire_joint",
    )
    return params, float(nll), restarts


def fit_joint_de_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    relationship_condition,
    response_desire,
    response_effort,
    table_kwargs,
    lr=0.1,
    max_steps=1000,
    verbose=True,
    n_restarts=5,
):
    """Study 1b — joint fit of utility weights + α_observer.

    Padded observer table is 6-D — (padded_slot, scenario, observed_action,
    relationship, desire[101], effort). The observed action sits in slot 0, so the
    per-trial joint over (desire, effort) is `table[0, scenario, action, relationship, :, :]`.
    Sums two per-slider losses: NLL of the continuous 0-1 desire rating against
    the marginal 101-bin desire posterior (`compute_desire_nll`) and binary
    cross-entropy on the 0-1 effort slider (P(effort=HIGH)).
    """

    def nll_trial(table, a, s, rel, r_desire, r_effort):
        joint = table[0, s, a, rel, :, :]  # (desire[101], effort[2])
        desire_post = joint.sum(axis=1)  # marginal over effort -> (101,)
        p_effort_high = joint[:, 1].sum()  # marginal over desire
        return compute_desire_nll(desire_post, r_desire) + compute_effort_nll(
            p_effort_high, r_effort
        )

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params):
        table = _build_observer_table(
            observer_fn, params, utility_param_names, table_kwargs
        )
        return jnp.sum(
            vmap_nll(
                table,
                action,
                scenario_idx,
                relationship_condition,
                response_desire,
                response_effort,
            )
        )

    params, nll, restarts = _fit_multistart(
        loss_fn,
        n_params=len(utility_param_names) + 1,
        n_restarts=n_restarts,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="joint_de_joint",
    )
    return params, float(nll), restarts


def fit_joint_ie_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    desire_condition,
    response_intimacy,
    response_effort,
    table_kwargs,
    lr=0.1,
    max_steps=1000,
    verbose=True,
    n_restarts=5,
):
    """Study 2b — joint fit of utility weights + α_observer.

    Padded observer table is 6-D — (padded_slot, scenario, observed_action,
    desire, relationship[101], effort). The observed action sits in slot 0, so
    the per-trial joint over (intimacy, effort) is
    `table[0, scenario, action, desire, :, :]`. Per-trial NLL =
    compute_intimacy_nll on the marginal intimacy posterior + compute_effort_nll
    on the marginal P(effort_high).
    """

    def nll_trial(table, a, s, r, r_intimacy, r_effort):
        joint = table[0, s, a, r, :, :]  # (101, 2)
        p_intimacy = joint.sum(axis=1)  # (101,) marginal over effort
        p_effort_high = joint[:, 1].sum()  # scalar marginal over intimacy
        return compute_intimacy_nll(p_intimacy, r_intimacy) + compute_effort_nll(
            p_effort_high, r_effort
        )

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params):
        table = _build_observer_table(
            observer_fn, params, utility_param_names, table_kwargs
        )
        return jnp.sum(
            vmap_nll(
                table,
                action,
                scenario_idx,
                desire_condition,
                response_intimacy,
                response_effort,
            )
        )

    params, nll, restarts = _fit_multistart(
        loss_fn,
        n_params=len(utility_param_names) + 1,
        n_restarts=n_restarts,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="joint_ie_joint",
    )
    return params, float(nll), restarts
