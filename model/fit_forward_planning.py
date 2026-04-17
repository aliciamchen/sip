"""
Fit forward planning models to human data.

This script fits three actor models to human forward planning data:
1. Full model: intimacy scales both reward and discomfort
2. Vanilla model: no intimacy scaling
3. Discomfort-only model: only considers discomfort

Uses maximum likelihood estimation with gradient descent (optax.adam).
Performs likelihood ratio tests to compare models.
"""

import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from model_utils import (
    LLM_TABLES,
    SCENARIO_TO_IDX,
    IntimacyLevels,
    RewardConditions,
    actions,
    actor_forw_access_full,
    actor_forw_access_full_llm,
    actor_forw_access_only,
    actor_forw_access_only_llm,
    actor_forw_discomfort_only,
    actor_forw_full,
    actor_forw_no_access,
    actor_forw_no_access_llm,
    actor_forw_vanilla,
)
from scipy import stats

from utils import get_project_root

# Data loading and preprocessing


def load_data(filepath: str = None):
    """Load and preprocess forward planning data.

    Converts:
    - intimacy: 0/50/75/100 -> 0.0/0.5/0.75/1.0
    - motivation: low/high -> 0/1 (RewardConditions enum)
    - scenario_label: alphabetical index (0-15)

    Returns:
        data: pandas DataFrame
        intimacy: JAX array of intimacy levels (0-1)
        reward_condition: JAX array of reward conditions (0 or 1)
        action: JAX array of actions (0-3)
        p_action: JAX array of human response probabilities
        scenario_idx: JAX array of scenario indices (0-15)
    """
    if filepath is None:
        filepath = get_project_root() / "data" / "forw_plan" / "main_trials_long.csv"
    print("Loading forward planning data...")
    data = pd.read_csv(filepath)

    # Convert intimacy to 0-1 scale
    intimacy_map = {0: 0.0, 50: 0.5, 75: 0.75, 100: 1.0}
    data["intimacy_scaled"] = data["intimacy"].map(intimacy_map)

    # Convert motivation to reward condition (0 = low, 1 = high)
    motivation_map = {"low": 0, "high": 1}
    data["reward_condition"] = data["motivation"].map(motivation_map)

    # Convert scenario_label to index
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    # Extract JAX arrays
    intimacy = jnp.array(data["intimacy_scaled"].values)
    reward_condition = jnp.array(data["reward_condition"].values)
    action = jnp.array(data["action"].values)
    p_action = jnp.array(data["p_action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)

    print(f"Loaded {len(data)} data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")

    return data, intimacy, reward_condition, action, p_action, scenario_idx


# Loss function


@jax.jit
def compute_nll(preds, responses):
    """Compute negative log-likelihood.

    NLL = -sum(responses * log(preds))

    Args:
        preds: model predictions (probabilities)
        responses: human responses (probability distribution)

    Returns:
        NLL (scalar)
    """
    epsilon = 1e-8
    preds_safe = jnp.clip(preds, epsilon, 1.0)
    responses_safe = jnp.clip(responses, epsilon, 1.0)
    nll = -jnp.sum(responses_safe * jnp.log(preds_safe))
    return nll


# Model comparison metrics


def compute_aic(nll, n_params):
    """Compute Akaike Information Criterion.

    AIC = 2k + 2*NLL where k is number of parameters.
    """
    return 2 * n_params + 2 * nll


def compute_bic(nll, n_params, n_obs):
    """Compute Bayesian Information Criterion.

    BIC = k*ln(n) + 2*NLL where k is number of parameters, n is observations.
    """
    return n_params * np.log(n_obs) + 2 * nll


def compute_pearson_r_by_condition(data, pred_col, human_col, group_cols, n_boot=1000):
    """Compute Pearson r at condition x action level with bootstrap CI.

    Per preregistration: correlation computed at condition x action level
    with 95% bootstrapped confidence intervals.

    Args:
        data: DataFrame with predictions and human responses
        pred_col: column name for model predictions
        human_col: column name for human responses
        group_cols: columns to group by (e.g., ['intimacy', 'motivation', 'action'])
        n_boot: number of bootstrap samples

    Returns:
        dict with r, p, ci_lower, ci_upper
    """
    # Aggregate to condition x action level
    agg = (
        data.groupby(group_cols)
        .agg({pred_col: "mean", human_col: "mean"})
        .reset_index()
    )

    # Compute correlation
    r, p = stats.pearsonr(agg[pred_col], agg[human_col])

    # Bootstrap CI
    np.random.seed(42)
    boot_rs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(agg), size=len(agg), replace=True)
        boot_pred = agg[pred_col].iloc[idx].values
        boot_human = agg[human_col].iloc[idx].values
        boot_r, _ = stats.pearsonr(boot_pred, boot_human)
        boot_rs.append(boot_r)

    ci_lower = np.percentile(boot_rs, 2.5)
    ci_upper = np.percentile(boot_rs, 97.5)

    return {"r": r, "p": p, "ci_lower": ci_lower, "ci_upper": ci_upper}


# Vectorized prediction functions


def get_intimacy_index(intimacy_value):
    """Convert intimacy value (0-1) to index in IntimacyLevels array."""
    return jnp.round(intimacy_value * 100).astype(int)


# JIT-compiled prediction functions for each model
@jax.jit
def get_full_prediction(intimacy, reward_condition, action, alpha, w_r, w_d, w_c):
    """Get prediction from full model for single data point."""
    intimacy_idx = get_intimacy_index(intimacy)
    # Call model to get probability matrix, then index into it
    return actor_forw_full(alpha, w_r, w_d, w_c)[action, intimacy_idx, reward_condition]


@jax.jit
def get_vanilla_prediction(intimacy, reward_condition, action, alpha, w_r, w_d, w_c):
    """Get prediction from vanilla model for single data point."""
    intimacy_idx = get_intimacy_index(intimacy)
    return actor_forw_vanilla(alpha, w_r, w_d, w_c)[
        action, intimacy_idx, reward_condition
    ]


@jax.jit
def get_discomfort_only_prediction(intimacy, reward_condition, action, alpha, w_d):
    """Get prediction from discomfort-only model for single data point."""
    intimacy_idx = get_intimacy_index(intimacy)
    return actor_forw_discomfort_only(alpha, w_d)[
        action, intimacy_idx, reward_condition
    ]


# Vectorized prediction functions
@jax.jit
def predict_full(intimacy, reward_condition, action, alpha, w_r, w_d, w_c):
    """Vectorized prediction function for full model."""
    return jax.vmap(
        lambda i, r, a: get_full_prediction(i, r, a, alpha, w_r, w_d, w_c),
        in_axes=(0, 0, 0),
    )(intimacy, reward_condition, action)


@jax.jit
def predict_vanilla(intimacy, reward_condition, action, alpha, w_r, w_d, w_c):
    """Vectorized prediction function for vanilla model."""
    return jax.vmap(
        lambda i, r, a: get_vanilla_prediction(i, r, a, alpha, w_r, w_d, w_c),
        in_axes=(0, 0, 0),
    )(intimacy, reward_condition, action)


@jax.jit
def predict_discomfort_only(intimacy, reward_condition, action, alpha, w_d):
    """Vectorized prediction function for discomfort-only model."""
    return jax.vmap(
        lambda i, r, a: get_discomfort_only_prediction(i, r, a, alpha, w_d),
        in_axes=(0, 0, 0),
    )(intimacy, reward_condition, action)


# Access-based model predictions


@jax.jit
def predict_access_full(intimacy, reward_condition, action, alpha, w_v, w_r, w_d, w_e):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_access_full(alpha, w_v, w_r, w_d, w_e)
    return jax.vmap(lambda i, r, a: probs[a, i, r])(intimacy_idx, reward_condition, action)


@jax.jit
def predict_access_only(intimacy, reward_condition, action, alpha, w_r, w_d):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_access_only(alpha, w_r, w_d)
    return jax.vmap(lambda i, r, a: probs[a, i, r])(intimacy_idx, reward_condition, action)


@jax.jit
def predict_no_access(intimacy, reward_condition, action, alpha, w_v, w_e):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_no_access(alpha, w_v, w_e)
    return jax.vmap(lambda i, r, a: probs[a, i, r])(intimacy_idx, reward_condition, action)


# LLM-parameterized predictions. Shape note: memo models return tables of shape
# (actions, scenarios, intimacy, reward_condition); we index per-trial.


@jax.jit
def predict_access_full_llm(
    intimacy, reward_condition, action, scenario_idx,
    alpha, w_v, w_r, w_d, w_e,
    access_table, effort_table, reward_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_access_full_llm(
        alpha, w_v, w_r, w_d, w_e, access_table, effort_table, reward_table
    )
    return jax.vmap(lambda i, r, a, s: probs[a, s, i, r])(
        intimacy_idx, reward_condition, action, scenario_idx
    )


@jax.jit
def predict_access_only_llm(
    intimacy, reward_condition, action, scenario_idx,
    alpha, w_r, w_d,
    access_table, effort_table, reward_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_access_only_llm(
        alpha, w_r, w_d, access_table, effort_table, reward_table
    )
    return jax.vmap(lambda i, r, a, s: probs[a, s, i, r])(
        intimacy_idx, reward_condition, action, scenario_idx
    )


@jax.jit
def predict_no_access_llm(
    intimacy, reward_condition, action, scenario_idx,
    alpha, w_v, w_e,
    access_table, effort_table, reward_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_no_access_llm(
        alpha, w_v, w_e, access_table, effort_table, reward_table
    )
    return jax.vmap(lambda i, r, a, s: probs[a, s, i, r])(
        intimacy_idx, reward_condition, action, scenario_idx
    )


# Model fitting


def fit_full_model(
    intimacy: jnp.ndarray,
    reward_condition: jnp.ndarray,
    action: jnp.ndarray,
    p_action: jnp.ndarray,
    lr: float = 0.01,
    max_steps: int = 5000,
    verbose: bool = True,
):
    """Fit full model parameters (w_r, w_d, w_c) with alpha fixed to 1.

    Alpha is fixed to 1 for identifiability - only the products alpha*w_i
    are identifiable from the data, so we fix alpha=1 and let the weights
    absorb the scaling.
    """
    ALPHA = 1.0  # Fixed for identifiability

    def loss_fn(params):
        w_r, w_d, w_c = params[0], params[1], params[2]
        preds = predict_full(intimacy, reward_condition, action, ALPHA, w_r, w_d, w_c)
        return compute_nll(preds, p_action)

    params = jnp.array([1.0, 1.0, 1.0])  # w_r, w_d, w_c
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_nll = None
    for step in range(max_steps):
        nll, grad = grad_fn(params)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 1e-6, jnp.inf)

        if verbose and step % 1000 == 0:
            print(f"  Step {step}, NLL: {nll:.4f}, params: {params}")

        if prev_nll is not None and nll > prev_nll + 1e-6:
            if verbose:
                print(f"  NLL increased at step {step}, stopping")
            break
        prev_nll = nll

    best_nll = float(loss_fn(params))
    if verbose:
        print(f"  Final NLL: {best_nll:.4f}")
        print(
            f"  Final params (alpha=1 fixed): w_r={params[0]:.4f}, w_d={params[1]:.4f}, w_c={params[2]:.4f}"
        )

    # Return with alpha=1 prepended for compatibility
    full_params = jnp.array([ALPHA, params[0], params[1], params[2]])
    return full_params, best_nll


def fit_vanilla_model(
    intimacy: jnp.ndarray,
    reward_condition: jnp.ndarray,
    action: jnp.ndarray,
    p_action: jnp.ndarray,
    lr: float = 0.01,
    max_steps: int = 5000,
    verbose: bool = True,
):
    """Fit vanilla model parameters (w_r, w_d, w_c) with alpha fixed to 1."""
    ALPHA = 1.0  # Fixed for identifiability

    def loss_fn(params):
        w_r, w_d, w_c = params[0], params[1], params[2]
        preds = predict_vanilla(
            intimacy, reward_condition, action, ALPHA, w_r, w_d, w_c
        )
        return compute_nll(preds, p_action)

    params = jnp.array([1.0, 1.0, 1.0])  # w_r, w_d, w_c
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_nll = None
    for step in range(max_steps):
        nll, grad = grad_fn(params)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 1e-6, jnp.inf)

        if verbose and step % 1000 == 0:
            print(f"  Step {step}, NLL: {nll:.4f}, params: {params}")

        if prev_nll is not None and nll > prev_nll + 1e-6:
            if verbose:
                print(f"  NLL increased at step {step}, stopping")
            break
        prev_nll = nll

    best_nll = float(loss_fn(params))
    if verbose:
        print(f"  Final NLL: {best_nll:.4f}")
        print(
            f"  Final params (alpha=1 fixed): w_r={params[0]:.4f}, w_d={params[1]:.4f}, w_c={params[2]:.4f}"
        )

    # Return with alpha=1 prepended for compatibility
    full_params = jnp.array([ALPHA, params[0], params[1], params[2]])
    return full_params, best_nll


def fit_discomfort_only_model(
    intimacy: jnp.ndarray,
    reward_condition: jnp.ndarray,
    action: jnp.ndarray,
    p_action: jnp.ndarray,
    lr: float = 0.01,
    max_steps: int = 5000,
    verbose: bool = True,
):
    """Fit discomfort-only model parameters (w_d) with alpha fixed to 1."""
    ALPHA = 1.0  # Fixed for identifiability

    def loss_fn(params):
        w_d = params[0]
        preds = predict_discomfort_only(intimacy, reward_condition, action, ALPHA, w_d)
        return compute_nll(preds, p_action)

    params = jnp.array([1.0])  # w_d only
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_nll = None
    for step in range(max_steps):
        nll, grad = grad_fn(params)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 1e-6, jnp.inf)

        if verbose and step % 1000 == 0:
            print(f"  Step {step}, NLL: {nll:.4f}, params: {params}")

        if prev_nll is not None and nll > prev_nll + 1e-6:
            if verbose:
                print(f"  NLL increased at step {step}, stopping")
            break
        prev_nll = nll

    best_nll = float(loss_fn(params))
    if verbose:
        print(f"  Final NLL: {best_nll:.4f}")
        print(f"  Final params (alpha=1 fixed): w_d={params[0]:.4f}")

    # Return with alpha=1 prepended for compatibility
    full_params = jnp.array([ALPHA, params[0]])
    return full_params, best_nll


# Access-based model fitting (alpha=1 fixed)


def _fit_with_adam(
    loss_fn, init_params, lr=0.01, max_steps=5000, verbose=True, label=""
):
    """Shared adam fit loop with non-negativity clipping and NLL monotonicity stop."""
    params = jnp.array(init_params)
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_nll = None
    for step in range(max_steps):
        nll, grad = grad_fn(params)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 1e-6, jnp.inf)

        if verbose and step % 1000 == 0:
            print(f"  Step {step}, NLL: {nll:.4f}, params: {params}")

        if prev_nll is not None and nll > prev_nll + 1e-6:
            if verbose:
                print(f"  NLL increased at step {step}, stopping")
            break
        prev_nll = nll

    best_nll = float(loss_fn(params))
    if verbose:
        print(f"  {label} final NLL: {best_nll:.4f}, params: {params}")
    return params, best_nll


def fit_access_full_model(intimacy, reward_condition, action, p_action, **kwargs):
    ALPHA = 1.0

    def loss_fn(params):
        w_v, w_r, w_d, w_e = params
        preds = predict_access_full(
            intimacy, reward_condition, action, ALPHA, w_v, w_r, w_d, w_e
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0, 1.0, 1.0], label="access_full", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1], params[2], params[3]]), nll


def fit_access_only_model(intimacy, reward_condition, action, p_action, **kwargs):
    ALPHA = 1.0

    def loss_fn(params):
        w_r, w_d = params
        preds = predict_access_only(
            intimacy, reward_condition, action, ALPHA, w_r, w_d
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0], label="access_only", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1]]), nll


def fit_no_access_model(intimacy, reward_condition, action, p_action, **kwargs):
    ALPHA = 1.0

    def loss_fn(params):
        w_v, w_e = params
        preds = predict_no_access(
            intimacy, reward_condition, action, ALPHA, w_v, w_e
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0], label="no_access", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1]]), nll


# LLM-parameterized fits. Scenario-specific access/effort/reward tables are
# passed in via `tables = (access_table, effort_table, reward_table)`.


def fit_access_full_llm_model(
    intimacy, reward_condition, action, scenario_idx, p_action, tables, **kwargs
):
    ALPHA = 1.0
    a_tab, e_tab, r_tab = tables

    def loss_fn(params):
        w_v, w_r, w_d, w_e = params
        preds = predict_access_full_llm(
            intimacy, reward_condition, action, scenario_idx,
            ALPHA, w_v, w_r, w_d, w_e, a_tab, e_tab, r_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0, 1.0, 1.0], label="access_full_llm", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1], params[2], params[3]]), nll


def fit_access_only_llm_model(
    intimacy, reward_condition, action, scenario_idx, p_action, tables, **kwargs
):
    ALPHA = 1.0
    a_tab, e_tab, r_tab = tables

    def loss_fn(params):
        w_r, w_d = params
        preds = predict_access_only_llm(
            intimacy, reward_condition, action, scenario_idx,
            ALPHA, w_r, w_d, a_tab, e_tab, r_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0], label="access_only_llm", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1]]), nll


def fit_no_access_llm_model(
    intimacy, reward_condition, action, scenario_idx, p_action, tables, **kwargs
):
    ALPHA = 1.0
    a_tab, e_tab, r_tab = tables

    def loss_fn(params):
        w_v, w_e = params
        preds = predict_no_access_llm(
            intimacy, reward_condition, action, scenario_idx,
            ALPHA, w_v, w_e, a_tab, e_tab, r_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0], label="no_access_llm", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1]]), nll


# Main script


def main():
    print("=" * 60)
    print("Forward Planning Model Fitting")
    print("=" * 60)

    # Load data
    data, intimacy, reward_condition, action, p_action, scenario_idx = load_data()

    # Fit models
    results = {}

    # Full model: 3 free params (w_r, w_d, w_c) with alpha=1 fixed
    print("\n" + "-" * 40)
    print("Fitting FULL model (alpha=1 fixed)...")
    print("-" * 40)
    full_params, full_nll = fit_full_model(intimacy, reward_condition, action, p_action)
    results["full"] = {
        "params": {
            "alpha": float(full_params[0]),
            "w_r": float(full_params[1]),
            "w_d": float(full_params[2]),
            "w_c": float(full_params[3]),
        },
        "nll": full_nll,
        "n_params": 3,  # w_r, w_d, w_c (alpha fixed to 1)
    }

    # Vanilla model: 3 free params (w_r, w_d, w_c) with alpha=1 fixed
    print("\n" + "-" * 40)
    print("Fitting VANILLA model (alpha=1 fixed)...")
    print("-" * 40)
    vanilla_params, vanilla_nll = fit_vanilla_model(
        intimacy, reward_condition, action, p_action
    )
    results["vanilla"] = {
        "params": {
            "alpha": float(vanilla_params[0]),
            "w_r": float(vanilla_params[1]),
            "w_d": float(vanilla_params[2]),
            "w_c": float(vanilla_params[3]),
        },
        "nll": vanilla_nll,
        "n_params": 3,  # w_r, w_d, w_c (alpha fixed to 1)
    }

    # Discomfort-only model: 1 free param (w_d) with alpha=1 fixed
    print("\n" + "-" * 40)
    print("Fitting DISCOMFORT-ONLY model (alpha=1 fixed)...")
    print("-" * 40)
    discomfort_params, discomfort_nll = fit_discomfort_only_model(
        intimacy, reward_condition, action, p_action
    )
    results["discomfort_only"] = {
        "params": {
            "alpha": float(discomfort_params[0]),
            "w_d": float(discomfort_params[1]),
        },
        "nll": discomfort_nll,
        "n_params": 1,  # w_d only (alpha fixed to 1)
    }

    # Access-based model comparison (canonical reformulation)
    # Three variants: Full model (access_full), Access only, Base/No access
    access_fits = {
        "access_full": (
            fit_access_full_model,
            predict_access_full,
            ["w_v", "w_r", "w_d", "w_e"],
        ),
        "access_only": (
            fit_access_only_model,
            predict_access_only,
            ["w_r", "w_d"],
        ),
        "no_access": (
            fit_no_access_model,
            predict_no_access,
            ["w_v", "w_e"],
        ),
    }
    access_param_arrays = {}
    for name, (fit_fn, _pred_fn, param_names) in access_fits.items():
        print("\n" + "-" * 40)
        print(f"Fitting {name.upper()} model (alpha=1 fixed)...")
        print("-" * 40)
        params, nll = fit_fn(intimacy, reward_condition, action, p_action)
        access_param_arrays[name] = params
        results[name] = {
            "params": {
                "alpha": float(params[0]),
                **{pn: float(params[i + 1]) for i, pn in enumerate(param_names)},
            },
            "nll": nll,
            "n_params": len(param_names),
        }

    # LLM-parameterized variants (same 3 models, scenario-specific access/effort/reward)
    llm_fits = {}
    if LLM_TABLES is not None:
        llm_tables = (LLM_TABLES["access"], LLM_TABLES["effort"], LLM_TABLES["reward"])
        llm_fits = {
            "access_full_llm": (
                fit_access_full_llm_model,
                predict_access_full_llm,
                ["w_v", "w_r", "w_d", "w_e"],
            ),
            "access_only_llm": (
                fit_access_only_llm_model,
                predict_access_only_llm,
                ["w_r", "w_d"],
            ),
            "no_access_llm": (
                fit_no_access_llm_model,
                predict_no_access_llm,
                ["w_v", "w_e"],
            ),
        }
        for name, (fit_fn, _pred_fn, param_names) in llm_fits.items():
            print("\n" + "-" * 40)
            print(f"Fitting {name.upper()} model (alpha=1 fixed)...")
            print("-" * 40)
            params, nll = fit_fn(
                intimacy, reward_condition, action, scenario_idx, p_action, llm_tables
            )
            access_param_arrays[name] = params
            results[name] = {
                "params": {
                    "alpha": float(params[0]),
                    **{pn: float(params[i + 1]) for i, pn in enumerate(param_names)},
                },
                "nll": nll,
                "n_params": len(param_names),
            }
    else:
        print(
            "\n(skipping _llm variants: model/outputs/lm_scenario_params.csv not found; "
            "run `uv run python model/lm_scenario_params.py` first)"
        )

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  NLL: {result['nll']:.4f}")
        print(f"  Params: {result['params']}")

    # Generate predictions and save to CSV
    print("\n" + "-" * 40)
    print("Saving predictions...")
    print("-" * 40)

    # Add predictions to dataframe
    data["pred_full"] = np.array(
        predict_full(
            intimacy,
            reward_condition,
            action,
            full_params[0],
            full_params[1],
            full_params[2],
            full_params[3],
        )
    )
    data["pred_vanilla"] = np.array(
        predict_vanilla(
            intimacy,
            reward_condition,
            action,
            vanilla_params[0],
            vanilla_params[1],
            vanilla_params[2],
            vanilla_params[3],
        )
    )
    data["pred_discomfort_only"] = np.array(
        predict_discomfort_only(
            intimacy,
            reward_condition,
            action,
            discomfort_params[0],
            discomfort_params[1],
        )
    )

    # Access-model predictions per datapoint (scenario-agnostic variants)
    for name, (_fit_fn, pred_fn, _param_names) in access_fits.items():
        params = access_param_arrays[name]
        data[f"pred_{name}"] = np.array(
            pred_fn(intimacy, reward_condition, action, *params)
        )

    # LLM-variant predictions (need scenario_idx + tables)
    for name, (_fit_fn, pred_fn, _param_names) in llm_fits.items():
        params = access_param_arrays[name]
        data[f"pred_{name}"] = np.array(
            pred_fn(
                intimacy, reward_condition, action, scenario_idx,
                *params, *llm_tables,
            )
        )

    # Save predictions
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "forward_planning_fits.csv"
    data.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    # Compute model comparison metrics (AIC, BIC, Pearson r)
    print("\n" + "-" * 40)
    print("Computing model comparison metrics...")
    print("-" * 40)

    n_obs = len(data)
    group_cols = ["intimacy", "motivation", "action"]
    model_metrics = {}
    all_model_names = [
        "full",
        "vanilla",
        "discomfort_only",
        "access_full",
        "access_only",
        "no_access",
    ] + list(llm_fits.keys())

    for model_name in all_model_names:
        nll = results[model_name]["nll"]
        n_params = results[model_name]["n_params"]

        # AIC and BIC
        aic = compute_aic(nll, n_params)
        bic = compute_bic(nll, n_params, n_obs)

        # Pearson r at condition x action level
        pred_col = f"pred_{model_name}"
        r_result = compute_pearson_r_by_condition(
            data, pred_col, "p_action", group_cols
        )

        model_metrics[model_name] = {
            "aic": aic,
            "bic": bic,
            "r": r_result["r"],
            "r_ci_lower": r_result["ci_lower"],
            "r_ci_upper": r_result["ci_upper"],
        }

        print(
            f"  {model_name}: AIC={aic:.2f}, BIC={bic:.2f}, r={r_result['r']:.3f} [{r_result['ci_lower']:.3f}, {r_result['ci_upper']:.3f}]"
        )

    # Save results summary with all metrics
    results_rows = []
    for model_name in all_model_names:
        row = {
            "model": model_name,
            "nll": results[model_name]["nll"],
            "n_params": results[model_name]["n_params"],
            "aic": model_metrics[model_name]["aic"],
            "bic": model_metrics[model_name]["bic"],
            "r": model_metrics[model_name]["r"],
            "r_ci_lower": model_metrics[model_name]["r_ci_lower"],
            "r_ci_upper": model_metrics[model_name]["r_ci_upper"],
            **{f"param_{k}": v for k, v in results[model_name]["params"].items()},
        }
        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)
    results_path = output_dir / "forward_planning_fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")

    # Also emit a focused access-model-only CSV for the comparison
    # (3 stipulated-vector variants + up to 3 LLM variants)
    access_model_names = [
        "access_full",
        "access_only",
        "no_access",
    ] + list(llm_fits.keys())
    access_rows = [r for r in results_rows if r["model"] in access_model_names]
    access_df = pd.DataFrame(access_rows)
    access_results_path = output_dir / "access_model_forward_fit_results.csv"
    access_df.to_csv(access_results_path, index=False)
    print(f"Saved access-model comparison to {access_results_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
