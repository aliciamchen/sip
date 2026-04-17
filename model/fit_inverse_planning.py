"""
Fit observer alpha parameter to inverse planning human data.

This script fits alpha_observer to human inverse planning data while keeping
all other parameters (alpha, w_r, w_d, w_c) frozen from forward planning fits.

Uses negative log-likelihood (NLL) with gradient descent (optax.adam).

For intimacy inference: NLL = -log(P(intimacy = response/100 | action, reward))
For reward inference: Binary cross-entropy between response/100 and P(high reward)
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
    SCENARIO_LABELS,
    SCENARIO_TO_IDX,
    IntimacyLevels,
    RelationshipConditions,
    RewardConditions,
    actions,
    # Access-based intimacy observer models (stipulated-vector)
    observer_intimacy_access_full,
    observer_intimacy_access_only,
    observer_intimacy_discomfort_only,
    # LLM-parameterized intimacy observers
    observer_intimacy_access_full_llm,
    observer_intimacy_access_only_llm,
    observer_intimacy_no_access_llm,
    # Pre-registered intimacy observer models
    observer_intimacy_full_model,
    # Modified intimacy observer models (effort scaled by intimacy)
    observer_intimacy_full_model_modified,
    observer_intimacy_no_access,
    observer_intimacy_vanilla_inv_plan,
    # Access-based reward observer models (stipulated-vector)
    observer_reward_access_full,
    observer_reward_access_only,
    observer_reward_discomfort_only,
    # LLM-parameterized reward observers
    observer_reward_access_full_llm,
    observer_reward_access_only_llm,
    observer_reward_no_access_llm,
    # Pre-registered reward observer models
    observer_reward_full_model,
    # Modified reward observer models (effort scaled by intimacy)
    observer_reward_full_model_modified,
    observer_reward_no_access,
    observer_reward_vanilla_inv_plan,
)

from utils import get_project_root

# ==============================================================================
# Data Loading
# ==============================================================================


def load_fitted_params(filepath: str = None) -> dict:
    """Load frozen actor parameters from forward planning fit results.

    Returns a dict: model_name -> dict of every `param_*` column present in that
    row (stripped of the `param_` prefix). Missing/NaN columns are omitted, so
    each model keeps only its own parameter set (e.g. access_full has w_v/w_r/w_d/w_e,
    whereas the pre-registered full model has w_r/w_d/w_c).
    """
    if filepath is None:
        filepath = (
            get_project_root()
            / "model"
            / "outputs"
            / "forward_planning_fit_results.csv"
        )
    df = pd.read_csv(filepath)
    params = {}
    for _, row in df.iterrows():
        model_name = row["model"]
        model_params = {}
        for col in df.columns:
            if col.startswith("param_") and pd.notna(row[col]):
                model_params[col.replace("param_", "")] = float(row[col])
        # Legacy defaults so existing callers that unconditionally look up
        # w_r/w_d/w_c against pre-registered models continue to work.
        for legacy_key in ("w_r", "w_d", "w_c"):
            model_params.setdefault(legacy_key, 0.0)
        params[model_name] = model_params
    return params


def load_intimacy_data(filepath: str = None):
    """Load and preprocess intimacy inference data.

    Filters to posterior only and converts action_condition to int.

    Returns:
        data: pandas DataFrame (filtered to posterior)
        action: JAX array of actions (0-3)
        reward_condition: JAX array of reward conditions (0 or 1)
        response: JAX array of human intimacy ratings (0-100)
        scenario_idx: JAX array of scenario indices (0-15)
    """
    if filepath is None:
        filepath = (
            get_project_root() / "data" / "inv_plan_intimacy" / "main_trials_long.csv"
        )
    print("Loading intimacy inference data...")
    data = pd.read_csv(filepath)

    # Filter to posterior only
    data = data[data["stage"] == "posterior"].copy()

    # Convert action_condition to int (action_0 -> 0, etc.)
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int)

    # Convert motivation to reward condition (0 = low, 1 = high)
    motivation_map = {"low": 0, "high": 1}
    data["reward_condition"] = data["motivation"].map(motivation_map)

    # Convert scenario_label to index
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    # Extract JAX arrays
    action = jnp.array(data["action"].values)
    reward_condition = jnp.array(data["reward_condition"].values)
    response = jnp.array(data["intimacy_rating"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)

    print(f"Loaded {len(data)} posterior data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")

    return data, action, reward_condition, response, scenario_idx


def load_reward_data(filepath: str = None):
    """Load and preprocess reward inference data.

    Filters to posterior only and converts action_condition to int.

    Returns:
        data: pandas DataFrame (filtered to posterior)
        action: JAX array of actions (0-3)
        intimacy_condition: JAX array of intimacy conditions (0-3 index)
        response: JAX array of human reward likelihood ratings (0-100)
        scenario_idx: JAX array of scenario indices (0-15)
    """
    if filepath is None:
        filepath = (
            get_project_root() / "data" / "inv_plan_desire" / "main_trials_long.csv"
        )
    print("Loading reward inference data...")
    data = pd.read_csv(filepath)

    # Filter to posterior only
    data = data[data["stage"] == "posterior"].copy()

    # Convert action_condition to int (action_0 -> 0, etc.)
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int)

    # Convert intimacy to index (0, 50, 75, 100 -> 0, 1, 2, 3)
    intimacy_map = {0: 0, 50: 1, 75: 2, 100: 3}
    data["intimacy_idx"] = data["intimacy"].map(intimacy_map)

    # Convert scenario_label to index
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    # Extract JAX arrays
    action = jnp.array(data["action"].values)
    intimacy_condition = jnp.array(data["intimacy_idx"].values)
    response = jnp.array(data["response"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)

    print(f"Loaded {len(data)} posterior data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")

    return data, action, intimacy_condition, response, scenario_idx


# ==============================================================================
# Loss Functions
# ==============================================================================


@jax.jit
def compute_intimacy_nll(posterior, response):
    """Compute NLL for intimacy inference.

    Args:
        posterior: model posterior distribution over intimacy levels (shape: 101,)
        response: human response (0-100 scale, integer)

    Returns:
        NLL = -log(P(intimacy = response/100))
    """
    epsilon = 1e-8
    # Response is on 0-100 scale, posterior is indexed 0-100
    response_idx = jnp.clip(jnp.round(response).astype(int), 0, 100)
    prob = posterior[response_idx]
    return -jnp.log(jnp.clip(prob, epsilon, 1.0))


@jax.jit
def compute_reward_nll(p_high, response):
    """Compute binary cross-entropy NLL for reward inference.

    Args:
        p_high: model P(high reward)
        response: human response (0-100 scale, interpreted as P(high)*100)

    Returns:
        NLL = -(p_human * log(p_model) + (1-p_human) * log(1-p_model))
    """
    epsilon = 1e-8
    p_human = response / 100.0  # Convert to probability
    p_model = jnp.clip(p_high, epsilon, 1.0 - epsilon)
    nll = -(p_human * jnp.log(p_model) + (1 - p_human) * jnp.log(1 - p_model))
    return nll


# ==============================================================================
# Intimacy Model Prediction Functions
# ==============================================================================


def get_intimacy_posterior(
    observer_fn, actor_params, alpha_observer, action, reward_condition
):
    """Get posterior distribution over intimacy from model for single data point."""
    # Get full posterior distribution
    posterior = observer_fn(
        alpha=actor_params["alpha"],
        w_r=actor_params["w_r"],
        w_d=actor_params["w_d"],
        w_c=actor_params["w_c"],
        alpha_observer=alpha_observer,
    )
    # posterior shape: (actions, intimacy_levels, reward_conditions)
    # Get posterior for this action and reward condition
    post = posterior[action, :, reward_condition]
    return post  # Shape: (101,) - probability for each intimacy level 0-100


# ==============================================================================
# Modified Model Prediction Functions (with beta parameter)
# ==============================================================================


def get_intimacy_posterior_modified(
    observer_fn, actor_params, alpha_observer, beta, action, reward_condition
):
    """Get posterior distribution over intimacy from modified model."""
    posterior = observer_fn(
        alpha=actor_params["alpha"],
        w_r=actor_params["w_r"],
        w_d=actor_params["w_d"],
        w_c=actor_params["w_c"],
        alpha_observer=alpha_observer,
        beta=beta,
    )
    post = posterior[action, :, reward_condition]
    return post


def get_reward_p_high_modified(
    observer_fn, actor_params, alpha_observer, beta, action, intimacy_idx
):
    """Get P(high reward) from modified model."""
    posterior = observer_fn(
        alpha=actor_params["alpha"],
        w_r=actor_params["w_r"],
        w_d=actor_params["w_d"],
        w_c=actor_params["w_c"],
        alpha_observer=alpha_observer,
        beta=beta,
    )
    p_high_reward = posterior[action, intimacy_idx, 1]
    return p_high_reward


# ==============================================================================
# Reward Model Prediction Functions
# ==============================================================================


def get_reward_p_high(observer_fn, actor_params, alpha_observer, action, intimacy_idx):
    """Get P(high reward) from model for single data point."""
    # Get full posterior distribution
    posterior = observer_fn(
        alpha=actor_params["alpha"],
        w_r=actor_params["w_r"],
        w_d=actor_params["w_d"],
        w_c=actor_params["w_c"],
        alpha_observer=alpha_observer,
    )
    # posterior shape: (actions, relationship_conditions, reward_conditions)
    # Get P(high reward | action, intimacy) - index 1 is high reward
    p_high_reward = posterior[action, intimacy_idx, 1]  # Probability [0, 1]
    return p_high_reward


# ==============================================================================
# Fitting Functions
# ==============================================================================


def fit_intimacy_alpha_observer(
    observer_fn,
    actor_params: dict,
    action: jnp.ndarray,
    reward_condition: jnp.ndarray,
    response: jnp.ndarray,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer for intimacy inference model using NLL."""

    def get_nll(alpha_observer, a, r, resp):
        posterior = get_intimacy_posterior(
            observer_fn, actor_params, alpha_observer, a, r
        )
        return compute_intimacy_nll(posterior, resp)

    vmap_get_nll = jax.vmap(
        lambda alpha_obs, a, r, resp: get_nll(alpha_obs, a, r, resp),
        in_axes=(None, 0, 0, 0),
    )

    def loss_fn(params):
        alpha_observer = params[0]
        nlls = vmap_get_nll(alpha_observer, action, reward_condition, response)
        return jnp.sum(nlls)

    # Initialize alpha_observer = 1.0
    params = jnp.array([1.0])
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_loss = None
    zero_grad_count = 0
    for step in range(max_steps):
        loss, grad = grad_fn(params)

        # Check for zero/NaN gradient (happens when model predictions don't depend on alpha_observer)
        grad_magnitude = float(jnp.abs(grad[0]))
        if jnp.isnan(grad[0]) or grad_magnitude < 1e-10:
            zero_grad_count += 1
            if zero_grad_count >= 5:
                if verbose:
                    print(f"  Gradient is zero/NaN for 5 consecutive steps.")
                    print(
                        f"  This typically means the model's likelihood doesn't depend on the latent variable"
                    )
                    print(
                        f"  (e.g., vanilla model for intimacy inference). Returning alpha_observer=1.0."
                    )
                return 1.0, float(loss)
        else:
            zero_grad_count = 0

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        # Keep alpha_observer positive
        params = jnp.clip(params, 0.01, 10.0)

        if verbose and step % 200 == 0:
            print(f"  Step {step}, NLL: {loss:.4f}, alpha_observer: {params[0]:.4f}")

        if prev_loss is not None and loss > prev_loss + 1e-4:
            if verbose:
                print(f"  Loss increased at step {step}, stopping")
            break
        prev_loss = loss

    best_nll = float(loss_fn(params))
    final_alpha = float(params[0])

    # Final check for NaN
    if jnp.isnan(final_alpha):
        if verbose:
            print(f"  Warning: alpha_observer is NaN, defaulting to 1.0")
        final_alpha = 1.0

    if verbose:
        print(f"  Final NLL: {best_nll:.4f}")
        print(f"  Final alpha_observer: {final_alpha:.4f}")

    return final_alpha, best_nll


def fit_reward_alpha_observer(
    observer_fn,
    actor_params: dict,
    action: jnp.ndarray,
    intimacy_condition: jnp.ndarray,
    response: jnp.ndarray,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer for reward inference model using NLL (binary cross-entropy)."""

    def get_nll(alpha_observer, a, i, resp):
        p_high = get_reward_p_high(observer_fn, actor_params, alpha_observer, a, i)
        return compute_reward_nll(p_high, resp)

    vmap_get_nll = jax.vmap(
        lambda alpha_obs, a, i, resp: get_nll(alpha_obs, a, i, resp),
        in_axes=(None, 0, 0, 0),
    )

    def loss_fn(params):
        alpha_observer = params[0]
        nlls = vmap_get_nll(alpha_observer, action, intimacy_condition, response)
        return jnp.sum(nlls)

    # Initialize alpha_observer = 1.0
    params = jnp.array([1.0])
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_loss = None
    zero_grad_count = 0
    for step in range(max_steps):
        loss, grad = grad_fn(params)

        # Check for zero/NaN gradient (happens when model predictions don't depend on alpha_observer)
        grad_magnitude = float(jnp.abs(grad[0]))
        if jnp.isnan(grad[0]) or grad_magnitude < 1e-10:
            zero_grad_count += 1
            if zero_grad_count >= 5:
                if verbose:
                    print(f"  Gradient is zero/NaN for 5 consecutive steps.")
                    print(
                        f"  This typically means the model's likelihood doesn't depend on the latent variable."
                    )
                    print(f"  Returning alpha_observer=1.0.")
                return 1.0, float(loss)
        else:
            zero_grad_count = 0

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        # Keep alpha_observer positive
        params = jnp.clip(params, 0.01, 10.0)

        if verbose and step % 200 == 0:
            print(f"  Step {step}, NLL: {loss:.4f}, alpha_observer: {params[0]:.4f}")

        if prev_loss is not None and loss > prev_loss + 1e-4:
            if verbose:
                print(f"  Loss increased at step {step}, stopping")
            break
        prev_loss = loss

    best_nll = float(loss_fn(params))
    final_alpha = float(params[0])

    # Final check for NaN
    if jnp.isnan(final_alpha):
        if verbose:
            print(f"  Warning: alpha_observer is NaN, defaulting to 1.0")
        final_alpha = 1.0

    if verbose:
        print(f"  Final NLL: {best_nll:.4f}")
        print(f"  Final alpha_observer: {final_alpha:.4f}")

    return final_alpha, best_nll


# ==============================================================================
# Fitting Functions for Modified Models (joint alpha_observer and beta)
# ==============================================================================


def fit_intimacy_alpha_observer_and_beta(
    observer_fn,
    actor_params: dict,
    action: jnp.ndarray,
    reward_condition: jnp.ndarray,
    response: jnp.ndarray,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer and beta for modified intimacy inference model.

    beta controls how much observers think actors' reward scales with intimacy.
    beta=0: no intimacy scaling on reward (vanilla-like)
    beta=1: pre-registered model (full intimacy scaling)
    """

    def get_nll(alpha_observer, beta, a, r, resp):
        posterior = get_intimacy_posterior_modified(
            observer_fn, actor_params, alpha_observer, beta, a, r
        )
        return compute_intimacy_nll(posterior, resp)

    vmap_get_nll = jax.vmap(
        lambda alpha_obs, beta, a, r, resp: get_nll(alpha_obs, beta, a, r, resp),
        in_axes=(None, None, 0, 0, 0),
    )

    def loss_fn(params):
        alpha_observer, beta = params[0], params[1]
        nlls = vmap_get_nll(alpha_observer, beta, action, reward_condition, response)
        return jnp.sum(nlls)

    # Initialize: alpha_observer = 1.0, beta = 0.5 (start with partial scaling)
    params = jnp.array([1.0, 0.5])
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_loss = None
    for step in range(max_steps):
        loss, grad = grad_fn(params)

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        # Keep alpha_observer positive, beta in [0, 1] range
        params = jnp.array(
            [
                jnp.clip(params[0], 0.01, 10.0),
                jnp.clip(params[1], 0.0, 1.0),
            ]
        )

        if verbose and step % 200 == 0:
            print(
                f"  Step {step}, NLL: {loss:.4f}, alpha_observer: {params[0]:.4f}, beta: {params[1]:.4f}"
            )

        if prev_loss is not None and loss > prev_loss + 1e-4:
            if verbose:
                print(f"  Loss increased at step {step}, stopping")
            break
        prev_loss = loss

    best_nll = float(loss_fn(params))
    final_alpha = float(params[0])
    final_beta = float(params[1])

    if verbose:
        print(f"  Final NLL: {best_nll:.4f}")
        print(f"  Final alpha_observer: {final_alpha:.4f}, beta: {final_beta:.4f}")

    return final_alpha, final_beta, best_nll


def fit_reward_alpha_observer_and_beta(
    observer_fn,
    actor_params: dict,
    action: jnp.ndarray,
    intimacy_condition: jnp.ndarray,
    response: jnp.ndarray,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer and beta for modified reward inference model.

    beta controls how much observers think actors' reward scales with intimacy.
    beta=0: no intimacy scaling on reward (vanilla-like)
    beta=1: pre-registered model (full intimacy scaling)
    """

    def get_nll(alpha_observer, beta, a, i, resp):
        p_high = get_reward_p_high_modified(
            observer_fn, actor_params, alpha_observer, beta, a, i
        )
        return compute_reward_nll(p_high, resp)

    vmap_get_nll = jax.vmap(
        lambda alpha_obs, beta, a, i, resp: get_nll(alpha_obs, beta, a, i, resp),
        in_axes=(None, None, 0, 0, 0),
    )

    def loss_fn(params):
        alpha_observer, beta = params[0], params[1]
        nlls = vmap_get_nll(alpha_observer, beta, action, intimacy_condition, response)
        return jnp.sum(nlls)

    # Initialize: alpha_observer = 1.0, beta = 0.5 (start with partial scaling)
    params = jnp.array([1.0, 0.5])
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_loss = None
    for step in range(max_steps):
        loss, grad = grad_fn(params)

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        # Keep alpha_observer positive, beta in [0, 1] range
        params = jnp.array(
            [
                jnp.clip(params[0], 0.01, 10.0),
                jnp.clip(params[1], 0.0, 1.0),
            ]
        )

        if verbose and step % 200 == 0:
            print(
                f"  Step {step}, NLL: {loss:.4f}, alpha_observer: {params[0]:.4f}, beta: {params[1]:.4f}"
            )

        if prev_loss is not None and loss > prev_loss + 1e-4:
            if verbose:
                print(f"  Loss increased at step {step}, stopping")
            break
        prev_loss = loss

    best_nll = float(loss_fn(params))
    final_alpha = float(params[0])
    final_beta = float(params[1])

    if verbose:
        print(f"  Final NLL: {best_nll:.4f}")
        print(f"  Final alpha_observer: {final_alpha:.4f}, beta: {final_beta:.4f}")

    return final_alpha, final_beta, best_nll


# ==============================================================================
# Access-Based Model Fitting (generic on param signature)
# ==============================================================================
#
# The pre-registered observer fitters hard-code the (alpha, w_r, w_d, w_c) actor
# signature. Access variants use different parameter sets (w_v/w_r/w_d/w_e or
# subsets), so we fit each via a generic routine that builds the observer_fn
# kwargs dynamically from a dict of actor-param names.


def _fit_alpha_observer_generic(
    observer_fn,
    actor_params: dict,
    actor_kwarg_names,
    action: jnp.ndarray,
    conditioning: jnp.ndarray,
    response: jnp.ndarray,
    nll_fn,
    posterior_slicer,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer for any observer model by dict-keyed actor params.

    Args:
        observer_fn: memo observer function.
        actor_params: dict of fitted actor params (e.g. {'alpha':1,'w_v':..., ...}).
        actor_kwarg_names: iterable of keys from actor_params to pass into observer_fn.
        conditioning: per-trial int array passed as the second indexing axis into the
            observer's output table (reward_condition for intimacy-inference, or
            intimacy_idx for reward-inference).
        nll_fn: (slice, response) -> scalar NLL.
        posterior_slicer: (posterior_tensor, action_i, cond_i) -> relevant slice to
            pass into nll_fn.
    """
    actor_kwargs = {k: actor_params[k] for k in actor_kwarg_names}

    def observer_table(alpha_observer):
        return observer_fn(**actor_kwargs, alpha_observer=alpha_observer)

    def get_nll(alpha_observer, a, c, resp):
        table = observer_table(alpha_observer)
        slc = posterior_slicer(table, a, c)
        return nll_fn(slc, resp)

    vmap_get_nll = jax.vmap(
        lambda alpha_obs, a, c, resp: get_nll(alpha_obs, a, c, resp),
        in_axes=(None, 0, 0, 0),
    )

    def loss_fn(params):
        alpha_observer = params[0]
        nlls = vmap_get_nll(alpha_observer, action, conditioning, response)
        return jnp.sum(nlls)

    params = jnp.array([1.0])
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_loss = None
    zero_grad_count = 0
    for step in range(max_steps):
        loss, grad = grad_fn(params)
        grad_mag = float(jnp.abs(grad[0]))
        if jnp.isnan(grad[0]) or grad_mag < 1e-10:
            zero_grad_count += 1
            if zero_grad_count >= 5:
                if verbose:
                    print("  Gradient zero/NaN for 5 consecutive steps; alpha_observer=1.0")
                return 1.0, float(loss)
        else:
            zero_grad_count = 0

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 0.01, 10.0)

        if verbose and step % 200 == 0:
            print(f"  Step {step}, NLL: {loss:.4f}, alpha_observer: {params[0]:.4f}")

        if prev_loss is not None and loss > prev_loss + 1e-4:
            if verbose:
                print(f"  Loss increased at step {step}, stopping")
            break
        prev_loss = loss

    best_nll = float(loss_fn(params))
    final_alpha = float(params[0])
    if jnp.isnan(final_alpha):
        final_alpha = 1.0
    if verbose:
        print(f"  Final NLL: {best_nll:.4f}, alpha_observer: {final_alpha:.4f}")
    return final_alpha, best_nll


def fit_access_intimacy_observer(
    observer_fn, actor_params, actor_kwarg_names,
    action, reward_condition, response, **kwargs,
):
    return _fit_alpha_observer_generic(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        conditioning=reward_condition,
        response=response,
        nll_fn=compute_intimacy_nll,
        posterior_slicer=lambda tab, a, r: tab[a, :, r],
        **kwargs,
    )


def fit_access_reward_observer(
    observer_fn, actor_params, actor_kwarg_names,
    action, intimacy_condition, response, **kwargs,
):
    return _fit_alpha_observer_generic(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        conditioning=intimacy_condition,
        response=response,
        nll_fn=compute_reward_nll,
        posterior_slicer=lambda tab, a, i: tab[a, i, 1],
        **kwargs,
    )


# ==============================================================================
# LLM-Parameterized Observer Fitting
# ==============================================================================
# LLM observer memo models add a `scenario_idx` dimension, so their output
# tables are 4D: (action, scenario, intimacy_or_relationship, reward_condition).
# Per-trial NLL indexes by (action, scenario, conditioning).


def _fit_alpha_observer_llm_generic(
    observer_fn,
    actor_params: dict,
    actor_kwarg_names,
    action: jnp.ndarray,
    scenario_idx: jnp.ndarray,
    conditioning: jnp.ndarray,
    response: jnp.ndarray,
    nll_fn,
    posterior_slicer,
    tables,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer for an _llm observer model.

    posterior_slicer signature: (table, action_i, scenario_i, cond_i) -> slice.
    tables: (access_table, effort_table, reward_table) passed to the observer.
    """
    actor_kwargs = {k: actor_params[k] for k in actor_kwarg_names}
    a_tab, e_tab, r_tab = tables

    def observer_table(alpha_observer):
        return observer_fn(
            **actor_kwargs, alpha_observer=alpha_observer,
            access_table=a_tab, effort_table=e_tab, reward_table=r_tab,
        )

    def get_nll(alpha_observer, a, s, c, resp):
        table = observer_table(alpha_observer)
        slc = posterior_slicer(table, a, s, c)
        return nll_fn(slc, resp)

    vmap_get_nll = jax.vmap(
        lambda alpha_obs, a, s, c, resp: get_nll(alpha_obs, a, s, c, resp),
        in_axes=(None, 0, 0, 0, 0),
    )

    def loss_fn(params):
        return jnp.sum(
            vmap_get_nll(params[0], action, scenario_idx, conditioning, response)
        )

    params = jnp.array([1.0])
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_loss = None
    zero_grad_count = 0
    for step in range(max_steps):
        loss, grad = grad_fn(params)
        grad_mag = float(jnp.abs(grad[0]))
        if jnp.isnan(grad[0]) or grad_mag < 1e-10:
            zero_grad_count += 1
            if zero_grad_count >= 5:
                if verbose:
                    print("  Gradient zero/NaN for 5 consecutive steps; alpha_observer=1.0")
                return 1.0, float(loss)
        else:
            zero_grad_count = 0

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 0.01, 10.0)

        if verbose and step % 200 == 0:
            print(f"  Step {step}, NLL: {loss:.4f}, alpha_observer: {params[0]:.4f}")

        if prev_loss is not None and loss > prev_loss + 1e-4:
            if verbose:
                print(f"  Loss increased at step {step}, stopping")
            break
        prev_loss = loss

    best_nll = float(loss_fn(params))
    final_alpha = float(params[0])
    if jnp.isnan(final_alpha):
        final_alpha = 1.0
    if verbose:
        print(f"  Final NLL: {best_nll:.4f}, alpha_observer: {final_alpha:.4f}")
    return final_alpha, best_nll


def fit_access_intimacy_observer_llm(
    observer_fn, actor_params, actor_kwarg_names,
    action, scenario_idx, reward_condition, response, tables, **kwargs,
):
    return _fit_alpha_observer_llm_generic(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=reward_condition,
        response=response,
        nll_fn=compute_intimacy_nll,
        posterior_slicer=lambda tab, a, s, r: tab[a, s, :, r],
        tables=tables,
        **kwargs,
    )


def fit_access_reward_observer_llm(
    observer_fn, actor_params, actor_kwarg_names,
    action, scenario_idx, intimacy_condition, response, tables, **kwargs,
):
    return _fit_alpha_observer_llm_generic(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=intimacy_condition,
        response=response,
        nll_fn=compute_reward_nll,
        posterior_slicer=lambda tab, a, s, i: tab[a, s, i, 1],
        tables=tables,
        **kwargs,
    )


# Access variant registry: name -> (intimacy_observer, reward_observer, actor_kwargs)
ACCESS_VARIANTS = {
    "access_full": (
        observer_intimacy_access_full,
        observer_reward_access_full,
        ["alpha", "w_v", "w_r", "w_d", "w_e"],
    ),
    "access_only": (
        observer_intimacy_access_only,
        observer_reward_access_only,
        ["alpha", "w_r", "w_d"],
    ),
    "no_access": (
        observer_intimacy_no_access,
        observer_reward_no_access,
        ["alpha", "w_v", "w_e"],
    ),
}

# LLM variant registry (same structure; actor_params are loaded from the
# _llm rows in forward_planning_fit_results.csv).
ACCESS_LLM_VARIANTS = {
    "access_full_llm": (
        observer_intimacy_access_full_llm,
        observer_reward_access_full_llm,
        ["alpha", "w_v", "w_r", "w_d", "w_e"],
    ),
    "access_only_llm": (
        observer_intimacy_access_only_llm,
        observer_reward_access_only_llm,
        ["alpha", "w_r", "w_d"],
    ),
    "no_access_llm": (
        observer_intimacy_no_access_llm,
        observer_reward_no_access_llm,
        ["alpha", "w_v", "w_e"],
    ),
}


# ==============================================================================
# Main Script
# ==============================================================================


def main():
    print("=" * 60)
    print("Inverse Planning Model Fitting")
    print("Fitting alpha_observer (frozen actor params)")
    print("=" * 60)

    # Load frozen actor parameters
    print("\nLoading frozen actor parameters...")
    actor_params = load_fitted_params()
    for model_name, p in actor_params.items():
        print(
            f"  {model_name}: alpha={p['alpha']:.3f}, w_r={p['w_r']:.3f}, w_d={p['w_d']:.3f}, w_c={p['w_c']:.3f}"
        )

    results = []

    # -------------------------------------------------------------------------
    # Fit Intimacy Inference Models
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INTIMACY INFERENCE")
    print("=" * 60)

    int_data, int_action, int_reward_condition, int_response, int_scenario_idx = (
        load_intimacy_data()
    )

    # Pre-registered models (fit alpha_observer only)
    intimacy_models = {
        "full": (observer_intimacy_full_model, "full"),
        "vanilla": (observer_intimacy_vanilla_inv_plan, "vanilla"),
        "discomfort_only": (observer_intimacy_discomfort_only, "discomfort_only"),
    }

    # Modified models (fit alpha_observer and beta)
    intimacy_models_modified = {
        "full_modified": (observer_intimacy_full_model_modified, "full"),
    }

    # Fit pre-registered models (alpha_observer only)
    for model_name, (observer_fn, actor_param_key) in intimacy_models.items():
        print(f"\n{'-' * 40}")
        print(f"Fitting {model_name}...")
        print(f"{'-' * 40}")

        alpha_observer, nll = fit_intimacy_alpha_observer(
            observer_fn=observer_fn,
            actor_params=actor_params[actor_param_key],
            action=int_action,
            reward_condition=int_reward_condition,
            response=int_response,
        )

        results.append(
            {
                "model": model_name,
                "experiment": "intimacy",
                "alpha_observer": alpha_observer,
                "beta": np.nan,
                "nll": nll,
                "n_params": 1,
            }
        )

    # Fit modified models (alpha_observer and beta)
    for model_name, (observer_fn, actor_param_key) in intimacy_models_modified.items():
        print(f"\n{'-' * 40}")
        print(f"Fitting {model_name} (with beta)...")
        print(f"{'-' * 40}")

        alpha_observer, beta, nll = fit_intimacy_alpha_observer_and_beta(
            observer_fn=observer_fn,
            actor_params=actor_params[actor_param_key],
            action=int_action,
            reward_condition=int_reward_condition,
            response=int_response,
        )

        results.append(
            {
                "model": model_name,
                "experiment": "intimacy",
                "alpha_observer": alpha_observer,
                "beta": beta,
                "nll": nll,
                "n_params": 2,
            }
        )

    # Fit access-based intimacy observers (actor params frozen from forward fit)
    for variant_name, (int_obs, _rew_obs, kw_names) in ACCESS_VARIANTS.items():
        if variant_name not in actor_params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        print(f"\n{'-' * 40}")
        print(f"Fitting {variant_name} (intimacy inference)...")
        print(f"{'-' * 40}")
        alpha_observer, nll = fit_access_intimacy_observer(
            observer_fn=int_obs,
            actor_params=actor_params[variant_name],
            actor_kwarg_names=kw_names,
            action=int_action,
            reward_condition=int_reward_condition,
            response=int_response,
        )
        results.append(
            {
                "model": variant_name,
                "experiment": "intimacy",
                "alpha_observer": alpha_observer,
                "beta": np.nan,
                "nll": nll,
                "n_params": 1,
            }
        )

    # LLM-parameterized variants (need per-trial scenario_idx + scenario tables)
    if LLM_TABLES is not None:
        llm_tables = (LLM_TABLES["access"], LLM_TABLES["effort"], LLM_TABLES["reward"])
        for variant_name, (int_obs, _rew_obs, kw_names) in ACCESS_LLM_VARIANTS.items():
            if variant_name not in actor_params:
                print(f"  (skipping {variant_name}: no forward fit available)")
                continue
            print(f"\n{'-' * 40}")
            print(f"Fitting {variant_name} (intimacy inference, LLM-param)...")
            print(f"{'-' * 40}")
            alpha_observer, nll = fit_access_intimacy_observer_llm(
                observer_fn=int_obs,
                actor_params=actor_params[variant_name],
                actor_kwarg_names=kw_names,
                action=int_action,
                scenario_idx=int_scenario_idx,
                reward_condition=int_reward_condition,
                response=int_response,
                tables=llm_tables,
            )
            results.append(
                {
                    "model": variant_name,
                    "experiment": "intimacy",
                    "alpha_observer": alpha_observer,
                    "beta": np.nan,
                    "nll": nll,
                    "n_params": 1,
                }
            )

    # -------------------------------------------------------------------------
    # Fit Reward Inference Models
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("REWARD INFERENCE")
    print("=" * 60)

    rew_data, rew_action, rew_intimacy_condition, rew_response, rew_scenario_idx = (
        load_reward_data()
    )

    # Pre-registered models (fit alpha_observer only)
    reward_models = {
        "full": (observer_reward_full_model, "full"),
        "vanilla": (observer_reward_vanilla_inv_plan, "vanilla"),
        "discomfort_only": (observer_reward_discomfort_only, "discomfort_only"),
    }

    # Modified models (fit alpha_observer and beta)
    reward_models_modified = {
        "full_modified": (observer_reward_full_model_modified, "full"),
    }

    # Fit pre-registered models (alpha_observer only)
    for model_name, (observer_fn, actor_param_key) in reward_models.items():
        print(f"\n{'-' * 40}")
        print(f"Fitting {model_name}...")
        print(f"{'-' * 40}")

        alpha_observer, nll = fit_reward_alpha_observer(
            observer_fn=observer_fn,
            actor_params=actor_params[actor_param_key],
            action=rew_action,
            intimacy_condition=rew_intimacy_condition,
            response=rew_response,
        )

        results.append(
            {
                "model": model_name,
                "experiment": "reward",
                "alpha_observer": alpha_observer,
                "beta": np.nan,
                "nll": nll,
                "n_params": 1,
            }
        )

    # Fit modified models (alpha_observer and beta)
    for model_name, (observer_fn, actor_param_key) in reward_models_modified.items():
        print(f"\n{'-' * 40}")
        print(f"Fitting {model_name} (with beta)...")
        print(f"{'-' * 40}")

        alpha_observer, beta, nll = fit_reward_alpha_observer_and_beta(
            observer_fn=observer_fn,
            actor_params=actor_params[actor_param_key],
            action=rew_action,
            intimacy_condition=rew_intimacy_condition,
            response=rew_response,
        )

        results.append(
            {
                "model": model_name,
                "experiment": "reward",
                "alpha_observer": alpha_observer,
                "beta": beta,
                "nll": nll,
                "n_params": 2,
            }
        )

    # Fit access-based reward observers (actor params frozen from forward fit)
    for variant_name, (_int_obs, rew_obs, kw_names) in ACCESS_VARIANTS.items():
        if variant_name not in actor_params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        print(f"\n{'-' * 40}")
        print(f"Fitting {variant_name} (reward inference)...")
        print(f"{'-' * 40}")
        alpha_observer, nll = fit_access_reward_observer(
            observer_fn=rew_obs,
            actor_params=actor_params[variant_name],
            actor_kwarg_names=kw_names,
            action=rew_action,
            intimacy_condition=rew_intimacy_condition,
            response=rew_response,
        )
        results.append(
            {
                "model": variant_name,
                "experiment": "reward",
                "alpha_observer": alpha_observer,
                "beta": np.nan,
                "nll": nll,
                "n_params": 1,
            }
        )

    # LLM-parameterized variants (need per-trial scenario_idx + scenario tables)
    if LLM_TABLES is not None:
        llm_tables = (LLM_TABLES["access"], LLM_TABLES["effort"], LLM_TABLES["reward"])
        for variant_name, (_int_obs, rew_obs, kw_names) in ACCESS_LLM_VARIANTS.items():
            if variant_name not in actor_params:
                print(f"  (skipping {variant_name}: no forward fit available)")
                continue
            print(f"\n{'-' * 40}")
            print(f"Fitting {variant_name} (reward inference, LLM-param)...")
            print(f"{'-' * 40}")
            alpha_observer, nll = fit_access_reward_observer_llm(
                observer_fn=rew_obs,
                actor_params=actor_params[variant_name],
                actor_kwarg_names=kw_names,
                action=rew_action,
                scenario_idx=rew_scenario_idx,
                intimacy_condition=rew_intimacy_condition,
                response=rew_response,
                tables=llm_tables,
            )
            results.append(
                {
                    "model": variant_name,
                    "experiment": "reward",
                    "alpha_observer": alpha_observer,
                    "beta": np.nan,
                    "nll": nll,
                    "n_params": 1,
                }
            )

    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Save
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "inverse_planning_fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")

    access_names = list(ACCESS_VARIANTS.keys()) + list(ACCESS_LLM_VARIANTS.keys())
    access_df = results_df[results_df["model"].isin(access_names)].copy()
    access_path = output_dir / "access_model_inverse_fit_results.csv"
    access_df.to_csv(access_path, index=False)
    print(f"Saved access-model comparison to {access_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    main()
