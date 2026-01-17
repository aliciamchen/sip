"""
Fit observer alpha parameter to inverse planning human data.

This script fits alpha_observer to human inverse planning data while keeping
all other parameters (alpha, w_r, w_d, w_c) frozen from forward planning fits.

Uses negative log-likelihood (NLL) with gradient descent (optax.adam).

For intimacy inference: NLL = -log(P(intimacy = response/100 | action, reward))
For reward inference: Binary cross-entropy between response/100 and P(high reward)
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
from pathlib import Path

from model_utils import (
    actions,
    IntimacyLevels,
    RewardConditions,
    RelationshipConditions,
    SCENARIO_LABELS,
    SCENARIO_TO_IDX,
    # Pre-registered intimacy observer models
    observer_intimacy_full_model,
    observer_intimacy_vanilla_inv_plan,
    observer_intimacy_discomfort_only,
    observer_intimacy_full_model_lm,
    observer_intimacy_vanilla_lm,
    observer_intimacy_discomfort_only_lm,
    # Modified intimacy observer models (effort scaled by intimacy)
    observer_intimacy_full_model_modified,
    observer_intimacy_full_model_lm_modified,
    # Pre-registered reward observer models
    observer_reward_full_model,
    observer_reward_vanilla_inv_plan,
    observer_reward_discomfort_only,
    observer_reward_full_model_lm,
    observer_reward_vanilla_lm,
    observer_reward_discomfort_only_lm,
    # Modified reward observer models (effort scaled by intimacy)
    observer_reward_full_model_modified,
    observer_reward_full_model_lm_modified,
)


# ==============================================================================
# Data Loading
# ==============================================================================


def load_fitted_params(filepath: str = "forward_planning_fit_results.csv") -> dict:
    """Load frozen actor parameters from forward planning fit results."""
    df = pd.read_csv(filepath)
    params = {}
    for _, row in df.iterrows():
        model_name = row["model"]
        params[model_name] = {
            "alpha": row["param_alpha"],
            "w_r": row.get("param_w_r", 0.0) if pd.notna(row.get("param_w_r")) else 0.0,
            "w_d": row["param_w_d"],
            "w_c": row.get("param_w_c", 0.0) if pd.notna(row.get("param_w_c")) else 0.0,
        }
    return params


def load_intimacy_data(filepath: str = "../data/inv_plan_intimacy/main_trials_long.csv"):
    """Load and preprocess intimacy inference data.

    Filters to posterior only and converts action_condition to int.

    Returns:
        data: pandas DataFrame (filtered to posterior)
        action: JAX array of actions (0-3)
        reward_condition: JAX array of reward conditions (0 or 1)
        response: JAX array of human intimacy ratings (0-100)
        scenario_idx: JAX array of scenario indices (0-15)
    """
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


def load_reward_data(filepath: str = "../data/inv_plan_reward/main_trials_long.csv"):
    """Load and preprocess reward inference data.

    Filters to posterior only and converts action_condition to int.

    Returns:
        data: pandas DataFrame (filtered to posterior)
        action: JAX array of actions (0-3)
        intimacy_condition: JAX array of intimacy conditions (0-3 index)
        response: JAX array of human reward likelihood ratings (0-100)
        scenario_idx: JAX array of scenario indices (0-15)
    """
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


def get_intimacy_posterior_stipulated(observer_fn, actor_params, alpha_observer, action, reward_condition):
    """Get posterior distribution over intimacy from stipulated model for single data point."""
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


def get_intimacy_posterior_lm(observer_fn, actor_params, alpha_observer, action, reward_condition, scenario_idx):
    """Get posterior distribution over intimacy from LM model for single data point."""
    # Get full posterior distribution
    posterior = observer_fn(
        scenario_idx=scenario_idx,
        alpha=actor_params["alpha"],
        w_r=actor_params["w_r"],
        w_d=actor_params["w_d"],
        w_c=actor_params["w_c"],
        alpha_observer=alpha_observer,
    )
    # Get posterior for this action and reward condition
    post = posterior[action, :, reward_condition]
    return post  # Shape: (101,) - probability for each intimacy level 0-100


# ==============================================================================
# Modified Model Prediction Functions (with delta parameter)
# ==============================================================================


def get_intimacy_posterior_stipulated_modified(observer_fn, actor_params, alpha_observer, delta, action, reward_condition):
    """Get posterior distribution over intimacy from modified stipulated model."""
    posterior = observer_fn(
        alpha=actor_params["alpha"],
        w_r=actor_params["w_r"],
        w_d=actor_params["w_d"],
        w_c=actor_params["w_c"],
        alpha_observer=alpha_observer,
        delta=delta,
    )
    post = posterior[action, :, reward_condition]
    return post


def get_intimacy_posterior_lm_modified(observer_fn, actor_params, alpha_observer, delta, action, reward_condition, scenario_idx):
    """Get posterior distribution over intimacy from modified LM model."""
    posterior = observer_fn(
        scenario_idx=scenario_idx,
        alpha=actor_params["alpha"],
        w_r=actor_params["w_r"],
        w_d=actor_params["w_d"],
        w_c=actor_params["w_c"],
        alpha_observer=alpha_observer,
        delta=delta,
    )
    post = posterior[action, :, reward_condition]
    return post


def get_reward_p_high_stipulated_modified(observer_fn, actor_params, alpha_observer, delta, action, intimacy_idx):
    """Get P(high reward) from modified stipulated model."""
    posterior = observer_fn(
        alpha=actor_params["alpha"],
        w_r=actor_params["w_r"],
        w_d=actor_params["w_d"],
        w_c=actor_params["w_c"],
        alpha_observer=alpha_observer,
        delta=delta,
    )
    p_high_reward = posterior[action, intimacy_idx, 1]
    return p_high_reward


def get_reward_p_high_lm_modified(observer_fn, actor_params, alpha_observer, delta, action, intimacy_idx, scenario_idx):
    """Get P(high reward) from modified LM model."""
    posterior = observer_fn(
        scenario_idx=scenario_idx,
        alpha=actor_params["alpha"],
        w_r=actor_params["w_r"],
        w_d=actor_params["w_d"],
        w_c=actor_params["w_c"],
        alpha_observer=alpha_observer,
        delta=delta,
    )
    p_high_reward = posterior[action, intimacy_idx, 1]
    return p_high_reward


# ==============================================================================
# Reward Model Prediction Functions
# ==============================================================================


def get_reward_p_high_stipulated(observer_fn, actor_params, alpha_observer, action, intimacy_idx):
    """Get P(high reward) from stipulated model for single data point."""
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


def get_reward_p_high_lm(observer_fn, actor_params, alpha_observer, action, intimacy_idx, scenario_idx):
    """Get P(high reward) from LM model for single data point."""
    # Get full posterior distribution
    posterior = observer_fn(
        scenario_idx=scenario_idx,
        alpha=actor_params["alpha"],
        w_r=actor_params["w_r"],
        w_d=actor_params["w_d"],
        w_c=actor_params["w_c"],
        alpha_observer=alpha_observer,
    )
    # Get P(high reward | action, intimacy)
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
    scenario_idx: jnp.ndarray = None,
    is_lm: bool = False,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer for intimacy inference model using NLL."""

    if is_lm:
        # LM model - vectorize over scenario_idx
        def get_nll(alpha_observer, a, r, s, resp):
            posterior = get_intimacy_posterior_lm(observer_fn, actor_params, alpha_observer, a, r, s)
            return compute_intimacy_nll(posterior, resp)

        vmap_get_nll = jax.vmap(
            lambda alpha_obs, a, r, s, resp: get_nll(alpha_obs, a, r, s, resp),
            in_axes=(None, 0, 0, 0, 0),
        )

        def loss_fn(params):
            alpha_observer = params[0]
            nlls = vmap_get_nll(alpha_observer, action, reward_condition, scenario_idx, response)
            return jnp.sum(nlls)
    else:
        # Stipulated model - no scenario_idx
        def get_nll(alpha_observer, a, r, resp):
            posterior = get_intimacy_posterior_stipulated(observer_fn, actor_params, alpha_observer, a, r)
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
                    print(f"  This typically means the model's likelihood doesn't depend on the latent variable")
                    print(f"  (e.g., vanilla model for intimacy inference). Returning alpha_observer=1.0.")
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
    scenario_idx: jnp.ndarray = None,
    is_lm: bool = False,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer for reward inference model using NLL (binary cross-entropy)."""

    if is_lm:
        # LM model - vectorize over scenario_idx
        def get_nll(alpha_observer, a, i, s, resp):
            p_high = get_reward_p_high_lm(observer_fn, actor_params, alpha_observer, a, i, s)
            return compute_reward_nll(p_high, resp)

        vmap_get_nll = jax.vmap(
            lambda alpha_obs, a, i, s, resp: get_nll(alpha_obs, a, i, s, resp),
            in_axes=(None, 0, 0, 0, 0),
        )

        def loss_fn(params):
            alpha_observer = params[0]
            nlls = vmap_get_nll(alpha_observer, action, intimacy_condition, scenario_idx, response)
            return jnp.sum(nlls)
    else:
        # Stipulated model - no scenario_idx
        def get_nll(alpha_observer, a, i, resp):
            p_high = get_reward_p_high_stipulated(observer_fn, actor_params, alpha_observer, a, i)
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
                    print(f"  This typically means the model's likelihood doesn't depend on the latent variable.")
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
# Fitting Functions for Modified Models (joint alpha_observer and delta)
# ==============================================================================


def fit_intimacy_alpha_observer_and_delta(
    observer_fn,
    actor_params: dict,
    action: jnp.ndarray,
    reward_condition: jnp.ndarray,
    response: jnp.ndarray,
    scenario_idx: jnp.ndarray = None,
    is_lm: bool = False,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer and delta for modified intimacy inference model.

    delta controls how much observers think actors' reward scales with intimacy.
    delta=0: no intimacy scaling on reward (vanilla-like)
    delta=1: pre-registered model (full intimacy scaling)
    """

    if is_lm:
        def get_nll(alpha_observer, delta, a, r, s, resp):
            posterior = get_intimacy_posterior_lm_modified(observer_fn, actor_params, alpha_observer, delta, a, r, s)
            return compute_intimacy_nll(posterior, resp)

        vmap_get_nll = jax.vmap(
            lambda alpha_obs, delta, a, r, s, resp: get_nll(alpha_obs, delta, a, r, s, resp),
            in_axes=(None, None, 0, 0, 0, 0),
        )

        def loss_fn(params):
            alpha_observer, delta = params[0], params[1]
            nlls = vmap_get_nll(alpha_observer, delta, action, reward_condition, scenario_idx, response)
            return jnp.sum(nlls)
    else:
        def get_nll(alpha_observer, delta, a, r, resp):
            posterior = get_intimacy_posterior_stipulated_modified(observer_fn, actor_params, alpha_observer, delta, a, r)
            return compute_intimacy_nll(posterior, resp)

        vmap_get_nll = jax.vmap(
            lambda alpha_obs, delta, a, r, resp: get_nll(alpha_obs, delta, a, r, resp),
            in_axes=(None, None, 0, 0, 0),
        )

        def loss_fn(params):
            alpha_observer, delta = params[0], params[1]
            nlls = vmap_get_nll(alpha_observer, delta, action, reward_condition, response)
            return jnp.sum(nlls)

    # Initialize: alpha_observer = 1.0, delta = 0.5 (start with partial scaling)
    params = jnp.array([1.0, 0.5])
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_loss = None
    for step in range(max_steps):
        loss, grad = grad_fn(params)

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        # Keep alpha_observer positive, delta in [0, 1] range
        params = jnp.array([
            jnp.clip(params[0], 0.01, 10.0),
            jnp.clip(params[1], 0.0, 1.0),
        ])

        if verbose and step % 200 == 0:
            print(f"  Step {step}, NLL: {loss:.4f}, alpha_observer: {params[0]:.4f}, delta: {params[1]:.4f}")

        if prev_loss is not None and loss > prev_loss + 1e-4:
            if verbose:
                print(f"  Loss increased at step {step}, stopping")
            break
        prev_loss = loss

    best_nll = float(loss_fn(params))
    final_alpha = float(params[0])
    final_delta = float(params[1])

    if verbose:
        print(f"  Final NLL: {best_nll:.4f}")
        print(f"  Final alpha_observer: {final_alpha:.4f}, delta: {final_delta:.4f}")

    return final_alpha, final_delta, best_nll


def fit_reward_alpha_observer_and_delta(
    observer_fn,
    actor_params: dict,
    action: jnp.ndarray,
    intimacy_condition: jnp.ndarray,
    response: jnp.ndarray,
    scenario_idx: jnp.ndarray = None,
    is_lm: bool = False,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer and delta for modified reward inference model.

    delta controls how much observers think actors' reward scales with intimacy.
    delta=0: no intimacy scaling on reward (vanilla-like)
    delta=1: pre-registered model (full intimacy scaling)
    """

    if is_lm:
        def get_nll(alpha_observer, delta, a, i, s, resp):
            p_high = get_reward_p_high_lm_modified(observer_fn, actor_params, alpha_observer, delta, a, i, s)
            return compute_reward_nll(p_high, resp)

        vmap_get_nll = jax.vmap(
            lambda alpha_obs, delta, a, i, s, resp: get_nll(alpha_obs, delta, a, i, s, resp),
            in_axes=(None, None, 0, 0, 0, 0),
        )

        def loss_fn(params):
            alpha_observer, delta = params[0], params[1]
            nlls = vmap_get_nll(alpha_observer, delta, action, intimacy_condition, scenario_idx, response)
            return jnp.sum(nlls)
    else:
        def get_nll(alpha_observer, delta, a, i, resp):
            p_high = get_reward_p_high_stipulated_modified(observer_fn, actor_params, alpha_observer, delta, a, i)
            return compute_reward_nll(p_high, resp)

        vmap_get_nll = jax.vmap(
            lambda alpha_obs, delta, a, i, resp: get_nll(alpha_obs, delta, a, i, resp),
            in_axes=(None, None, 0, 0, 0),
        )

        def loss_fn(params):
            alpha_observer, delta = params[0], params[1]
            nlls = vmap_get_nll(alpha_observer, delta, action, intimacy_condition, response)
            return jnp.sum(nlls)

    # Initialize: alpha_observer = 1.0, delta = 0.5 (start with partial scaling)
    params = jnp.array([1.0, 0.5])
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_loss = None
    for step in range(max_steps):
        loss, grad = grad_fn(params)

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        # Keep alpha_observer positive, delta in [0, 1] range
        params = jnp.array([
            jnp.clip(params[0], 0.01, 10.0),
            jnp.clip(params[1], 0.0, 1.0),
        ])

        if verbose and step % 200 == 0:
            print(f"  Step {step}, NLL: {loss:.4f}, alpha_observer: {params[0]:.4f}, delta: {params[1]:.4f}")

        if prev_loss is not None and loss > prev_loss + 1e-4:
            if verbose:
                print(f"  Loss increased at step {step}, stopping")
            break
        prev_loss = loss

    best_nll = float(loss_fn(params))
    final_alpha = float(params[0])
    final_delta = float(params[1])

    if verbose:
        print(f"  Final NLL: {best_nll:.4f}")
        print(f"  Final alpha_observer: {final_alpha:.4f}, delta: {final_delta:.4f}")

    return final_alpha, final_delta, best_nll


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
        print(f"  {model_name}: alpha={p['alpha']:.3f}, w_r={p['w_r']:.3f}, w_d={p['w_d']:.3f}, w_c={p['w_c']:.3f}")

    results = []

    # -------------------------------------------------------------------------
    # Fit Intimacy Inference Models
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INTIMACY INFERENCE")
    print("=" * 60)

    int_data, int_action, int_reward_condition, int_response, int_scenario_idx = load_intimacy_data()

    # Pre-registered models (fit alpha_observer only)
    intimacy_models = {
        "full": (observer_intimacy_full_model, "full", False),
        "vanilla": (observer_intimacy_vanilla_inv_plan, "vanilla", False),
        "discomfort_only": (observer_intimacy_discomfort_only, "discomfort_only", False),
        "full_lm": (observer_intimacy_full_model_lm, "full_lm", True),
        "vanilla_lm": (observer_intimacy_vanilla_lm, "vanilla_lm", True),
        "discomfort_only_lm": (observer_intimacy_discomfort_only_lm, "discomfort_only_lm", True),
    }

    # Modified models (fit alpha_observer and beta)
    intimacy_models_modified = {
        "full_modified": (observer_intimacy_full_model_modified, "full", False),
        "full_lm_modified": (observer_intimacy_full_model_lm_modified, "full_lm", True),
    }

    # Fit pre-registered models (alpha_observer only)
    for model_name, (observer_fn, actor_param_key, is_lm) in intimacy_models.items():
        print(f"\n{'-' * 40}")
        print(f"Fitting {model_name}...")
        print(f"{'-' * 40}")

        alpha_observer, nll = fit_intimacy_alpha_observer(
            observer_fn=observer_fn,
            actor_params=actor_params[actor_param_key],
            action=int_action,
            reward_condition=int_reward_condition,
            response=int_response,
            scenario_idx=int_scenario_idx if is_lm else None,
            is_lm=is_lm,
        )

        results.append({
            "model": model_name,
            "experiment": "intimacy",
            "alpha_observer": alpha_observer,
            "delta": np.nan,
            "nll": nll,
            "n_params": 1,
        })

    # Fit modified models (alpha_observer and delta)
    for model_name, (observer_fn, actor_param_key, is_lm) in intimacy_models_modified.items():
        print(f"\n{'-' * 40}")
        print(f"Fitting {model_name} (with delta)...")
        print(f"{'-' * 40}")

        alpha_observer, delta, nll = fit_intimacy_alpha_observer_and_delta(
            observer_fn=observer_fn,
            actor_params=actor_params[actor_param_key],
            action=int_action,
            reward_condition=int_reward_condition,
            response=int_response,
            scenario_idx=int_scenario_idx if is_lm else None,
            is_lm=is_lm,
        )

        results.append({
            "model": model_name,
            "experiment": "intimacy",
            "alpha_observer": alpha_observer,
            "delta": delta,
            "nll": nll,
            "n_params": 2,
        })

    # -------------------------------------------------------------------------
    # Fit Reward Inference Models
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("REWARD INFERENCE")
    print("=" * 60)

    rew_data, rew_action, rew_intimacy_condition, rew_response, rew_scenario_idx = load_reward_data()

    # Pre-registered models (fit alpha_observer only)
    reward_models = {
        "full": (observer_reward_full_model, "full", False),
        "vanilla": (observer_reward_vanilla_inv_plan, "vanilla", False),
        "discomfort_only": (observer_reward_discomfort_only, "discomfort_only", False),
        "full_lm": (observer_reward_full_model_lm, "full_lm", True),
        "vanilla_lm": (observer_reward_vanilla_lm, "vanilla_lm", True),
        "discomfort_only_lm": (observer_reward_discomfort_only_lm, "discomfort_only_lm", True),
    }

    # Modified models (fit alpha_observer and delta)
    reward_models_modified = {
        "full_modified": (observer_reward_full_model_modified, "full", False),
        "full_lm_modified": (observer_reward_full_model_lm_modified, "full_lm", True),
    }

    # Fit pre-registered models (alpha_observer only)
    for model_name, (observer_fn, actor_param_key, is_lm) in reward_models.items():
        print(f"\n{'-' * 40}")
        print(f"Fitting {model_name}...")
        print(f"{'-' * 40}")

        alpha_observer, nll = fit_reward_alpha_observer(
            observer_fn=observer_fn,
            actor_params=actor_params[actor_param_key],
            action=rew_action,
            intimacy_condition=rew_intimacy_condition,
            response=rew_response,
            scenario_idx=rew_scenario_idx if is_lm else None,
            is_lm=is_lm,
        )

        results.append({
            "model": model_name,
            "experiment": "reward",
            "alpha_observer": alpha_observer,
            "delta": np.nan,
            "nll": nll,
            "n_params": 1,
        })

    # Fit modified models (alpha_observer and delta)
    for model_name, (observer_fn, actor_param_key, is_lm) in reward_models_modified.items():
        print(f"\n{'-' * 40}")
        print(f"Fitting {model_name} (with delta)...")
        print(f"{'-' * 40}")

        alpha_observer, delta, nll = fit_reward_alpha_observer_and_delta(
            observer_fn=observer_fn,
            actor_params=actor_params[actor_param_key],
            action=rew_action,
            intimacy_condition=rew_intimacy_condition,
            response=rew_response,
            scenario_idx=rew_scenario_idx if is_lm else None,
            is_lm=is_lm,
        )

        results.append({
            "model": model_name,
            "experiment": "reward",
            "alpha_observer": alpha_observer,
            "delta": delta,
            "nll": nll,
            "n_params": 2,
        })

    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Save
    results_path = "inverse_planning_fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    main()
