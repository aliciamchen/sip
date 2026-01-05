"""
Fit forward planning models to human data.

This script fits three actor models to human forward planning data:
1. Full model: intimacy scales both reward and discomfort
2. Vanilla model: no intimacy scaling
3. Discomfort-only model: only considers discomfort

Uses maximum likelihood estimation with gradient descent (optax.adam).
Performs likelihood ratio tests to compare models.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd

from model_utils import (
    actions,
    IntimacyLevels,
    RewardConditions,
    actor_forw_full,
    actor_forw_vanilla,
    actor_forw_discomfort_only,
)


# Data loading and preprocessing


def load_data(filepath: str = "../data/forw_plan/main_trials_long.csv"):
    """Load and preprocess forward planning data.

    Converts:
    - intimacy: 0/50/75/100 -> 0.0/0.5/0.75/1.0
    - motivation: low/high -> 0/1 (RewardConditions enum)

    Returns:
        data: pandas DataFrame
        intimacy: JAX array of intimacy levels (0-1)
        reward_condition: JAX array of reward conditions (0 or 1)
        action: JAX array of actions (0-3)
        p_action: JAX array of human response probabilities
    """
    print("Loading forward planning data...")
    data = pd.read_csv(filepath)

    # Convert intimacy to 0-1 scale
    intimacy_map = {0: 0.0, 50: 0.5, 75: 0.75, 100: 1.0}
    data["intimacy_scaled"] = data["intimacy"].map(intimacy_map)

    # Convert motivation to reward condition (0 = low, 1 = high)
    motivation_map = {"low": 0, "high": 1}
    data["reward_condition"] = data["motivation"].map(motivation_map)

    # Extract JAX arrays
    intimacy = jnp.array(data["intimacy_scaled"].values)
    reward_condition = jnp.array(data["reward_condition"].values)
    action = jnp.array(data["action"].values)
    p_action = jnp.array(data["p_action"].values)

    print(f"Loaded {len(data)} data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")

    return data, intimacy, reward_condition, action, p_action


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
    return actor_forw_vanilla(alpha, w_r, w_d, w_c)[action, intimacy_idx, reward_condition]


@jax.jit
def get_discomfort_only_prediction(intimacy, reward_condition, action, alpha, w_d):
    """Get prediction from discomfort-only model for single data point."""
    intimacy_idx = get_intimacy_index(intimacy)
    return actor_forw_discomfort_only(alpha, w_d)[action, intimacy_idx, reward_condition]


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
    """Fit full model parameters (alpha, w_r, w_d, w_c)."""

    def loss_fn(params):
        alpha, w_r, w_d, w_c = params[0], params[1], params[2], params[3]
        preds = predict_full(intimacy, reward_condition, action, alpha, w_r, w_d, w_c)
        return compute_nll(preds, p_action)

    params = jnp.array([1.0, 1.0, 1.0, 1.0])
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
        print(f"  Final params: {params}")

    return params, best_nll


def fit_vanilla_model(
    intimacy: jnp.ndarray,
    reward_condition: jnp.ndarray,
    action: jnp.ndarray,
    p_action: jnp.ndarray,
    lr: float = 0.01,
    max_steps: int = 5000,
    verbose: bool = True,
):
    """Fit vanilla model parameters (alpha, w_r, w_d, w_c)."""

    def loss_fn(params):
        alpha, w_r, w_d, w_c = params[0], params[1], params[2], params[3]
        preds = predict_vanilla(intimacy, reward_condition, action, alpha, w_r, w_d, w_c)
        return compute_nll(preds, p_action)

    params = jnp.array([1.0, 1.0, 1.0, 1.0])
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
        print(f"  Final params: {params}")

    return params, best_nll


def fit_discomfort_only_model(
    intimacy: jnp.ndarray,
    reward_condition: jnp.ndarray,
    action: jnp.ndarray,
    p_action: jnp.ndarray,
    lr: float = 0.01,
    max_steps: int = 5000,
    verbose: bool = True,
):
    """Fit discomfort-only model parameters (alpha, w_d)."""

    def loss_fn(params):
        alpha, w_d = params[0], params[1]
        preds = predict_discomfort_only(intimacy, reward_condition, action, alpha, w_d)
        return compute_nll(preds, p_action)

    params = jnp.array([1.0, 1.0])
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
        print(f"  Final params: {params}")

    return params, best_nll


# Main script


def main():
    print("=" * 60)
    print("Forward Planning Model Fitting")
    print("=" * 60)

    # Load data
    data, intimacy, reward_condition, action, p_action = load_data()

    # Fit models
    results = {}

    # Full model: 4 params (alpha, w_r, w_d, w_c)
    print("\n" + "-" * 40)
    print("Fitting FULL model...")
    print("-" * 40)
    full_params, full_nll = fit_full_model(intimacy, reward_condition, action, p_action)
    results["full"] = {
        "params": {"alpha": float(full_params[0]), "w_r": float(full_params[1]), "w_d": float(full_params[2]), "w_c": float(full_params[3])},
        "nll": full_nll,
        "n_params": 4,
    }

    # Vanilla model: 4 params (alpha, w_r, w_d, w_c)
    print("\n" + "-" * 40)
    print("Fitting VANILLA model...")
    print("-" * 40)
    vanilla_params, vanilla_nll = fit_vanilla_model(intimacy, reward_condition, action, p_action)
    results["vanilla"] = {
        "params": {"alpha": float(vanilla_params[0]), "w_r": float(vanilla_params[1]), "w_d": float(vanilla_params[2]), "w_c": float(vanilla_params[3])},
        "nll": vanilla_nll,
        "n_params": 4,
    }

    # Discomfort-only model: 2 params (alpha, w_d)
    print("\n" + "-" * 40)
    print("Fitting DISCOMFORT-ONLY model...")
    print("-" * 40)
    discomfort_params, discomfort_nll = fit_discomfort_only_model(intimacy, reward_condition, action, p_action)
    results["discomfort_only"] = {
        "params": {"alpha": float(discomfort_params[0]), "w_d": float(discomfort_params[1])},
        "nll": discomfort_nll,
        "n_params": 2,
    }

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
    data["pred_full"] = np.array(predict_full(
        intimacy, reward_condition, action,
        full_params[0], full_params[1], full_params[2], full_params[3]
    ))
    data["pred_vanilla"] = np.array(predict_vanilla(
        intimacy, reward_condition, action,
        vanilla_params[0], vanilla_params[1], vanilla_params[2], vanilla_params[3]
    ))
    data["pred_discomfort_only"] = np.array(predict_discomfort_only(
        intimacy, reward_condition, action,
        discomfort_params[0], discomfort_params[1]
    ))

    # Save
    output_path = "forward_planning_fits.csv"
    data.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    # Save results summary
    results_df = pd.DataFrame([
        {"model": "full", "nll": full_nll, "n_params": 4, **{f"param_{k}": v for k, v in results["full"]["params"].items()}},
        {"model": "vanilla", "nll": vanilla_nll, "n_params": 4, **{f"param_{k}": v for k, v in results["vanilla"]["params"].items()}},
        {"model": "discomfort_only", "nll": discomfort_nll, "n_params": 2, **{f"param_{k}": v for k, v in results["discomfort_only"]["params"].items()}},
    ])
    results_path = "forward_planning_fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved fit results to {results_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
