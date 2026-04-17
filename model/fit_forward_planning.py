"""
Fit forward planning models to human data.

Three access-model ablations are fit (scenario-specific LLM-parameterized):
1. access_full: full utility with food reward, positive/negative access, and effort
2. access_only: only the two access terms (w_r*access*I - w_d*access*(1-I))
3. no_access:   base model (w_v*V - w_e*effort)

Uses maximum likelihood estimation with gradient descent (optax.adam).
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
    actor_forw_access_full,
    actor_forw_access_only,
    actor_forw_no_access,
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

    intimacy_map = {0: 0.0, 50: 0.5, 75: 0.75, 100: 1.0}
    data["intimacy_scaled"] = data["intimacy"].map(intimacy_map)

    motivation_map = {"low": 0, "high": 1}
    data["reward_condition"] = data["motivation"].map(motivation_map)

    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

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
    """Compute negative log-likelihood. NLL = -sum(responses * log(preds))."""
    epsilon = 1e-8
    preds_safe = jnp.clip(preds, epsilon, 1.0)
    responses_safe = jnp.clip(responses, epsilon, 1.0)
    return -jnp.sum(responses_safe * jnp.log(preds_safe))


# Model comparison metrics


def compute_aic(nll, n_params):
    """AIC = 2k + 2*NLL."""
    return 2 * n_params + 2 * nll


def compute_bic(nll, n_params, n_obs):
    """BIC = k*ln(n) + 2*NLL."""
    return n_params * np.log(n_obs) + 2 * nll


def compute_pearson_r_by_condition(data, pred_col, human_col, group_cols, n_boot=1000):
    """Compute Pearson r at condition x action level with bootstrap CI.

    Per preregistration: correlation computed at condition x action level
    with 95% bootstrapped confidence intervals.
    """
    agg = (
        data.groupby(group_cols)
        .agg({pred_col: "mean", human_col: "mean"})
        .reset_index()
    )

    r, p = stats.pearsonr(agg[pred_col], agg[human_col])

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


# Memo models return tables of shape (actions, scenarios, intimacy, reward_condition);
# we index per-trial with scenario_idx from the dataset.


@jax.jit
def predict_access_full(
    intimacy, reward_condition, action, scenario_idx,
    alpha, w_v, w_r, w_d, w_e,
    access_table, effort_table, reward_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_access_full(
        alpha, w_v, w_r, w_d, w_e, access_table, effort_table, reward_table
    )
    return jax.vmap(lambda i, r, a, s: probs[a, s, i, r])(
        intimacy_idx, reward_condition, action, scenario_idx
    )


@jax.jit
def predict_access_only(
    intimacy, reward_condition, action, scenario_idx,
    alpha, w_r, w_d,
    access_table, effort_table, reward_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_access_only(
        alpha, w_r, w_d, access_table, effort_table, reward_table
    )
    return jax.vmap(lambda i, r, a, s: probs[a, s, i, r])(
        intimacy_idx, reward_condition, action, scenario_idx
    )


@jax.jit
def predict_no_access(
    intimacy, reward_condition, action, scenario_idx,
    alpha, w_v, w_e,
    access_table, effort_table, reward_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_no_access(
        alpha, w_v, w_e, access_table, effort_table, reward_table
    )
    return jax.vmap(lambda i, r, a, s: probs[a, s, i, r])(
        intimacy_idx, reward_condition, action, scenario_idx
    )


# Model fitting (alpha=1 fixed for identifiability; see fit_access_*_model docstrings)


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


def fit_access_full_model(
    intimacy, reward_condition, action, scenario_idx, p_action, tables, **kwargs
):
    ALPHA = 1.0
    a_tab, e_tab, r_tab = tables

    def loss_fn(params):
        w_v, w_r, w_d, w_e = params
        preds = predict_access_full(
            intimacy, reward_condition, action, scenario_idx,
            ALPHA, w_v, w_r, w_d, w_e, a_tab, e_tab, r_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0, 1.0, 1.0], label="access_full", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1], params[2], params[3]]), nll


def fit_access_only_model(
    intimacy, reward_condition, action, scenario_idx, p_action, tables, **kwargs
):
    ALPHA = 1.0
    a_tab, e_tab, r_tab = tables

    def loss_fn(params):
        w_r, w_d = params
        preds = predict_access_only(
            intimacy, reward_condition, action, scenario_idx,
            ALPHA, w_r, w_d, a_tab, e_tab, r_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0], label="access_only", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1]]), nll


def fit_no_access_model(
    intimacy, reward_condition, action, scenario_idx, p_action, tables, **kwargs
):
    ALPHA = 1.0
    a_tab, e_tab, r_tab = tables

    def loss_fn(params):
        w_v, w_e = params
        preds = predict_no_access(
            intimacy, reward_condition, action, scenario_idx,
            ALPHA, w_v, w_e, a_tab, e_tab, r_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0], label="no_access", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1]]), nll


# Main script


def main():
    print("=" * 60)
    print("Forward Planning Model Fitting")
    print("=" * 60)

    data, intimacy, reward_condition, action, p_action, scenario_idx = load_data()

    tables = (LLM_TABLES["access"], LLM_TABLES["effort"], LLM_TABLES["reward"])
    fits = {
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

    results = {}
    param_arrays = {}
    for name, (fit_fn, _pred_fn, param_names) in fits.items():
        print("\n" + "-" * 40)
        print(f"Fitting {name.upper()} model (alpha=1 fixed)...")
        print("-" * 40)
        params, nll = fit_fn(
            intimacy, reward_condition, action, scenario_idx, p_action, tables
        )
        param_arrays[name] = params
        results[name] = {
            "params": {
                "alpha": float(params[0]),
                **{pn: float(params[i + 1]) for i, pn in enumerate(param_names)},
            },
            "nll": nll,
            "n_params": len(param_names),
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

    for name, (_fit_fn, pred_fn, _param_names) in fits.items():
        params = param_arrays[name]
        data[f"pred_{name}"] = np.array(
            pred_fn(
                intimacy, reward_condition, action, scenario_idx,
                *params, *tables,
            )
        )

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

    for model_name in fits.keys():
        nll = results[model_name]["nll"]
        n_params = results[model_name]["n_params"]

        aic = compute_aic(nll, n_params)
        bic = compute_bic(nll, n_params, n_obs)

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
            f"  {model_name}: AIC={aic:.2f}, BIC={bic:.2f}, "
            f"r={r_result['r']:.3f} [{r_result['ci_lower']:.3f}, {r_result['ci_upper']:.3f}]"
        )

    # Save results summary
    results_rows = []
    for model_name in fits.keys():
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

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
