"""
Fit forward planning models to the effort-manipulation experiment (food_forw_intimacy_effort).

Parallel to model/fit_forward_planning.py, adapted for:
  - 2 actions per scenario (non-saliva-share vs saliva-share; CSV action=1/2 -> 0/1).
  - An effort_condition covariate (low, high) carried by the vignette text.
  - Reward fixed at HIGH — V(a|s) = 1 for both actions is stipulated in
    model_utils_effort.get_stipulated_reward_effort. w_v is retained in the
    utility for ablation-parallelism with the canonical pipeline but is
    non-identified in the softmax and will stay near its initialization.

Three access-model ablations: full, discomfort_only, base. α = 1
fixed (matches canonical for identifiability).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_forward_planning import (
    _fit_with_adam,
    compute_aic,
    compute_bic,
    compute_nll,
    compute_pearson_r_by_condition,
)
from model_utils_effort import (
    EFFORT_CONDITION_TO_IDX,
    LLM_TABLES_EFFORT,
    actor_forw_effort_full,
    actor_forw_effort_discomfort_only,
    actor_forw_effort_base,
)
from model_utils import SCENARIO_TO_IDX

from utils import get_project_root


# Data loading and preprocessing

INTIMACY_MAP = {0: 0.0, 50: 0.5, 75: 0.75, 100: 1.0}


def load_data(filepath: str = None):
    """Load and preprocess food_forw_intimacy_effort data.

    Converts:
      - intimacy 0/50/75/100 -> 0.0/0.5/0.75/1.0
      - effort low/high -> 0/1
      - action 1/2 -> 0/1 (internal index: 0 = non-share, 1 = saliva-share)
      - scenario_label -> idx (alphabetical 0-15, shared with canonical pipeline)
    """
    if filepath is None:
        filepath = get_project_root() / "data" / "food_forw_intimacy_effort" / "main_trials_long.csv"
    print("Loading food_forw_intimacy_effort data...")
    data = pd.read_csv(filepath)

    data["intimacy_scaled"] = data["intimacy"].map(INTIMACY_MAP)
    data["effort_condition"] = data["effort"].map(EFFORT_CONDITION_TO_IDX)
    data["action_idx"] = data["action"].astype(int) - 1
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    intimacy = jnp.array(data["intimacy_scaled"].values)
    effort_condition = jnp.array(data["effort_condition"].values)
    action = jnp.array(data["action_idx"].values)
    p_action = jnp.array(data["p_action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)

    print(f"Loaded {len(data)} data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")

    return data, intimacy, effort_condition, action, p_action, scenario_idx


# Vectorized prediction functions
# Memo models return tables of shape (actions, scenarios, intimacy, effort_condition);
# index per-trial by scenario / intimacy / effort_condition / action.


def get_intimacy_index(intimacy_value):
    return jnp.round(intimacy_value * 100).astype(int)


@jax.jit
def predict_full(
    intimacy, effort_condition, action, scenario_idx,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_effort_full(
        alpha, w_v, w_d, w_e, gamma, access_table, effort_table,
    )
    return jax.vmap(lambda i, e, a, s: probs[a, s, i, e])(
        intimacy_idx, effort_condition, action, scenario_idx
    )


@jax.jit
def predict_discomfort_only(
    intimacy, effort_condition, action, scenario_idx,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_effort_discomfort_only(
        alpha, w_d, gamma, access_table, effort_table,
    )
    return jax.vmap(lambda i, e, a, s: probs[a, s, i, e])(
        intimacy_idx, effort_condition, action, scenario_idx
    )


@jax.jit
def predict_base(
    intimacy, effort_condition, action, scenario_idx,
    alpha, w_v, w_e,
    access_table, effort_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_effort_base(
        alpha, w_v, w_e, access_table, effort_table,
    )
    return jax.vmap(lambda i, e, a, s: probs[a, s, i, e])(
        intimacy_idx, effort_condition, action, scenario_idx
    )


# Model fitting (alpha=1 fixed for identifiability)


def fit_full_model(
    intimacy, effort_condition, action, scenario_idx, p_action, tables, **kwargs
):
    ALPHA = 1.0
    a_tab, e_tab = tables

    def loss_fn(params):
        w_v, w_d, w_e, gamma = params
        preds = predict_full(
            intimacy, effort_condition, action, scenario_idx,
            ALPHA, w_v, w_d, w_e, gamma, a_tab, e_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0, 1.0, 1.0], label="full", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1], params[2], params[3]]), nll


def fit_discomfort_only_model(
    intimacy, effort_condition, action, scenario_idx, p_action, tables, **kwargs
):
    ALPHA = 1.0
    a_tab, e_tab = tables

    def loss_fn(params):
        w_d, gamma = params
        preds = predict_discomfort_only(
            intimacy, effort_condition, action, scenario_idx,
            ALPHA, w_d, gamma, a_tab, e_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0], label="discomfort_only", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1]]), nll


def fit_base_model(
    intimacy, effort_condition, action, scenario_idx, p_action, tables, **kwargs
):
    ALPHA = 1.0
    a_tab, e_tab = tables

    def loss_fn(params):
        w_v, w_e = params
        preds = predict_base(
            intimacy, effort_condition, action, scenario_idx,
            ALPHA, w_v, w_e, a_tab, e_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0], label="base", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1]]), nll


# Main script


def main():
    print("=" * 60)
    print("Forward Planning Model Fitting — effort experiment")
    print("=" * 60)

    data, intimacy, effort_condition, action, p_action, scenario_idx = load_data()

    tables = (LLM_TABLES_EFFORT["access"], LLM_TABLES_EFFORT["effort"])

    fits = {
        "full": (
            fit_full_model, predict_full, ["w_v", "w_d", "w_e", "gamma"], tables,
        ),
        "discomfort_only": (
            fit_discomfort_only_model, predict_discomfort_only, ["w_d", "gamma"], tables,
        ),
        "base": (
            fit_base_model, predict_base, ["w_v", "w_e"], tables,
        ),
    }

    results = {}
    param_arrays = {}
    for name, (fit_fn, _pred_fn, param_names, tab) in fits.items():
        print("\n" + "-" * 40)
        print(f"Fitting {name.upper()} model (alpha=1 fixed)...")
        print("-" * 40)
        params, nll = fit_fn(
            intimacy, effort_condition, action, scenario_idx, p_action, tab
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

    # Per-trial predictions
    print("\n" + "-" * 40)
    print("Saving predictions...")
    print("-" * 40)

    for name, (_fit_fn, pred_fn, _param_names, tab) in fits.items():
        params = param_arrays[name]
        data[f"pred_{name}"] = np.array(
            pred_fn(
                intimacy, effort_condition, action, scenario_idx,
                *params, *tab,
            )
        )

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "forward_planning_effort_fits.csv"
    data.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    # Model comparison
    print("\n" + "-" * 40)
    print("Computing model comparison metrics...")
    print("-" * 40)

    n_obs = len(data)
    group_cols = ["intimacy", "effort", "action"]
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

    # Results summary
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
    results_path = output_dir / "forward_planning_effort_fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
