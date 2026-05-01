"""
Fit observer alpha parameter to inverse planning human data.

Fits alpha_observer for each access-model ablation (full, discomfort_only,
base), with all actor parameters frozen from forward planning fits.

For intimacy inference: NLL = -log(P(intimacy = response/100 | action, reward))
For reward inference: binary cross-entropy between response/100 and P(high reward)

Observer memo models take scenario-specific LLM-parameterized tables; per-trial
NLL indexes the 4D output (action, scenario, intimacy_or_relationship, reward_condition).
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
    load_lm_v,
    observer_intimacy_full,
    observer_intimacy_discomfort_only,
    observer_intimacy_base,
    observer_reward_full,
    observer_reward_discomfort_only,
    observer_reward_base,
)

from utils import get_project_root

# ==============================================================================
# Data Loading
# ==============================================================================


def load_fitted_params(filepath: str = None) -> dict:
    """Load frozen actor parameters from forward planning fit results.

    Returns a dict: model_name -> dict of every `param_*` column present in that
    row (stripped of the `param_` prefix). Missing/NaN columns are omitted, so
    each model keeps only its own parameter set.
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
        params[model_name] = model_params
    return params


def load_intimacy_data(filepath: str = None):
    """Load and preprocess intimacy inference data (posterior stage only)."""
    if filepath is None:
        filepath = (
            get_project_root() / "data" / "inv_plan_intimacy_alt" / "main_trials_long.csv"
        )
    print("Loading intimacy inference data...")
    data = pd.read_csv(filepath)

    data = data[data["stage"] == "posterior"].copy()
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int)

    motivation_map = {"low": 0, "high": 1}
    data["reward_condition"] = data["motivation"].map(motivation_map)

    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    action = jnp.array(data["action"].values)
    reward_condition = jnp.array(data["reward_condition"].values)
    response = jnp.array(data["intimacy_rating"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)

    print(f"Loaded {len(data)} posterior data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")

    return data, action, reward_condition, response, scenario_idx


def load_reward_data(filepath: str = None):
    """Load and preprocess reward inference data (posterior stage only)."""
    if filepath is None:
        filepath = (
            get_project_root() / "data" / "inv_plan_desire_alt" / "main_trials_long.csv"
        )
    print("Loading reward inference data...")
    data = pd.read_csv(filepath)

    data = data[data["stage"] == "posterior"].copy()
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int)

    intimacy_map = {0: 0, 50: 1, 75: 2, 100: 3}
    data["intimacy_idx"] = data["intimacy"].map(intimacy_map)

    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

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
    """NLL = -log(P(intimacy = response/100)).

    posterior: shape (101,) over intimacy levels 0-100.
    response: integer 0-100.
    """
    epsilon = 1e-8
    response_idx = jnp.clip(jnp.round(response).astype(int), 0, 100)
    prob = posterior[response_idx]
    return -jnp.log(jnp.clip(prob, epsilon, 1.0))


@jax.jit
def compute_reward_nll(p_high, response):
    """Binary cross-entropy NLL for reward inference.

    response is 0-100 (interpreted as P(high)*100).
    """
    epsilon = 1e-8
    p_human = response / 100.0
    p_model = jnp.clip(p_high, epsilon, 1.0 - epsilon)
    return -(p_human * jnp.log(p_model) + (1 - p_human) * jnp.log(1 - p_model))


# ==============================================================================
# Observer Fitting
# ==============================================================================
# Observer memo models take (actor kwargs, alpha_observer, access_table,
# effort_table) and return 4D tables:
# (action, scenario, intimacy_or_relationship, reward_condition).


def _fit_alpha_observer(
    observer_fn,
    actor_params: dict,
    actor_kwarg_names,
    action: jnp.ndarray,
    scenario_idx: jnp.ndarray,
    conditioning: jnp.ndarray,
    response: jnp.ndarray,
    nll_fn,
    posterior_slicer,
    table_kwargs: dict,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer by dict-keyed actor params.

    posterior_slicer: (table, action_i, scenario_i, cond_i) -> slice to pass to nll_fn.
    table_kwargs: dict of table-argument kwargs threaded into observer_fn
      (e.g. {"access_table": ..., "effort_table": ...}).
    """
    actor_kwargs = {k: actor_params[k] for k in actor_kwarg_names}

    def observer_table(alpha_observer):
        return observer_fn(
            **actor_kwargs, alpha_observer=alpha_observer,
            **table_kwargs,
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


def fit_intimacy_observer(
    observer_fn, actor_params, actor_kwarg_names,
    action, scenario_idx, reward_condition, response, table_kwargs, **kwargs,
):
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=reward_condition,
        response=response,
        nll_fn=compute_intimacy_nll,
        posterior_slicer=lambda tab, a, s, r: tab[a, s, :, r],
        table_kwargs=table_kwargs,
        **kwargs,
    )


def fit_reward_observer(
    observer_fn, actor_params, actor_kwarg_names,
    action, scenario_idx, intimacy_condition, response, table_kwargs, **kwargs,
):
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=intimacy_condition,
        response=response,
        nll_fn=compute_reward_nll,
        posterior_slicer=lambda tab, a, s, i: tab[a, s, i, 1],
        table_kwargs=table_kwargs,
        **kwargs,
    )


# Variant registry: name -> (intimacy_observer, reward_observer, actor_kwarg_names,
# uses_v). discomfort_only is V-independent and doesn't take v_table.
ACCESS_VARIANTS = {
    "full": (
        observer_intimacy_full,
        observer_reward_full,
        ["alpha", "w_v", "w_d", "w_e", "gamma"],
        True,
    ),
    "discomfort_only": (
        observer_intimacy_discomfort_only,
        observer_reward_discomfort_only,
        ["alpha", "w_d", "gamma"],
        False,
    ),
    "base": (
        observer_intimacy_base,
        observer_reward_base,
        ["alpha", "w_v", "w_e"],
        True,
    ),
}


def _table_kwargs(uses_v: bool):
    """full and base need v_table; discomfort_only is V-independent."""
    kw = {"access_table": LLM_TABLES["access"], "effort_table": LLM_TABLES["effort"]}
    if uses_v:
        kw["v_table"] = load_lm_v("food")
    return kw


# ==============================================================================
# Main Script
# ==============================================================================


def main():
    print("=" * 60)
    print("Inverse Planning Model Fitting")
    print("Fitting alpha_observer (frozen actor params)")
    print("=" * 60)

    print("\nLoading frozen actor parameters...")
    actor_params = load_fitted_params()
    for model_name, p in actor_params.items():
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in p.items())
        print(f"  {model_name}: {param_str}")

    results = []

    # -------------------------------------------------------------------------
    # Intimacy Inference
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INTIMACY INFERENCE")
    print("=" * 60)

    int_data, int_action, int_reward_condition, int_response, int_scenario_idx = (
        load_intimacy_data()
    )

    for variant_name, (int_obs, _rew_obs, kw_names, uses_v) in ACCESS_VARIANTS.items():
        if variant_name not in actor_params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        print(f"\n{'-' * 40}")
        print(f"Fitting {variant_name} (intimacy inference)...")
        print(f"{'-' * 40}")
        alpha_observer, nll = fit_intimacy_observer(
            observer_fn=int_obs,
            actor_params=actor_params[variant_name],
            actor_kwarg_names=kw_names,
            action=int_action,
            scenario_idx=int_scenario_idx,
            reward_condition=int_reward_condition,
            response=int_response,
            table_kwargs=_table_kwargs(uses_v),
        )
        results.append(
            {
                "model": variant_name,
                "experiment": "intimacy",
                "alpha_observer": alpha_observer,
                "nll": nll,
                "n_params": 1,
            }
        )

    # -------------------------------------------------------------------------
    # Reward Inference
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("REWARD INFERENCE")
    print("=" * 60)

    rew_data, rew_action, rew_intimacy_condition, rew_response, rew_scenario_idx = (
        load_reward_data()
    )

    for variant_name, (_int_obs, rew_obs, kw_names, uses_v) in ACCESS_VARIANTS.items():
        if variant_name not in actor_params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        print(f"\n{'-' * 40}")
        print(f"Fitting {variant_name} (reward inference)...")
        print(f"{'-' * 40}")
        alpha_observer, nll = fit_reward_observer(
            observer_fn=rew_obs,
            actor_params=actor_params[variant_name],
            actor_kwarg_names=kw_names,
            action=rew_action,
            scenario_idx=rew_scenario_idx,
            intimacy_condition=rew_intimacy_condition,
            response=rew_response,
            table_kwargs=_table_kwargs(uses_v),
        )
        results.append(
            {
                "model": variant_name,
                "experiment": "reward",
                "alpha_observer": alpha_observer,
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

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "inverse_planning_fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    main()
