"""
Fit alpha_observer for the no-alternatives-shown intimacy inference variant.

Fits three access-utility ablations (same three as the main Exp 2a fit):
  - access_full  : w_v * V - w_d * access * (1-I) - w_e * effort
  - access_only  : -w_d * access * (1-I)
  - no_access    : w_v * V - w_e * effort

All variants use the padded observer with a trial-specific action space
(observed canonical action at slot 0, LM-generated alternatives at slots 1..k,
null padding at remaining slots). Actor params are frozen from the forward-
planning fit (per variant). Only alpha_observer is fitted against the no-alt
data.

Output: model/outputs/inverse_planning_noalt_fit_results.csv
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import jax
import jax.numpy as jnp
import optax
import pandas as pd
from model_utils import (
    SCENARIO_TO_IDX,
    load_padded_lm_tables,
    observer_intimacy_access_full_padded,
    observer_intimacy_access_only_padded,
    observer_intimacy_no_access_padded,
)

from utils import get_project_root

from fit_inverse_planning import compute_intimacy_nll, load_fitted_params


# Variant registry: name -> (observer_fn, actor_kwarg_names)
PADDED_VARIANTS = {
    "access_full": (
        observer_intimacy_access_full_padded,
        ["alpha", "w_v", "w_d", "w_e"],
    ),
    "access_only": (
        observer_intimacy_access_only_padded,
        ["alpha", "w_d"],
    ),
    "no_access": (
        observer_intimacy_no_access_padded,
        ["alpha", "w_v", "w_e"],
    ),
}


def load_intimacy_noalt_data(filepath=None):
    """Load and preprocess the no-alt intimacy data (posterior stage only)."""
    if filepath is None:
        filepath = (
            get_project_root()
            / "data"
            / "inv_plan_intimacy_noalt"
            / "main_trials_long.csv"
        )
    print("Loading no-alt intimacy inference data...")
    data = pd.read_csv(filepath)

    data = data[data["stage"] == "posterior"].copy()
    data["observed_action"] = (
        data["action_condition"].str.replace("action_", "").astype(int)
    )

    motivation_map = {"low": 0, "high": 1}
    data["reward_condition"] = data["motivation"].map(motivation_map)

    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    observed_action = jnp.array(data["observed_action"].values)
    reward_condition = jnp.array(data["reward_condition"].values)
    response = jnp.array(data["intimacy_rating"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)

    print(f"Loaded {len(data)} posterior data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")

    return data, observed_action, reward_condition, response, scenario_idx


def fit_padded_alpha_observer(
    observer_fn,
    actor_params,
    actor_kwarg_names,
    observed_action,
    scenario_idx,
    reward_condition,
    response,
    access_table,
    effort_table,
    is_share_table,
    prior_table,
    lr=0.1,
    max_steps=1000,
    verbose=True,
):
    actor_kwargs = {k: actor_params[k] for k in actor_kwarg_names}

    def observer_table(alpha_observer):
        return observer_fn(
            **actor_kwargs,
            alpha_observer=alpha_observer,
            access_table=access_table,
            effort_table=effort_table,
            is_share_table=is_share_table,
            prior_table=prior_table,
        )

    def get_nll(alpha_observer, obs_a, s, r, resp):
        table = observer_table(alpha_observer)
        # table shape: (padded_slot, scenario, observed_action, intimacy, reward_condition)
        # Slot 0 is always the observed canonical action.
        posterior_over_intimacy = table[0, s, obs_a, :, r]
        return compute_intimacy_nll(posterior_over_intimacy, resp)

    vmap_get_nll = jax.vmap(
        lambda alpha_obs, a, s, r, resp: get_nll(alpha_obs, a, s, r, resp),
        in_axes=(None, 0, 0, 0, 0),
    )

    def loss_fn(params):
        return jnp.sum(
            vmap_get_nll(
                params[0], observed_action, scenario_idx, reward_condition, response
            )
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


def main():
    print("=" * 60)
    print("No-alt inverse planning — alpha_observer fit (3 variants)")
    print("Padded observer; frozen actor params per variant; uniform actor prior")
    print("=" * 60)

    print("\nLoading frozen actor parameters...")
    actor_params_by_model = load_fitted_params()
    for variant in PADDED_VARIANTS:
        if variant not in actor_params_by_model:
            print(f"  (missing {variant}; will skip)")
        else:
            param_str = ", ".join(
                f"{k}={v:.3f}" for k, v in actor_params_by_model[variant].items()
            )
            print(f"  {variant}: {param_str}")

    data, observed_action, reward_condition, response, scenario_idx = (
        load_intimacy_noalt_data()
    )

    padded = load_padded_lm_tables()
    if padded is None:
        print("  Error: lm_alternatives.csv or lm_alternatives_features.csv not found.")
        sys.exit(1)
    print(f"  access shape: {padded['access'].shape}")
    print(f"  prior shape: {padded['prior'].shape}")

    results = []
    for variant, (observer_fn, kw_names) in PADDED_VARIANTS.items():
        if variant not in actor_params_by_model:
            continue
        print(f"\n{'-' * 40}")
        print(f"Fitting {variant}_padded...")
        print(f"{'-' * 40}")
        alpha_observer, nll = fit_padded_alpha_observer(
            observer_fn=observer_fn,
            actor_params=actor_params_by_model[variant],
            actor_kwarg_names=kw_names,
            observed_action=observed_action,
            scenario_idx=scenario_idx,
            reward_condition=reward_condition,
            response=response,
            access_table=padded["access"],
            effort_table=padded["effort"],
            is_share_table=padded["is_share"],
            prior_table=padded["prior"],
        )
        results.append({
            "model": f"{variant}_padded",
            "experiment": "intimacy_noalt",
            "alpha_observer": alpha_observer,
            "nll": nll,
            "n_params": 1,
        })

    results_df = pd.DataFrame(results)
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "inverse_planning_noalt_fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
