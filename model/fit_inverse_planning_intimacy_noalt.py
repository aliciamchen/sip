"""
Fit alpha_observer for the no-alternatives-shown intimacy inference variant.

Fits three access-utility ablations (same three as the alt-shown inverse-planning fit):
  - full  : w_v * V - w_d * access * (1-I) - w_e * effort
  - discomfort_only  : -w_d * access * (1-I)
  - base    : w_v * V - w_e * effort

All variants use the padded observer with a trial-specific action space
(observed canonical action at slot 0, LM-generated alternatives at slots 1..k,
null padding at remaining slots). Actor params are frozen from the forward-
planning fit (per variant). Only alpha_observer is fitted against the no-alt
data.

Output: model/outputs/inverse_planning_intimacy_noalt_fit_results.csv
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import jax
import jax.numpy as jnp
import optax
import pandas as pd
from tables import SCENARIO_TO_IDX, load_padded_lm_tables
from observers import (
    observer_intimacy_base_padded,
    observer_intimacy_discomfort_only_padded,
    observer_intimacy_full_padded,
)

from utils import get_project_root

from fit_forward import _fit_with_adam
from fit_inverse_planning_alt import compute_intimacy_nll, load_fitted_params


# Variant registry: name -> (observer_fn, actor_kwarg_names, utility_param_names).
# `actor_kwarg_names` is the full kwargs needed by the observer memo (including
# `alpha` for the actor softmax, which is fixed at 1). `utility_param_names` is
# the subset that is actually fitted jointly with α_observer in the no-alt
# pipeline (α_actor is not fitted).
# Tuple values: (observer_fn, full_kw_names, utility_names, uses_v).
# discomfort_only is V-independent and doesn't take v_padded_table.
PADDED_VARIANTS = {
    "full": (
        observer_intimacy_full_padded,
        ["alpha", "w_v", "w_d", "w_e", "gamma"],
        ["w_v", "w_d", "w_e", "gamma"],
        True,
    ),
    "discomfort_only": (
        observer_intimacy_discomfort_only_padded,
        ["alpha", "w_d", "gamma"],
        ["w_d", "gamma"],
        False,
    ),
    "base": (
        observer_intimacy_base_padded,
        ["alpha", "w_v", "w_e"],
        ["w_v", "w_e"],
        True,
    ),
}


def load_intimacy_noalt_data(filepath=None):
    """Load and preprocess the no-alt intimacy data (posterior stage only)."""
    if filepath is None:
        filepath = (
            get_project_root()
            / "data"
            / "food_inv-intimacy_desire_noalt"
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
    prior_table,
    v_padded_table=None,
    lr=0.1,
    max_steps=1000,
    verbose=True,
):
    actor_kwargs = {k: actor_params[k] for k in actor_kwarg_names}
    table_kwargs = dict(
        access_table=access_table, effort_table=effort_table, prior_table=prior_table,
    )
    if v_padded_table is not None:
        table_kwargs["v_padded_table"] = v_padded_table

    def observer_table(alpha_observer):
        return observer_fn(
            **actor_kwargs,
            alpha_observer=alpha_observer,
            **table_kwargs,
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


def fit_padded_joint_model(
    observer_fn,
    utility_param_names,
    observed_action,
    scenario_idx,
    reward_condition,
    response,
    access_table,
    effort_table,
    prior_table,
    v_padded_table=None,
    lr=0.05,
    max_steps=2000,
    verbose=True,
    label="padded_joint",
):
    """Jointly fit actor utility weights + α_observer for the padded observer.

    Free params (in order): utility_param_names, then α_observer at the end.
    α_actor is fixed at 1 (same as everywhere else).
    """
    ALPHA_ACTOR = 1.0
    n_utility = len(utility_param_names)
    table_kwargs = dict(
        access_table=access_table, effort_table=effort_table, prior_table=prior_table,
    )
    if v_padded_table is not None:
        table_kwargs["v_padded_table"] = v_padded_table

    def build_observer_table(params):
        actor_kwargs = {"alpha": ALPHA_ACTOR}
        for i, name in enumerate(utility_param_names):
            actor_kwargs[name] = params[i]
        return observer_fn(
            **actor_kwargs,
            alpha_observer=params[-1],
            **table_kwargs,
        )

    def nll_trial(table, obs_a, s, r, resp):
        post = table[0, s, obs_a, :, r]
        return compute_intimacy_nll(post, resp)

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0))

    def loss_fn(params):
        table = build_observer_table(params)
        return jnp.sum(
            vmap_nll(table, observed_action, scenario_idx, reward_condition, response)
        )

    init_params = jnp.ones(n_utility + 1)
    params, nll = _fit_with_adam(
        loss_fn, init_params, lr=lr, max_steps=max_steps, verbose=verbose, label=label,
    )
    return params, float(nll)


def main():
    print("=" * 60)
    print("No-alt inverse planning — joint fit (actor weights + α_observer)")
    print("Padded observer; uniform-over-valid-slots actor prior")
    print("=" * 60)

    data, observed_action, reward_condition, response, scenario_idx = (
        load_intimacy_noalt_data()
    )

    padded = load_padded_lm_tables()
    if padded is None:
        print(
            "  Error: missing one of lm_alternatives.csv, lm_alternatives_features.csv, "
            "lm_scenario_v.csv, lm_alternatives_v.csv. Run lm/generate_alternatives.py "
            "and lm/scenario_params.py --feature {v,v_alternatives} first."
        )
        sys.exit(1)
    print(f"  access shape: {padded['access'].shape}")
    print(f"  prior shape: {padded['prior'].shape}")

    results = []
    for variant, (observer_fn, _kw_names, utility_names, uses_v) in PADDED_VARIANTS.items():
        print(f"\n{'-' * 40}")
        print(f"Jointly fitting {variant}_padded ({len(utility_names)} utility weights + α_observer)...")
        print(f"{'-' * 40}")
        params, nll = fit_padded_joint_model(
            observer_fn=observer_fn,
            utility_param_names=utility_names,
            observed_action=observed_action,
            scenario_idx=scenario_idx,
            reward_condition=reward_condition,
            response=response,
            access_table=padded["access"],
            effort_table=padded["effort"],
            prior_table=padded["prior"],
            v_padded_table=padded["v"] if uses_v else None,
        )
        row = {
            "model": f"{variant}_padded",
            "experiment": "intimacy_noalt",
            "nll": nll,
            "n_params": len(utility_names) + 1,
            "param_alpha": 1.0,
            "alpha_observer": float(params[-1]),
        }
        for i, name in enumerate(utility_names):
            row[f"param_{name}"] = float(params[i])
        results.append(row)

    results_df = pd.DataFrame(results)
    output_dir = Path(__file__).parent / "outputs" / "food_inv-intimacy_desire_noalt"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
