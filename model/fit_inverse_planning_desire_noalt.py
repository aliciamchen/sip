"""
Fit actor utility weights + alpha_observer for the no-alternatives-shown
desire (reward) inference variant.

Mirrors fit_inverse_planning_intimacy_noalt.py (intimacy variant) but flips the
inference target. The observer infers reward_condition (motivation) from a
single observed action, conditioned on the actor's relationship. Three
ablations are jointly fit:
  - full  : w_v * V - w_d * access * (1-I) - w_e * effort
  - discomfort_only  : -w_d * access * (1-I)
  - base    : w_v * V - w_e * effort

The action space is **relationship-keyed** — the LM-generated counterfactual
alternatives are conditioned on (scenario, observed_action, relationship)
rather than on motivation, since motivation is the latent and relationship is
what the observer sees. Tables come from `load_padded_lm_tables_relationship`
and have shapes (16, 4, 4, MAX_ACTIONS) for access/effort/prior and
(16, 4, 4, MAX_ACTIONS, 2) for V (extra motivation_query axis).

Loss is binary cross-entropy between human slider/100 and the model's
P(reward = HIGH).

Output: model/outputs/inverse_planning_desire_noalt_fit_results.csv
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
    load_padded_lm_tables_relationship,
    observer_reward_full_padded_rel,
    observer_reward_discomfort_only_padded_rel,
    observer_reward_base_padded_rel,
)

from utils import get_project_root

from fit_inverse_planning_alt import compute_reward_nll


# Variant registry: name -> (observer_fn, utility_param_names, uses_v).
# alpha_actor is fixed at 1; alpha_observer is appended to the fit params.
# discomfort_only is V-independent.
PADDED_VARIANTS = {
    "full": (
        observer_reward_full_padded_rel,
        ["w_v", "w_d", "w_e", "gamma"],
        True,
    ),
    "discomfort_only": (
        observer_reward_discomfort_only_padded_rel,
        ["w_d", "gamma"],
        False,
    ),
    "base": (
        observer_reward_base_padded_rel,
        ["w_v", "w_e"],
        True,
    ),
}


# Map intimacy column value (0, 50, 75, 100) → RelationshipConditions enum
# index (0..3) used by the relationship-keyed memos.
_INTIMACY_TO_RELATIONSHIP_IDX = {0: 0, 50: 1, 75: 2, 100: 3}


def load_desire_noalt_data(filepath=None):
    """Load and preprocess the no-alt desire data (posterior stage only).

    Returns observed_action, relationship_idx, response, scenario_idx as
    jnp arrays. `relationship_idx` is the `RelationshipConditions` enum index
    (0..3) for the relationship-keyed memo dim, mapped from the intimacy
    column values {0, 50, 75, 100}.
    """
    if filepath is None:
        filepath = (
            get_project_root()
            / "data"
            / "food_inv-desire_intimacy_noalt"
            / "main_trials_long.csv"
        )
    print("Loading no-alt desire inference data...")
    data = pd.read_csv(filepath)

    data = data[data["stage"] == "posterior"].copy()
    data["observed_action"] = (
        data["action_condition"].str.replace("action_", "").astype(int)
    )
    data["relationship_idx"] = data["intimacy"].astype(int).map(
        _INTIMACY_TO_RELATIONSHIP_IDX
    )
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    observed_action = jnp.array(data["observed_action"].values)
    relationship_idx = jnp.array(data["relationship_idx"].values)
    response = jnp.array(data["response"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)

    print(f"Loaded {len(data)} posterior data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")

    return data, observed_action, relationship_idx, response, scenario_idx


def fit_padded_joint_model(
    observer_fn,
    utility_param_names,
    observed_action,
    scenario_idx,
    relationship_idx,
    response,
    access_table,
    effort_table,
    prior_table,
    v_padded_table=None,
    lr=0.01,
    max_steps=5000,
    verbose=True,
    label="padded_joint",
    param_max=10.0,
    plateau_patience=200,
    plateau_tol=1e-3,
):
    """Jointly fit actor utility weights + alpha_observer for the padded reward observer.

    Free params (in order): utility_param_names, then alpha_observer at the end.
    alpha_actor is fixed at 1. Params are clipped to [1e-6, param_max] to keep
    the BCE-on-softmax landscape from blowing up via unbounded weights. Stops
    when best-so-far NLL hasn't improved by `plateau_tol` for `plateau_patience`
    consecutive steps.
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

    def nll_trial(table, obs_a, s, rel, resp):
        # table shape: (padded_slot, scenario, observed_action, relationship, reward_condition)
        # Slot 0 is the observed canonical action; reward index 1 = HIGH.
        p_high = table[0, s, obs_a, rel, 1]
        return compute_reward_nll(p_high, resp)

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0))

    def loss_fn(params):
        table = build_observer_table(params)
        return jnp.sum(
            vmap_nll(table, observed_action, scenario_idx, relationship_idx, response)
        )

    params = jnp.ones(n_utility + 1)
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    best_nll = float("inf")
    best_params = params
    steps_since_improvement = 0

    for step in range(max_steps):
        nll, grad = grad_fn(params)
        nll_f = float(nll)

        if jnp.any(jnp.isnan(grad)) or jnp.isnan(nll):
            if verbose:
                print(f"  NaN at step {step}, stopping")
            break

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 1e-6, param_max)

        if nll_f < best_nll - plateau_tol:
            best_nll = nll_f
            best_params = params
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        if verbose and step % 500 == 0:
            print(f"  Step {step}, NLL: {nll_f:.4f}, params: {params}")

        if steps_since_improvement >= plateau_patience:
            if verbose:
                print(f"  Plateau at step {step} (best NLL: {best_nll:.4f})")
            break

    final_nll = float(loss_fn(best_params))
    if verbose:
        print(f"  {label} final NLL: {final_nll:.4f}, params: {best_params}")
    return best_params, final_nll


def main():
    print("=" * 60)
    print("No-alt desire inference — joint fit (actor weights + α_observer)")
    print("Padded reward observer; uniform-over-valid-slots actor prior")
    print("=" * 60)

    data, observed_action, relationship_idx, response, scenario_idx = (
        load_desire_noalt_data()
    )

    padded = load_padded_lm_tables_relationship()
    if padded is None:
        print(
            "  Error: missing one of lm_alternatives_relationship.csv, "
            "lm_alternatives_relationship_features.csv, lm_scenario_v.csv, "
            "lm_alternatives_relationship_v.csv. Run "
            "`lm_generate_alternatives.py --conditioning relationship`, "
            "`lm_scenario_params.py --feature access_effort_alternatives_relationship`, "
            "and `lm_scenario_params.py --feature v_alternatives_relationship` first."
        )
        sys.exit(1)
    print(f"  access shape: {padded['access'].shape}")
    print(f"  v shape: {padded['v'].shape}")
    print(f"  prior shape: {padded['prior'].shape}")

    results = []
    for variant, (observer_fn, utility_names, uses_v) in PADDED_VARIANTS.items():
        print(f"\n{'-' * 40}")
        print(f"Jointly fitting {variant}_padded ({len(utility_names)} utility weights + α_observer)...")
        print(f"{'-' * 40}")
        params, nll = fit_padded_joint_model(
            observer_fn=observer_fn,
            utility_param_names=utility_names,
            observed_action=observed_action,
            scenario_idx=scenario_idx,
            relationship_idx=relationship_idx,
            response=response,
            access_table=padded["access"],
            effort_table=padded["effort"],
            prior_table=padded["prior"],
            v_padded_table=padded["v"] if uses_v else None,
        )
        row = {
            "model": f"{variant}_padded",
            "experiment": "desire_noalt",
            "nll": nll,
            "n_params": len(utility_names) + 1,
            "param_alpha": 1.0,
            "alpha_observer": float(params[-1]),
        }
        for i, name in enumerate(utility_names):
            row[f"param_{name}"] = float(params[i])
        results.append(row)

    results_df = pd.DataFrame(results)
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "inverse_planning_desire_noalt_fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
