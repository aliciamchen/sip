"""
Fit observer alpha for the inverse-planning effort experiment (inv_plan_effort).

Parallel to model/fit_inverse_planning.py for the alt-shown intimacy observer,
adapted for:
  - 2 actions per scenario (action_1, action_2 -> internal 0, 1).
  - effort_condition (low, high) covariate instead of motivation.
  - frozen actor weights from forw_plan_effort (forward_planning_effort_fit_results.csv),
    NOT the canonical forw_plan fit.

Fits only alpha_observer per variant; actor weights stay frozen.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import jax.numpy as jnp
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_inverse_planning import (
    _fit_alpha_observer,
    compute_intimacy_nll,
    load_fitted_params,
)
from model_utils import SCENARIO_TO_IDX
from model_utils_effort import (
    EFFORT_CONDITION_TO_IDX,
    LLM_TABLES_EFFORT,
    observer_intimacy_effort_access_full,
    observer_intimacy_effort_access_only,
    observer_intimacy_effort_no_access,
)

from utils import get_project_root


# ==============================================================================
# Data loading
# ==============================================================================


def load_intimacy_effort_data(filepath: str = None):
    """Load and preprocess inv_plan_effort data (posterior stage only).

    Returns JAX arrays: action (0-1), effort_condition (0-1), response (0-100),
    scenario_idx (0-15) plus the metadata DataFrame.
    """
    if filepath is None:
        filepath = (
            get_project_root() / "data" / "inv_plan_effort" / "main_trials_long.csv"
        )
    print("Loading inv_plan_effort data...")
    data = pd.read_csv(filepath)

    data = data[data["stage"] == "posterior"].copy()
    # action_condition is "action_1" or "action_2"; map to 0 or 1 internal index
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int) - 1
    data["effort_condition"] = data["effort"].map(EFFORT_CONDITION_TO_IDX)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    action = jnp.array(data["action"].values)
    effort_condition = jnp.array(data["effort_condition"].values)
    response = jnp.array(data["intimacy_rating"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)

    print(f"Loaded {len(data)} posterior data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")

    return data, action, effort_condition, response, scenario_idx


def load_forw_plan_effort_actor_params(filepath: str = None) -> dict:
    """Load frozen actor parameters from the forw_plan_effort fit."""
    if filepath is None:
        filepath = (
            get_project_root()
            / "model"
            / "outputs"
            / "forward_planning_effort_fit_results.csv"
        )
    return load_fitted_params(filepath=filepath)


# ==============================================================================
# Fit-loop wrappers (effort observer threads effort_condition not reward)
# ==============================================================================


def fit_intimacy_effort_observer(
    observer_fn, actor_params, actor_kwarg_names,
    action, scenario_idx, effort_condition, response, table_kwargs, **kwargs,
):
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=effort_condition,
        response=response,
        nll_fn=compute_intimacy_nll,
        # observer table is (action, scenario, intimacy, effort_condition) → slice on intimacy
        posterior_slicer=lambda tab, a, s, e: tab[a, s, :, e],
        table_kwargs=table_kwargs,
        **kwargs,
    )


# Variant registry: name -> (observer_fn, actor_kwarg_names)
ACCESS_VARIANTS_EFFORT = {
    "access_full": (
        observer_intimacy_effort_access_full,
        ["alpha", "w_v", "w_d", "w_e"],
    ),
    "access_only": (
        observer_intimacy_effort_access_only,
        ["alpha", "w_d"],
    ),
    "no_access": (
        observer_intimacy_effort_no_access,
        ["alpha", "w_v", "w_e"],
    ),
}


def _table_kwargs():
    return {
        "access_table": LLM_TABLES_EFFORT["access"],
        "effort_table": LLM_TABLES_EFFORT["effort"],
    }


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 60)
    print("Inverse Planning Model Fitting — effort experiment")
    print("Fitting alpha_observer (frozen actor params from forw_plan_effort)")
    print("=" * 60)

    print("\nLoading frozen actor parameters from forw_plan_effort...")
    actor_params = load_forw_plan_effort_actor_params()
    for model_name, p in actor_params.items():
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in p.items())
        print(f"  {model_name}: {param_str}")

    data, action, effort_condition, response, scenario_idx = load_intimacy_effort_data()

    results = []
    for variant_name, (obs_fn, kw_names) in ACCESS_VARIANTS_EFFORT.items():
        if variant_name not in actor_params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        print(f"\n{'-' * 40}")
        print(f"Fitting {variant_name}...")
        print(f"{'-' * 40}")
        alpha_observer, nll = fit_intimacy_effort_observer(
            observer_fn=obs_fn,
            actor_params=actor_params[variant_name],
            actor_kwarg_names=kw_names,
            action=action,
            scenario_idx=scenario_idx,
            effort_condition=effort_condition,
            response=response,
            table_kwargs=_table_kwargs(),
        )
        results.append({
            "model": variant_name,
            "experiment": "intimacy_effort",
            "alpha_observer": alpha_observer,
            "nll": nll,
            "n_params": 1,
        })

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "inverse_planning_effort_fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    return results_df


if __name__ == "__main__":
    main()
