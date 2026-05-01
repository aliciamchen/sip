"""
Fit observer alpha for the effort-inference experiment (food_inv-effort_intimacy_alt).

Parallel to model/fit_inverse_planning_intimacy_effort.py, but flips the inference
direction:
  - Manipulation: observed action (2) × intimacy (4 levels: 0, 50, 75, 100).
  - Latent: effort_condition (low/high). Slider response 0-100 = P(effort_high)*100.
  - Frozen actor weights from food_forw_intimacy_effort (forward_planning_effort_fit_results.csv).

Fits only alpha_observer per variant; actor weights stay frozen.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import jax.numpy as jnp
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_inverse_planning_alt import (
    _fit_alpha_observer,
    compute_reward_nll,
    load_fitted_params,
)
from tables import LLM_TABLES_EFFORT, SCENARIO_TO_IDX
from observers import (
    observer_effort_intimacy_base,
    observer_effort_intimacy_discomfort_only,
    observer_effort_intimacy_full,
)

from utils import get_project_root


# ==============================================================================
# Data loading
# ==============================================================================


def load_effort_intimacy_data(filepath: str = None):
    """Load and preprocess food_inv-effort_intimacy_alt data (posterior stage only).

    Returns JAX arrays: action (0-1), intimacy_idx (integer index 0/50/75/100
    into the 101-level IntimacyLevels axis of the actor), response (0-100
    encoding P(effort_high)*100), scenario_idx (0-15), plus the metadata
    DataFrame.
    """
    if filepath is None:
        filepath = (
            get_project_root()
            / "data"
            / "food_inv-effort_intimacy_alt"
            / "main_trials_long.csv"
        )
    print("Loading food_inv-effort_intimacy_alt data...")
    data = pd.read_csv(filepath)

    data = data[data["stage"] == "posterior"].copy()
    # action_condition is "action_1" or "action_2"; map to 0 or 1 internal index
    data["action"] = (
        data["action_condition"].str.replace("action_", "").astype(int) - 1
    )
    # intimacy is one of {0, 50, 75, 100}; the actor's IntimacyLevels axis has
    # 101 levels indexed 0..100 in 0.01 increments, so intimacy as int doubles
    # as the index.
    data["intimacy_idx"] = data["intimacy"].astype(int)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    action = jnp.array(data["action"].values)
    intimacy_idx = jnp.array(data["intimacy_idx"].values)
    response = jnp.array(data["response"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)

    print(f"Loaded {len(data)} posterior data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")

    return data, action, intimacy_idx, response, scenario_idx


def load_food_forw_intimacy_effort_actor_params(filepath: str = None) -> dict:
    """Load frozen actor parameters from the food_forw_intimacy_effort fit."""
    if filepath is None:
        filepath = (
            get_project_root()
            / "model"
            / "outputs"
            / "food_forw_intimacy_effort" / "fit_results.csv"
        )
    return load_fitted_params(filepath=filepath)


# ==============================================================================
# Fit-loop wrapper
# ==============================================================================


def fit_effort_intimacy_observer(
    observer_fn, actor_params, actor_kwarg_names,
    action, scenario_idx, intimacy_idx, response, table_kwargs, **kwargs,
):
    """Fit α_observer for the effort-inference observer.

    The observer table has shape (action, scenario, intimacy, effort_condition).
    For each trial we want P(effort_high) = tab[a, s, i, 1]; the binary
    cross-entropy NLL (compute_reward_nll) takes this directly as `p_high`.
    """
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=intimacy_idx,
        response=response,
        nll_fn=compute_reward_nll,
        # tab[a, s, i, 1] → scalar P(effort_high)
        posterior_slicer=lambda tab, a, s, i: tab[a, s, i, 1],
        table_kwargs=table_kwargs,
        **kwargs,
    )


# Variant registry: name -> (observer_fn, actor_kwarg_names)
ACCESS_VARIANTS_EFFORT_INFERRED = {
    "full": (
        observer_effort_intimacy_full,
        ["alpha", "w_v", "w_d", "w_e", "gamma"],
    ),
    "discomfort_only": (
        observer_effort_intimacy_discomfort_only,
        ["alpha", "w_d", "gamma"],
    ),
    "base": (
        observer_effort_intimacy_base,
        ["alpha", "w_v", "w_e"],
    ),
}


def _table_kwargs():
    # The observer in this experiment does NOT see the effort paragraph, so
    # the access values they use to reason about the actor must also not
    # depend on effort_condition. We use the effort-marginal access table
    # (vignette only) when available; the effort term itself is unaffected.
    access_table = LLM_TABLES_EFFORT.get(
        "access_marg", LLM_TABLES_EFFORT["access"]
    )
    return {
        "access_table": access_table,
        "effort_table": LLM_TABLES_EFFORT["effort"],
    }


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 60)
    print("Inverse Planning Model Fitting — effort-inference experiment")
    print("Fitting alpha_observer (frozen actor params from food_forw_intimacy_effort)")
    print("=" * 60)

    print("\nLoading frozen actor parameters from food_forw_intimacy_effort...")
    actor_params = load_food_forw_intimacy_effort_actor_params()
    for model_name, p in actor_params.items():
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in p.items())
        print(f"  {model_name}: {param_str}")

    data, action, intimacy_idx, response, scenario_idx = load_effort_intimacy_data()

    results = []
    for variant_name, (obs_fn, kw_names) in ACCESS_VARIANTS_EFFORT_INFERRED.items():
        if variant_name not in actor_params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        print(f"\n{'-' * 40}")
        print(f"Fitting {variant_name}...")
        print(f"{'-' * 40}")
        alpha_observer, nll = fit_effort_intimacy_observer(
            observer_fn=obs_fn,
            actor_params=actor_params[variant_name],
            actor_kwarg_names=kw_names,
            action=action,
            scenario_idx=scenario_idx,
            intimacy_idx=intimacy_idx,
            response=response,
            table_kwargs=_table_kwargs(),
        )
        results.append({
            "model": variant_name,
            "experiment": "effort_intimacy",
            "alpha_observer": alpha_observer,
            "nll": nll,
            "n_params": 1,
        })

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    output_dir = Path(__file__).parent / "outputs" / "food_inv-effort_intimacy_alt"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    return results_df


if __name__ == "__main__":
    main()
