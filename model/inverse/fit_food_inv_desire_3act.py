"""Fit alpha_observer for food_inv_desire_3act.

Study 3b — observer knows (effort, intimacy), infers desire. Actor params frozen from the canonical food forward fit
(food_forw_intimacy_desire). Writes outputs/food_inv_desire_3act/fit_results.csv.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    load_desire_3act_data,
    fit_desire_3act_observer,
    desire_3act_table_kwargs,
    load_fitted_params,
)
from observers import (  # noqa: E402
    observer_reward_3act_base,
    observer_reward_3act_discomfort_only,
    observer_reward_3act_full,
)

EXPERIMENT_SLUG = "food_inv_desire_3act"

VARIANTS = {
    "full": (observer_reward_3act_full, ["alpha", "w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_reward_3act_discomfort_only, ["alpha", "w_d", "gamma"], False),
    "base": (observer_reward_3act_base, ["alpha", "w_v", "w_e"], True),
}


def main():
    print("=" * 60)
    print(f"Inverse planning fit: {EXPERIMENT_SLUG}")
    print("Fitting alpha_observer (frozen actor params)")
    print("=" * 60)

    actor_params = load_fitted_params()
    data, action, scenario_idx, effort_condition, relationship_condition, response = load_desire_3act_data(EXPERIMENT_SLUG)

    results = []
    for variant_name, (obs_fn, kw_names, uses_v) in VARIANTS.items():
        if variant_name not in actor_params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        print(f"\n{'-' * 40}\nFitting {variant_name}...\n{'-' * 40}")
        alpha_observer, nll = fit_desire_3act_observer(
            observer_fn=obs_fn,
            actor_params=actor_params[variant_name],
            actor_kwarg_names=kw_names,
            action=action, scenario_idx=scenario_idx, effort_condition=effort_condition, relationship_condition=relationship_condition, response=response,
            table_kwargs=desire_3act_table_kwargs(uses_v),
        )
        results.append({
            "model": variant_name,
            "experiment": EXPERIMENT_SLUG,
            "alpha_observer": alpha_observer,
            "nll": nll,
            "n_params": 1,
        })

    output_dir = _project_root / "model" / "outputs" / EXPERIMENT_SLUG
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60 + "\nRESULTS SUMMARY\n" + "=" * 60)
    print(results_df.to_string(index=False))
    results_path = output_dir / "fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")


if __name__ == "__main__":
    main()
