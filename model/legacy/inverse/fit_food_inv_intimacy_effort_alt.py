"""Fit alpha_observer for food_inv_intimacy_effort_alt.

Effort-experiment 2-action observer infers intimacy. Actor params frozen from
food_forw_intimacy_effort fit.

Writes outputs/food_inv_intimacy_effort_alt/fit_results.csv.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    fit_intimacy_observer,
    load_food_forw_intimacy_effort_actor_params,
    load_intimacy_effort_data,
)
from observers import (  # noqa: E402
    observer_intimacy_effort_base,
    observer_intimacy_effort_discomfort_only,
    observer_intimacy_effort_full,
)
from tables import LLM_TABLES_EFFORT  # noqa: E402

EXPERIMENT_SLUG = "food_inv_intimacy_effort_alt"

ACCESS_VARIANTS_EFFORT = {
    "full": (observer_intimacy_effort_full, ["alpha", "w_v", "w_d", "w_e", "gamma"]),
    "discomfort_only": (observer_intimacy_effort_discomfort_only, ["alpha", "w_d", "gamma"]),
    "base": (observer_intimacy_effort_base, ["alpha", "w_v", "w_e"]),
}


def _table_kwargs():
    return {
        "access_table": LLM_TABLES_EFFORT["access"],
        "effort_table": LLM_TABLES_EFFORT["effort"],
    }


def main():
    print("=" * 60)
    print(f"Inverse planning fit: {EXPERIMENT_SLUG}")
    print("=" * 60)

    actor_params = load_food_forw_intimacy_effort_actor_params()
    data, action, effort_condition, response, scenario_idx = load_intimacy_effort_data()

    results = []
    for variant_name, (obs_fn, kw_names) in ACCESS_VARIANTS_EFFORT.items():
        if variant_name not in actor_params:
            print(f"  (skipping {variant_name})")
            continue
        print(f"\n{'-' * 40}")
        print(f"Fitting {variant_name}...")
        print(f"{'-' * 40}")
        alpha_observer, nll = fit_intimacy_observer(
            observer_fn=obs_fn,
            actor_params=actor_params[variant_name],
            actor_kwarg_names=kw_names,
            action=action,
            scenario_idx=scenario_idx,
            conditioning=effort_condition,
            response=response,
            table_kwargs=_table_kwargs(),
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
    results_df.to_csv(output_dir / "fit_results.csv", index=False)
    print(f"\nSaved fit results to {output_dir / 'fit_results.csv'}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
