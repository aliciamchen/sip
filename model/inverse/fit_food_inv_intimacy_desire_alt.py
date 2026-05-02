"""Fit alpha_observer for food_inv_intimacy_desire_alt.

Alt-shown intimacy observer; actor params frozen from the canonical food
forward fit. Writes outputs/food_inv_intimacy_desire_alt/fit_results.csv.
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
    load_fitted_params,
    load_intimacy_alt_data,
)
from observers import (  # noqa: E402
    observer_intimacy_base,
    observer_intimacy_discomfort_only,
    observer_intimacy_full,
)
from tables import LLM_TABLES, load_lm_v  # noqa: E402

EXPERIMENT_SLUG = "food_inv_intimacy_desire_alt"

# Variant registry: name -> (observer_fn, actor_kwarg_names, uses_v).
ACCESS_VARIANTS = {
    "full": (observer_intimacy_full, ["alpha", "w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_intimacy_discomfort_only, ["alpha", "w_d", "gamma"], False),
    "base": (observer_intimacy_base, ["alpha", "w_v", "w_e"], True),
}


def _table_kwargs(uses_v):
    kw = {"access_table": LLM_TABLES["access"], "effort_table": LLM_TABLES["effort"]}
    if uses_v:
        kw["v_table"] = load_lm_v("food")
    return kw


def main():
    print("=" * 60)
    print(f"Inverse planning fit: {EXPERIMENT_SLUG}")
    print("Fitting alpha_observer (frozen actor params)")
    print("=" * 60)

    actor_params = load_fitted_params()
    data, action, reward_condition, response, scenario_idx = load_intimacy_alt_data()

    results = []
    for variant_name, (obs_fn, kw_names, uses_v) in ACCESS_VARIANTS.items():
        if variant_name not in actor_params:
            print(f"  (skipping {variant_name}: no forward fit available)")
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
            conditioning=reward_condition,
            response=response,
            table_kwargs=_table_kwargs(uses_v),
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
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    results_path = output_dir / "fit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")


if __name__ == "__main__":
    main()
