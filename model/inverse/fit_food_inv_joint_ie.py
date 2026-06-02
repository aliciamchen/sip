"""Fit observer + actor utility weights for food_inv_joint_ie.

Study 2b — joint over (intimacy, effort) given desire. Each variant jointly fits
its utility weights and alpha_observer from this experiment's posterior data
(no transfer between studies). Writes outputs/food_inv_joint_ie/fit_results.csv.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    fit_joint_ie_observer_joint,
    joint_ie_table_kwargs,
    load_joint_ie_data,
)
from observers import (  # noqa: E402
    observer_joint_ie_base,
    observer_joint_ie_discomfort_only,
    observer_joint_ie_full,
)

EXPERIMENT_SLUG = "food_inv_joint_ie"

# (observer_fn, utility_param_names, uses_v)
VARIANTS = {
    "full": (observer_joint_ie_full, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (
        observer_joint_ie_discomfort_only,
        ["w_d", "gamma"],
        False,
    ),
    "base": (observer_joint_ie_base, ["w_v", "w_e"], True),
}


def main():
    print("=" * 60)
    print(f"Joint inverse fit: {EXPERIMENT_SLUG}")
    print("Fitting utility weights + alpha_observer per variant")
    print("=" * 60)

    data, action, scenario_idx, desire_condition, resp_intimacy, resp_effort = (
        load_joint_ie_data(EXPERIMENT_SLUG)
    )

    results = []
    for variant_name, (obs_fn, utility_names, uses_v) in VARIANTS.items():
        print(
            f"\n{'-' * 40}\nJointly fitting {variant_name} ({len(utility_names)} weights + alpha_observer)...\n{'-' * 40}"
        )
        params, nll = fit_joint_ie_observer_joint(
            observer_fn=obs_fn,
            utility_param_names=utility_names,
            action=action,
            scenario_idx=scenario_idx,
            desire_condition=desire_condition,
            response_intimacy=resp_intimacy,
            response_effort=resp_effort,
            table_kwargs=joint_ie_table_kwargs(uses_v),
        )
        row = {
            "model": variant_name,
            "experiment": EXPERIMENT_SLUG,
            "nll": nll,
            "n_params": len(utility_names) + 1,
            "param_alpha": 1.0,
            "alpha_observer": float(params[-1]),
        }
        for i, name in enumerate(utility_names):
            row[f"param_{name}"] = float(params[i])
        results.append(row)

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
