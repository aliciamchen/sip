"""Fit padded intimacy observer for food_inv-intimacy_desire_noalt (no-alt).

Jointly fits all actor utility weights + alpha_observer per ablation. Padded
observer uses LM-generated counterfactual alternatives (slots 1..k) plus the
observed canonical action (slot 0).

Writes outputs/food_inv-intimacy_desire_noalt/fit_results.csv.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    fit_padded_joint_intimacy,
    load_intimacy_noalt_data,
)
from observers import (  # noqa: E402
    observer_intimacy_base_padded,
    observer_intimacy_discomfort_only_padded,
    observer_intimacy_full_padded,
)
from tables import load_padded_lm_tables  # noqa: E402

EXPERIMENT_SLUG = "food_inv-intimacy_desire_noalt"

# (observer_fn, utility_param_names, uses_v)
PADDED_VARIANTS = {
    "full": (observer_intimacy_full_padded, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_intimacy_discomfort_only_padded, ["w_d", "gamma"], False),
    "base": (observer_intimacy_base_padded, ["w_v", "w_e"], True),
}


def main():
    print("=" * 60)
    print(f"No-alt joint fit: {EXPERIMENT_SLUG}")
    print("=" * 60)

    data, observed_action, reward_condition, response, scenario_idx = load_intimacy_noalt_data()
    padded = load_padded_lm_tables()
    if padded is None:
        print(
            "Error: missing one of lm_alternatives.csv, lm_alternatives_features.csv, "
            "lm_scenario_v.csv, lm_alternatives_v.csv. Run lm/generate_alternatives_motivation.py "
            "and lm/score_canonical_features.py --feature {v,v_alternatives} first."
        )
        sys.exit(1)

    results = []
    for variant, (obs_fn, utility_names, uses_v) in PADDED_VARIANTS.items():
        print(f"\n{'-' * 40}")
        print(f"Jointly fitting {variant}_padded ({len(utility_names)} weights + α_observer)...")
        print(f"{'-' * 40}")
        params, nll = fit_padded_joint_intimacy(
            observer_fn=obs_fn,
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
    results_df.to_csv(output_dir / "fit_results.csv", index=False)
    print(f"\nSaved fit results to {output_dir / 'fit_results.csv'}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
