"""Generate in-sample predictions for food_inv_joint_ie.

Reads this experiment's own fit_results.json, runs the joint_ie observer across
all elicitation runs on the full scenario grid, and writes the model belief
updates `delta_intimacy` and `delta_effort` per (scenario, action, desire) cell
to preds_summary.json. (Headline model-vs-human comparisons are out-of-sample — see cv_*.)
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import numpy as np  # noqa: E402

from _helpers import (  # noqa: E402
    EFFORT_PRIOR_MEAN,
    GRID,
    PRIOR_MEAN,
    _build_observer_tables_runs,
    joint_ie_table_kwargs,
    load_fit_results,
    params_dict_to_array,
    write_json,
)
from observers import (  # noqa: E402
    observer_joint_ie_base,
    observer_joint_ie_discomfort_only,
    observer_joint_ie_full,
)
from tables import N_ACTIONS, SCENARIO_LABELS  # noqa: E402

EXPERIMENT_SLUG = "food_inv_joint_ie"

VARIANTS = {
    "full": (observer_joint_ie_full, ["w_v", "w_d", "w_e", "gamma"]),
    "discomfort_only": (observer_joint_ie_discomfort_only, ["w_d", "gamma"]),
    "base": (observer_joint_ie_base, ["w_v", "w_e"]),
}

GRID_NP = np.asarray(GRID)
PRIOR_MEAN_F = float(PRIOR_MEAN)
EFFORT_PRIOR_MEAN_F = float(EFFORT_PRIOR_MEAN)


def main():
    print("=" * 60)
    print(f"Generating predictions: {EXPERIMENT_SLUG}")
    print("=" * 60)

    fit_params = load_fit_results(EXPERIMENT_SLUG)
    output_dir = _project_root / "model" / "outputs" / EXPERIMENT_SLUG
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_rows = []
    for variant, (obs_fn, utility_names) in VARIANTS.items():
        if variant not in fit_params:
            print(f"  (skipping {variant}: no fit row)")
            continue
        p = fit_params[variant]
        print(f"  {variant} (alpha_observer={p['alpha_observer']:.3f})...")
        params_arr = params_dict_to_array(p, utility_names)
        # (run, slot, scenario, observed_action, desire, intimacy_101, effort_2)
        tables = np.asarray(
            _build_observer_tables_runs(
                obs_fn, params_arr, utility_names, joint_ie_table_kwargs(utility_names)
            )
        )
        for s_idx, scenario_label in enumerate(SCENARIO_LABELS):
            for a_idx in range(N_ACTIONS):
                for r in (0, 1):
                    joint_runs = tables[:, 0, s_idx, a_idx, r, :, :]  # (K,101,2)
                    intimacy_mean = joint_runs.sum(axis=2) @ GRID_NP  # (K,)
                    p_high = joint_runs[:, :, 1].sum(axis=1)  # (K,)
                    pred_rows.append(
                        {
                            "scenario_label": scenario_label,
                            "action": a_idx,
                            "desire_condition": "low" if r == 0 else "high",
                            "delta_intimacy": float(
                                (intimacy_mean - PRIOR_MEAN_F).mean()
                            ),
                            "delta_effort": float(
                                (p_high - EFFORT_PRIOR_MEAN_F).mean()
                            ),
                            "model": variant,
                        }
                    )

    write_json(output_dir / "preds_summary.json", pred_rows)
    print(f"\nSaved per-cell delta predictions to {output_dir / 'preds_summary.json'}")


if __name__ == "__main__":
    main()
