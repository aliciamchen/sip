"""Generate in-sample predictions for food_inv_intimacy.

Reads this experiment's own fit_results.json, runs the intimacy observer across
all elicitation runs on the full scenario grid, and writes the model belief
update `delta_intimacy` per (scenario, action, desire, effort) cell to
preds_summary.json. (Headline model-vs-human comparisons are out-of-sample — see cv_*.)
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import numpy as np  # noqa: E402

from _helpers import (  # noqa: E402
    GRID,
    PRIOR_MEAN,
    _build_observer_tables_runs,
    intimacy_table_kwargs,
    load_fit_results,
    params_dict_to_array,
    write_json,
)
from observers import (  # noqa: E402
    observer_intimacy_base,
    observer_intimacy_discomfort_only,
    observer_intimacy_full,
)
from tables import N_ACTIONS, SCENARIO_LABELS  # noqa: E402

EXPERIMENT_SLUG = "food_inv_intimacy"

VARIANTS = {
    "full": (observer_intimacy_full, ["w_v", "w_d", "w_e", "gamma"]),
    "discomfort_only": (observer_intimacy_discomfort_only, ["w_d", "gamma"]),
    "base": (observer_intimacy_base, ["w_v", "w_e"]),
}

GRID_NP = np.asarray(GRID)
PRIOR_MEAN_F = float(PRIOR_MEAN)


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
        # (run, slot, scenario, observed_action, desire, effort, intimacy_101)
        tables = np.asarray(
            _build_observer_tables_runs(
                obs_fn, params_arr, utility_names, intimacy_table_kwargs(utility_names)
            )
        )
        for s_idx, scenario_label in enumerate(SCENARIO_LABELS):
            for a_idx in range(N_ACTIONS):
                for r in (0, 1):
                    for e in (0, 1):
                        density_runs = tables[:, 0, s_idx, a_idx, r, e, :]
                        delta = float((density_runs @ GRID_NP).mean() - PRIOR_MEAN_F)
                        pred_rows.append(
                            {
                                "scenario_label": scenario_label,
                                "action": a_idx,
                                "desire_condition": "low" if r == 0 else "high",
                                "effort_condition": "low" if e == 0 else "high",
                                "delta_intimacy": delta,
                                "model": variant,
                            }
                        )

    write_json(output_dir / "preds_summary.json", pred_rows)
    print(f"\nSaved per-cell delta predictions to {output_dir / 'preds_summary.json'}")


if __name__ == "__main__":
    main()
