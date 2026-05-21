"""Generate predictions for food_inv_joint_de_3act.

Runs the joint_de observer on a per-scenario grid; writes preds_full.csv
and preds_summary.csv.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    joint_3act_table_kwargs,
    load_fitted_alpha_observer,
    load_fitted_params,
)
from observers import (  # noqa: E402
    observer_joint_de_3act_base,
    observer_joint_de_3act_discomfort_only,
    observer_joint_de_3act_full,
)
from tables import SCENARIO_LABELS  # noqa: E402

EXPERIMENT_SLUG = "food_inv_joint_de_3act"

VARIANTS = {
    "full": (observer_joint_de_3act_full, ["alpha", "w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_joint_de_3act_discomfort_only, ["alpha", "w_d", "gamma"], False),
    "base": (observer_joint_de_3act_base, ["alpha", "w_v", "w_e"], True),
}


def main():
    print("=" * 60)
    print(f"Generating predictions: {EXPERIMENT_SLUG}")
    print("=" * 60)

    params = load_fitted_params()
    alpha_obs = load_fitted_alpha_observer(
        filepath=_project_root / "model" / "outputs" / EXPERIMENT_SLUG / "fit_results.csv"
    )

    output_dir = _project_root / "model" / "outputs" / EXPERIMENT_SLUG
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant, (obs_fn, kw_names, uses_v) in VARIANTS.items():
        if variant not in params:
            print(f"  (skipping {variant})")
            continue
        a_obs = alpha_obs.get((variant, EXPERIMENT_SLUG), 1.0)
        print(f"  {variant} (alpha_observer={a_obs:.3f})...")
        actor_kwargs = {k: params[variant][k] for k in kw_names}
        actor_kwargs["alpha_observer"] = a_obs
        result = obs_fn(**actor_kwargs, **joint_3act_table_kwargs(uses_v))
        # Save raw table as .npy and a flat CSV with shape metadata.
        import numpy as np
        np.save(output_dir / f"preds_{variant}.npy", np.asarray(result))

    # Placeholder summary: real summary requires the per-trial human grid which
    # only exists once data is collected. For now we just record table shapes.
    summary = []
    for variant in VARIANTS:
        path = output_dir / f"preds_{variant}.npy"
        if path.exists():
            import numpy as np
            arr = np.load(path)
            summary.append({"model": variant, "shape": str(arr.shape), "sum": float(arr.sum())})
    pd.DataFrame(summary).to_csv(output_dir / "preds_summary.csv", index=False)
    print(f"\nSaved per-variant prediction arrays to {output_dir}")


if __name__ == "__main__":
    main()
