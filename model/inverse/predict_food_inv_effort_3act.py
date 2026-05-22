"""Generate predictions for food_inv_effort_3act.

Reads jointly-fit utility weights + alpha_observer from this experiment's own
fit_results.csv (NOT from the forward fit), runs the effort observer on a
per-scenario grid, and writes preds_<variant>.npy + preds_summary.csv.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    effort_3act_table_kwargs,
    load_3act_fit_results,
)
from observers import (  # noqa: E402
    observer_effort_3act_base,
    observer_effort_3act_discomfort_only,
    observer_effort_3act_full,
)

EXPERIMENT_SLUG = "food_inv_effort_3act"

# (observer_fn, utility_param_names, uses_v)
VARIANTS = {
    "full": (observer_effort_3act_full, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_effort_3act_discomfort_only, ["w_d", "gamma"], False),
    "base": (observer_effort_3act_base, ["w_v", "w_e"], True),
}


def main():
    print("=" * 60)
    print(f"Generating predictions: {EXPERIMENT_SLUG}")
    print("=" * 60)

    fit_params = load_3act_fit_results(EXPERIMENT_SLUG)
    output_dir = _project_root / "model" / "outputs" / EXPERIMENT_SLUG
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant, (obs_fn, utility_names, uses_v) in VARIANTS.items():
        if variant not in fit_params:
            print(f"  (skipping {variant}: no fit row)")
            continue
        p = fit_params[variant]
        print(f"  {variant} (alpha_observer={p['alpha_observer']:.3f})...")
        kwargs = {"alpha": p["alpha"]}
        for name in utility_names:
            kwargs[name] = p[name]
        kwargs["alpha_observer"] = p["alpha_observer"]
        result = obs_fn(**kwargs, **effort_3act_table_kwargs(uses_v))
        np.save(output_dir / f"preds_{variant}.npy", np.asarray(result))

    summary = []
    for variant in VARIANTS:
        path = output_dir / f"preds_{variant}.npy"
        if path.exists():
            arr = np.load(path)
            summary.append(
                {"model": variant, "shape": str(arr.shape), "sum": float(arr.sum())}
            )
    pd.DataFrame(summary).to_csv(output_dir / "preds_summary.csv", index=False)
    print(f"\nSaved per-variant prediction arrays to {output_dir}")


if __name__ == "__main__":
    main()
