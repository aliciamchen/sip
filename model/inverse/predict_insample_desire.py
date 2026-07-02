"""In-sample model predictions for Study 1a (food_inv_desire) — VIZ ONLY.

This is NOT part of the canonical pipeline: the out-of-sample predictions that
back every reported model-vs-human number come from leave-one-scenario-out CV
(`cv_food_inv_desire.py` → `cv_preds_summary.json`). This script instead runs each
ablation's observer *forward* through its already-fitted weights
(`fit_results.json`) to get each cell's predicted desire belief update, purely so
the analysis qmd can eyeball "what is each fitted model saying" before the (hours-
long) CV is run. In-sample, so it will look optimistic — use only for inspection.

Writes `outputs/food_inv_desire/insample_preds.json`: one record per
(model, scenario, action, effort, intimacy) cell with the model's predicted
belief update `delta_pred` (mean over the K elicitation runs) and the run spread
`delta_pred_sd`, for all three ablations (full / discomfort_only / base). The
model update is `posterior_mean − PRIOR_MEAN` on the 0-1 desire scale, exactly as
the fit's likelihood defines it — note PRIOR_MEAN is the model's uniform-prior
mean (0.5), which differs from participants' empirical prior (~0.67), so model and
human updates sit on slightly different baselines. (discomfort_only has no reward
term, so its desire posterior equals the prior → delta_pred ≈ 0 for every cell.)
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
    desire_table_kwargs,
    load_fit_results,
    params_dict_to_array,
    write_json,
)
from observers import (  # noqa: E402
    observer_desire_base,
    observer_desire_discomfort_only,
    observer_desire_full,
)
from tables import (  # noqa: E402
    ACTION_LABEL_TO_IDX,
    EFFORT_CONDITION_TO_IDX,
    INTIMACY_CONDITION_TO_IDX,
    SCENARIO_TO_IDX,
)

SLUG = "food_inv_desire"

# (observer_fn, utility_param_names) per ablation — mirrors fit_food_inv_desire.py.
VARIANTS = {
    "full": (observer_desire_full, ["w_v", "w_d", "w_e", "gamma"]),
    "discomfort_only": (observer_desire_discomfort_only, ["w_d", "gamma"]),
    "base": (observer_desire_base, ["w_v", "w_e"]),
}


def predict_variant(name, obs_fn, utility_names, fits, inv):
    params = params_dict_to_array(fits[name], utility_names)
    table_kwargs = desire_table_kwargs(utility_names, base=(name == "base"))
    # (K, slot, scenario, action, effort, relationship, desire[101]); slot 0 is the
    # observed action — exactly the slice the fit/CV score.
    tables = np.asarray(
        _build_observer_tables_runs(obs_fn, params, utility_names, table_kwargs)
    )
    grid = np.asarray(GRID)
    prior_mean = float(PRIOR_MEAN)
    rows = []
    for s in range(len(inv["scn"])):
        for a in range(len(inv["act"])):
            for e in range(len(inv["eff"])):
                for rel in range(len(inv["rel"])):
                    deltas = tables[:, 0, s, a, e, rel, :] @ grid - prior_mean  # (K,)
                    rows.append(
                        {
                            "model": name,
                            "scenario_label": inv["scn"][s],
                            "action_condition": inv["act"][a],
                            "effort": inv["eff"][e],
                            "intimacy": inv["rel"][rel],
                            "delta_pred": float(deltas.mean()),
                            "delta_pred_sd": float(deltas.std()),
                            # per-run mixture components (for the SI run-spread figure)
                            "delta_pred_runs": [float(x) for x in deltas],
                        }
                    )
    return rows


def main():
    fits = load_fit_results(SLUG)
    inv = {
        "scn": {v: k for k, v in SCENARIO_TO_IDX.items()},
        "act": {v: k for k, v in ACTION_LABEL_TO_IDX.items()},
        "eff": {v: k for k, v in EFFORT_CONDITION_TO_IDX.items()},
        "rel": {v: k for k, v in INTIMACY_CONDITION_TO_IDX.items()},
    }

    all_rows = []
    for name, (obs_fn, utility_names) in VARIANTS.items():
        rows = predict_variant(name, obs_fn, utility_names, fits, inv)
        all_rows.extend(rows)
        import pandas as pd

        by_action = (
            pd.DataFrame(rows).groupby("action_condition").delta_pred.mean() * 100
        )
        print(
            f"{name:16s} mean predicted update by action (x100): "
            f"{by_action.round(1).to_dict()}"
        )

    out = _project_root / "model" / "outputs" / SLUG / "insample_preds.json"
    write_json(out, all_rows)
    print(
        f"\nWrote {len(all_rows)} per-cell predictions "
        f"({len(VARIANTS)} variants x 384 cells) to {out}"
    )


if __name__ == "__main__":
    main()
