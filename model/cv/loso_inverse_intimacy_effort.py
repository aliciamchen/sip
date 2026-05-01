"""
Leave-one-scenario-out CV for the food_inv-intimacy_effort_alt observer.

Mirrors model/cv/loso_inverse_alt.py but adapted for:
  - 2 actions per scenario (action_1, action_2 → internal 0, 1).
  - effort_condition (low/high) covariate.
  - Frozen actor weights from food_forw_intimacy_effort (NOT the canonical food_forw_intimacy_desire fit).

Per fold: hold out one scenario, refit α_observer on remaining 15, generate
held-out posterior predictions using the refit α.

Outputs:
  - cv_loso_food_inv-intimacy_effort_alt_preds_summary.csv  — one row per (scenario, action, effort, model);
    expected_intimacy from LOSO-refit observer.
  - cv_loso_inverse_intimacy_effort_folds.csv  — per-fold α_observer, train/test NLL.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))

import numpy as np
import pandas as pd

from fit_inverse_planning_alt import _fit_alpha_observer, compute_intimacy_nll
from fit_inverse_planning_intimacy_effort import (
    ACCESS_VARIANTS_EFFORT,
    _table_kwargs,
    load_food_forw_intimacy_effort_actor_params,
    load_intimacy_effort_data,
)
from tables import IntimacyLevels, SCENARIO_LABELS, actions_effort

from utils import get_project_root


N_SCENARIOS = len(SCENARIO_LABELS)
VARIANTS = ["full", "discomfort_only", "base"]
EFFORT_LABELS = {0: "low", 1: "high"}


def _loso_intimacy_effort(actor_params_by_model):
    data, action, effort_condition, response, scenario_idx = load_intimacy_effort_data()
    scenario_idx_np = np.asarray(scenario_idx)

    pred_rows = []
    fold_rows = []
    intimacy_grid = np.asarray(IntimacyLevels) * 100.0  # 0..100

    for variant in VARIANTS:
        if variant not in actor_params_by_model:
            print(f"  (skipping {variant}: no forward fit)")
            continue
        obs_fn, kw_names = ACCESS_VARIANTS_EFFORT[variant]
        tk = _table_kwargs()
        actor_params = actor_params_by_model[variant]
        actor_kwargs = {k: actor_params[k] for k in kw_names}

        for fold in range(N_SCENARIOS):
            scenario_label = SCENARIO_LABELS[fold]
            train_mask = scenario_idx_np != fold
            test_mask = scenario_idx_np == fold
            n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
            print(
                f"  {variant} / fold {fold + 1}/{N_SCENARIOS} "
                f"({scenario_label}): train={n_train}, test={n_test}"
            )

            alpha_obs, train_nll = _fit_alpha_observer(
                observer_fn=obs_fn,
                actor_params=actor_params,
                actor_kwarg_names=kw_names,
                action=action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                conditioning=effort_condition[train_mask],
                response=response[train_mask],
                nll_fn=compute_intimacy_nll,
                posterior_slicer=lambda tab, a, s, e: tab[a, s, :, e],
                table_kwargs=tk,
                verbose=False,
            )

            # Generate full observer table with refit α and slice held-out scenario
            result = obs_fn(**actor_kwargs, alpha_observer=alpha_obs, **tk)
            # Shape: (actions, scenarios, intimacy_levels, effort_conditions)
            held_out_table = np.asarray(result[:, fold, :, :])
            for a_idx, a_internal in enumerate(actions_effort):
                csv_action = int(a_internal) + 1  # internal 0/1 -> CSV 1/2
                for e in [0, 1]:
                    density = held_out_table[a_idx, :, e]
                    expected_intimacy = float(np.sum(intimacy_grid * density))
                    pred_rows.append({
                        "scenario_label": scenario_label,
                        "action": csv_action,
                        "effort_condition": EFFORT_LABELS[e],
                        "model": variant,
                        "expected_intimacy": expected_intimacy,
                    })

            # Test NLL per trial on held-out
            test_nll = 0.0
            for i in np.where(test_mask)[0]:
                post = np.asarray(
                    result[int(action[i]), int(scenario_idx[i]), :, int(effort_condition[i])]
                )
                resp_idx = int(np.clip(round(float(response[i])), 0, 100))
                prob = max(post[resp_idx], 1e-8)
                test_nll += -float(np.log(prob))

            fold_rows.append({
                "experiment": "intimacy_effort",
                "variant": variant,
                "fold": fold,
                "held_out_scenario": scenario_label,
                "alpha_observer": float(alpha_obs),
                "train_nll": float(train_nll),
                "test_nll": test_nll,
                "n_train": n_train,
                "n_test": n_test,
            })

    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def main():
    print("=" * 60)
    print("LOSO cross-validation: food_inv-intimacy_effort_alt")
    print("=" * 60)

    print("\nLoading frozen actor parameters (food_forw_intimacy_effort all-data fit)...")
    actor_params_by_model = load_food_forw_intimacy_effort_actor_params()

    output_dir = get_project_root() / "model" / "outputs" / "food_inv-intimacy_effort_alt"
    output_dir.mkdir(parents=True, exist_ok=True)

    preds, folds = _loso_intimacy_effort(actor_params_by_model)
    preds_path = output_dir / "cv_preds_summary.csv"
    folds_path = output_dir / "cv_folds.csv"
    preds.to_csv(preds_path, index=False)
    folds.to_csv(folds_path, index=False)
    print(f"\nWrote {preds_path}")
    print(f"Wrote {folds_path}")

    print("\n=== Per-variant summary ===")
    for variant, sub in folds.groupby("variant"):
        print(
            f"  {variant}: "
            f"α_obs = {sub['alpha_observer'].mean():.3f} ± {sub['alpha_observer'].std():.3f}, "
            f"mean test NLL/trial = {(sub['test_nll'] / sub['n_test']).mean():.4f}"
        )


if __name__ == "__main__":
    main()
