"""
Leave-one-scenario-out CV for the inv_plan_effort_inferred observer.

Mirrors model/cv/loso_inverse_effort.py but flips the inference direction:
  - Manipulation: observed action × intimacy.
  - Latent: effort_condition. Slider response 0-100 = P(effort_high)*100.
  - Frozen actor weights from forw_plan_effort.

Per fold: hold out one scenario, refit α_observer on the remaining 15
scenarios, generate held-out posterior predictions using the refit α.

Outputs:
  - cv_loso_inv_plan_effort_inferred_preds_summary.csv  — one row per
    (scenario, action, intimacy, model); column `p_effort_high` is the
    LOSO-refit observer's P(effort_high).
  - cv_loso_inverse_effort_inferred_folds.csv  — per-fold α_observer,
    train/test NLL.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))

import numpy as np
import pandas as pd

from fit_inverse_planning import _fit_alpha_observer, compute_reward_nll
from fit_inverse_planning_effort_inferred import (
    ACCESS_VARIANTS_EFFORT_INFERRED,
    _table_kwargs,
    load_effort_inferred_data,
    load_forw_plan_effort_actor_params,
)
from model_utils import SCENARIO_LABELS
from model_utils_effort import actions_effort

from utils import get_project_root


N_SCENARIOS = len(SCENARIO_LABELS)
VARIANTS = ["full", "discomfort_only", "base"]
INTIMACY_DISPLAY_LEVELS = [0, 50, 75, 100]


def _loso_effort_inferred(actor_params_by_model):
    data, action, intimacy_idx, response, scenario_idx = load_effort_inferred_data()
    scenario_idx_np = np.asarray(scenario_idx)

    pred_rows = []
    fold_rows = []

    for variant in VARIANTS:
        if variant not in actor_params_by_model:
            print(f"  (skipping {variant}: no forward fit)")
            continue
        obs_fn, kw_names = ACCESS_VARIANTS_EFFORT_INFERRED[variant]
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
                conditioning=intimacy_idx[train_mask],
                response=response[train_mask],
                nll_fn=compute_reward_nll,
                posterior_slicer=lambda tab, a, s, i: tab[a, s, i, 1],
                table_kwargs=tk,
                verbose=False,
            )

            # Generate full observer table with refit α and slice held-out scenario
            result = obs_fn(**actor_kwargs, alpha_observer=alpha_obs, **tk)
            # Shape: (actions, scenarios, intimacy_levels, effort_conditions)
            held_out_table = np.asarray(result[:, fold, :, :])
            for a_idx, a_internal in enumerate(actions_effort):
                csv_action = int(a_internal) + 1  # internal 0/1 -> CSV 1/2
                for intimacy_int in INTIMACY_DISPLAY_LEVELS:
                    p_high = float(held_out_table[a_idx, intimacy_int, 1])
                    pred_rows.append({
                        "scenario_label": scenario_label,
                        "action": csv_action,
                        "intimacy": intimacy_int,
                        "model": variant,
                        "p_effort_high": p_high,
                    })

            # Test NLL per trial on held-out (binary cross-entropy)
            test_nll = 0.0
            for i in np.where(test_mask)[0]:
                p_high = float(
                    result[int(action[i]), int(scenario_idx[i]), int(intimacy_idx[i]), 1]
                )
                p_high = float(np.clip(p_high, 1e-8, 1.0 - 1e-8))
                p_human = float(response[i]) / 100.0
                test_nll += -(p_human * np.log(p_high) + (1.0 - p_human) * np.log(1.0 - p_high))

            fold_rows.append({
                "experiment": "effort_inferred",
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
    print("LOSO cross-validation: inv_plan_effort_inferred")
    print("=" * 60)

    print("\nLoading frozen actor parameters (forw_plan_effort all-data fit)...")
    actor_params_by_model = load_forw_plan_effort_actor_params()

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)

    preds, folds = _loso_effort_inferred(actor_params_by_model)
    preds_path = output_dir / "cv_loso_inv_plan_effort_inferred_preds_summary.csv"
    folds_path = output_dir / "cv_loso_inverse_effort_inferred_folds.csv"
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
