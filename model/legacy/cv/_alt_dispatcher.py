"""
Leave-one-scenario-out CV for the alt-shown inverse-planning experiments
(intimacy inference — Exp 2a; desire inference — Exp 2b).

For each of the 16 scenarios, hold it out, refit α_observer on the remaining
15 scenarios (actor weights frozen at the all-data Exp 1 fit, as always), and
generate predictions for the held-out scenario using the refit α_observer.

Runs over the three canonical uniform-prior variants
(full, discomfort_only, base).

Outputs (in model/outputs/):
  - cv_loso_food_inv_intimacy_desire_alt_preds_summary.csv
    One row per (scenario, action, motivation, model), same schema as
    food_inv_intimacy_desire_alt_preds_summary.csv; `expected_intimacy` comes from
    the LOSO-refit observer for that scenario's held-out fold.
  - cv_loso_food_inv_desire_intimacy_alt_preds_summary.csv
    One row per (scenario, action, intimacy_condition, model); `p_high_reward`
    from LOSO-refit observer.
  - cv_loso_inverse_alt_folds.csv
    Per-fold fitted α_observer and train/test NLL, per experiment × variant.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import jax.numpy as jnp
import numpy as np
import pandas as pd

from _helpers import (  # noqa: E402
    ACCESS_VARIANTS,
    _fit_alpha_observer,
    alt_table_kwargs as _table_kwargs,
    compute_intimacy_nll,
    compute_reward_nll,
    load_fitted_params,
    load_intimacy_alt_data as load_intimacy_data,
    load_desire_alt_data as load_reward_data,
)
from tables import IntimacyLevels, SCENARIO_LABELS, actions  # noqa: E402

from utils import get_project_root


N_SCENARIOS = len(SCENARIO_LABELS)

# Canonical uniform-prior variants.
VARIANTS = ["full", "discomfort_only", "base"]
VARIANT_LABEL = {
    "full": "Full model",
    "discomfort_only": "Discomfort-only",
    "base": "Base model",
}


def _loso_intimacy(actor_params_by_model):
    data, action, reward_condition, response, scenario_idx = load_intimacy_data()
    scenario_idx_np = np.asarray(scenario_idx)

    pred_rows = []
    fold_rows = []

    for variant in VARIANTS:
        if variant not in actor_params_by_model:
            print(f"  (skipping {variant}: no forward fit)")
            continue
        obs_fn, _rew_fn, kw_names, uses_v = ACCESS_VARIANTS[variant]
        tk = _table_kwargs(uses_v)
        actor_params = actor_params_by_model[variant]
        actor_kwargs = {k: actor_params[k] for k in kw_names}

        for fold in range(N_SCENARIOS):
            scenario_label = SCENARIO_LABELS[fold]
            train_mask = scenario_idx_np != fold
            test_mask = scenario_idx_np == fold
            n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
            print(
                f"  intimacy / {variant} / fold {fold + 1}/{N_SCENARIOS} "
                f"({scenario_label}): train={n_train}, test={n_test}"
            )

            alpha_obs, train_nll = _fit_alpha_observer(
                observer_fn=obs_fn,
                actor_params=actor_params,
                actor_kwarg_names=kw_names,
                action=action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                conditioning=reward_condition[train_mask],
                response=response[train_mask],
                nll_fn=compute_intimacy_nll,
                posterior_slicer=lambda tab, a, s, r: tab[a, s, :, r],
                table_kwargs=tk,
                verbose=False,
            )

            # Generate the full observer table with the refit α_observer,
            # then slice held-out scenario's predictions.
            result = obs_fn(
                **actor_kwargs, alpha_observer=alpha_obs, **tk,
            )
            # Shape: (actions, scenarios, intimacy_levels, reward_conditions)
            held_out_table = np.asarray(result[:, fold, :, :])
            intimacy_grid = np.asarray(IntimacyLevels) * 100.0  # 0..100
            for a_idx, a in enumerate(actions):
                for r in [0, 1]:
                    density = held_out_table[a_idx, :, r]
                    expected_intimacy = float(np.sum(intimacy_grid * density))
                    pred_rows.append({
                        "scenario_label": scenario_label,
                        "action": int(a),
                        "reward_condition": "low" if r == 0 else "high",
                        "model": variant,
                        "expected_intimacy": expected_intimacy,
                    })

            # Test NLL per trial on held-out
            test_nll = 0.0
            for i in np.where(test_mask)[0]:
                post = np.asarray(result[int(action[i]), int(scenario_idx[i]), :, int(reward_condition[i])])
                resp_idx = int(np.clip(round(float(response[i])), 0, 100))
                prob = max(post[resp_idx], 1e-8)
                test_nll += -float(np.log(prob))

            fold_rows.append({
                "experiment": "food_inv_intimacy_desire_alt",
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


def _loso_desire(actor_params_by_model):
    data, action, intimacy_condition, response, scenario_idx = load_reward_data()
    scenario_idx_np = np.asarray(scenario_idx)
    intimacy_map_back = {0: 0, 1: 50, 2: 75, 3: 100}

    pred_rows = []
    fold_rows = []

    for variant in VARIANTS:
        if variant not in actor_params_by_model:
            print(f"  (skipping {variant}: no forward fit)")
            continue
        _int_fn, obs_fn, kw_names, uses_v = ACCESS_VARIANTS[variant]
        tk = _table_kwargs(uses_v)
        actor_params = actor_params_by_model[variant]
        actor_kwargs = {k: actor_params[k] for k in kw_names}

        for fold in range(N_SCENARIOS):
            scenario_label = SCENARIO_LABELS[fold]
            train_mask = scenario_idx_np != fold
            test_mask = scenario_idx_np == fold
            n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
            print(
                f"  desire / {variant} / fold {fold + 1}/{N_SCENARIOS} "
                f"({scenario_label}): train={n_train}, test={n_test}"
            )

            alpha_obs, train_nll = _fit_alpha_observer(
                observer_fn=obs_fn,
                actor_params=actor_params,
                actor_kwarg_names=kw_names,
                action=action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                conditioning=intimacy_condition[train_mask],
                response=response[train_mask],
                nll_fn=compute_reward_nll,
                posterior_slicer=lambda tab, a, s, i: tab[a, s, i, 1],
                table_kwargs=tk,
                verbose=False,
            )

            result = obs_fn(
                **actor_kwargs, alpha_observer=alpha_obs, **tk,
            )
            # Shape: (actions, scenarios, relationship_conditions, reward_conditions)
            held_out_table = np.asarray(result[:, fold, :, :])
            for a_idx, a in enumerate(actions):
                for rel_idx in range(4):
                    p_high = float(held_out_table[a_idx, rel_idx, 1]) * 100.0
                    pred_rows.append({
                        "scenario_label": scenario_label,
                        "action": int(a),
                        "intimacy_condition": intimacy_map_back[rel_idx],
                        "p_high_reward": p_high,
                        "model": variant,
                    })

            # Test NLL per trial (binary cross-entropy on held-out)
            test_nll = 0.0
            for i in np.where(test_mask)[0]:
                p_model = float(result[int(action[i]), int(scenario_idx[i]),
                                        int(intimacy_condition[i]), 1])
                p_human = float(response[i]) / 100.0
                p_m = min(max(p_model, 1e-8), 1 - 1e-8)
                test_nll += -(p_human * float(np.log(p_m)) + (1 - p_human) * float(np.log(1 - p_m)))

            fold_rows.append({
                "experiment": "food_inv_desire_intimacy_alt",
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


def main_intimacy_alt():
    print("=" * 60)
    print("LOSO CV: food_inv_intimacy_desire_alt")
    print("=" * 60)

    actor_params_by_model = load_fitted_params()
    outputs_dir = get_project_root() / "model" / "outputs" / "food_inv_intimacy_desire_alt"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    int_preds, int_folds = _loso_intimacy(actor_params_by_model)
    int_preds.to_csv(outputs_dir / "cv_preds_summary.csv", index=False)
    int_folds.to_csv(outputs_dir / "cv_folds.csv", index=False)
    print(f"Wrote {outputs_dir / 'cv_preds_summary.csv'}")
    print(f"Wrote {outputs_dir / 'cv_folds.csv'}")

    print("\n=== Per-variant summary ===")
    for variant, sub in int_folds.groupby("variant"):
        print(
            f"  {variant}: α_obs = {sub['alpha_observer'].mean():.3f} ± {sub['alpha_observer'].std():.3f}, "
            f"mean test NLL/trial = {(sub['test_nll'] / sub['n_test']).mean():.4f}"
        )


def main_desire_alt():
    print("=" * 60)
    print("LOSO CV: food_inv_desire_intimacy_alt")
    print("=" * 60)

    actor_params_by_model = load_fitted_params()
    outputs_dir = get_project_root() / "model" / "outputs" / "food_inv_desire_intimacy_alt"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    des_preds, des_folds = _loso_desire(actor_params_by_model)
    des_preds.to_csv(outputs_dir / "cv_preds_summary.csv", index=False)
    des_folds.to_csv(outputs_dir / "cv_folds.csv", index=False)
    print(f"Wrote {outputs_dir / 'cv_preds_summary.csv'}")
    print(f"Wrote {outputs_dir / 'cv_folds.csv'}")

    print("\n=== Per-variant summary ===")
    for variant, sub in des_folds.groupby("variant"):
        print(
            f"  {variant}: α_obs = {sub['alpha_observer'].mean():.3f} ± {sub['alpha_observer'].std():.3f}, "
            f"mean test NLL/trial = {(sub['test_nll'] / sub['n_test']).mean():.4f}"
        )
