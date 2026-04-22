"""
Leave-one-scenario-out CV for the no-alternatives-shown inverse-planning
experiment (intimacy inference — Exp 2c variant).

For each of the 16 scenarios, hold it out, refit α_observer on the remaining
15 scenarios (actor weights frozen at the all-data Exp 1 fit, as always), and
generate predictions for the held-out scenario using the refit α_observer.

Runs over the three padded-observer variants (access_full_padded, access_only_padded,
no_access_padded). These use the uniform-over-valid-slots prior baked into the
padded observer architecture — not the LM action prior (which only applies to
the canonical-4-action variants).

Output (in model/outputs/):
  - cv_loso_inv_plan_intimacy_noalt_preds_summary.csv
    One row per (scenario, observed_action, motivation, model), same schema
    as inv_plan_intimacy_noalt_preds_summary.csv; `expected_intimacy` from
    the LOSO-refit observer for that scenario's held-out fold.
  - cv_loso_inverse_noalt_folds.csv
    Per-fold fitted α_observer and train/test NLL, per variant.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))

import jax.numpy as jnp
import numpy as np
import pandas as pd

from fit_inverse_planning import compute_intimacy_nll, load_fitted_params
from fit_inverse_planning_noalt import (
    PADDED_VARIANTS,
    fit_padded_alpha_observer,
    load_intimacy_noalt_data,
)
from model_utils import IntimacyLevels, SCENARIO_LABELS, load_padded_lm_tables, padded_slots

from utils import get_project_root


N_SCENARIOS = len(SCENARIO_LABELS)


def _held_out_test_nll(result, action, scenario_idx, reward_condition, response, test_idx):
    """Per-trial NLL on held-out trials. result shape: (padded_slot, scenario,
    observed_action, intimacy_levels, reward_condition). Canonical observed
    action sits at padded_slot=0."""
    total = 0.0
    for i in test_idx:
        post = np.asarray(result[0, int(scenario_idx[i]), int(action[i]), :, int(reward_condition[i])])
        resp_idx = int(np.clip(round(float(response[i])), 0, 100))
        prob = max(post[resp_idx], 1e-8)
        total += -float(np.log(prob))
    return total


def run_loso_noalt(actor_params_by_model, padded_tables):
    data, observed_action, reward_condition, response, scenario_idx = (
        load_intimacy_noalt_data()
    )
    scenario_idx_np = np.asarray(scenario_idx)

    pred_rows = []
    fold_rows = []

    intimacy_grid = np.asarray(IntimacyLevels) * 100.0  # 0..100 scale

    for variant, (obs_fn, kw_names) in PADDED_VARIANTS.items():
        if variant not in actor_params_by_model:
            print(f"  (skipping {variant}: no forward fit)")
            continue
        actor_params = actor_params_by_model[variant]
        actor_kwargs = {k: actor_params[k] for k in kw_names}

        for fold in range(N_SCENARIOS):
            scenario_label = SCENARIO_LABELS[fold]
            train_mask = scenario_idx_np != fold
            test_mask = scenario_idx_np == fold
            n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
            if n_test == 0:
                print(f"  skipping {variant} fold {fold} ({scenario_label}): no test trials")
                continue
            print(
                f"  {variant} / fold {fold + 1}/{N_SCENARIOS} "
                f"({scenario_label}): train={n_train}, test={n_test}"
            )

            alpha_obs, train_nll = fit_padded_alpha_observer(
                observer_fn=obs_fn,
                actor_params=actor_params,
                actor_kwarg_names=kw_names,
                observed_action=observed_action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                reward_condition=reward_condition[train_mask],
                response=response[train_mask],
                access_table=padded_tables["access"],
                effort_table=padded_tables["effort"],
                is_share_table=padded_tables["is_share"],
                prior_table=padded_tables["prior"],
                verbose=False,
            )

            result = obs_fn(
                **actor_kwargs,
                alpha_observer=alpha_obs,
                access_table=padded_tables["access"],
                effort_table=padded_tables["effort"],
                is_share_table=padded_tables["is_share"],
                prior_table=padded_tables["prior"],
            )
            # Shape: (padded_slot, scenario, observed_action, intimacy_levels, reward_condition)
            # Slot 0 is always the observed canonical action.
            held_out = np.asarray(result[0, fold, :, :, :])  # (observed_action, intimacy, reward)

            for obs_a in range(4):
                for r in [0, 1]:
                    density = held_out[obs_a, :, r]
                    expected_intimacy = float(np.sum(intimacy_grid * density))
                    pred_rows.append({
                        "scenario_label": scenario_label,
                        "observed_action": obs_a,
                        "motivation": "low" if r == 0 else "high",
                        "model": f"{variant}_padded",
                        "expected_intimacy": expected_intimacy,
                    })

            test_idx = np.where(test_mask)[0]
            test_nll = _held_out_test_nll(
                result, observed_action, scenario_idx, reward_condition, response, test_idx
            )

            fold_rows.append({
                "experiment": "intimacy_noalt",
                "variant": f"{variant}_padded",
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
    print("LOSO cross-validation: no-alt inverse planning (padded observer)")
    print("=" * 60)

    print("\nLoading frozen actor parameters (all-data Exp 1 fit)...")
    actor_params_by_model = load_fitted_params()

    padded = load_padded_lm_tables()
    if padded is None:
        print("ERROR: lm_alternatives.csv or lm_alternatives_features.csv missing.")
        sys.exit(1)
    print(f"  padded access shape: {padded['access'].shape}")

    preds_df, fold_df = run_loso_noalt(actor_params_by_model, padded)

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    preds_path = output_dir / "cv_loso_inv_plan_intimacy_noalt_preds_summary.csv"
    fold_path = output_dir / "cv_loso_inverse_noalt_folds.csv"
    preds_df.to_csv(preds_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    print(f"\nWrote {preds_path}")
    print(f"Wrote {fold_path}")

    print("\n=== Per-variant summary ===")
    for variant, sub in fold_df.groupby("variant"):
        print(
            f"  {variant}: "
            f"α_obs = {sub['alpha_observer'].mean():.3f} ± {sub['alpha_observer'].std():.3f}, "
            f"mean test NLL/trial = {(sub['test_nll'] / sub['n_test']).mean():.4f}"
        )


if __name__ == "__main__":
    main()
