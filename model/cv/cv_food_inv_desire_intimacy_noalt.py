"""
Leave-one-scenario-out CV for the no-alternatives-shown desire (reward) inference
experiment, using the JOINT fit (all actor weights + α_observer refit per fold).

Mirrors cv_food_inv_intimacy_desire_noalt.py but for the relationship-keyed desire-noalt
pipeline. For each of the 16 scenarios, hold it out, jointly fit all actor
weights + α_observer on the remaining 15 scenarios, and emit predictions for
the held-out scenario using the refit weights.

Runs over the three relationship-keyed padded-observer variants
(full_padded_rel, discomfort_only_padded_rel, base_padded_rel), via
the PADDED_VARIANTS registry in inverse/fit_food_inv_desire_intimacy_noalt.

Output (in model/outputs/):
  - food_inv_desire_intimacy_noalt/cv_preds_summary.csv
    One row per (scenario, observed_action, intimacy, model). Same shape as
    food_inv_desire_intimacy_noalt_preds_summary.csv; `p_high` is what the slider
    response 0-100 encodes (P(motivation = HIGH)).
  - cv_folds.csv
    Per-fold fitted weights + α_observer + train/test NLL, per variant.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import numpy as np
import pandas as pd

from _helpers import (  # noqa: E402
    PADDED_VARIANTS_REWARD as PADDED_VARIANTS,
    fit_padded_joint_desire as fit_padded_joint_model,
    load_desire_noalt_data,
)
from tables import SCENARIO_LABELS, load_padded_lm_tables_relationship  # noqa: E402

from utils import get_project_root


N_SCENARIOS = len(SCENARIO_LABELS)
N_RELATIONSHIPS = 4  # 0, 50, 75, 100 → enum indices 0..3
RELATIONSHIP_LABELS = [0, 50, 75, 100]
EXPERIMENT_SLUG = "food_inv_desire_intimacy_noalt"


def _bce(p_high, response_pct):
    """Per-trial binary cross-entropy NLL. Mirrors compute_reward_nll."""
    eps = 1e-8
    p_human = response_pct / 100.0
    p_model = max(min(float(p_high), 1.0 - eps), eps)
    return -(p_human * np.log(p_model) + (1 - p_human) * np.log(1 - p_model))


def _held_out_test_nll(result, observed_action, scenario_idx, relationship_idx, response, test_idx):
    """Per-trial BCE NLL on held-out trials; sum. result shape:
    (padded_slot, scenario, observed_action, relationship, reward_condition).
    Canonical observed action sits at padded_slot=0; reward index 1 = HIGH."""
    total = 0.0
    for i in test_idx:
        p_high = float(result[
            0, int(scenario_idx[i]), int(observed_action[i]), int(relationship_idx[i]), 1
        ])
        total += _bce(p_high, float(response[i]))
    return total


def _build_observer_table(observer_fn, utility_names, uses_v, params, padded):
    """Reconstruct the observer table from fitted joint params
    (utility weights then α_observer, matching fit_padded_joint_model)."""
    actor_kwargs = {"alpha": 1.0}
    for i, name in enumerate(utility_names):
        actor_kwargs[name] = float(params[i])
    table_kwargs = dict(
        access_table=padded["access"],
        effort_table=padded["effort"],
        prior_table=padded["prior"],
    )
    if uses_v:
        table_kwargs["v_padded_table"] = padded["v"]
    return observer_fn(
        **actor_kwargs,
        alpha_observer=float(params[-1]),
        **table_kwargs,
    )


def run_loso_desire_noalt_joint(padded):
    data, observed_action, relationship_idx, response, scenario_idx = (
        load_desire_noalt_data()
    )
    scenario_idx_np = np.asarray(scenario_idx)

    pred_rows = []
    fold_rows = []

    for variant, (observer_fn, utility_names, uses_v) in PADDED_VARIANTS.items():
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

            params, train_nll = fit_padded_joint_model(
                observer_fn=observer_fn,
                utility_param_names=utility_names,
                observed_action=observed_action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                relationship_condition=relationship_idx[train_mask],
                response=response[train_mask],
                access_table=padded["access"],
                effort_table=padded["effort"],
                prior_table=padded["prior"],
                v_padded_table=padded["v"] if uses_v else None,
                verbose=False,
            )

            result = _build_observer_table(observer_fn, utility_names, uses_v, params, padded)
            # Shape: (padded_slot, scenario, observed_action, relationship, reward)
            held_out = np.asarray(result[0, fold, :, :, :])  # (observed_action, relationship, reward)

            for obs_a in range(4):
                for r_idx in range(N_RELATIONSHIPS):
                    p_low = float(held_out[obs_a, r_idx, 0])
                    p_high = float(held_out[obs_a, r_idx, 1])
                    pred_rows.append({
                        "scenario_label": scenario_label,
                        "observed_action": obs_a,
                        "intimacy": RELATIONSHIP_LABELS[r_idx],
                        "model": f"{variant}_padded",
                        "p_low": p_low,
                        "p_high": p_high,
                    })

            test_idx = np.where(test_mask)[0]
            test_nll = _held_out_test_nll(
                result, observed_action, scenario_idx, relationship_idx, response, test_idx
            )

            fold_row = {
                "experiment": EXPERIMENT_SLUG,
                "variant": f"{variant}_padded",
                "fold": fold,
                "held_out_scenario": scenario_label,
                "alpha_observer": float(params[-1]),
                "train_nll": float(train_nll),
                "test_nll": test_nll,
                "n_train": n_train,
                "n_test": n_test,
            }
            for i, name in enumerate(utility_names):
                fold_row[f"param_{name}"] = float(params[i])
            fold_rows.append(fold_row)

    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def main():
    print("=" * 60)
    print("LOSO CV: no-alt desire inference (joint fit, relationship-keyed)")
    print("=" * 60)

    padded = load_padded_lm_tables_relationship()
    if padded is None:
        print(
            "ERROR: relationship-keyed LM tables missing. Run "
            "lm/generate_alternatives_relationship.py and the "
            "two relationship feature/V passes first."
        )
        sys.exit(1)
    print(f"  padded access shape: {padded['access'].shape}")
    print(f"  padded v shape: {padded['v'].shape}")

    preds_df, fold_df = run_loso_desire_noalt_joint(padded)

    output_dir = get_project_root() / "model" / "outputs" / "legacy" / "food_inv_desire_intimacy_noalt"
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "cv_preds_summary.csv"
    fold_path = output_dir / "cv_folds.csv"
    preds_df.to_csv(preds_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    print(f"\nWrote {preds_path}")
    print(f"Wrote {fold_path}")

    print("\n=== Per-variant LOSO summary (joint fit) ===")
    for variant, sub in fold_df.groupby("variant"):
        alpha_obs_str = f"{sub['alpha_observer'].mean():.3f} ± {sub['alpha_observer'].std():.3f}"
        mean_test_nll = (sub['test_nll'] / sub['n_test']).mean()
        utility_cols = [c for c in sub.columns if c.startswith("param_")]
        param_summaries = []
        for c in utility_cols:
            vals = sub[c].dropna()
            if len(vals):
                param_summaries.append(f"{c.replace('param_', '')}={vals.mean():.2f}±{vals.std():.2f}")
        print(f"  {variant}: α_obs = {alpha_obs_str}, mean test BCE/trial = {mean_test_nll:.4f}")
        if param_summaries:
            print(f"    utility weights: {', '.join(param_summaries)}")


if __name__ == "__main__":
    main()
