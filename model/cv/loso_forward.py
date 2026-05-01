"""
Leave-one-scenario-out CV for the forward-planning canonical variants.

For each of the 16 scenarios, hold it out, fit the three uniform-prior
variants (full, discomfort_only, base) on the remaining 15
scenarios, and predict the held-out scenario's human action probabilities.

Reports per-fold fitted parameters, train/test NLL, and Pearson r at the
(intimacy, motivation, action) cell-mean level — both per held-out scenario
and pooled across folds. The pooled metric is directly comparable to the
in-sample r reported in forward_planning_fit_results.csv.

Outputs (in model/outputs/):
  - cv_loso_forward.csv — one row per (fold, variant) with fitted params + metrics
  - cv_loso_preds.csv   — per-trial predictions across all 16 held-out folds
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))

import numpy as np
import pandas as pd
from scipy import stats

from fit_forward_planning import (
    compute_nll,
    fit_full_model,
    fit_discomfort_only_model,
    fit_base_model,
    load_data,
    predict_full,
    predict_discomfort_only,
    predict_base,
)
from tables import LLM_TABLES, SCENARIO_LABELS, load_domain_assets, load_lm_v

from utils import get_project_root


N_SCENARIOS = len(SCENARIO_LABELS)

# variant_name -> (fit_fn, predict_fn, utility_param_names). All three variants
# now take LM-V as canonical V; discomfort_only is V-independent and uses only
# (access, effort) but the dispatcher still gives it the trimmed tables tuple.
VARIANTS = {
    "full": (
        fit_full_model,
        predict_full,
        ["w_v", "w_d", "w_e", "gamma"],
    ),
    "discomfort_only": (
        fit_discomfort_only_model,
        predict_discomfort_only,
        ["w_d", "gamma"],
    ),
    "base": (
        fit_base_model,
        predict_base,
        ["w_v", "w_e"],
    ),
}


def _cell_mean_r(pred_df, variant, scenario=None):
    """Pearson r between model and human p_action at (intimacy, motivation,
    action) cell-means. If `scenario` is given, restrict to that held-out
    scenario; otherwise pool across all folds."""
    sub = pred_df[pred_df["variant"] == variant]
    if scenario is not None:
        sub = sub[sub["held_out_scenario"] == scenario]
    cell = (
        sub.groupby(["intimacy", "motivation", "action"])
        .agg(humans=("p_action", "mean"), model=("p_action_pred", "mean"))
        .reset_index()
    )
    if len(cell) < 3 or cell["model"].std() == 0:
        return np.nan
    r, _ = stats.pearsonr(cell["model"], cell["humans"])
    return float(r)


def run_loso(domain: str = "food"):
    scenario_labels, scenario_to_idx, llm_tables = load_domain_assets(domain)
    n_scenarios = len(scenario_labels)
    v_table = load_lm_v(domain)

    if domain == "food":
        data_path = get_project_root() / "data" / "food_forw_intimacy_desire" / "main_trials_long.csv"
    elif domain == "nonfood":
        data_path = get_project_root() / "data" / "nonfood_forw_intimacy_desire" / "main_trials_long.csv"
    else:
        raise ValueError(f"Unknown domain: {domain!r}")

    data, intimacy, reward_condition, action, p_action, scenario_idx = load_data(
        filepath=data_path, scenario_to_idx=scenario_to_idx,
    )
    tables = (llm_tables["access"], llm_tables["effort"], v_table)
    # discomfort_only is V-independent; the rest take the full 3-tuple.
    tables_per_variant = {
        "full": tables,
        "discomfort_only": tables[:2],
        "base":   tables,
    }
    scenario_idx_np = np.asarray(scenario_idx)

    fold_rows = []
    pred_rows = []

    for fold in range(n_scenarios):
        scenario_label = scenario_labels[fold]
        train_mask = scenario_idx_np != fold
        test_mask = scenario_idx_np == fold
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        print(
            f"\n=== Fold {fold + 1}/{n_scenarios} (holding out '{scenario_label}') ==="
        )
        print(f"  train trials: {n_train}, test trials: {n_test}")

        train_args = (
            intimacy[train_mask],
            reward_condition[train_mask],
            action[train_mask],
            scenario_idx[train_mask],
            p_action[train_mask],
        )
        test_args = (
            intimacy[test_mask],
            reward_condition[test_mask],
            action[test_mask],
            scenario_idx[test_mask],
        )

        for variant, (fit_fn, pred_fn, param_names) in VARIANTS.items():
            print(f"  Fitting {variant}...")
            tab = tables_per_variant[variant]
            params, train_nll = fit_fn(*train_args, tab, verbose=False)
            test_preds = pred_fn(*test_args, *params, *tab)
            test_nll = float(compute_nll(test_preds, p_action[test_mask]))

            fold_rows.append(
                {
                    "fold": fold,
                    "held_out_scenario": scenario_label,
                    "variant": variant,
                    "train_nll": float(train_nll),
                    "test_nll": test_nll,
                    "train_nll_per_trial": float(train_nll) / n_train,
                    "test_nll_per_trial": test_nll / n_test,
                    "n_train": n_train,
                    "n_test": n_test,
                    "param_alpha": float(params[0]),
                    **{
                        f"param_{pn}": float(params[i + 1])
                        for i, pn in enumerate(param_names)
                    },
                }
            )

            test_preds_np = np.asarray(test_preds)
            test_idx = np.where(test_mask)[0]
            for i, trial_idx in enumerate(test_idx):
                pred_rows.append(
                    {
                        "fold": fold,
                        "held_out_scenario": scenario_label,
                        "variant": variant,
                        "subject_id": data["subject_id"].iloc[trial_idx],
                        "intimacy": int(data["intimacy"].iloc[trial_idx]),
                        "motivation": data["motivation"].iloc[trial_idx],
                        "action": int(data["action"].iloc[trial_idx]),
                        "p_action": float(p_action[trial_idx]),
                        "p_action_pred": float(test_preds_np[i]),
                    }
                )

    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def attach_per_scenario_r(fold_df, pred_df):
    """Add `test_cell_r` column: per-fold (intimacy, motivation, action)
    cell-mean Pearson r on that fold's held-out scenario."""
    rs = []
    for _, row in fold_df.iterrows():
        rs.append(_cell_mean_r(pred_df, row["variant"], scenario=row["held_out_scenario"]))
    fold_df = fold_df.copy()
    fold_df["test_cell_r"] = rs
    return fold_df


def main(domain: str = "food"):
    print("=" * 60)
    print(f"LOSO cross-validation: forward planning (domain={domain})")
    print("=" * 60)

    fold_df, pred_df = run_loso(domain=domain)
    fold_df = attach_per_scenario_r(fold_df, pred_df)

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    if domain == "food":
        fold_filename = "cv_loso_forward.csv"
        pred_filename = "cv_loso_preds.csv"
    else:
        fold_filename = f"cv_loso_forward_{domain}.csv"
        pred_filename = f"cv_loso_preds_{domain}.csv"
    fold_path = output_dir / fold_filename
    pred_path = output_dir / pred_filename
    fold_df.to_csv(fold_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved fold-level results to {fold_path}")
    print(f"Saved per-trial predictions to {pred_path}")

    print("\n=== Pooled out-of-sample Pearson r (cell means across 16 folds) ===")
    for variant in VARIANTS:
        r = _cell_mean_r(pred_df, variant)
        sub = fold_df[fold_df["variant"] == variant]
        per_fold = sub["test_cell_r"].dropna()
        print(
            f"  {variant}: pooled r = {r:.3f}, "
            f"per-fold r = {per_fold.mean():.3f} ± {per_fold.std():.3f}, "
            f"mean test NLL/trial = {sub['test_nll_per_trial'].mean():.4f}"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LOSO CV for forward planning.")
    parser.add_argument(
        "--domain", choices=("food", "nonfood"), default="food",
        help="Which experiment to CV: 'food' (default) or 'nonfood'.",
    )
    args = parser.parse_args()
    main(domain=args.domain)
