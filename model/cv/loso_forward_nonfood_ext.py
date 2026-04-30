"""
LOSO CV for forward-planning extensions (food and non-food). The file
name keeps the "_nonfood_ext" suffix from when this only supported
non-food, but it now accepts --domain food|nonfood.

Currently: only `access_full_gamma` (Full + power-law intimacy).

For each of the 16 scenarios in the chosen domain, hold it out, fit the
gamma variant on the remaining 15 scenarios, and predict the held-out
scenario's human action probabilities. Mirrors `cv/loso_forward.py` but
on the extension variant only — does NOT refit the canonical
access_full / access_only / no_access (those are produced by the existing
`cv/loso_forward.py`).

Outputs (in model/outputs/):
  - food:    cv_loso_forward_ext.csv,         cv_loso_preds_ext.csv
  - nonfood: cv_loso_forward_nonfood_ext.csv, cv_loso_preds_nonfood_ext.csv
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))

import numpy as np
import pandas as pd
from fit_forward_planning import compute_nll, load_data
from fit_forward_planning_nonfood_ext import (
    fit_access_full_gamma_alpha_model,
    fit_access_full_gamma_model,
    fit_access_full_gamma_vpow_model,
    fit_access_full_typed_gamma_model,
    predict_access_full_gamma,
    predict_access_full_gamma_vpow,
    predict_access_full_typed_gamma,
)
from model_utils import load_domain_assets, load_lm_v
from scipy import stats

from utils import get_project_root


def variants_for_domain(domain: str):
    v = {
        "access_full_gamma": (
            fit_access_full_gamma_model,
            predict_access_full_gamma,
            ["w_v", "w_d", "w_e", "gamma"],
        ),
    }
    if domain == "nonfood":
        v["access_full_typed_gamma"] = (
            fit_access_full_typed_gamma_model,
            predict_access_full_typed_gamma,
            ["w_v", "w_d_substance", "w_d_space", "w_d_privacy", "w_e", "gamma"],
        )
        v["access_full_gamma_alpha"] = (
            fit_access_full_gamma_alpha_model,
            predict_access_full_gamma,
            ["w_v", "w_d", "w_e", "gamma"],
        )
        v["access_full_gamma_vpow"] = (
            fit_access_full_gamma_vpow_model,
            predict_access_full_gamma_vpow,
            ["w_v", "w_d", "w_e", "gamma", "beta"],
        )
    return v


def _cell_mean_r(pred_df, variant, scenario=None):
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


def run_loso(domain: str = "nonfood"):
    if domain not in ("food", "nonfood"):
        raise ValueError(f"Unknown domain: {domain!r} (expected 'food' or 'nonfood')")

    scenario_labels, scenario_to_idx, llm_tables = load_domain_assets(domain)
    n_scenarios = len(scenario_labels)
    v_table = load_lm_v(domain)
    variants = variants_for_domain(domain)

    if domain == "food":
        data_path = get_project_root() / "data" / "forw_plan" / "main_trials_long.csv"
    else:
        data_path = get_project_root() / "data" / "nonfood_forw_plan" / "main_trials_long.csv"

    data, intimacy, reward_condition, action, p_action, scenario_idx = load_data(
        filepath=data_path, scenario_to_idx=scenario_to_idx,
    )
    tables = (llm_tables["access"], llm_tables["effort"], v_table)
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

        for variant, (fit_fn, pred_fn, param_names) in variants.items():
            print(f"  Fitting {variant}...")
            params, train_nll = fit_fn(*train_args, tables, verbose=False)
            test_preds = pred_fn(*test_args, *params, *tables)
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
    rs = []
    for _, row in fold_df.iterrows():
        rs.append(_cell_mean_r(pred_df, row["variant"], scenario=row["held_out_scenario"]))
    fold_df = fold_df.copy()
    fold_df["test_cell_r"] = rs
    return fold_df


def main(domain: str = "nonfood"):
    print("=" * 60)
    print(f"LOSO cross-validation: forward-planning extensions (domain={domain})")
    print("=" * 60)

    fold_df, pred_df = run_loso(domain=domain)
    fold_df = attach_per_scenario_r(fold_df, pred_df)

    output_dir = get_project_root() / "model" / "outputs"
    output_dir.mkdir(exist_ok=True)
    if domain == "food":
        fold_path = output_dir / "cv_loso_forward_ext.csv"
        pred_path = output_dir / "cv_loso_preds_ext.csv"
    else:
        fold_path = output_dir / "cv_loso_forward_nonfood_ext.csv"
        pred_path = output_dir / "cv_loso_preds_nonfood_ext.csv"
    fold_df.to_csv(fold_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved fold-level results to {fold_path}")
    print(f"Saved per-trial predictions to {pred_path}")

    print("\n=== Pooled out-of-sample Pearson r (cell means across 16 folds) ===")
    for variant in variants_for_domain(domain):
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
    parser = argparse.ArgumentParser(
        description="LOSO CV for forward-planning extensions (currently: access_full_gamma) on food or non-food."
    )
    parser.add_argument(
        "--domain", choices=("food", "nonfood"), default="nonfood",
        help="Which experiment to CV: 'food' (writes *_ext.csv) or 'nonfood' (writes *_nonfood_ext.csv, default).",
    )
    args = parser.parse_args()
    main(domain=args.domain)
