"""
Leave-one-scenario-out CV for forward-planning experiments — unified driver.

Holds out each of the 16 scenarios in turn, refits the three ablations
(full / discomfort_only / base) on the remaining 15, and predicts the
held-out scenario's human action probabilities.

Reports per-fold fitted parameters, train/test NLL, and Pearson r at the
(intimacy, IV, action) cell-mean level — both per held-out scenario and
pooled across folds. The pooled metric is directly comparable to the
in-sample r reported in the corresponding fit_results CSV.

Three experiments are dispatched by `--experiment <slug>`:
  - food_forw_intimacy_desire: canonical 4-action × motivation IV (food)
  - nonfood_forw_intimacy_desire: canonical 4-action × motivation IV (nonfood)
  - food_forw_intimacy_effort: 2-action × effort IV (food)

Outputs (in model/outputs/) preserve the pre-existing filename convention
and will be reorganized in a later pass (Step 7 of the model/ refactor).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(_project_root / "model" / "forward"))

from _shared import (  # noqa: E402
    compute_nll,
    fit_canonical_base,
    fit_canonical_discomfort_only,
    fit_canonical_full,
    fit_effort_base,
    fit_effort_discomfort_only,
    fit_effort_full,
    load_data_canonical,
    load_data_effort,
    predict_canonical_base,
    predict_canonical_discomfort_only,
    predict_canonical_full,
    predict_effort_base,
    predict_effort_discomfort_only,
    predict_effort_full,
)
from tables import (
    LLM_TABLES,
    LLM_TABLES_EFFORT,
    SCENARIO_LABELS,
    load_domain_assets,
    load_lm_v,
)

from utils import get_project_root


def _cell_mean_r(pred_df, variant, group_cols, scenario=None):
    """Pearson r between model and human p_action at cell-mean level. If
    `scenario` is given, restrict to that held-out scenario; otherwise pool
    across all folds."""
    sub = pred_df[pred_df["variant"] == variant]
    if scenario is not None:
        sub = sub[sub["held_out_scenario"] == scenario]
    cell = (
        sub.groupby(group_cols)
        .agg(humans=("p_action", "mean"), model=("p_action_pred", "mean"))
        .reset_index()
    )
    if len(cell) < 3 or cell["model"].std() == 0:
        return np.nan
    r, _ = stats.pearsonr(cell["model"], cell["humans"])
    return float(r)


# ==============================================================================
# Experiment registry
# ==============================================================================
# For each slug: scenario labels, data path, data loader, tables_per_variant,
# fit + predict + param-name registries, the per-trial IV column name (used
# in pred_rows), and the cell-mean group columns.


def _canonical_config(domain: str):
    scenario_labels, scenario_to_idx, llm_tables = load_domain_assets(domain)
    v_table = load_lm_v(domain)
    if domain == "food":
        data_path = get_project_root() / "data" / "food_forw_intimacy_desire" / "main_trials_long.csv"
        slug = "food_forw_intimacy_desire"; fold_filename = "cv_folds.csv"
        pred_filename = "cv_preds.csv"
    elif domain == "nonfood":
        data_path = get_project_root() / "data" / "nonfood_forw_intimacy_desire" / "main_trials_long.csv"
        slug = "nonfood_forw_intimacy_desire"; fold_filename = "cv_folds.csv"
        pred_filename = "cv_preds.csv"
    else:
        raise ValueError(f"Unknown domain: {domain!r}")

    tables = (llm_tables["access"], llm_tables["effort"], v_table)
    return {
        "scenario_labels": scenario_labels,
        "data_path": data_path,
        "data_loader": lambda fp: load_data_canonical(fp, scenario_to_idx),
        "tables_per_variant": {
            "full": tables,
            "discomfort_only": tables[:2],
            "base": tables,
        },
        "variants": {
            "full": (fit_canonical_full, predict_canonical_full, ["w_v", "w_d", "w_e", "gamma"]),
            "discomfort_only": (fit_canonical_discomfort_only, predict_canonical_discomfort_only, ["w_d", "gamma"]),
            "base": (fit_canonical_base, predict_canonical_base, ["w_v", "w_e"]),
        },
        "iv_column": "motivation",       # column name in human data CSV
        "iv_dtype": "object",            # leave as string ("low"/"high")
        "group_cols": ["intimacy", "motivation", "action"],
        "slug": slug,
        "fold_filename": fold_filename,
        "pred_filename": pred_filename,
    }


def _effort_config():
    tables = (LLM_TABLES_EFFORT["access"], LLM_TABLES_EFFORT["effort"])
    return {
        "scenario_labels": SCENARIO_LABELS,
        "data_path": get_project_root() / "data" / "food_forw_intimacy_effort" / "main_trials_long.csv",
        "data_loader": lambda fp: load_data_effort(fp),
        "tables_per_variant": {
            "full": tables,
            "discomfort_only": tables,
            "base": tables,
        },
        "variants": {
            "full": (fit_effort_full, predict_effort_full, ["w_v", "w_d", "w_e", "gamma"]),
            "discomfort_only": (fit_effort_discomfort_only, predict_effort_discomfort_only, ["w_d", "gamma"]),
            "base": (fit_effort_base, predict_effort_base, ["w_v", "w_e"]),
        },
        "iv_column": "effort",
        "iv_dtype": "object",            # "low" / "high"
        "group_cols": ["intimacy", "effort", "action"],
        "slug": "food_forw_intimacy_effort", "fold_filename": "cv_folds.csv",
        "pred_filename": "cv_preds.csv",
    }


EXPERIMENT_CONFIGS = {
    "food_forw_intimacy_desire": lambda: _canonical_config("food"),
    "nonfood_forw_intimacy_desire": lambda: _canonical_config("nonfood"),
    "food_forw_intimacy_effort": _effort_config,
}


# ==============================================================================
# CV core
# ==============================================================================


def run_loso(config: dict):
    scenario_labels = config["scenario_labels"]
    n_scenarios = len(scenario_labels)
    data, intimacy, condition_iv, action, p_action, scenario_idx = config["data_loader"](config["data_path"])
    scenario_idx_np = np.asarray(scenario_idx)

    fold_rows = []
    pred_rows = []

    for fold in range(n_scenarios):
        scenario_label = scenario_labels[fold]
        train_mask = scenario_idx_np != fold
        test_mask = scenario_idx_np == fold
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        print(f"\n=== Fold {fold + 1}/{n_scenarios} (holding out '{scenario_label}') ===")
        print(f"  train trials: {n_train}, test trials: {n_test}")

        train_args = (
            intimacy[train_mask],
            condition_iv[train_mask],
            action[train_mask],
            scenario_idx[train_mask],
            p_action[train_mask],
        )
        test_args = (
            intimacy[test_mask],
            condition_iv[test_mask],
            action[test_mask],
            scenario_idx[test_mask],
        )

        for variant, (fit_fn, pred_fn, param_names) in config["variants"].items():
            print(f"  Fitting {variant}...")
            tab = config["tables_per_variant"][variant]
            params, train_nll = fit_fn(*train_args, tab, verbose=False)
            test_preds = pred_fn(*test_args, *params, *tab)
            test_nll = float(compute_nll(test_preds, p_action[test_mask]))

            fold_rows.append({
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
            })

            test_preds_np = np.asarray(test_preds)
            test_idx = np.where(test_mask)[0]
            iv_col = config["iv_column"]
            for i, trial_idx in enumerate(test_idx):
                pred_rows.append({
                    "fold": fold,
                    "held_out_scenario": scenario_label,
                    "variant": variant,
                    "subject_id": data["subject_id"].iloc[trial_idx],
                    "intimacy": int(data["intimacy"].iloc[trial_idx]),
                    iv_col: data[iv_col].iloc[trial_idx],
                    "action": int(data["action"].iloc[trial_idx]),
                    "p_action": float(p_action[trial_idx]),
                    "p_action_pred": float(test_preds_np[i]),
                })

    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def attach_per_scenario_r(fold_df, pred_df, group_cols):
    rs = []
    for _, row in fold_df.iterrows():
        rs.append(_cell_mean_r(pred_df, row["variant"], group_cols, scenario=row["held_out_scenario"]))
    fold_df = fold_df.copy()
    fold_df["test_cell_r"] = rs
    return fold_df


def main(experiment: str):
    if experiment not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment: {experiment!r}. "
            f"Choose from {list(EXPERIMENT_CONFIGS)}"
        )

    print("=" * 60)
    print(f"LOSO cross-validation: forward planning (experiment={experiment})")
    print("=" * 60)

    config = EXPERIMENT_CONFIGS[experiment]()
    fold_df, pred_df = run_loso(config)
    fold_df = attach_per_scenario_r(fold_df, pred_df, config["group_cols"])

    output_dir = get_project_root() / "model" / "outputs" / config["slug"]
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / config["fold_filename"]
    pred_path = output_dir / config["pred_filename"]
    fold_df.to_csv(fold_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved fold-level results to {fold_path}")
    print(f"Saved per-trial predictions to {pred_path}")

    print("\n=== Pooled out-of-sample Pearson r (cell means across 16 folds) ===")
    for variant in config["variants"]:
        r = _cell_mean_r(pred_df, variant, config["group_cols"])
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
        "--experiment",
        choices=tuple(EXPERIMENT_CONFIGS),
        default="food_forw_intimacy_desire",
        help="Which forward-planning experiment to CV.",
    )
    args = parser.parse_args()
    main(args.experiment)
