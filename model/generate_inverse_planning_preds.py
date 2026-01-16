"""
Generate inverse planning predictions using frozen forward planning parameters.

This script generates predictions for two inverse planning experiments:
1. inv_plan_intimacy: Infer intimacy from observed action (given reward condition)
2. inv_plan_reward: Infer reward condition from observed action (given intimacy)

Uses parameters fitted from forward planning experiment (frozen, not re-fitted).
Generates predictions for both stipulated and LM-derived scenario parameters.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path

from model_utils import (
    # Constants
    actions,
    IntimacyLevels,
    RewardConditions,
    RelationshipConditions,
    SCENARIO_LABELS,
    SCENARIO_TO_IDX,

    # Stipulated observer models for intimacy inference
    observer_intimacy_full_model,
    observer_intimacy_vanilla_inv_plan,
    observer_intimacy_discomfort_only,

    # LM-based observer models for intimacy inference
    observer_intimacy_full_model_lm,
    observer_intimacy_vanilla_lm,
    observer_intimacy_discomfort_only_lm,

    # Stipulated observer models for reward inference
    observer_reward_full_model,
    observer_reward_vanilla_inv_plan,
    observer_reward_discomfort_only,

    # LM-based observer models for reward inference
    observer_reward_full_model_lm,
    observer_reward_vanilla_lm,
    observer_reward_discomfort_only_lm,
)


# ==============================================================================
# Load Fitted Parameters
# ==============================================================================

def load_fitted_params(filepath: str = "forward_planning_fit_results.csv") -> dict:
    """Load fitted parameters from forward planning fit results."""
    df = pd.read_csv(filepath)
    params = {}
    for _, row in df.iterrows():
        model_name = row["model"]
        params[model_name] = {
            "alpha": row["param_alpha"],
            "w_r": row.get("param_w_r", 0.0) if pd.notna(row.get("param_w_r")) else 0.0,
            "w_d": row["param_w_d"],
            "w_c": row.get("param_w_c", 0.0) if pd.notna(row.get("param_w_c")) else 0.0,
        }
    return params


def load_fitted_alpha_observer(filepath: str = "inverse_planning_fit_results.csv") -> dict:
    """Load fitted alpha_observer values from inverse planning fit results.

    Returns dict with (model, experiment) -> alpha_observer
    """
    df = pd.read_csv(filepath)
    alpha_obs = {}
    for _, row in df.iterrows():
        key = (row["model"], row["experiment"])
        alpha_obs[key] = row["alpha_observer"]
    return alpha_obs


# ==============================================================================
# Intimacy Inference Predictions
# ==============================================================================

def generate_intimacy_preds_stipulated(params: dict, model_name: str, alpha_observer: float = 1.0) -> pd.DataFrame:
    """Generate intimacy inference predictions using stipulated parameters.

    Returns DataFrame with posterior distribution over intimacy for each (action, reward_condition).
    """
    if model_name == "full":
        result = observer_intimacy_full_model(
            alpha=params["alpha"], w_r=params["w_r"], w_d=params["w_d"], w_c=params["w_c"],
            alpha_observer=alpha_observer
        )
    elif model_name == "vanilla":
        result = observer_intimacy_vanilla_inv_plan(
            alpha=params["alpha"], w_r=params["w_r"], w_d=params["w_d"], w_c=params["w_c"],
            alpha_observer=alpha_observer
        )
    elif model_name == "discomfort_only":
        result = observer_intimacy_discomfort_only(
            alpha=params["alpha"], w_r=params["w_r"], w_d=params["w_d"], w_c=params["w_c"],
            alpha_observer=alpha_observer
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Result is a JAX array with shape (actions, intimacy_levels, reward_conditions)
    # Convert to DataFrame
    data = []
    for a_idx, a in enumerate(actions):
        for r in [0, 1]:  # LOW, HIGH
            for i_idx, i in enumerate(IntimacyLevels):
                data.append({
                    "action": int(a),
                    "reward_condition": "low" if r == 0 else "high",
                    "intimacy": float(i),
                    "density": float(result[a_idx, i_idx, r]),
                })

    df = pd.DataFrame(data)
    df["model"] = model_name
    df["param_source"] = "stipulated"

    return df


def generate_intimacy_preds_lm(params: dict, model_name: str, alpha_observer: float = 1.0) -> pd.DataFrame:
    """Generate intimacy inference predictions using LM-derived parameters.

    Returns DataFrame with scenario-specific posterior distribution.
    """
    if model_name == "full":
        observer_fn = observer_intimacy_full_model_lm
    elif model_name == "vanilla":
        observer_fn = observer_intimacy_vanilla_lm
    elif model_name == "discomfort_only":
        observer_fn = observer_intimacy_discomfort_only_lm
    else:
        raise ValueError(f"Unknown model: {model_name}")

    dfs = []
    for scenario_label in SCENARIO_LABELS:
        scenario_idx = SCENARIO_TO_IDX[scenario_label]

        result = observer_fn(
            scenario_idx=scenario_idx,
            alpha=params["alpha"],
            w_r=params["w_r"],
            w_d=params["w_d"],
            w_c=params["w_c"],
            alpha_observer=alpha_observer,
        )

        # Convert to DataFrame
        data = []
        for a_idx, a in enumerate(actions):
            for r in [0, 1]:
                for i_idx, i in enumerate(IntimacyLevels):
                    data.append({
                        "scenario_label": scenario_label,
                        "action": int(a),
                        "reward_condition": "low" if r == 0 else "high",
                        "intimacy": float(i),
                        "density": float(result[a_idx, i_idx, r]),
                    })

        dfs.append(pd.DataFrame(data))

    df = pd.concat(dfs, ignore_index=True)
    df["model"] = model_name
    df["param_source"] = "lm"

    return df


# ==============================================================================
# Reward Inference Predictions
# ==============================================================================

def generate_reward_preds_stipulated(params: dict, model_name: str, alpha_observer: float = 1.0) -> pd.DataFrame:
    """Generate reward inference predictions using stipulated parameters.

    Returns DataFrame with P(reward_condition | action, intimacy_condition).
    """
    if model_name == "full":
        result = observer_reward_full_model(
            alpha=params["alpha"], w_r=params["w_r"], w_d=params["w_d"], w_c=params["w_c"],
            alpha_observer=alpha_observer
        )
    elif model_name == "vanilla":
        result = observer_reward_vanilla_inv_plan(
            alpha=params["alpha"], w_r=params["w_r"], w_d=params["w_d"], w_c=params["w_c"],
            alpha_observer=alpha_observer
        )
    elif model_name == "discomfort_only":
        result = observer_reward_discomfort_only(
            alpha=params["alpha"], w_r=params["w_r"], w_d=params["w_d"], w_c=params["w_c"],
            alpha_observer=alpha_observer
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Result is a JAX array with shape (actions, relationship_conditions, reward_conditions)
    # Convert to DataFrame
    intimacy_map = {0: 0, 1: 50, 2: 75, 3: 100}
    data = []
    for a_idx, a in enumerate(actions):
        for rel_idx in range(4):
            for r in [0, 1]:
                data.append({
                    "action": int(a),
                    "intimacy_condition": intimacy_map[rel_idx],
                    "reward_condition": "low" if r == 0 else "high",
                    "density": float(result[a_idx, rel_idx, r]),
                })

    df = pd.DataFrame(data)
    df["model"] = model_name
    df["param_source"] = "stipulated"

    return df


def generate_reward_preds_lm(params: dict, model_name: str, alpha_observer: float = 1.0) -> pd.DataFrame:
    """Generate reward inference predictions using LM-derived parameters.

    Returns DataFrame with scenario-specific P(reward_condition | action, intimacy_condition).
    """
    if model_name == "full":
        observer_fn = observer_reward_full_model_lm
    elif model_name == "vanilla":
        observer_fn = observer_reward_vanilla_lm
    elif model_name == "discomfort_only":
        observer_fn = observer_reward_discomfort_only_lm
    else:
        raise ValueError(f"Unknown model: {model_name}")

    intimacy_map = {0: 0, 1: 50, 2: 75, 3: 100}
    dfs = []
    for scenario_label in SCENARIO_LABELS:
        scenario_idx = SCENARIO_TO_IDX[scenario_label]

        result = observer_fn(
            scenario_idx=scenario_idx,
            alpha=params["alpha"],
            w_r=params["w_r"],
            w_d=params["w_d"],
            w_c=params["w_c"],
            alpha_observer=alpha_observer,
        )

        # Convert to DataFrame
        data = []
        for a_idx, a in enumerate(actions):
            for rel_idx in range(4):
                for r in [0, 1]:
                    data.append({
                        "scenario_label": scenario_label,
                        "action": int(a),
                        "intimacy_condition": intimacy_map[rel_idx],
                        "reward_condition": "low" if r == 0 else "high",
                        "density": float(result[a_idx, rel_idx, r]),
                    })

        dfs.append(pd.DataFrame(data))

    df = pd.concat(dfs, ignore_index=True)
    df["model"] = model_name
    df["param_source"] = "lm"

    return df


# ==============================================================================
# Summary Statistics
# ==============================================================================

def compute_expected_intimacy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute expected value of intimacy from posterior distribution."""
    df = df.copy()
    df["intimacy_scaled"] = df["intimacy"] * 100  # Convert to 0-100 scale

    # Group by condition and compute expected value
    summary = df.groupby(
        ["scenario_label", "action", "reward_condition", "model", "param_source"],
        dropna=False
    ).apply(
        lambda g: pd.Series({
            "expected_intimacy": (g["intimacy_scaled"] * g["density"]).sum(),
        })
    ).reset_index()

    return summary


def compute_p_high_reward(df: pd.DataFrame) -> pd.DataFrame:
    """Extract P(high reward) from posterior distribution."""
    # Filter to just high reward condition
    df_high = df[df["reward_condition"] == "high"].copy()
    df_high = df_high.rename(columns={"density": "p_high_reward"})
    # Scale to 0-100 for comparison with slider responses
    df_high["p_high_reward"] = df_high["p_high_reward"] * 100
    df_high = df_high.drop(columns=["reward_condition"])

    return df_high


# ==============================================================================
# Main Script
# ==============================================================================

def main():
    print("=" * 60)
    print("Generating Inverse Planning Predictions")
    print("Using Frozen Forward Planning Parameters + Fitted alpha_observer")
    print("=" * 60)

    output_dir = Path(__file__).parent

    # Load fitted actor parameters
    print("\nLoading fitted actor parameters...")
    params = load_fitted_params()
    for model_name, p in params.items():
        print(f"  {model_name}: alpha={p['alpha']:.3f}, w_r={p['w_r']:.3f}, w_d={p['w_d']:.3f}, w_c={p['w_c']:.3f}")

    # Load fitted alpha_observer values
    print("\nLoading fitted alpha_observer values...")
    alpha_obs = load_fitted_alpha_observer()
    for (model, exp), alpha in alpha_obs.items():
        print(f"  {model} ({exp}): alpha_observer={alpha:.3f}")

    # Note: Empirical priors are loaded in model_utils.py at module init
    print("\nEmpirical priors loaded from model_utils.py (integrated into observer models)")

    # -------------------------------------------------------------------------
    # Intimacy Inference Predictions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Generating INTIMACY inference predictions...")
    print("-" * 40)

    intimacy_dfs = []

    # Stipulated models
    for model_name in ["full", "vanilla", "discomfort_only"]:
        alpha_observer = alpha_obs.get((model_name, "intimacy"), 1.0)
        print(f"  {model_name} (stipulated, alpha_observer={alpha_observer:.3f})...")
        df = generate_intimacy_preds_stipulated(params[model_name], model_name, alpha_observer=alpha_observer)
        # Add scenario_label column (same prediction for all scenarios)
        scenario_dfs = []
        for scenario_label in SCENARIO_LABELS:
            df_scenario = df.copy()
            df_scenario["scenario_label"] = scenario_label
            scenario_dfs.append(df_scenario)
        intimacy_dfs.append(pd.concat(scenario_dfs, ignore_index=True))

    # LM models
    for model_name, lm_model_name in [("full", "full_lm"), ("vanilla", "vanilla_lm"), ("discomfort_only", "discomfort_only_lm")]:
        alpha_observer = alpha_obs.get((lm_model_name, "intimacy"), 1.0)
        print(f"  {model_name} (LM, alpha_observer={alpha_observer:.3f})...")
        df = generate_intimacy_preds_lm(params[lm_model_name], model_name, alpha_observer=alpha_observer)
        intimacy_dfs.append(df)

    # Combine all intimacy predictions
    df_intimacy_full = pd.concat(intimacy_dfs, ignore_index=True)

    # Compute summary (expected intimacy)
    print("  Computing expected intimacy...")
    df_intimacy_summary = compute_expected_intimacy(df_intimacy_full)

    # Save
    full_path = output_dir / "inv_plan_intimacy_preds_full.csv"
    summary_path = output_dir / "inv_plan_intimacy_preds_summary.csv"
    df_intimacy_full.to_csv(full_path, index=False)
    df_intimacy_summary.to_csv(summary_path, index=False)
    print(f"  Saved {len(df_intimacy_full)} rows to {full_path}")
    print(f"  Saved {len(df_intimacy_summary)} rows to {summary_path}")

    # -------------------------------------------------------------------------
    # Reward Inference Predictions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Generating REWARD inference predictions...")
    print("-" * 40)

    reward_dfs = []

    # Stipulated models
    for model_name in ["full", "vanilla", "discomfort_only"]:
        alpha_observer = alpha_obs.get((model_name, "reward"), 1.0)
        print(f"  {model_name} (stipulated, alpha_observer={alpha_observer:.3f})...")
        df = generate_reward_preds_stipulated(params[model_name], model_name, alpha_observer=alpha_observer)
        # Add scenario_label column (same prediction for all scenarios)
        scenario_dfs = []
        for scenario_label in SCENARIO_LABELS:
            df_scenario = df.copy()
            df_scenario["scenario_label"] = scenario_label
            scenario_dfs.append(df_scenario)
        reward_dfs.append(pd.concat(scenario_dfs, ignore_index=True))

    # LM models
    for model_name, lm_model_name in [("full", "full_lm"), ("vanilla", "vanilla_lm"), ("discomfort_only", "discomfort_only_lm")]:
        alpha_observer = alpha_obs.get((lm_model_name, "reward"), 1.0)
        print(f"  {model_name} (LM, alpha_observer={alpha_observer:.3f})...")
        df = generate_reward_preds_lm(params[lm_model_name], model_name, alpha_observer=alpha_observer)
        reward_dfs.append(df)

    # Combine all reward predictions
    df_reward_full = pd.concat(reward_dfs, ignore_index=True)

    # Compute summary (P(high reward))
    print("  Computing P(high reward)...")
    df_reward_summary = compute_p_high_reward(df_reward_full)

    # Save
    full_path = output_dir / "inv_plan_reward_preds_full.csv"
    summary_path = output_dir / "inv_plan_reward_preds_summary.csv"
    df_reward_full.to_csv(full_path, index=False)
    df_reward_summary.to_csv(summary_path, index=False)
    print(f"  Saved {len(df_reward_full)} rows to {full_path}")
    print(f"  Saved {len(df_reward_summary)} rows to {summary_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
