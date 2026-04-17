"""
Generate inverse planning predictions using frozen forward planning parameters.

This script generates predictions for two inverse planning experiments:
1. inv_plan_intimacy: Infer intimacy from observed action (given reward condition)
2. inv_plan_desire: Infer desire condition from observed action (given intimacy)

Uses parameters fitted from forward planning experiment (frozen, not re-fitted).
"""

import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import jax.numpy as jnp
import numpy as np
import pandas as pd

from utils import get_project_root
from model_utils import (
    # Constants
    actions,
    IntimacyLevels,
    RewardConditions,
    RelationshipConditions,
    SCENARIO_LABELS,
    SCENARIO_TO_IDX,

    # Observer models for intimacy inference (pre-registered)
    observer_intimacy_full_model,
    observer_intimacy_vanilla_inv_plan,
    observer_intimacy_discomfort_only,

    # Modified observer for intimacy inference (effort scaled by intimacy)
    observer_intimacy_full_model_modified,

    # Observer models for reward inference (pre-registered)
    observer_reward_full_model,
    observer_reward_vanilla_inv_plan,
    observer_reward_discomfort_only,

    # Modified observer for reward inference (effort scaled by intimacy)
    observer_reward_full_model_modified,

    # Access-based observer models (stipulated-vector)
    observer_intimacy_access_full,
    observer_intimacy_access_only,
    observer_intimacy_no_access,
    observer_reward_access_full,
    observer_reward_access_only,
    observer_reward_no_access,

    # LLM-parameterized observer models
    LLM_TABLES,
    observer_intimacy_access_full_llm,
    observer_intimacy_access_only_llm,
    observer_intimacy_no_access_llm,
    observer_reward_access_full_llm,
    observer_reward_access_only_llm,
    observer_reward_no_access_llm,
)


# ==============================================================================
# Load Fitted Parameters
# ==============================================================================

def load_fitted_params(filepath: str = None) -> dict:
    """Load fitted actor parameters from forward planning fit results.

    Returns a dict: model_name -> dict of every `param_*` column present in that
    row (stripped of the `param_` prefix). Missing/NaN columns are omitted.
    """
    if filepath is None:
        filepath = get_project_root() / "model" / "outputs" / "forward_planning_fit_results.csv"
    df = pd.read_csv(filepath)
    params = {}
    for _, row in df.iterrows():
        model_name = row["model"]
        model_params = {}
        for col in df.columns:
            if col.startswith("param_") and pd.notna(row[col]):
                model_params[col.replace("param_", "")] = float(row[col])
        # Legacy defaults used by the pre-registered-model codepath
        for legacy_key in ("w_r", "w_d", "w_c"):
            model_params.setdefault(legacy_key, 0.0)
        params[model_name] = model_params
    return params


def load_fitted_alpha_observer(filepath: str = None) -> dict:
    """Load fitted alpha_observer values from inverse planning fit results.

    Returns dict with (model, experiment) -> alpha_observer

    Note: If alpha_observer is NaN (e.g., for vanilla intimacy model where the
    utility doesn't depend on intimacy, making the gradient zero), we default
    to 1.0. This produces the uniform prior as expected.
    """
    if filepath is None:
        filepath = get_project_root() / "model" / "outputs" / "inverse_planning_fit_results.csv"
    df = pd.read_csv(filepath)
    alpha_obs = {}
    for _, row in df.iterrows():
        key = (row["model"], row["experiment"])
        alpha_val = row["alpha_observer"]
        # Default to 1.0 if NaN (happens when gradient is zero, e.g., vanilla intimacy)
        alpha_obs[key] = alpha_val if pd.notna(alpha_val) else 1.0
    return alpha_obs


def load_fitted_beta(filepath: str = None) -> dict:
    """Load fitted beta values from inverse planning fit results.

    Returns dict with (model, experiment) -> beta

    Beta is only defined for modified models. Non-modified models have NaN.
    """
    if filepath is None:
        filepath = get_project_root() / "model" / "outputs" / "inverse_planning_fit_results.csv"
    df = pd.read_csv(filepath)
    beta_vals = {}
    for _, row in df.iterrows():
        key = (row["model"], row["experiment"])
        beta_val = row.get("beta", np.nan)
        if pd.notna(beta_val):
            beta_vals[key] = beta_val
    return beta_vals


# ==============================================================================
# Intimacy Inference Predictions
# ==============================================================================

def generate_intimacy_preds_stipulated(params: dict, model_name: str, alpha_observer: float = 1.0, modified: bool = False, beta: float = 1.0) -> pd.DataFrame:
    """Generate intimacy inference predictions using stipulated parameters.

    Returns DataFrame with posterior distribution over intimacy for each (action, reward_condition).

    Args:
        params: Dictionary with alpha, w_r, w_d, w_c
        model_name: "full", "vanilla", or "discomfort_only"
        alpha_observer: Observer inverse temperature
        modified: If True, use the modified model (reward scaled by 1 + beta*intimacy)
        beta: Reward-intimacy scaling parameter (only used for modified models)
    """
    if model_name == "full":
        if modified:
            result = observer_intimacy_full_model_modified(
                alpha=params["alpha"], w_r=params["w_r"], w_d=params["w_d"], w_c=params["w_c"],
                alpha_observer=alpha_observer, beta=beta
            )
        else:
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


# ==============================================================================
# Reward Inference Predictions
# ==============================================================================

def generate_reward_preds_stipulated(params: dict, model_name: str, alpha_observer: float = 1.0, modified: bool = False, beta: float = 1.0) -> pd.DataFrame:
    """Generate reward inference predictions using stipulated parameters.

    Returns DataFrame with P(reward_condition | action, intimacy_condition).

    Args:
        params: Dictionary with alpha, w_r, w_d, w_c
        model_name: "full", "vanilla", or "discomfort_only"
        alpha_observer: Observer inverse temperature
        modified: If True, use the modified model (reward scaled by 1 + beta*intimacy)
        beta: Reward-intimacy scaling parameter (only used for modified models)
    """
    if model_name == "full":
        if modified:
            result = observer_reward_full_model_modified(
                alpha=params["alpha"], w_r=params["w_r"], w_d=params["w_d"], w_c=params["w_c"],
                alpha_observer=alpha_observer, beta=beta
            )
        else:
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


# ==============================================================================
# Access-Based Prediction Generators
# ==============================================================================
#
# Each access variant has its own actor-param signature. We hold those as a
# registry keyed by variant name and dispatch generically.

ACCESS_INTIMACY_OBSERVERS = {
    "access_full": (observer_intimacy_access_full, ["alpha", "w_v", "w_r", "w_d", "w_e"]),
    "access_only": (observer_intimacy_access_only, ["alpha", "w_r", "w_d"]),
    "no_access": (observer_intimacy_no_access, ["alpha", "w_v", "w_e"]),
}

ACCESS_REWARD_OBSERVERS = {
    "access_full": (observer_reward_access_full, ["alpha", "w_v", "w_r", "w_d", "w_e"]),
    "access_only": (observer_reward_access_only, ["alpha", "w_r", "w_d"]),
    "no_access": (observer_reward_no_access, ["alpha", "w_v", "w_e"]),
}

# LLM variants: observer produces a 4D table indexed by scenario_idx, so
# predictions are scenario-specific (no duplicate-across-scenarios in main()).
ACCESS_LLM_INTIMACY_OBSERVERS = {
    "access_full_llm": (observer_intimacy_access_full_llm, ["alpha", "w_v", "w_r", "w_d", "w_e"]),
    "access_only_llm": (observer_intimacy_access_only_llm, ["alpha", "w_r", "w_d"]),
    "no_access_llm": (observer_intimacy_no_access_llm, ["alpha", "w_v", "w_e"]),
}

ACCESS_LLM_REWARD_OBSERVERS = {
    "access_full_llm": (observer_reward_access_full_llm, ["alpha", "w_v", "w_r", "w_d", "w_e"]),
    "access_only_llm": (observer_reward_access_only_llm, ["alpha", "w_r", "w_d"]),
    "no_access_llm": (observer_reward_no_access_llm, ["alpha", "w_v", "w_e"]),
}


def generate_intimacy_preds_access(
    params: dict, variant_name: str, alpha_observer: float = 1.0
) -> pd.DataFrame:
    """Intimacy-inference predictions for one access variant."""
    observer_fn, kw_names = ACCESS_INTIMACY_OBSERVERS[variant_name]
    kwargs = {k: params[k] for k in kw_names}
    kwargs["alpha_observer"] = alpha_observer
    result = observer_fn(**kwargs)

    data = []
    for a_idx, a in enumerate(actions):
        for r in [0, 1]:
            for i_idx, i in enumerate(IntimacyLevels):
                data.append({
                    "action": int(a),
                    "reward_condition": "low" if r == 0 else "high",
                    "intimacy": float(i),
                    "density": float(result[a_idx, i_idx, r]),
                })
    df = pd.DataFrame(data)
    df["model"] = variant_name
    df["param_source"] = "stipulated"
    return df


def generate_reward_preds_access(
    params: dict, variant_name: str, alpha_observer: float = 1.0
) -> pd.DataFrame:
    """Reward-inference predictions for one access variant."""
    observer_fn, kw_names = ACCESS_REWARD_OBSERVERS[variant_name]
    kwargs = {k: params[k] for k in kw_names}
    kwargs["alpha_observer"] = alpha_observer
    result = observer_fn(**kwargs)

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
    df["model"] = variant_name
    df["param_source"] = "stipulated"
    return df


def generate_intimacy_preds_access_llm(
    params: dict, variant_name: str, alpha_observer, tables
) -> pd.DataFrame:
    """Intimacy-inference predictions for an LLM variant (per scenario).

    Returns one row per (scenario, action, reward_condition, intimacy_level).
    """
    observer_fn, kw_names = ACCESS_LLM_INTIMACY_OBSERVERS[variant_name]
    kwargs = {k: params[k] for k in kw_names}
    kwargs["alpha_observer"] = alpha_observer
    a_tab, e_tab, r_tab = tables
    result = observer_fn(
        **kwargs, access_table=a_tab, effort_table=e_tab, reward_table=r_tab
    )
    # result shape: (actions, scenarios, intimacy_levels, reward_conditions)
    data = []
    for s_idx, scenario_label in enumerate(SCENARIO_LABELS):
        for a_idx, a in enumerate(actions):
            for r in [0, 1]:
                for i_idx, i in enumerate(IntimacyLevels):
                    data.append({
                        "scenario_label": scenario_label,
                        "action": int(a),
                        "reward_condition": "low" if r == 0 else "high",
                        "intimacy": float(i),
                        "density": float(result[a_idx, s_idx, i_idx, r]),
                    })
    df = pd.DataFrame(data)
    df["model"] = variant_name
    df["param_source"] = "llm"
    return df


def generate_reward_preds_access_llm(
    params: dict, variant_name: str, alpha_observer, tables
) -> pd.DataFrame:
    """Reward-inference predictions for an LLM variant (per scenario)."""
    observer_fn, kw_names = ACCESS_LLM_REWARD_OBSERVERS[variant_name]
    kwargs = {k: params[k] for k in kw_names}
    kwargs["alpha_observer"] = alpha_observer
    a_tab, e_tab, r_tab = tables
    result = observer_fn(
        **kwargs, access_table=a_tab, effort_table=e_tab, reward_table=r_tab
    )
    # result shape: (actions, scenarios, relationship_conditions, reward_conditions)
    intimacy_map = {0: 0, 1: 50, 2: 75, 3: 100}
    data = []
    for s_idx, scenario_label in enumerate(SCENARIO_LABELS):
        for a_idx, a in enumerate(actions):
            for rel_idx in range(4):
                for r in [0, 1]:
                    data.append({
                        "scenario_label": scenario_label,
                        "action": int(a),
                        "intimacy_condition": intimacy_map[rel_idx],
                        "reward_condition": "low" if r == 0 else "high",
                        "density": float(result[a_idx, s_idx, rel_idx, r]),
                    })
    df = pd.DataFrame(data)
    df["model"] = variant_name
    df["param_source"] = "llm"
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
    print("\nGenerating predictions for:")
    print("  - Pre-registered models: full, vanilla, discomfort_only")
    print("  - Modified model: full_modified (effort scaled by intimacy)")

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

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

    # Load fitted beta values (only for modified models)
    print("\nLoading fitted beta values...")
    beta_vals = load_fitted_beta()
    for (model, exp), beta in beta_vals.items():
        print(f"  {model} ({exp}): beta={beta:.3f}")


    # -------------------------------------------------------------------------
    # Intimacy Inference Predictions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Generating INTIMACY inference predictions...")
    print("-" * 40)

    intimacy_dfs = []

    # Pre-registered models
    for model_name in ["full", "vanilla", "discomfort_only"]:
        alpha_observer = alpha_obs.get((model_name, "intimacy"), 1.0)
        print(f"  {model_name} (pre-reg, alpha_observer={alpha_observer:.3f})...")
        df = generate_intimacy_preds_stipulated(params[model_name], model_name, alpha_observer=alpha_observer, modified=False)
        # Add scenario_label column (same prediction for all scenarios)
        scenario_dfs = []
        for scenario_label in SCENARIO_LABELS:
            df_scenario = df.copy()
            df_scenario["scenario_label"] = scenario_label
            scenario_dfs.append(df_scenario)
        intimacy_dfs.append(pd.concat(scenario_dfs, ignore_index=True))

    # Modified full model - use its own fitted alpha_observer and beta
    alpha_observer = alpha_obs.get(("full_modified", "intimacy"), 1.0)
    beta = beta_vals.get(("full_modified", "intimacy"), 1.0)
    print(f"  full_modified (modified, alpha_observer={alpha_observer:.3f}, beta={beta:.3f})...")
    df = generate_intimacy_preds_stipulated(params["full"], "full", alpha_observer=alpha_observer, modified=True, beta=beta)
    df["model"] = "full_modified"  # Rename to distinguish from pre-reg
    scenario_dfs = []
    for scenario_label in SCENARIO_LABELS:
        df_scenario = df.copy()
        df_scenario["scenario_label"] = scenario_label
        scenario_dfs.append(df_scenario)
    intimacy_dfs.append(pd.concat(scenario_dfs, ignore_index=True))

    # Access-based variants (canonical reformulation)
    for variant_name in ACCESS_INTIMACY_OBSERVERS:
        if variant_name not in params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        alpha_observer = alpha_obs.get((variant_name, "intimacy"), 1.0)
        print(f"  {variant_name} (access, alpha_observer={alpha_observer:.3f})...")
        df = generate_intimacy_preds_access(
            params[variant_name], variant_name, alpha_observer=alpha_observer
        )
        scenario_dfs = []
        for scenario_label in SCENARIO_LABELS:
            df_scenario = df.copy()
            df_scenario["scenario_label"] = scenario_label
            scenario_dfs.append(df_scenario)
        intimacy_dfs.append(pd.concat(scenario_dfs, ignore_index=True))

    # LLM-parameterized access variants (scenario-specific predictions)
    if LLM_TABLES is not None:
        llm_tables = (LLM_TABLES["access"], LLM_TABLES["effort"], LLM_TABLES["reward"])
        for variant_name in ACCESS_LLM_INTIMACY_OBSERVERS:
            if variant_name not in params:
                print(f"  (skipping {variant_name}: no forward fit available)")
                continue
            alpha_observer = alpha_obs.get((variant_name, "intimacy"), 1.0)
            print(f"  {variant_name} (LLM, alpha_observer={alpha_observer:.3f})...")
            df = generate_intimacy_preds_access_llm(
                params[variant_name], variant_name,
                alpha_observer=alpha_observer, tables=llm_tables,
            )
            # Predictions already include scenario_label — no duplication needed
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

    # Pre-registered models
    for model_name in ["full", "vanilla", "discomfort_only"]:
        alpha_observer = alpha_obs.get((model_name, "reward"), 1.0)
        print(f"  {model_name} (pre-reg, alpha_observer={alpha_observer:.3f})...")
        df = generate_reward_preds_stipulated(params[model_name], model_name, alpha_observer=alpha_observer, modified=False)
        # Add scenario_label column (same prediction for all scenarios)
        scenario_dfs = []
        for scenario_label in SCENARIO_LABELS:
            df_scenario = df.copy()
            df_scenario["scenario_label"] = scenario_label
            scenario_dfs.append(df_scenario)
        reward_dfs.append(pd.concat(scenario_dfs, ignore_index=True))

    # Modified full model - use its own fitted alpha_observer and beta
    alpha_observer = alpha_obs.get(("full_modified", "reward"), 1.0)
    beta = beta_vals.get(("full_modified", "reward"), 1.0)
    print(f"  full_modified (modified, alpha_observer={alpha_observer:.3f}, beta={beta:.3f})...")
    df = generate_reward_preds_stipulated(params["full"], "full", alpha_observer=alpha_observer, modified=True, beta=beta)
    df["model"] = "full_modified"  # Rename to distinguish from pre-reg
    scenario_dfs = []
    for scenario_label in SCENARIO_LABELS:
        df_scenario = df.copy()
        df_scenario["scenario_label"] = scenario_label
        scenario_dfs.append(df_scenario)
    reward_dfs.append(pd.concat(scenario_dfs, ignore_index=True))

    # Access-based variants (canonical reformulation)
    for variant_name in ACCESS_REWARD_OBSERVERS:
        if variant_name not in params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        alpha_observer = alpha_obs.get((variant_name, "reward"), 1.0)
        print(f"  {variant_name} (access, alpha_observer={alpha_observer:.3f})...")
        df = generate_reward_preds_access(
            params[variant_name], variant_name, alpha_observer=alpha_observer
        )
        scenario_dfs = []
        for scenario_label in SCENARIO_LABELS:
            df_scenario = df.copy()
            df_scenario["scenario_label"] = scenario_label
            scenario_dfs.append(df_scenario)
        reward_dfs.append(pd.concat(scenario_dfs, ignore_index=True))

    # LLM-parameterized access variants (scenario-specific predictions)
    if LLM_TABLES is not None:
        llm_tables = (LLM_TABLES["access"], LLM_TABLES["effort"], LLM_TABLES["reward"])
        for variant_name in ACCESS_LLM_REWARD_OBSERVERS:
            if variant_name not in params:
                print(f"  (skipping {variant_name}: no forward fit available)")
                continue
            alpha_observer = alpha_obs.get((variant_name, "reward"), 1.0)
            print(f"  {variant_name} (LLM, alpha_observer={alpha_observer:.3f})...")
            df = generate_reward_preds_access_llm(
                params[variant_name], variant_name,
                alpha_observer=alpha_observer, tables=llm_tables,
            )
            reward_dfs.append(df)

    # Combine all reward predictions
    df_reward_full = pd.concat(reward_dfs, ignore_index=True)

    # Compute summary (P(high reward))
    print("  Computing P(high reward)...")
    df_reward_summary = compute_p_high_reward(df_reward_full)

    # Save
    full_path = output_dir / "inv_plan_desire_preds_full.csv"
    summary_path = output_dir / "inv_plan_desire_preds_summary.csv"
    df_reward_full.to_csv(full_path, index=False)
    df_reward_summary.to_csv(summary_path, index=False)
    print(f"  Saved {len(df_reward_full)} rows to {full_path}")
    print(f"  Saved {len(df_reward_summary)} rows to {summary_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
