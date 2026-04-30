"""
Generate inverse planning predictions using frozen forward planning parameters.

Generates predictions for two inverse planning experiments:
1. inv_plan_intimacy_alt: Infer intimacy from observed action (given reward condition), alternatives shown to participants
2. inv_plan_desire_alt:   Infer desire (reward condition) from observed action (given intimacy), alternatives shown

Uses actor parameters fitted from the forward planning experiment (frozen, not
re-fitted) and alpha_observer fitted from the inverse planning experiments.
Every variant reads scenario-specific access/effort from LLM_TABLES (reward is
stipulated as a binary goal-satisfaction gate) and emits per-scenario predictions.
"""

import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd

from utils import get_project_root
from model_utils import (
    LLM_TABLES,
    SCENARIO_LABELS,
    IntimacyLevels,
    actions,
    load_lm_v,
    observer_intimacy_access_full,
    observer_intimacy_access_only,
    observer_intimacy_no_access,
    observer_reward_access_full,
    observer_reward_access_only,
    observer_reward_no_access,
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
        params[model_name] = model_params
    return params


def load_fitted_alpha_observer(filepath: str = None) -> dict:
    """Load fitted alpha_observer values from inverse planning fit results.

    Returns dict with (model, experiment) -> alpha_observer. Defaults to 1.0 if NaN.
    """
    if filepath is None:
        filepath = get_project_root() / "model" / "outputs" / "inverse_planning_fit_results.csv"
    df = pd.read_csv(filepath)
    alpha_obs = {}
    for _, row in df.iterrows():
        key = (row["model"], row["experiment"])
        alpha_val = row["alpha_observer"]
        alpha_obs[key] = alpha_val if pd.notna(alpha_val) else 1.0
    return alpha_obs


# ==============================================================================
# Prediction Generators
# ==============================================================================
# Each observer produces a 4D table (actions, scenarios, intimacy_or_relationship,
# reward_condition). We unroll the table into a long-format DataFrame with one
# row per (scenario, action, conditioning, reward_condition).

# Tuple values: (observer_fn, actor_kwarg_names, uses_v).
INTIMACY_OBSERVERS = {
    "access_full": (observer_intimacy_access_full, ["alpha", "w_v", "w_d", "w_e", "gamma"], True),
    "access_only": (observer_intimacy_access_only, ["alpha", "w_d", "gamma"], False),
    "no_access":   (observer_intimacy_no_access,   ["alpha", "w_v", "w_e"], True),
}

REWARD_OBSERVERS = {
    "access_full": (observer_reward_access_full, ["alpha", "w_v", "w_d", "w_e", "gamma"], True),
    "access_only": (observer_reward_access_only, ["alpha", "w_d", "gamma"], False),
    "no_access":   (observer_reward_no_access,   ["alpha", "w_v", "w_e"], True),
}


def _table_kwargs(uses_v: bool):
    kw = {"access_table": LLM_TABLES["access"], "effort_table": LLM_TABLES["effort"]}
    if uses_v:
        kw["v_table"] = load_lm_v("food")
    return kw


def generate_intimacy_preds(
    params: dict, variant_name: str, alpha_observer
) -> pd.DataFrame:
    """Intimacy-inference predictions for one access variant (per scenario).

    Returns one row per (scenario, action, reward_condition, intimacy_level).
    """
    observer_fn, kw_names, uses_v = INTIMACY_OBSERVERS[variant_name]
    kwargs = {k: params[k] for k in kw_names}
    kwargs["alpha_observer"] = alpha_observer
    result = observer_fn(**kwargs, **_table_kwargs(uses_v))
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
    return df


def generate_reward_preds(
    params: dict, variant_name: str, alpha_observer
) -> pd.DataFrame:
    """Reward-inference predictions for one access variant (per scenario)."""
    observer_fn, kw_names, uses_v = REWARD_OBSERVERS[variant_name]
    kwargs = {k: params[k] for k in kw_names}
    kwargs["alpha_observer"] = alpha_observer
    result = observer_fn(**kwargs, **_table_kwargs(uses_v))
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
    return df


# ==============================================================================
# Summary Statistics
# ==============================================================================


def compute_expected_intimacy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute expected value of intimacy from posterior distribution."""
    df = df.copy()
    df["intimacy_scaled"] = df["intimacy"] * 100  # Convert to 0-100 scale

    summary = df.groupby(
        ["scenario_label", "action", "reward_condition", "model"],
        dropna=False
    ).apply(
        lambda g: pd.Series({
            "expected_intimacy": (g["intimacy_scaled"] * g["density"]).sum(),
        })
    ).reset_index()

    return summary


def compute_p_high_reward(df: pd.DataFrame) -> pd.DataFrame:
    """Extract P(high reward) from posterior distribution."""
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
    print("Frozen forward params + fitted alpha_observer")
    print("=" * 60)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Load fitted actor parameters
    print("\nLoading fitted actor parameters...")
    params = load_fitted_params()
    for model_name, p in params.items():
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in p.items())
        print(f"  {model_name}: {param_str}")

    print("\nLoading fitted alpha_observer values...")
    alpha_obs = load_fitted_alpha_observer()
    for (model, exp), alpha in alpha_obs.items():
        print(f"  {model} ({exp}): alpha_observer={alpha:.3f}")

    # -------------------------------------------------------------------------
    # Intimacy Inference Predictions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Generating INTIMACY inference predictions...")
    print("-" * 40)

    intimacy_dfs = []
    for variant_name, (_obs, _kw, _uv) in INTIMACY_OBSERVERS.items():
        if variant_name not in params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        alpha_observer = alpha_obs.get((variant_name, "intimacy"), 1.0)
        print(f"  {variant_name} (alpha_observer={alpha_observer:.3f})...")
        df = generate_intimacy_preds(
            params[variant_name], variant_name, alpha_observer=alpha_observer,
        )
        intimacy_dfs.append(df)

    df_intimacy_full = pd.concat(intimacy_dfs, ignore_index=True)

    print("  Computing expected intimacy...")
    df_intimacy_summary = compute_expected_intimacy(df_intimacy_full)

    full_path = output_dir / "inv_plan_intimacy_alt_preds_full.csv"
    summary_path = output_dir / "inv_plan_intimacy_alt_preds_summary.csv"
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
    for variant_name, (_obs, _kw, _uv) in REWARD_OBSERVERS.items():
        if variant_name not in params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        alpha_observer = alpha_obs.get((variant_name, "reward"), 1.0)
        print(f"  {variant_name} (alpha_observer={alpha_observer:.3f})...")
        df = generate_reward_preds(
            params[variant_name], variant_name, alpha_observer=alpha_observer,
        )
        reward_dfs.append(df)

    df_reward_full = pd.concat(reward_dfs, ignore_index=True)

    print("  Computing P(high reward)...")
    df_reward_summary = compute_p_high_reward(df_reward_full)

    full_path = output_dir / "inv_plan_desire_alt_preds_full.csv"
    summary_path = output_dir / "inv_plan_desire_alt_preds_summary.csv"
    df_reward_full.to_csv(full_path, index=False)
    df_reward_summary.to_csv(summary_path, index=False)
    print(f"  Saved {len(df_reward_full)} rows to {full_path}")
    print(f"  Saved {len(df_reward_summary)} rows to {summary_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
