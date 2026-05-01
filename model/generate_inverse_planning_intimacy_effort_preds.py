"""
Generate inverse-planning predictions for the effort experiment (food_inv-intimacy_effort_alt).

Uses frozen actor parameters from food_forw_intimacy_effort + fitted alpha_observer from
inverse_planning_intimacy_effort_fit_results.csv. For each variant produces a 4D
posterior over intimacy and unrolls to long-format CSVs:

  food_inv-intimacy_effort_alt_preds_full.csv     — one row per (scenario, action, effort, intimacy_level)
  food_inv-intimacy_effort_alt_preds_summary.csv  — expected_intimacy collapsed across the IntimacyLevels grid
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_inverse_planning_intimacy_effort import (
    ACCESS_VARIANTS_EFFORT,
    load_food_forw_intimacy_effort_actor_params,
)
from generate_inverse_planning_alt_preds import load_fitted_alpha_observer
from tables import IntimacyLevels, LLM_TABLES_EFFORT, SCENARIO_LABELS, actions_effort

from utils import get_project_root


EFFORT_LABELS = {0: "low", 1: "high"}


def _table_kwargs():
    return {
        "access_table": LLM_TABLES_EFFORT["access"],
        "effort_table": LLM_TABLES_EFFORT["effort"],
    }


def generate_intimacy_effort_preds(
    params: dict, variant_name: str, alpha_observer
) -> pd.DataFrame:
    """Posterior intimacy for one variant.

    Returns one row per (scenario, action [coded as 1/2 in CSV space],
    effort_condition, intimacy_level). The observer table shape is
    (actions, scenarios, intimacy_levels, effort_conditions).
    """
    obs_fn, kw_names = ACCESS_VARIANTS_EFFORT[variant_name]
    kwargs = {k: params[k] for k in kw_names}
    kwargs["alpha_observer"] = alpha_observer
    result = obs_fn(**kwargs, **_table_kwargs())

    rows = []
    for s_idx, scenario_label in enumerate(SCENARIO_LABELS):
        for a_idx, a_internal in enumerate(actions_effort):
            csv_action = int(a_internal) + 1  # internal 0/1 -> CSV 1/2
            for e in [0, 1]:
                for i_idx, i in enumerate(IntimacyLevels):
                    rows.append({
                        "scenario_label": scenario_label,
                        "action": csv_action,
                        "effort_condition": EFFORT_LABELS[e],
                        "intimacy": float(i),
                        "density": float(result[a_idx, s_idx, i_idx, e]),
                    })
    df = pd.DataFrame(rows)
    df["model"] = variant_name
    return df


def compute_expected_intimacy_effort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["intimacy_scaled"] = df["intimacy"] * 100
    summary = df.groupby(
        ["scenario_label", "action", "effort_condition", "model"], dropna=False
    ).apply(
        lambda g: pd.Series({
            "expected_intimacy": (g["intimacy_scaled"] * g["density"]).sum(),
        })
    ).reset_index()
    return summary


def main():
    print("=" * 60)
    print("Generating food_inv-intimacy_effort_alt predictions")
    print("=" * 60)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("\nLoading frozen actor parameters from food_forw_intimacy_effort...")
    params = load_food_forw_intimacy_effort_actor_params()
    for model_name, p in params.items():
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in p.items())
        print(f"  {model_name}: {param_str}")

    print("\nLoading fitted alpha_observer values from inverse_planning_intimacy_effort_fit_results.csv...")
    alpha_obs = load_fitted_alpha_observer(
        get_project_root() / "model" / "outputs" / "inverse_planning_intimacy_effort_fit_results.csv"
    )
    for (model, exp), alpha in alpha_obs.items():
        print(f"  {model} ({exp}): alpha_observer={alpha:.3f}")

    print("\n" + "-" * 40)
    print("Generating predictions...")
    print("-" * 40)

    dfs = []
    for variant_name, (_obs, _kw) in ACCESS_VARIANTS_EFFORT.items():
        if variant_name not in params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        alpha_observer = alpha_obs.get((variant_name, "intimacy_effort"), 1.0)
        print(f"  {variant_name} (alpha_observer={alpha_observer:.3f})...")
        dfs.append(generate_intimacy_effort_preds(
            params[variant_name], variant_name, alpha_observer=alpha_observer,
        ))

    df_full = pd.concat(dfs, ignore_index=True)
    df_summary = compute_expected_intimacy_effort(df_full)

    full_path = output_dir / "food_inv-intimacy_effort_alt_preds_full.csv"
    summary_path = output_dir / "food_inv-intimacy_effort_alt_preds_summary.csv"
    df_full.to_csv(full_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    print(f"\n  Saved {len(df_full)} rows to {full_path}")
    print(f"  Saved {len(df_summary)} rows to {summary_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
