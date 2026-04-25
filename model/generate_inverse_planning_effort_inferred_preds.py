"""
Generate effort-inference predictions for the inv_plan_effort_inferred experiment.

Uses frozen actor parameters from forw_plan_effort + fitted alpha_observer from
inverse_planning_effort_inferred_fit_results.csv. The observer-table shape is
(actions, scenarios, intimacy_levels, effort_conditions). For the canonical
4-level intimacy slice we keep only intimacy ∈ {0, 50, 75, 100} (indices into
the 101-level IntimacyLevels grid).

Outputs:
  inv_plan_effort_inferred_preds_full.csv     — one row per (scenario, action,
    intimacy ∈ {0,50,75,100}, effort_condition); column `density` is the
    posterior probability of that effort condition.
  inv_plan_effort_inferred_preds_summary.csv  — one row per (scenario, action,
    intimacy); column `p_effort_high` is the posterior probability of the
    high-effort condition (i.e. what the slider response 0-100 encodes).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_inverse_planning_effort_inferred import (
    ACCESS_VARIANTS_EFFORT_INFERRED,
    load_forw_plan_effort_actor_params,
)
from generate_inverse_planning_preds import load_fitted_alpha_observer
from model_utils import SCENARIO_LABELS
from model_utils_effort import LLM_TABLES_EFFORT, actions_effort

from utils import get_project_root


EFFORT_LABELS = {0: "low", 1: "high"}
INTIMACY_DISPLAY_LEVELS = [0, 50, 75, 100]


def _table_kwargs_for(needs_prior):
    tk = {
        "access_table": LLM_TABLES_EFFORT["access"],
        "effort_table": LLM_TABLES_EFFORT["effort"],
    }
    if needs_prior:
        tk["prior_table"] = LLM_TABLES_EFFORT["action_prior"]
    return tk


def generate_effort_inferred_preds(
    params: dict, variant_name: str, alpha_observer
) -> pd.DataFrame:
    """Posterior over effort condition for one variant.

    Returns one row per (scenario, action [coded as 1/2 in CSV space],
    intimacy ∈ {0, 50, 75, 100}, effort_condition). The observer table shape
    is (actions, scenarios, intimacy_levels [101], effort_conditions).
    """
    obs_fn, kw_names, needs_prior = ACCESS_VARIANTS_EFFORT_INFERRED[variant_name]
    kwargs = {k: params[k] for k in kw_names}
    kwargs["alpha_observer"] = alpha_observer
    result = obs_fn(**kwargs, **_table_kwargs_for(needs_prior))

    rows = []
    for s_idx, scenario_label in enumerate(SCENARIO_LABELS):
        for a_idx, a_internal in enumerate(actions_effort):
            csv_action = int(a_internal) + 1  # internal 0/1 -> CSV 1/2
            for intimacy_int in INTIMACY_DISPLAY_LEVELS:
                for e in [0, 1]:
                    rows.append({
                        "scenario_label": scenario_label,
                        "action": csv_action,
                        "intimacy": intimacy_int,
                        "effort_condition": EFFORT_LABELS[e],
                        "density": float(result[a_idx, s_idx, intimacy_int, e]),
                    })
    df = pd.DataFrame(rows)
    df["model"] = variant_name
    return df


def compute_p_effort_high_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the effort axis: one row per (scenario, action, intimacy, model)
    with `p_effort_high`."""
    df_high = df[df["effort_condition"] == "high"].copy()
    df_high = df_high.rename(columns={"density": "p_effort_high"})
    return df_high[
        ["scenario_label", "action", "intimacy", "model", "p_effort_high"]
    ].reset_index(drop=True)


def main():
    print("=" * 60)
    print("Generating inv_plan_effort_inferred predictions")
    print("=" * 60)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("\nLoading frozen actor parameters from forw_plan_effort...")
    params = load_forw_plan_effort_actor_params()
    for model_name, p in params.items():
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in p.items())
        print(f"  {model_name}: {param_str}")

    print(
        "\nLoading fitted alpha_observer values from "
        "inverse_planning_effort_inferred_fit_results.csv..."
    )
    alpha_obs = load_fitted_alpha_observer(
        get_project_root()
        / "model"
        / "outputs"
        / "inverse_planning_effort_inferred_fit_results.csv"
    )
    for (model, exp), alpha in alpha_obs.items():
        print(f"  {model} ({exp}): alpha_observer={alpha:.3f}")

    print("\n" + "-" * 40)
    print("Generating predictions...")
    print("-" * 40)

    dfs = []
    for variant_name, (_obs, _kw, needs_prior) in ACCESS_VARIANTS_EFFORT_INFERRED.items():
        if variant_name not in params:
            print(f"  (skipping {variant_name}: no forward fit available)")
            continue
        if needs_prior and "action_prior" not in LLM_TABLES_EFFORT:
            print(f"  (skipping {variant_name}: lm_action_priors_effort.csv missing)")
            continue
        alpha_observer = alpha_obs.get((variant_name, "effort_inferred"), 1.0)
        print(f"  {variant_name} (alpha_observer={alpha_observer:.3f})...")
        dfs.append(generate_effort_inferred_preds(
            params[variant_name], variant_name, alpha_observer=alpha_observer,
        ))

    df_full = pd.concat(dfs, ignore_index=True)
    df_summary = compute_p_effort_high_summary(df_full)

    full_path = output_dir / "inv_plan_effort_inferred_preds_full.csv"
    summary_path = output_dir / "inv_plan_effort_inferred_preds_summary.csv"
    df_full.to_csv(full_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    print(f"\n  Saved {len(df_full)} rows to {full_path}")
    print(f"  Saved {len(df_summary)} rows to {summary_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
