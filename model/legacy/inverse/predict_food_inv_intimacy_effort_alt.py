"""Generate predictions for food_inv_intimacy_effort_alt."""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    load_fitted_alpha_observer,
    load_food_forw_intimacy_effort_actor_params,
)
from observers import (  # noqa: E402
    observer_intimacy_effort_base,
    observer_intimacy_effort_discomfort_only,
    observer_intimacy_effort_full,
)
from tables import IntimacyLevels, LLM_TABLES_EFFORT, SCENARIO_LABELS, actions_effort  # noqa: E402

EXPERIMENT_SLUG = "food_inv_intimacy_effort_alt"
EFFORT_LABELS = {0: "low", 1: "high"}

ACCESS_VARIANTS_EFFORT = {
    "full": (observer_intimacy_effort_full, ["alpha", "w_v", "w_d", "w_e", "gamma"]),
    "discomfort_only": (observer_intimacy_effort_discomfort_only, ["alpha", "w_d", "gamma"]),
    "base": (observer_intimacy_effort_base, ["alpha", "w_v", "w_e"]),
}


def _table_kwargs():
    return {
        "access_table": LLM_TABLES_EFFORT["access"],
        "effort_table": LLM_TABLES_EFFORT["effort"],
    }


def generate_preds(params, variant, alpha_observer):
    obs_fn, kw_names = ACCESS_VARIANTS_EFFORT[variant]
    kwargs = {k: params[k] for k in kw_names}
    kwargs["alpha_observer"] = alpha_observer
    result = obs_fn(**kwargs, **_table_kwargs())
    rows = []
    for s_idx, scenario_label in enumerate(SCENARIO_LABELS):
        for a_idx, a_internal in enumerate(actions_effort):
            csv_action = int(a_internal) + 1
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
    df["model"] = variant
    return df


def compute_summary(df):
    df = df.copy()
    df["intimacy_scaled"] = df["intimacy"] * 100
    return df.groupby(
        ["scenario_label", "action", "effort_condition", "model"], dropna=False,
    ).apply(
        lambda g: pd.Series({"expected_intimacy": (g["intimacy_scaled"] * g["density"]).sum()})
    ).reset_index()


def main():
    print("=" * 60)
    print(f"Generating predictions: {EXPERIMENT_SLUG}")
    print("=" * 60)

    params = load_food_forw_intimacy_effort_actor_params()
    fit_path = _project_root / "model" / "outputs" / EXPERIMENT_SLUG / "fit_results.csv"
    alpha_obs = load_fitted_alpha_observer(fit_path)

    dfs = []
    for variant in ACCESS_VARIANTS_EFFORT:
        if variant not in params:
            continue
        a_obs = alpha_obs.get((variant, EXPERIMENT_SLUG), 1.0)
        print(f"  {variant} (alpha_observer={a_obs:.3f})...")
        dfs.append(generate_preds(params[variant], variant, a_obs))
    df_full = pd.concat(dfs, ignore_index=True)
    df_summary = compute_summary(df_full)

    output_dir = _project_root / "model" / "outputs" / EXPERIMENT_SLUG
    df_full.to_csv(output_dir / "preds_full.csv", index=False)
    df_summary.to_csv(output_dir / "preds_summary.csv", index=False)
    print(f"\nSaved {len(df_full)} rows to {output_dir / 'preds_full.csv'}")
    print(f"Saved {len(df_summary)} rows to {output_dir / 'preds_summary.csv'}")


if __name__ == "__main__":
    main()
