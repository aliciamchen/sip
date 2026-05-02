"""Generate predictions for food_inv_desire_intimacy_alt.

Reads fit_results.csv, runs the reward observer, writes preds_full.csv +
preds_summary.csv.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    compute_p_high_reward,
    load_fitted_alpha_observer,
    load_fitted_params,
)
from observers import (  # noqa: E402
    observer_reward_base,
    observer_reward_discomfort_only,
    observer_reward_full,
)
from tables import LLM_TABLES, SCENARIO_LABELS, actions, load_lm_v  # noqa: E402

EXPERIMENT_SLUG = "food_inv_desire_intimacy_alt"

REWARD_OBSERVERS = {
    "full": (observer_reward_full, ["alpha", "w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_reward_discomfort_only, ["alpha", "w_d", "gamma"], False),
    "base": (observer_reward_base, ["alpha", "w_v", "w_e"], True),
}


def _table_kwargs(uses_v):
    kw = {"access_table": LLM_TABLES["access"], "effort_table": LLM_TABLES["effort"]}
    if uses_v:
        kw["v_table"] = load_lm_v("food")
    return kw


def generate_preds(params: dict, variant: str, alpha_observer):
    obs_fn, kw_names, uses_v = REWARD_OBSERVERS[variant]
    kwargs = {k: params[k] for k in kw_names}
    kwargs["alpha_observer"] = alpha_observer
    result = obs_fn(**kwargs, **_table_kwargs(uses_v))
    intimacy_map = {0: 0, 1: 50, 2: 75, 3: 100}
    rows = []
    for s_idx, scenario_label in enumerate(SCENARIO_LABELS):
        for a_idx, a in enumerate(actions):
            for rel_idx in range(4):
                for r in [0, 1]:
                    rows.append({
                        "scenario_label": scenario_label,
                        "action": int(a),
                        "intimacy_condition": intimacy_map[rel_idx],
                        "reward_condition": "low" if r == 0 else "high",
                        "density": float(result[a_idx, s_idx, rel_idx, r]),
                    })
    df = pd.DataFrame(rows)
    df["model"] = variant
    return df


def main():
    print("=" * 60)
    print(f"Generating predictions: {EXPERIMENT_SLUG}")
    print("=" * 60)

    params = load_fitted_params()
    alpha_obs = load_fitted_alpha_observer()

    dfs = []
    for variant in REWARD_OBSERVERS:
        if variant not in params:
            print(f"  (skipping {variant})")
            continue
        a_obs = alpha_obs.get((variant, EXPERIMENT_SLUG), 1.0)
        print(f"  {variant} (alpha_observer={a_obs:.3f})...")
        dfs.append(generate_preds(params[variant], variant, a_obs))
    df_full = pd.concat(dfs, ignore_index=True)
    df_summary = compute_p_high_reward(df_full)

    output_dir = _project_root / "model" / "outputs" / EXPERIMENT_SLUG
    output_dir.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(output_dir / "preds_full.csv", index=False)
    df_summary.to_csv(output_dir / "preds_summary.csv", index=False)
    print(f"\nSaved {len(df_full)} rows to {output_dir / 'preds_full.csv'}")
    print(f"Saved {len(df_summary)} rows to {output_dir / 'preds_summary.csv'}")


if __name__ == "__main__":
    main()
