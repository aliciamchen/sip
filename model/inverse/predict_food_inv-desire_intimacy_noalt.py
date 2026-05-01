"""Generate predictions for food_inv-desire_intimacy_noalt (no-alt joint fit, relationship-keyed)."""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import pandas as pd  # noqa: E402

from observers import (  # noqa: E402
    observer_reward_base_padded_rel,
    observer_reward_discomfort_only_padded_rel,
    observer_reward_full_padded_rel,
)
from tables import (  # noqa: E402
    RelationshipConditions,
    SCENARIO_LABELS,
    load_padded_lm_tables_relationship,
)

EXPERIMENT_SLUG = "food_inv-desire_intimacy_noalt"

PADDED_VARIANTS = {
    "full": (observer_reward_full_padded_rel, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_reward_discomfort_only_padded_rel, ["w_d", "gamma"], False),
    "base": (observer_reward_base_padded_rel, ["w_v", "w_e"], True),
}

RELATIONSHIP_LEVELS = [
    (RelationshipConditions.ZERO, 0, 0),
    (RelationshipConditions.FIFTY, 50, 1),
    (RelationshipConditions.SEVENTY_FIVE, 75, 2),
    (RelationshipConditions.ONE_HUNDRED, 100, 3),
]


def load_fit_results():
    path = _project_root / "model" / "outputs" / EXPERIMENT_SLUG / "fit_results.csv"
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        variant = str(row["model"]).replace("_padded", "")
        params = {
            "alpha": float(row["param_alpha"]) if pd.notna(row.get("param_alpha", None)) else 1.0,
            "alpha_observer": float(row["alpha_observer"]),
        }
        for pn in ["w_v", "w_d", "w_e", "gamma"]:
            col = f"param_{pn}"
            if col in row and pd.notna(row[col]):
                params[pn] = float(row[col])
        out[variant] = params
    return out


def generate_preds(params, variant, padded):
    obs_fn, utility_names, uses_v = PADDED_VARIANTS[variant]
    kwargs = {"alpha": params["alpha"]}
    for name in utility_names:
        kwargs[name] = params[name]
    kwargs["alpha_observer"] = params["alpha_observer"]
    table_kwargs = dict(
        access_table=padded["access"], effort_table=padded["effort"], prior_table=padded["prior"],
    )
    if uses_v:
        table_kwargs["v_padded_table"] = padded["v"]
    result = obs_fn(**kwargs, **table_kwargs)
    rows = []
    for s_idx, scenario_label in enumerate(SCENARIO_LABELS):
        for observed in range(4):
            for _rel_enum, rel_label, rel_idx in RELATIONSHIP_LEVELS:
                p_low = float(result[0, s_idx, observed, rel_idx, 0])
                p_high = float(result[0, s_idx, observed, rel_idx, 1])
                rows.append({
                    "scenario_label": scenario_label,
                    "observed_action": observed,
                    "intimacy": rel_label,
                    "p_low": p_low,
                    "p_high": p_high,
                })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print(f"Generating predictions: {EXPERIMENT_SLUG}")
    print("=" * 60)

    params_by_variant = load_fit_results()
    padded = load_padded_lm_tables_relationship()
    if padded is None:
        print("Error: relationship-keyed padded tables unavailable.")
        sys.exit(1)

    dfs = []
    for variant in PADDED_VARIANTS:
        if variant not in params_by_variant:
            continue
        df = generate_preds(params_by_variant[variant], variant, padded)
        df["model"] = f"{variant}_padded"
        dfs.append(df)
    df_full = pd.concat(dfs, ignore_index=True)
    df_summary = df_full[["scenario_label", "observed_action", "intimacy", "model", "p_high"]].copy()

    output_dir = _project_root / "model" / "outputs" / EXPERIMENT_SLUG
    df_full.to_csv(output_dir / "preds_full.csv", index=False)
    df_summary.to_csv(output_dir / "preds_summary.csv", index=False)
    print(f"\nSaved {len(df_full)} rows to {output_dir / 'preds_full.csv'}")
    print(f"Saved {len(df_summary)} rows to {output_dir / 'preds_summary.csv'}")


if __name__ == "__main__":
    main()
