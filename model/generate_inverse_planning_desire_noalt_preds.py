"""
Generate predictions for the no-alternatives-shown desire (reward) inference variant.

Mirrors generate_inverse_planning_intimacy_noalt_preds.py but flips the inference target.
For each (scenario, observed_action, relationship_condition) cell, emits the
posterior P(reward = HIGH) under all three padded reward observer ablations.
The summary CSV's `p_high` column is what the slider response 0-100 encodes.

The action space is **relationship-keyed** — alternatives elicited per
(scenario, observed_action, relationship_condition) cell. Tables come from
`load_padded_lm_tables_relationship`. All actor weights and α_observer come
from the JOINT fit on no-alt desire data
(`inverse_planning_desire_noalt_fit_results.csv`).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd

from utils import get_project_root
from model_utils import (
    SCENARIO_LABELS,
    RelationshipConditions,
    load_padded_lm_tables_relationship,
    observer_reward_full_padded_rel,
    observer_reward_discomfort_only_padded_rel,
    observer_reward_base_padded_rel,
)


# Variant registry: name -> (observer_fn, utility_param_names, uses_v).
# discomfort_only is V-independent.
PADDED_VARIANTS = {
    "full": (observer_reward_full_padded_rel, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_reward_discomfort_only_padded_rel, ["w_d", "gamma"], False),
    "base":   (observer_reward_base_padded_rel,   ["w_v", "w_e"], True),
}


# Map RelationshipConditions enum index → human-readable intimacy label
# (matches the experiment's intimacy slider values).
RELATIONSHIP_LEVELS = [
    (RelationshipConditions.ZERO,         0,   0),
    (RelationshipConditions.FIFTY,        50,  1),
    (RelationshipConditions.SEVENTY_FIVE, 75,  2),
    (RelationshipConditions.ONE_HUNDRED,  100, 3),
]


def load_desire_noalt_fit_results(filepath=None):
    """Return dict: variant_name -> dict of all fitted params (utility + α_obs + α_actor)."""
    if filepath is None:
        filepath = (
            get_project_root()
            / "model"
            / "outputs"
            / "inverse_planning_desire_noalt_fit_results.csv"
        )
    df = pd.read_csv(filepath)
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


def generate_desire_noalt_preds(params, variant_name, padded):
    observer_fn, utility_names, uses_v = PADDED_VARIANTS[variant_name]
    kwargs = {"alpha": params["alpha"]}
    for name in utility_names:
        kwargs[name] = params[name]
    kwargs["alpha_observer"] = params["alpha_observer"]
    table_kwargs = dict(
        access_table=padded["access"],
        effort_table=padded["effort"],
        prior_table=padded["prior"],
    )
    if uses_v:
        table_kwargs["v_padded_table"] = padded["v"]
    result = observer_fn(**kwargs, **table_kwargs)
    # result shape: (padded_slot, scenario, observed_action, relationship, reward_condition)
    # Slot 0 always holds the observed canonical action.
    data = []
    for s_idx, scenario_label in enumerate(SCENARIO_LABELS):
        for observed in range(4):
            for _rel_enum, rel_label, rel_idx in RELATIONSHIP_LEVELS:
                p_low = float(result[0, s_idx, observed, rel_idx, 0])
                p_high = float(result[0, s_idx, observed, rel_idx, 1])
                data.append({
                    "scenario_label": scenario_label,
                    "observed_action": observed,
                    "intimacy": rel_label,
                    "p_low": p_low,
                    "p_high": p_high,
                })
    return pd.DataFrame(data)


def main():
    print("=" * 60)
    print("No-alt desire inference predictions (3 variants, joint fit)")
    print("=" * 60)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("\nLoading joint-fit parameters (actor weights + α_observer)...")
    params_by_variant = load_desire_noalt_fit_results()
    for variant, params in params_by_variant.items():
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in params.items())
        print(f"  {variant}: {param_str}")

    padded = load_padded_lm_tables_relationship()
    if padded is None:
        print(
            "  Error: relationship-keyed padded tables unavailable. Run the "
            "relationship-conditioned LM elicitation first (see "
            "fit_inverse_planning_desire_noalt.py for the command list)."
        )
        return

    print("\nGenerating predictions per variant...")
    dfs_full = []
    for variant in PADDED_VARIANTS:
        if variant not in params_by_variant:
            continue
        model_name = f"{variant}_padded"
        print(f"  {model_name}...")
        df = generate_desire_noalt_preds(params_by_variant[variant], variant, padded)
        df["model"] = model_name
        dfs_full.append(df)

    df_full = pd.concat(dfs_full, ignore_index=True)
    # The "summary" for desire is the same per-cell P(high); no marginalization needed
    # since reward is binary (unlike intimacy, where the summary takes E[I] over 101 levels).
    df_summary = df_full[
        ["scenario_label", "observed_action", "intimacy", "model", "p_high"]
    ].copy()

    full_path = output_dir / "food_inv-desire_intimacy_noalt_preds_full.csv"
    summary_path = output_dir / "food_inv-desire_intimacy_noalt_preds_summary.csv"
    df_full.to_csv(full_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved {len(df_full)} rows to {full_path}")
    print(f"Saved {len(df_summary)} rows to {summary_path}")


if __name__ == "__main__":
    main()
