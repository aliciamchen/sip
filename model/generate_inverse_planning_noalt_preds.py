"""
Generate predictions for the no-alternatives-shown intimacy inference variant.

Generates predictions for all three access-utility ablations (access_full,
access_only, no_access) using their respective padded observer memos. Emits
one prediction row per (scenario, observed_action, motivation, intimacy_level,
model). Summary adds expected intimacy per (scenario, observed, motivation,
model). Actor prior is uniform over valid padded slots.

All actor weights and α_observer come from the JOINT fit on no-alt data
(`inverse_planning_noalt_fit_results.csv`). This is different from the alt-
shown experiments, which freeze actor weights from the forward fit. The
padded observer's variable-length action space makes the Exp 1 weights a
poor transplant, so the no-alt pipeline refits all weights directly.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd

from utils import get_project_root
from model_utils import (
    SCENARIO_LABELS,
    IntimacyLevels,
    load_padded_lm_tables,
    observer_intimacy_access_full_padded,
    observer_intimacy_access_only_padded,
    observer_intimacy_no_access_padded,
)


# Variant registry: name -> (observer_fn, utility_param_names, uses_v).
# α_actor is fixed at 1; utility names are what we pull from the fit-results CSV.
# access_only is V-independent and doesn't take v_padded_table.
PADDED_VARIANTS = {
    "access_full": (observer_intimacy_access_full_padded, ["w_v", "w_d", "w_e"], True),
    "access_only": (observer_intimacy_access_only_padded, ["w_d"], False),
    "no_access":   (observer_intimacy_no_access_padded,   ["w_v", "w_e"], True),
}


def load_noalt_fit_results(filepath=None):
    """Return dict: variant_name -> dict of all fitted params (utility + α_obs + α_actor).

    Reads the joint-fit CSV written by fit_inverse_planning_noalt.py:main().
    Keys are "access_full", "access_only", "no_access" (no `_padded` suffix).
    """
    if filepath is None:
        filepath = (
            get_project_root()
            / "model"
            / "outputs"
            / "inverse_planning_noalt_fit_results.csv"
        )
    df = pd.read_csv(filepath)
    out = {}
    for _, row in df.iterrows():
        variant = str(row["model"]).replace("_padded", "")
        params = {
            "alpha": float(row["param_alpha"]) if pd.notna(row.get("param_alpha", None)) else 1.0,
            "alpha_observer": float(row["alpha_observer"]),
        }
        for pn in ["w_v", "w_d", "w_e"]:
            col = f"param_{pn}"
            if col in row and pd.notna(row[col]):
                params[pn] = float(row[col])
        out[variant] = params
    return out


def generate_noalt_preds(params, variant_name, padded):
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
    # result shape: (padded_slot, scenario, observed_action, intimacy, reward_condition)
    # Slot 0 always holds the observed canonical action.
    data = []
    for s_idx, scenario_label in enumerate(SCENARIO_LABELS):
        for observed in range(4):
            for r in (0, 1):
                for i_idx, i in enumerate(IntimacyLevels):
                    data.append({
                        "scenario_label": scenario_label,
                        "observed_action": observed,
                        "motivation": "low" if r == 0 else "high",
                        "intimacy": float(i),
                        "density": float(result[0, s_idx, observed, i_idx, r]),
                    })
    df = pd.DataFrame(data)
    return df


def compute_expected_intimacy_noalt(df):
    df = df.copy()
    df["intimacy_scaled"] = df["intimacy"] * 100
    summary = (
        df.groupby(
            ["scenario_label", "observed_action", "motivation", "model"],
            dropna=False,
        )
        .apply(
            lambda g: pd.Series(
                {"expected_intimacy": (g["intimacy_scaled"] * g["density"]).sum()}
            )
        )
        .reset_index()
    )
    return summary


def main():
    print("=" * 60)
    print("No-alt intimacy inference predictions (3 variants, joint fit)")
    print("=" * 60)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("\nLoading joint-fit parameters (actor weights + α_observer)...")
    params_by_variant = load_noalt_fit_results()
    for variant, params in params_by_variant.items():
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in params.items())
        print(f"  {variant}: {param_str}")

    padded = load_padded_lm_tables()
    if padded is None:
        print("  Error: padded tables unavailable (missing LM alternatives CSVs).")
        return

    print("\nGenerating predictions per variant...")
    dfs_full = []
    for variant in PADDED_VARIANTS:
        if variant not in params_by_variant:
            continue
        model_name = f"{variant}_padded"
        print(f"  {model_name}...")
        df = generate_noalt_preds(params_by_variant[variant], variant, padded)
        df["model"] = model_name
        dfs_full.append(df)

    df_full = pd.concat(dfs_full, ignore_index=True)
    df_summary = compute_expected_intimacy_noalt(df_full)

    full_path = output_dir / "inv_plan_intimacy_noalt_preds_full.csv"
    summary_path = output_dir / "inv_plan_intimacy_noalt_preds_summary.csv"
    df_full.to_csv(full_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved {len(df_full)} rows to {full_path}")
    print(f"Saved {len(df_summary)} rows to {summary_path}")


if __name__ == "__main__":
    main()
