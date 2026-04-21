"""
Generate predictions for the no-alternatives-shown intimacy inference variant.

Generates predictions for all three access-utility ablations (access_full,
access_only, no_access) using their respective padded observer memos. Emits
one prediction row per (scenario, observed_action, motivation, intimacy_level,
model). Summary adds expected intimacy per (scenario, observed, motivation,
model). Actor prior is uniform over valid padded slots.

Actor params are frozen from forward planning (per variant).
alpha_observer is loaded from inverse_planning_noalt_fit_results.csv.
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

from generate_inverse_planning_preds import load_fitted_params


PADDED_VARIANTS = {
    "access_full": (observer_intimacy_access_full_padded, ["alpha", "w_v", "w_d", "w_e"]),
    "access_only": (observer_intimacy_access_only_padded, ["alpha", "w_d"]),
    "no_access":   (observer_intimacy_no_access_padded,   ["alpha", "w_v", "w_e"]),
}


def load_fitted_alpha_observer_noalt(filepath=None):
    """Return dict: variant_name -> alpha_observer.

    model column looks like "access_full_padded".
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
        alpha = row["alpha_observer"]
        out[variant] = float(alpha) if pd.notna(alpha) else 1.0
    return out


def generate_noalt_preds(actor_params, variant_name, alpha_observer, padded):
    observer_fn, kw_names = PADDED_VARIANTS[variant_name]
    kwargs = {k: actor_params[k] for k in kw_names}
    kwargs["alpha_observer"] = alpha_observer
    result = observer_fn(
        **kwargs,
        access_table=padded["access"],
        effort_table=padded["effort"],
        is_share_table=padded["is_share"],
        prior_table=padded["prior"],
    )
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
    print("No-alt intimacy inference predictions (3 variants, uniform prior)")
    print("=" * 60)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("\nLoading fitted actor parameters...")
    params_by_variant = load_fitted_params()
    print("\nLoading fitted alpha_observer (no-alt)...")
    alpha_by_variant = load_fitted_alpha_observer_noalt()
    for variant, alpha in alpha_by_variant.items():
        print(f"  {variant}: alpha_observer = {alpha:.3f}")

    padded = load_padded_lm_tables()
    if padded is None:
        print("  Error: padded tables unavailable (missing LM alternatives CSVs).")
        return

    print("\nGenerating predictions per variant...")
    dfs_full = []
    for variant in PADDED_VARIANTS:
        if variant not in params_by_variant or variant not in alpha_by_variant:
            continue
        model_name = f"{variant}_padded"
        print(f"  {model_name}...")
        df = generate_noalt_preds(
            params_by_variant[variant],
            variant,
            alpha_by_variant[variant],
            padded,
        )
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
