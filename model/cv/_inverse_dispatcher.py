"""
Leave-one-scenario-out (LOSO) CV for the active inverse experiments
(Studies 1a, 1b, 2a, 2b).

For each variant (full / discomfort_only / base) and each of the 16 scenarios,
hold the scenario out, jointly refit the actor utility weights and
`alpha_observer` on the remaining 15 scenarios, then predict the held-out
scenario from that refit. Per-fold rows go to `cv_folds.csv`; held-out
predictions are aggregated into `cv_preds_summary.csv` (one row per
held-out cell × variant).

Each `main_*()` runs end-to-end for one experiment and is exposed
through the corresponding `cv/cv_food_inv_*.py` thin wrapper.

The experiments differ in which latent the observer infers and how
many slider responses participants give per trial:

  Study 2a (`food_inv_intimacy`)   — infer intimacy given (desire, effort)
  Study 1a (`food_inv_desire`)     — infer desire given (effort, intimacy)
  Study 1b (`food_inv_joint_de`)   — joint over (desire, effort) given intimacy
  Study 2b (`food_inv_joint_ie`)   — joint over (intimacy, effort) given desire

All share the joint-fit logic in `model/inverse/_helpers.py` — there is
no transfer between studies.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    desire_table_kwargs,
    fit_desire_observer_joint,
    fit_intimacy_observer_joint,
    fit_joint_de_observer_joint,
    fit_joint_ie_observer_joint,
    intimacy_table_kwargs,
    joint_de_table_kwargs,
    joint_ie_table_kwargs,
    load_desire_data,
    load_intimacy_data,
    load_joint_de_data,
    load_joint_ie_data,
)
from observers import (  # noqa: E402
    observer_intimacy_base,
    observer_intimacy_discomfort_only,
    observer_intimacy_full,
    observer_joint_de_base,
    observer_joint_de_discomfort_only,
    observer_joint_de_full,
    observer_joint_ie_base,
    observer_joint_ie_discomfort_only,
    observer_joint_ie_full,
    observer_desire_base,
    observer_desire_discomfort_only,
    observer_desire_full,
)
from tables import (  # noqa: E402
    DesireLevels,
    INTIMACY_CONDITIONS,
    IntimacyLevels,
    SCENARIO_LABELS,
    actions,
)
from utils import get_project_root  # noqa: E402


N_SCENARIOS = len(SCENARIO_LABELS)
# Latent grids are already on the [0, 1] scale (so are the human ratings and
# the model's predicted ratings / belief updates). The 101-bin index of a 0-1
# response is round(response * 100).
INTIMACY_GRID = np.asarray(IntimacyLevels)
DESIRE_GRID = np.asarray(DesireLevels)
# Map the RelationshipConditions axis index back to the verbal condition slug
# written into the prediction CSVs (so they merge with the human data, which
# stores intimacy_condition as a slug — never a numeric code).
INTIMACY_IDX_TO_LEVEL = dict(enumerate(INTIMACY_CONDITIONS))
N_ACTIONS = int(len(actions))
# Multi-start restarts per fold refit — lower than the full fits' default of 5
# to keep 16 folds x 3 variants x 4 studies tractable.
N_RESTARTS_CV = 3


# Per-variant (observer_fn, utility_param_names, uses_v). Each registry pairs
# one of the three ablations with the matching observer for that experiment.
VARIANTS_INTIMACY = {
    "full": (observer_intimacy_full, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (
        observer_intimacy_discomfort_only,
        ["w_d", "gamma"],
        False,
    ),
    "base": (observer_intimacy_base, ["w_v", "w_e"], True),
}
VARIANTS_DESIRE = {
    "full": (observer_desire_full, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_desire_discomfort_only, ["w_d", "gamma"], False),
    "base": (observer_desire_base, ["w_v", "w_e"], True),
}
VARIANTS_JOINT_DE = {
    "full": (observer_joint_de_full, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (
        observer_joint_de_discomfort_only,
        ["w_d", "gamma"],
        False,
    ),
    "base": (observer_joint_de_base, ["w_v", "w_e"], True),
}
VARIANTS_JOINT_IE = {
    "full": (observer_joint_ie_full, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (
        observer_joint_ie_discomfort_only,
        ["w_d", "gamma"],
        False,
    ),
    "base": (observer_joint_ie_base, ["w_v", "w_e"], True),
}
# ------------------------------------------------------------------------------
# Helpers shared across the five LOSO mains.
# ------------------------------------------------------------------------------


def _build_observer_table(obs_fn, params_arr, utility_param_names, table_kwargs):
    """Reproduce the observer table from a joint-fit parameter array."""
    actor_kwargs = {"alpha": 1.0}
    for i, name in enumerate(utility_param_names):
        actor_kwargs[name] = float(params_arr[i])
    return obs_fn(
        **actor_kwargs,
        alpha_observer=float(params_arr[-1]),
        **table_kwargs,
    )


def _fold_row(
    slug,
    variant,
    fold,
    scenario_label,
    params_arr,
    utility_param_names,
    train_nll,
    test_nll,
    n_train,
    n_test,
):
    row = {
        "experiment": slug,
        "variant": variant,
        "fold": fold,
        "held_out_scenario": scenario_label,
        "alpha_observer": float(params_arr[-1]),
        "train_nll": float(train_nll),
        "test_nll": float(test_nll),
        "n_train": int(n_train),
        "n_test": int(n_test),
    }
    for i, name in enumerate(utility_param_names):
        row[f"param_{name}"] = float(params_arr[i])
    return row


def _write_outputs(slug, preds_df, folds_df):
    outputs_dir = get_project_root() / "model" / "outputs" / slug
    outputs_dir.mkdir(parents=True, exist_ok=True)
    preds_path = outputs_dir / "cv_preds_summary.csv"
    folds_path = outputs_dir / "cv_folds.csv"
    preds_df.to_csv(preds_path, index=False)
    folds_df.to_csv(folds_path, index=False)
    print(f"\nWrote {preds_path}")
    print(f"Wrote {folds_path}")

    print("\n=== Per-variant summary ===")
    for variant, sub in folds_df.groupby("variant"):
        per_trial = (sub["test_nll"] / sub["n_test"]).mean()
        print(
            f"  {variant}: alpha_obs = {sub['alpha_observer'].mean():.3f} "
            f"+/- {sub['alpha_observer'].std():.3f}, "
            f"mean test NLL/trial = {per_trial:.4f}"
        )


def _print_fold_header(slug, variant, fold, scenario_label, n_train, n_test):
    print(
        f"  {slug} / {variant} / fold {fold + 1}/{N_SCENARIOS} "
        f"({scenario_label}): train={n_train}, test={n_test}"
    )


# ==============================================================================
# Study 2a — infer intimacy given (desire, effort)
# ==============================================================================


def _loso_intimacy(slug):
    data, action, scenario_idx, desire_condition, effort_condition, response = (
        load_intimacy_data(slug)
    )
    scenario_idx_np = np.asarray(scenario_idx)

    pred_rows, fold_rows = [], []

    for variant, (obs_fn, utility_names, uses_v) in VARIANTS_INTIMACY.items():
        tk = intimacy_table_kwargs(uses_v)
        for fold in range(N_SCENARIOS):
            scenario_label = SCENARIO_LABELS[fold]
            train_mask = scenario_idx_np != fold
            test_mask = scenario_idx_np == fold
            n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
            _print_fold_header(slug, variant, fold, scenario_label, n_train, n_test)

            params, train_nll, _ = fit_intimacy_observer_joint(
                observer_fn=obs_fn,
                utility_param_names=utility_names,
                action=action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                desire_condition=desire_condition[train_mask],
                effort_condition=effort_condition[train_mask],
                response=response[train_mask],
                table_kwargs=tk,
                verbose=False,
                n_restarts=N_RESTARTS_CV,
            )
            table = np.asarray(_build_observer_table(obs_fn, params, utility_names, tk))
            # Padded shape: (padded_slot, scenario, observed_action, desire, effort, intimacy_101)
            # The observed action sits in slot 0.

            held_out = table[
                0, fold, :, :, :, :
            ]  # (observed_action, desire, effort, 101)
            for a_idx in range(N_ACTIONS):
                for r in (0, 1):
                    for e in (0, 1):
                        density = held_out[a_idx, r, e, :]
                        expected_intimacy = float(np.sum(INTIMACY_GRID * density))
                        pred_rows.append(
                            {
                                "scenario_label": scenario_label,
                                "action": a_idx,
                                "desire_condition": "low" if r == 0 else "high",
                                "effort_condition": "low" if e == 0 else "high",
                                "expected_intimacy": expected_intimacy,
                                "model": variant,
                            }
                        )

            test_nll = 0.0
            for i in np.where(test_mask)[0]:
                post = table[
                    0,
                    int(scenario_idx[i]),
                    int(action[i]),
                    int(desire_condition[i]),
                    int(effort_condition[i]),
                    :,
                ]
                resp_idx = int(np.clip(round(float(response[i]) * 100), 0, 100))
                prob = max(float(post[resp_idx]), 1e-8)
                test_nll += -float(np.log(prob))

            fold_rows.append(
                _fold_row(
                    slug,
                    variant,
                    fold,
                    scenario_label,
                    params,
                    utility_names,
                    train_nll,
                    test_nll,
                    n_train,
                    n_test,
                )
            )

    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def main_intimacy():
    slug = "food_inv_intimacy"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    preds_df, folds_df = _loso_intimacy(slug)
    _write_outputs(slug, preds_df, folds_df)


# ==============================================================================
# Study 1a — infer desire given (effort, intimacy)
# ==============================================================================


def _loso_desire(slug):
    data, action, scenario_idx, effort_condition, relationship_condition, response = (
        load_desire_data(slug)
    )
    scenario_idx_np = np.asarray(scenario_idx)

    pred_rows, fold_rows = [], []

    for variant, (obs_fn, utility_names, uses_v) in VARIANTS_DESIRE.items():
        tk = desire_table_kwargs(uses_v)
        for fold in range(N_SCENARIOS):
            scenario_label = SCENARIO_LABELS[fold]
            train_mask = scenario_idx_np != fold
            test_mask = scenario_idx_np == fold
            n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
            _print_fold_header(slug, variant, fold, scenario_label, n_train, n_test)

            params, train_nll, _ = fit_desire_observer_joint(
                observer_fn=obs_fn,
                utility_param_names=utility_names,
                action=action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                effort_condition=effort_condition[train_mask],
                relationship_condition=relationship_condition[train_mask],
                response=response[train_mask],
                table_kwargs=tk,
                verbose=False,
                n_restarts=N_RESTARTS_CV,
            )
            table = np.asarray(_build_observer_table(obs_fn, params, utility_names, tk))
            # Shape: (padded_slot, scenario, observed_action, effort, intimacy, desire)
            # The canonical observed action sits in slot 0; slots 1..k are LM alts.

            held_out = table[
                0, fold, :, :, :, :
            ]  # (observed_action, effort, intimacy, desire[101])
            for a_idx in range(N_ACTIONS):
                for rel_idx in range(4):
                    for e in (0, 1):
                        density = held_out[a_idx, e, rel_idx, :]
                        expected_desire = float(np.sum(DESIRE_GRID * density))
                        pred_rows.append(
                            {
                                "scenario_label": scenario_label,
                                "action": a_idx,
                                "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel_idx],
                                "effort_condition": "low" if e == 0 else "high",
                                "expected_desire": expected_desire,
                                "model": variant,
                            }
                        )

            # Desire DV is a continuous 0-1 rating; test loss is the NLL of the
            # response bin under the 101-bin desire posterior.
            test_nll = 0.0
            for i in np.where(test_mask)[0]:
                post = table[
                    0,
                    int(scenario_idx[i]),
                    int(action[i]),
                    int(effort_condition[i]),
                    int(relationship_condition[i]),
                    :,
                ]
                resp_idx = int(np.clip(round(float(response[i]) * 100), 0, 100))
                test_nll += -float(np.log(max(float(post[resp_idx]), 1e-8)))

            fold_rows.append(
                _fold_row(
                    slug,
                    variant,
                    fold,
                    scenario_label,
                    params,
                    utility_names,
                    train_nll,
                    test_nll,
                    n_train,
                    n_test,
                )
            )

    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def main_desire():
    slug = "food_inv_desire"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    preds_df, folds_df = _loso_desire(slug)
    _write_outputs(slug, preds_df, folds_df)


# ==============================================================================
# Study 1b — joint over (desire, effort) given intimacy
# ==============================================================================
# Two slider responses per trial: P(desire=HIGH) and P(effort=HIGH). Per-trial
# test NLL sums the two binary cross-entropies, matching the training loss.


def _loso_joint_de(slug):
    (
        data,
        action,
        scenario_idx,
        relationship_condition,
        response_desire,
        response_effort,
    ) = load_joint_de_data(slug)
    scenario_idx_np = np.asarray(scenario_idx)

    pred_rows, fold_rows = [], []

    for variant, (obs_fn, utility_names, uses_v) in VARIANTS_JOINT_DE.items():
        tk = joint_de_table_kwargs(uses_v)
        for fold in range(N_SCENARIOS):
            scenario_label = SCENARIO_LABELS[fold]
            train_mask = scenario_idx_np != fold
            test_mask = scenario_idx_np == fold
            n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
            _print_fold_header(slug, variant, fold, scenario_label, n_train, n_test)

            params, train_nll, _ = fit_joint_de_observer_joint(
                observer_fn=obs_fn,
                utility_param_names=utility_names,
                action=action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                relationship_condition=relationship_condition[train_mask],
                response_desire=response_desire[train_mask],
                response_effort=response_effort[train_mask],
                table_kwargs=tk,
                verbose=False,
                n_restarts=N_RESTARTS_CV,
            )
            table = np.asarray(_build_observer_table(obs_fn, params, utility_names, tk))
            # Padded shape: (padded_slot, scenario, observed_action, relationship_4, desire, effort)
            # joint over (desire, effort); observed action sits in slot 0.

            held_out = table[
                0, fold, :, :, :, :
            ]  # (observed_action, rel_4, desire[101], effort[2])
            for a_idx in range(N_ACTIONS):
                for rel_idx in range(4):
                    joint = held_out[a_idx, rel_idx, :, :]  # (101, 2)
                    desire_post = joint.sum(axis=1)  # marginal over effort
                    expected_desire = float(np.sum(DESIRE_GRID * desire_post))
                    p_effort_high = float(joint[:, 1].sum())
                    pred_rows.append(
                        {
                            "scenario_label": scenario_label,
                            "action": a_idx,
                            "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel_idx],
                            "expected_desire": expected_desire,
                            "p_effort_high": p_effort_high,
                            "model": variant,
                        }
                    )

            # Desire slider is a continuous 0-1 rating (NLL over the 101-bin
            # posterior); effort slider is a 0-1 rating (BCE on P(effort=HIGH)).
            test_nll = 0.0
            for i in np.where(test_mask)[0]:
                joint = table[
                    0,
                    int(scenario_idx[i]),
                    int(action[i]),
                    int(relationship_condition[i]),
                    :,
                    :,
                ]
                desire_post = joint.sum(axis=1)
                p_e_high = float(joint[:, 1].sum())
                # desire: NLL of the response bin under the 101-bin posterior
                resp_idx = int(np.clip(round(float(response_desire[i]) * 100), 0, 100))
                test_nll += -float(np.log(max(float(desire_post[resp_idx]), 1e-8)))
                # effort: binary cross-entropy on the 0-1 slider
                p_human = float(response_effort[i])
                p_m = min(max(p_e_high, 1e-8), 1 - 1e-8)
                test_nll += -(
                    p_human * float(np.log(p_m))
                    + (1 - p_human) * float(np.log(1 - p_m))
                )

            fold_rows.append(
                _fold_row(
                    slug,
                    variant,
                    fold,
                    scenario_label,
                    params,
                    utility_names,
                    train_nll,
                    test_nll,
                    n_train,
                    n_test,
                )
            )

    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def main_joint_de():
    slug = "food_inv_joint_de"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    preds_df, folds_df = _loso_joint_de(slug)
    _write_outputs(slug, preds_df, folds_df)


# ==============================================================================
# Study 2b — joint over (intimacy, effort) given desire
# ==============================================================================


def _loso_joint_ie(slug):
    (
        data,
        action,
        scenario_idx,
        desire_condition,
        response_intimacy,
        response_effort,
    ) = load_joint_ie_data(slug)
    scenario_idx_np = np.asarray(scenario_idx)

    pred_rows, fold_rows = [], []

    for variant, (obs_fn, utility_names, uses_v) in VARIANTS_JOINT_IE.items():
        tk = joint_ie_table_kwargs(uses_v)
        for fold in range(N_SCENARIOS):
            scenario_label = SCENARIO_LABELS[fold]
            train_mask = scenario_idx_np != fold
            test_mask = scenario_idx_np == fold
            n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
            _print_fold_header(slug, variant, fold, scenario_label, n_train, n_test)

            params, train_nll, _ = fit_joint_ie_observer_joint(
                observer_fn=obs_fn,
                utility_param_names=utility_names,
                action=action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                desire_condition=desire_condition[train_mask],
                response_intimacy=response_intimacy[train_mask],
                response_effort=response_effort[train_mask],
                table_kwargs=tk,
                verbose=False,
                n_restarts=N_RESTARTS_CV,
            )
            table = np.asarray(_build_observer_table(obs_fn, params, utility_names, tk))
            # Padded shape: (padded_slot, scenario, observed_action, desire, intimacy_101, effort)
            # joint over (intimacy, effort); observed action sits in slot 0.

            held_out = table[
                0, fold, :, :, :, :
            ]  # (observed_action, desire, intimacy_101, effort)
            for a_idx in range(N_ACTIONS):
                for r in (0, 1):
                    joint = held_out[a_idx, r, :, :]  # (101, 2)
                    p_intimacy = joint.sum(axis=1)  # marginal over effort
                    expected_intimacy = float(np.sum(INTIMACY_GRID * p_intimacy))
                    p_effort_high = float(joint[:, 1].sum())
                    pred_rows.append(
                        {
                            "scenario_label": scenario_label,
                            "action": a_idx,
                            "desire_condition": "low" if r == 0 else "high",
                            "expected_intimacy": expected_intimacy,
                            "p_effort_high": p_effort_high,
                            "model": variant,
                        }
                    )

            test_nll = 0.0
            for i in np.where(test_mask)[0]:
                joint = table[
                    0,
                    int(scenario_idx[i]),
                    int(action[i]),
                    int(desire_condition[i]),
                    :,
                    :,
                ]
                # intimacy slider NLL (101-bin posterior)
                p_intimacy = joint.sum(axis=1)
                resp_idx = int(
                    np.clip(round(float(response_intimacy[i]) * 100), 0, 100)
                )
                test_nll += -float(np.log(max(float(p_intimacy[resp_idx]), 1e-8)))
                # effort slider NLL (binary cross-entropy)
                p_e_high = float(joint[:, 1].sum())
                p_human = float(response_effort[i])
                p_m = min(max(p_e_high, 1e-8), 1 - 1e-8)
                test_nll += -(
                    p_human * float(np.log(p_m))
                    + (1 - p_human) * float(np.log(1 - p_m))
                )

            fold_rows.append(
                _fold_row(
                    slug,
                    variant,
                    fold,
                    scenario_label,
                    params,
                    utility_names,
                    train_nll,
                    test_nll,
                    n_train,
                    n_test,
                )
            )

    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def main_joint_ie():
    slug = "food_inv_joint_ie"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    preds_df, folds_df = _loso_joint_ie(slug)
    _write_outputs(slug, preds_df, folds_df)


# ==============================================================================
