"""
Leave-one-scenario-out (LOSO) CV for the active inverse experiments
(Studies 1a, 1b, 2a — Study 2b scaffolded separately).

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

  Study 2a (`food_inv_intimacy`)   — infer intimacy given (reward, effort)
  Study 1a (`food_inv_desire`)     — infer reward given (effort, intimacy)
  Study 1b (`food_inv_joint_de`)   — joint over (reward, effort) given intimacy

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
    observer_reward_base,
    observer_reward_discomfort_only,
    observer_reward_full,
)
from tables import IntimacyLevels, SCENARIO_LABELS, actions_3act  # noqa: E402
from utils import get_project_root  # noqa: E402


N_SCENARIOS = len(SCENARIO_LABELS)
INTIMACY_GRID_100 = np.asarray(IntimacyLevels) * 100.0
INTIMACY_IDX_TO_LEVEL = {0: 0, 1: 50, 2: 75, 3: 100}
N_ACTIONS = int(len(actions_3act))


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
    "full": (observer_reward_full, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_reward_discomfort_only, ["w_d", "gamma"], False),
    "base": (observer_reward_base, ["w_v", "w_e"], True),
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
# Study 2 — infer intimacy given (reward, effort)
# ==============================================================================


def _loso_intimacy(slug):
    data, action, scenario_idx, reward_condition, effort_condition, response = (
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

            params, train_nll = fit_intimacy_observer_joint(
                observer_fn=obs_fn,
                utility_param_names=utility_names,
                action=action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                reward_condition=reward_condition[train_mask],
                effort_condition=effort_condition[train_mask],
                response=response[train_mask],
                table_kwargs=tk,
                verbose=False,
            )
            table = np.asarray(_build_observer_table(obs_fn, params, utility_names, tk))
            # Padded shape: (padded_slot, scenario, observed_action, reward, effort, intimacy_101)
            # The observed action sits in slot 0.

            held_out = table[
                0, fold, :, :, :, :
            ]  # (observed_action, reward, effort, 101)
            for a_idx in range(N_ACTIONS):
                for r in (0, 1):
                    for e in (0, 1):
                        density = held_out[a_idx, r, e, :]
                        expected_intimacy = float(np.sum(INTIMACY_GRID_100 * density))
                        pred_rows.append(
                            {
                                "scenario_label": scenario_label,
                                "action": a_idx,
                                "reward_condition": "low" if r == 0 else "high",
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
                    int(reward_condition[i]),
                    int(effort_condition[i]),
                    :,
                ]
                resp_idx = int(np.clip(round(float(response[i])), 0, 100))
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


# ==============================================================================
# Study 3b — infer reward (desire) given (effort, intimacy)
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

            params, train_nll = fit_desire_observer_joint(
                observer_fn=obs_fn,
                utility_param_names=utility_names,
                action=action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                effort_condition=effort_condition[train_mask],
                relationship_condition=relationship_condition[train_mask],
                response=response[train_mask],
                table_kwargs=tk,
                verbose=False,
            )
            table = np.asarray(_build_observer_table(obs_fn, params, utility_names, tk))
            # Shape: (padded_slot, scenario, observed_action, effort, intimacy, reward)
            # The canonical observed action sits in slot 0; slots 1..k are LM alts.

            held_out = table[
                0, fold, :, :, :, :
            ]  # (observed_action, effort, intimacy, reward)
            for a_idx in range(N_ACTIONS):
                for rel_idx in range(4):
                    for e in (0, 1):
                        p_high_reward = float(held_out[a_idx, e, rel_idx, 1])
                        pred_rows.append(
                            {
                                "scenario_label": scenario_label,
                                "action": a_idx,
                                "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel_idx],
                                "effort_condition": "low" if e == 0 else "high",
                                "p_high_reward": p_high_reward * 100.0,
                                # Continuous 0-100 desire prediction: 100 * P(reward=HIGH)
                                "pred_desire": 100.0 * p_high_reward,
                                "model": variant,
                            }
                        )

            # Desire DV is a continuous 0-100 slider; test loss is squared error
            # between the normalized prediction P(high) and the observed rating/100.
            test_nll = 0.0
            for i in np.where(test_mask)[0]:
                p_model = float(
                    table[
                        0,
                        int(scenario_idx[i]),
                        int(action[i]),
                        int(effort_condition[i]),
                        int(relationship_condition[i]),
                        1,
                    ]
                )
                test_nll += (
                    min(max(p_model, 0.0), 1.0) - float(response[i]) / 100.0
                ) ** 2

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
# Study 4a — joint over (reward, effort) given intimacy
# ==============================================================================
# Two slider responses per trial: P(reward=HIGH) and P(effort=HIGH). Per-trial
# test NLL sums the two binary cross-entropies, matching the training loss.


def _loso_joint_de(slug):
    (
        data,
        action,
        scenario_idx,
        relationship_condition,
        response_reward,
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

            params, train_nll = fit_joint_de_observer_joint(
                observer_fn=obs_fn,
                utility_param_names=utility_names,
                action=action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                relationship_condition=relationship_condition[train_mask],
                response_reward=response_reward[train_mask],
                response_effort=response_effort[train_mask],
                table_kwargs=tk,
                verbose=False,
            )
            table = np.asarray(_build_observer_table(obs_fn, params, utility_names, tk))
            # Padded shape: (padded_slot, scenario, observed_action, relationship_4, reward, effort)
            # joint over (reward, effort); observed action sits in slot 0.

            held_out = table[
                0, fold, :, :, :, :
            ]  # (observed_action, rel_4, reward, effort)
            for a_idx in range(N_ACTIONS):
                for rel_idx in range(4):
                    joint = held_out[a_idx, rel_idx, :, :]  # (2, 2)
                    p_reward_high = float(joint[1, :].sum())
                    p_effort_high = float(joint[:, 1].sum()) * 100.0
                    pred_rows.append(
                        {
                            "scenario_label": scenario_label,
                            "action": a_idx,
                            "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel_idx],
                            "p_high_reward": p_reward_high * 100.0,
                            # Continuous 0-100 desire prediction: 100 * P(reward=HIGH)
                            "pred_desire": 100.0 * p_reward_high,
                            "p_effort_high": p_effort_high,
                            "model": variant,
                        }
                    )

            # Desire slider is a continuous 0-100 rating (squared error on the
            # normalized prediction); effort slider is a 0-100 continuous rating
            # (binary cross-entropy on P(effort=HIGH)).
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
                p_r_high = float(joint[1, :].sum())
                p_e_high = float(joint[:, 1].sum())
                # desire: squared error on the normalized 0-100 rating
                test_nll += (
                    min(max(p_r_high, 0.0), 1.0) - float(response_reward[i]) / 100.0
                ) ** 2
                # effort: binary cross-entropy on the 0-100 slider
                p_human = float(response_effort[i]) / 100.0
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
# Study 2b — joint over (intimacy, effort) given reward
# ==============================================================================


def _loso_joint_ie(slug):
    (
        data,
        action,
        scenario_idx,
        reward_condition,
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

            params, train_nll = fit_joint_ie_observer_joint(
                observer_fn=obs_fn,
                utility_param_names=utility_names,
                action=action[train_mask],
                scenario_idx=scenario_idx[train_mask],
                reward_condition=reward_condition[train_mask],
                response_intimacy=response_intimacy[train_mask],
                response_effort=response_effort[train_mask],
                table_kwargs=tk,
                verbose=False,
            )
            table = np.asarray(_build_observer_table(obs_fn, params, utility_names, tk))
            # Padded shape: (padded_slot, scenario, observed_action, reward, intimacy_101, effort)
            # joint over (intimacy, effort); observed action sits in slot 0.

            held_out = table[
                0, fold, :, :, :, :
            ]  # (observed_action, reward, intimacy_101, effort)
            for a_idx in range(N_ACTIONS):
                for r in (0, 1):
                    joint = held_out[a_idx, r, :, :]  # (101, 2)
                    p_intimacy = joint.sum(axis=1)  # marginal over effort
                    expected_intimacy = float(np.sum(INTIMACY_GRID_100 * p_intimacy))
                    p_effort_high = float(joint[:, 1].sum()) * 100.0
                    pred_rows.append(
                        {
                            "scenario_label": scenario_label,
                            "action": a_idx,
                            "reward_condition": "low" if r == 0 else "high",
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
                    int(reward_condition[i]),
                    :,
                    :,
                ]
                # intimacy slider NLL (101-bin posterior)
                p_intimacy = joint.sum(axis=1)
                resp_idx = int(np.clip(round(float(response_intimacy[i])), 0, 100))
                test_nll += -float(np.log(max(float(p_intimacy[resp_idx]), 1e-8)))
                # effort slider NLL (binary cross-entropy)
                p_e_high = float(joint[:, 1].sum())
                p_human = float(response_effort[i]) / 100.0
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
