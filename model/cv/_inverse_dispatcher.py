"""
Leave-one-scenario-out (LOSO) CV for the active inverse experiments
(Studies 1a, 1b, 2a, 2b).

For each variant (full / discomfort_only / base) and each of the 16 scenarios,
hold the scenario out, jointly refit the actor utility weights, `alpha_observer`,
and the response-noise `sigma` on the remaining 15 scenarios, then predict the
held-out scenario from that refit.

Outputs per `outputs/<slug>/`:
  - `cv_trial_ll.jsonl` — the PRIMARY metric: one record per held-out trial with
    its held-out log-likelihood `log p(u)` under the cross-validated mixture and
    its `subject_id` (so the analysis can bootstrap the full−ablation difference
    by participant).
  - `cv_preds_summary.json` — the secondary descriptive prediction: the model
    belief update `delta_<latent>` per held-out cell × variant (for the
    condition-averaged model-vs-human correlation).
  - `cv_folds.jsonl` — per-fold refit diagnostics (params, train/test NLL).

Each `main_*()` runs end-to-end for one experiment and is exposed through the
corresponding `cv/cv_food_inv_*.py` thin wrapper.

The experiments differ in which latent the observer infers and how many slider
responses participants give per trial:

  Study 2a (`food_inv_intimacy`)   — infer intimacy given (desire, effort)
  Study 1a (`food_inv_desire`)     — infer desire given (effort, intimacy)
  Study 1b (`food_inv_joint_de`)   — joint over (desire, effort) given intimacy
  Study 2b (`food_inv_joint_ie`)   — joint over (intimacy, effort) given desire

All share the joint-fit logic in `model/inverse/_helpers.py` — there is no
transfer between studies.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    EFFORT_PRIOR_MEAN,
    GRID,
    PRIOR_MEAN,
    _build_observer_tables_runs,
    desire_table_kwargs,
    fit_desire_observer_joint,
    fit_intimacy_observer_joint,
    fit_joint_de_observer_joint,
    fit_joint_ie_observer_joint,
    intimacy_table_kwargs,
    joint_de_table_kwargs,
    joint_ie_table_kwargs,
    load_desire_data,
    load_fit_results,
    load_intimacy_data,
    load_joint_de_data,
    load_joint_ie_data,
    mixture_nll_1d,
    mixture_nll_2d,
    params_dict_to_array,
    write_json,
    write_jsonl,
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
    INTIMACY_CONDITIONS,
    SCENARIO_LABELS,
    actions,
)
from utils import get_project_root  # noqa: E402


N_SCENARIOS = len(SCENARIO_LABELS)
GRID_NP = np.asarray(GRID)  # 101-bin [0, 1] latent grid
PRIOR_MEAN_F = float(PRIOR_MEAN)  # model prior mean of a continuous latent (= 0.5)
EFFORT_PRIOR_MEAN_F = float(EFFORT_PRIOR_MEAN)  # 2-state effort prior mean (= 0.5)
# Map the RelationshipConditions axis index back to the verbal condition slug
# written into the prediction outputs (so they merge with the human data, which
# stores intimacy_condition as a slug — never a numeric code).
INTIMACY_IDX_TO_LEVEL = dict(enumerate(INTIMACY_CONDITIONS))
N_ACTIONS = int(len(actions))
# Restarts per fold refit. Each refit warm-starts from the full-data fit (see
# `full_fit` below) — a leave-one-scenario-out refit only perturbs it slightly —
# so a single restart from that warm init converges fast and reliably, replacing
# the old 3 cold restarts. Falls back to a cold start if no full fit exists.
N_RESTARTS_CV = 1


# Per-variant (observer_fn, utility_param_names). Each registry pairs one of the
# three ablations with the matching observer for that experiment; which optional
# LM tables a variant needs is derived from its param names in *_table_kwargs.
VARIANTS_INTIMACY = {
    "full": (observer_intimacy_full, ["w_v", "w_d", "w_e", "gamma"]),
    "discomfort_only": (observer_intimacy_discomfort_only, ["w_d", "gamma"]),
    "base": (observer_intimacy_base, ["w_v", "w_e"]),
}
VARIANTS_DESIRE = {
    "full": (observer_desire_full, ["w_v", "w_d", "w_e", "gamma"]),
    "discomfort_only": (observer_desire_discomfort_only, ["w_d", "gamma"]),
    "base": (observer_desire_base, ["w_v", "w_e"]),
}
VARIANTS_JOINT_DE = {
    "full": (observer_joint_de_full, ["w_v", "w_d", "w_e", "gamma"]),
    "discomfort_only": (observer_joint_de_discomfort_only, ["w_d", "gamma"]),
    "base": (observer_joint_de_base, ["w_v", "w_e"]),
}
VARIANTS_JOINT_IE = {
    "full": (observer_joint_ie_full, ["w_v", "w_d", "w_e", "gamma"]),
    "discomfort_only": (observer_joint_ie_discomfort_only, ["w_d", "gamma"]),
    "base": (observer_joint_ie_base, ["w_v", "w_e"]),
}
# ------------------------------------------------------------------------------
# Helpers shared across the four LOSO mains.
# ------------------------------------------------------------------------------


def _held_out_ll_1d(deltas_per_trial, u_per_trial, sigma):
    """Held-out log-likelihood log p(u) per trial under the 1-D mixture, using
    the same `mixture_nll_1d` as the fit. `deltas_per_trial` is (n_test, K),
    `u_per_trial` is (n_test,). Returns a numpy array (n_test,)."""
    lls = jax.vmap(lambda u, d: -mixture_nll_1d(u, d, sigma))(
        jnp.asarray(u_per_trial), jnp.asarray(deltas_per_trial)
    )
    return np.asarray(lls)


def _held_out_ll_2d(deltas_per_trial, u_per_trial, sigma):
    """Held-out log p(u) per trial under the bivariate mixture. `deltas_per_trial`
    is (n_test, K, 2), `u_per_trial` is (n_test, 2)."""
    lls = jax.vmap(lambda u, d: -mixture_nll_2d(u, d, sigma))(
        jnp.asarray(u_per_trial), jnp.asarray(deltas_per_trial)
    )
    return np.asarray(lls)


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
        "alpha_observer": float(params_arr[-2]),
        "sigma": float(params_arr[-1]),
        "train_nll": float(train_nll),
        "test_nll": float(test_nll),
        "n_train": int(n_train),
        "n_test": int(n_test),
    }
    for i, name in enumerate(utility_param_names):
        row[f"param_{name}"] = float(params_arr[i])
    return row


def _write_outputs(slug, pred_rows, fold_rows, trial_ll_rows):
    outputs_dir = get_project_root() / "model" / "outputs" / slug
    outputs_dir.mkdir(parents=True, exist_ok=True)
    write_json(outputs_dir / "cv_preds_summary.json", pred_rows)
    write_jsonl(outputs_dir / "cv_folds.jsonl", fold_rows)
    write_jsonl(outputs_dir / "cv_trial_ll.jsonl", trial_ll_rows)
    print(f"\nWrote {outputs_dir / 'cv_trial_ll.jsonl'} (primary metric)")
    print(f"Wrote {outputs_dir / 'cv_preds_summary.json'}")
    print(f"Wrote {outputs_dir / 'cv_folds.jsonl'}")

    print("\n=== Per-variant summary (held-out log-likelihood) ===")
    trial_df = pd.DataFrame(trial_ll_rows)
    folds_df = pd.DataFrame(fold_rows)
    for variant, sub in trial_df.groupby("model"):
        fsub = folds_df[folds_df["variant"] == variant]
        print(
            f"  {variant}: mean held-out LL/trial = {sub['held_out_ll'].mean():.4f} "
            f"(alpha_obs = {fsub['alpha_observer'].mean():.3f}, "
            f"sigma = {fsub['sigma'].mean():.3f})"
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
    action_np = np.asarray(action)
    desire_np = np.asarray(desire_condition)
    effort_np = np.asarray(effort_condition)
    response_np = np.asarray(response)
    subject_ids = np.asarray(data["subject_id"].values)
    # Warm-start source: the full-data fit (refits perturb it only slightly).
    full_fit = (
        load_fit_results(slug)
        if (
            get_project_root() / "model" / "outputs" / slug / "fit_results.json"
        ).exists()
        else {}
    )

    pred_rows, fold_rows, trial_ll_rows = [], [], []

    for variant, (obs_fn, utility_names) in VARIANTS_INTIMACY.items():
        tk = intimacy_table_kwargs(utility_names)
        warm = (
            params_dict_to_array(full_fit[variant], utility_names)
            if variant in full_fit
            else None
        )
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
                init_params=warm,
            )
            sigma = float(params[-1])
            # (run, slot, scenario, observed_action, desire, effort, intimacy_101)
            tables = np.asarray(
                _build_observer_tables_runs(obs_fn, params, utility_names, tk)
            )

            # Predicted belief update δ per held-out cell (mean over runs).
            for a_idx in range(N_ACTIONS):
                for r in (0, 1):
                    for e in (0, 1):
                        density_runs = tables[:, 0, fold, a_idx, r, e, :]  # (K, 101)
                        deltas = density_runs @ GRID_NP - PRIOR_MEAN_F  # (K,)
                        pred_rows.append(
                            {
                                "scenario_label": scenario_label,
                                "action": a_idx,
                                "desire_condition": "low" if r == 0 else "high",
                                "effort_condition": "low" if e == 0 else "high",
                                "delta_intimacy": float(deltas.mean()),
                                "model": variant,
                            }
                        )

            # Per-trial held-out log-likelihood under the mixture.
            ti = np.where(test_mask)[0]
            if len(ti):
                post = tables[
                    :,
                    0,
                    scenario_idx_np[ti],
                    action_np[ti],
                    desire_np[ti],
                    effort_np[ti],
                    :,
                ]  # (K, n_test, 101)
                deltas_t = (post @ GRID_NP).T - PRIOR_MEAN_F  # (n_test, K)
                lls = _held_out_ll_1d(deltas_t, response_np[ti], sigma)
                test_nll = -float(lls.sum())
                for j, i in enumerate(ti):
                    trial_ll_rows.append(
                        {
                            "experiment": slug,
                            "model": variant,
                            "subject_id": str(subject_ids[i]),
                            "scenario_label": scenario_label,
                            "held_out_ll": float(lls[j]),
                        }
                    )
            else:
                test_nll = 0.0

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

    return pred_rows, fold_rows, trial_ll_rows


def main_intimacy():
    slug = "food_inv_intimacy"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(slug, *_loso_intimacy(slug))


# ==============================================================================
# Study 1a — infer desire given (effort, intimacy)
# ==============================================================================


def _loso_desire(slug):
    data, action, scenario_idx, effort_condition, relationship_condition, response = (
        load_desire_data(slug)
    )
    scenario_idx_np = np.asarray(scenario_idx)
    action_np = np.asarray(action)
    effort_np = np.asarray(effort_condition)
    rel_np = np.asarray(relationship_condition)
    response_np = np.asarray(response)
    subject_ids = np.asarray(data["subject_id"].values)
    # Warm-start source: the full-data fit (refits perturb it only slightly).
    full_fit = (
        load_fit_results(slug)
        if (
            get_project_root() / "model" / "outputs" / slug / "fit_results.json"
        ).exists()
        else {}
    )

    pred_rows, fold_rows, trial_ll_rows = [], [], []

    for variant, (obs_fn, utility_names) in VARIANTS_DESIRE.items():
        tk = desire_table_kwargs(utility_names)
        warm = (
            params_dict_to_array(full_fit[variant], utility_names)
            if variant in full_fit
            else None
        )
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
                init_params=warm,
            )
            sigma = float(params[-1])
            # (run, slot, scenario, observed_action, effort, intimacy, desire_101)
            tables = np.asarray(
                _build_observer_tables_runs(obs_fn, params, utility_names, tk)
            )

            for a_idx in range(N_ACTIONS):
                for rel_idx in range(4):
                    for e in (0, 1):
                        density_runs = tables[
                            :, 0, fold, a_idx, e, rel_idx, :
                        ]  # (K,101)
                        deltas = density_runs @ GRID_NP - PRIOR_MEAN_F
                        pred_rows.append(
                            {
                                "scenario_label": scenario_label,
                                "action": a_idx,
                                "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel_idx],
                                "effort_condition": "low" if e == 0 else "high",
                                "delta_desire": float(deltas.mean()),
                                "model": variant,
                            }
                        )

            ti = np.where(test_mask)[0]
            if len(ti):
                post = tables[
                    :,
                    0,
                    scenario_idx_np[ti],
                    action_np[ti],
                    effort_np[ti],
                    rel_np[ti],
                    :,
                ]  # (K, n_test, 101)
                deltas_t = (post @ GRID_NP).T - PRIOR_MEAN_F
                lls = _held_out_ll_1d(deltas_t, response_np[ti], sigma)
                test_nll = -float(lls.sum())
                for j, i in enumerate(ti):
                    trial_ll_rows.append(
                        {
                            "experiment": slug,
                            "model": variant,
                            "subject_id": str(subject_ids[i]),
                            "scenario_label": scenario_label,
                            "held_out_ll": float(lls[j]),
                        }
                    )
            else:
                test_nll = 0.0

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

    return pred_rows, fold_rows, trial_ll_rows


def main_desire():
    slug = "food_inv_desire"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(slug, *_loso_desire(slug))


# ==============================================================================
# Study 1b — joint over (desire, effort) given intimacy
# ==============================================================================


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
    action_np = np.asarray(action)
    rel_np = np.asarray(relationship_condition)
    rd_np = np.asarray(response_desire)
    re_np = np.asarray(response_effort)
    subject_ids = np.asarray(data["subject_id"].values)
    # Warm-start source: the full-data fit (refits perturb it only slightly).
    full_fit = (
        load_fit_results(slug)
        if (
            get_project_root() / "model" / "outputs" / slug / "fit_results.json"
        ).exists()
        else {}
    )

    pred_rows, fold_rows, trial_ll_rows = [], [], []

    for variant, (obs_fn, utility_names) in VARIANTS_JOINT_DE.items():
        tk = joint_de_table_kwargs(utility_names)
        warm = (
            params_dict_to_array(full_fit[variant], utility_names)
            if variant in full_fit
            else None
        )
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
                init_params=warm,
            )
            sigma = float(params[-1])
            # (run, slot, scenario, observed_action, relationship_4, desire_101, effort_2)
            tables = np.asarray(
                _build_observer_tables_runs(obs_fn, params, utility_names, tk)
            )

            for a_idx in range(N_ACTIONS):
                for rel_idx in range(4):
                    joint_runs = tables[:, 0, fold, a_idx, rel_idx, :, :]  # (K,101,2)
                    desire_mean = joint_runs.sum(axis=2) @ GRID_NP  # (K,)
                    p_high = joint_runs[:, :, 1].sum(axis=1)  # (K,)
                    pred_rows.append(
                        {
                            "scenario_label": scenario_label,
                            "action": a_idx,
                            "intimacy_condition": INTIMACY_IDX_TO_LEVEL[rel_idx],
                            "delta_desire": float((desire_mean - PRIOR_MEAN_F).mean()),
                            "delta_effort": float(
                                (p_high - EFFORT_PRIOR_MEAN_F).mean()
                            ),
                            "model": variant,
                        }
                    )

            ti = np.where(test_mask)[0]
            if len(ti):
                joint_t = tables[
                    :, 0, scenario_idx_np[ti], action_np[ti], rel_np[ti], :, :
                ]  # (K, n_test, 101, 2)
                desire_mean_t = joint_t.sum(axis=3) @ GRID_NP  # (K, n_test)
                p_high_t = joint_t[:, :, :, 1].sum(axis=2)  # (K, n_test)
                deltas_t = np.stack(
                    [desire_mean_t - PRIOR_MEAN_F, p_high_t - EFFORT_PRIOR_MEAN_F],
                    axis=-1,
                )  # (K, n_test, 2)
                deltas_t = np.transpose(deltas_t, (1, 0, 2))  # (n_test, K, 2)
                u_t = np.stack([rd_np[ti], re_np[ti]], axis=1)  # (n_test, 2)
                lls = _held_out_ll_2d(deltas_t, u_t, sigma)
                test_nll = -float(lls.sum())
                for j, i in enumerate(ti):
                    trial_ll_rows.append(
                        {
                            "experiment": slug,
                            "model": variant,
                            "subject_id": str(subject_ids[i]),
                            "scenario_label": scenario_label,
                            "held_out_ll": float(lls[j]),
                        }
                    )
            else:
                test_nll = 0.0

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

    return pred_rows, fold_rows, trial_ll_rows


def main_joint_de():
    slug = "food_inv_joint_de"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(slug, *_loso_joint_de(slug))


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
    action_np = np.asarray(action)
    desire_np = np.asarray(desire_condition)
    ri_np = np.asarray(response_intimacy)
    re_np = np.asarray(response_effort)
    subject_ids = np.asarray(data["subject_id"].values)
    # Warm-start source: the full-data fit (refits perturb it only slightly).
    full_fit = (
        load_fit_results(slug)
        if (
            get_project_root() / "model" / "outputs" / slug / "fit_results.json"
        ).exists()
        else {}
    )

    pred_rows, fold_rows, trial_ll_rows = [], [], []

    for variant, (obs_fn, utility_names) in VARIANTS_JOINT_IE.items():
        tk = joint_ie_table_kwargs(utility_names)
        warm = (
            params_dict_to_array(full_fit[variant], utility_names)
            if variant in full_fit
            else None
        )
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
                init_params=warm,
            )
            sigma = float(params[-1])
            # (run, slot, scenario, observed_action, desire, intimacy_101, effort_2)
            tables = np.asarray(
                _build_observer_tables_runs(obs_fn, params, utility_names, tk)
            )

            for a_idx in range(N_ACTIONS):
                for r in (0, 1):
                    joint_runs = tables[:, 0, fold, a_idx, r, :, :]  # (K,101,2)
                    intimacy_mean = joint_runs.sum(axis=2) @ GRID_NP  # (K,)
                    p_high = joint_runs[:, :, 1].sum(axis=1)  # (K,)
                    pred_rows.append(
                        {
                            "scenario_label": scenario_label,
                            "action": a_idx,
                            "desire_condition": "low" if r == 0 else "high",
                            "delta_intimacy": float(
                                (intimacy_mean - PRIOR_MEAN_F).mean()
                            ),
                            "delta_effort": float(
                                (p_high - EFFORT_PRIOR_MEAN_F).mean()
                            ),
                            "model": variant,
                        }
                    )

            ti = np.where(test_mask)[0]
            if len(ti):
                joint_t = tables[
                    :, 0, scenario_idx_np[ti], action_np[ti], desire_np[ti], :, :
                ]  # (K, n_test, 101, 2)
                intimacy_mean_t = joint_t.sum(axis=3) @ GRID_NP  # (K, n_test)
                p_high_t = joint_t[:, :, :, 1].sum(axis=2)  # (K, n_test)
                deltas_t = np.stack(
                    [intimacy_mean_t - PRIOR_MEAN_F, p_high_t - EFFORT_PRIOR_MEAN_F],
                    axis=-1,
                )
                deltas_t = np.transpose(deltas_t, (1, 0, 2))  # (n_test, K, 2)
                u_t = np.stack([ri_np[ti], re_np[ti]], axis=1)  # (n_test, 2)
                lls = _held_out_ll_2d(deltas_t, u_t, sigma)
                test_nll = -float(lls.sum())
                for j, i in enumerate(ti):
                    trial_ll_rows.append(
                        {
                            "experiment": slug,
                            "model": variant,
                            "subject_id": str(subject_ids[i]),
                            "scenario_label": scenario_label,
                            "held_out_ll": float(lls[j]),
                        }
                    )
            else:
                test_nll = 0.0

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

    return pred_rows, fold_rows, trial_ll_rows


def main_joint_ie():
    slug = "food_inv_joint_ie"
    print("=" * 60)
    print(f"LOSO CV: {slug}")
    print("=" * 60)
    _write_outputs(slug, *_loso_joint_ie(slug))
