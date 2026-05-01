"""
Shared infrastructure for forward-planning fit + predict scripts.

Two flavors of prediction/fit code:
- canonical 4-action × motivation IV (food + nonfood)
- effort 2-action × effort IV (food)

Each per-experiment fit script imports the shared helpers it needs and
runs the 3 ablation fits (full / discomfort_only / base). Each predict
script imports the predict_* functions and runs them with parameters
read back from fit_results.csv.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy import stats

from actors import (
    actor_forw_base,
    actor_forw_discomfort_only,
    actor_forw_effort_base,
    actor_forw_effort_discomfort_only,
    actor_forw_effort_full,
    actor_forw_full,
)
from tables import EFFORT_CONDITION_TO_IDX, SCENARIO_TO_IDX

INTIMACY_MAP = {0: 0.0, 50: 0.5, 75: 0.75, 100: 1.0}


# ==============================================================================
# Loss + model comparison helpers
# ==============================================================================


@jax.jit
def compute_nll(preds, responses):
    """Negative log-likelihood. NLL = -sum(responses · log(preds))."""
    epsilon = 1e-8
    preds_safe = jnp.clip(preds, epsilon, 1.0)
    responses_safe = jnp.clip(responses, epsilon, 1.0)
    return -jnp.sum(responses_safe * jnp.log(preds_safe))


def compute_aic(nll, n_params):
    return 2 * n_params + 2 * nll


def compute_bic(nll, n_params, n_obs):
    return n_params * np.log(n_obs) + 2 * nll


def compute_pearson_r_by_condition(data, pred_col, human_col, group_cols, n_boot=1000):
    """Pearson r at condition × action level with bootstrap 95% CI."""
    agg = (
        data.groupby(group_cols)
        .agg({pred_col: "mean", human_col: "mean"})
        .reset_index()
    )
    r, p = stats.pearsonr(agg[pred_col], agg[human_col])

    np.random.seed(42)
    boot_rs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(agg), size=len(agg), replace=True)
        boot_pred = agg[pred_col].iloc[idx].values
        boot_human = agg[human_col].iloc[idx].values
        boot_r, _ = stats.pearsonr(boot_pred, boot_human)
        boot_rs.append(boot_r)
    return {
        "r": r, "p": p,
        "ci_lower": np.percentile(boot_rs, 2.5),
        "ci_upper": np.percentile(boot_rs, 97.5),
    }


def _fit_with_adam(loss_fn, init_params, lr=0.01, max_steps=5000, verbose=True, label=""):
    """Adam fit loop with non-negativity clipping and NLL monotonicity stop."""
    params = jnp.array(init_params)
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_nll = None
    for step in range(max_steps):
        nll, grad = grad_fn(params)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 1e-6, jnp.inf)

        if verbose and step % 1000 == 0:
            print(f"  Step {step}, NLL: {nll:.4f}, params: {params}")

        if prev_nll is not None and nll > prev_nll + 1e-6:
            if verbose:
                print(f"  NLL increased at step {step}, stopping")
            break
        prev_nll = nll

    best_nll = float(loss_fn(params))
    if verbose:
        print(f"  {label} final NLL: {best_nll:.4f}, params: {params}")
    return params, best_nll


def get_intimacy_index(intimacy_value):
    """Convert intimacy in [0, 1] to index into the 101-level IntimacyLevels axis."""
    return jnp.round(intimacy_value * 100).astype(int)


# ==============================================================================
# Canonical (4-action) prediction + fit
# ==============================================================================


@jax.jit
def predict_canonical_full(
    intimacy, condition_iv, action, scenario_idx,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_full(
        alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_table,
    )
    return jax.vmap(lambda i, c, a, s: probs[a, s, i, c])(
        intimacy_idx, condition_iv, action, scenario_idx,
    )


@jax.jit
def predict_canonical_discomfort_only(
    intimacy, condition_iv, action, scenario_idx,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_discomfort_only(
        alpha, w_d, gamma, access_table, effort_table,
    )
    return jax.vmap(lambda i, c, a, s: probs[a, s, i, c])(
        intimacy_idx, condition_iv, action, scenario_idx,
    )


@jax.jit
def predict_canonical_base(
    intimacy, condition_iv, action, scenario_idx,
    alpha, w_v, w_e,
    access_table, effort_table, v_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_base(
        alpha, w_v, w_e, access_table, effort_table, v_table,
    )
    return jax.vmap(lambda i, c, a, s: probs[a, s, i, c])(
        intimacy_idx, condition_iv, action, scenario_idx,
    )


def fit_canonical_full(intimacy, condition_iv, action, scenario_idx, p_action, tables, **kwargs):
    """tables = (access, effort, v). 4 free params: w_v, w_d, w_e, gamma."""
    ALPHA = 1.0
    a_tab, e_tab, v_tab = tables

    def loss_fn(params):
        w_v, w_d, w_e, gamma = params
        preds = predict_canonical_full(
            intimacy, condition_iv, action, scenario_idx,
            ALPHA, w_v, w_d, w_e, gamma, a_tab, e_tab, v_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(loss_fn, [1.0, 1.0, 1.0, 1.0], label="full", **kwargs)
    return jnp.array([ALPHA, params[0], params[1], params[2], params[3]]), nll


def fit_canonical_discomfort_only(intimacy, condition_iv, action, scenario_idx, p_action, tables, **kwargs):
    """tables = (access, effort). 2 free params: w_d, gamma."""
    ALPHA = 1.0
    a_tab, e_tab = tables[:2]

    def loss_fn(params):
        w_d, gamma = params
        preds = predict_canonical_discomfort_only(
            intimacy, condition_iv, action, scenario_idx,
            ALPHA, w_d, gamma, a_tab, e_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(loss_fn, [1.0, 1.0], label="discomfort_only", **kwargs)
    return jnp.array([ALPHA, params[0], params[1]]), nll


def fit_canonical_base(intimacy, condition_iv, action, scenario_idx, p_action, tables, **kwargs):
    """tables = (access, effort, v). 2 free params: w_v, w_e."""
    ALPHA = 1.0
    a_tab, e_tab, v_tab = tables

    def loss_fn(params):
        w_v, w_e = params
        preds = predict_canonical_base(
            intimacy, condition_iv, action, scenario_idx,
            ALPHA, w_v, w_e, a_tab, e_tab, v_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(loss_fn, [1.0, 1.0], label="base", **kwargs)
    return jnp.array([ALPHA, params[0], params[1]]), nll


# ==============================================================================
# Effort (2-action) prediction + fit (V stipulated to 1)
# ==============================================================================


@jax.jit
def predict_effort_full(
    intimacy, condition_iv, action, scenario_idx,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_effort_full(
        alpha, w_v, w_d, w_e, gamma, access_table, effort_table,
    )
    return jax.vmap(lambda i, c, a, s: probs[a, s, i, c])(
        intimacy_idx, condition_iv, action, scenario_idx,
    )


@jax.jit
def predict_effort_discomfort_only(
    intimacy, condition_iv, action, scenario_idx,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_effort_discomfort_only(
        alpha, w_d, gamma, access_table, effort_table,
    )
    return jax.vmap(lambda i, c, a, s: probs[a, s, i, c])(
        intimacy_idx, condition_iv, action, scenario_idx,
    )


@jax.jit
def predict_effort_base(
    intimacy, condition_iv, action, scenario_idx,
    alpha, w_v, w_e,
    access_table, effort_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_effort_base(
        alpha, w_v, w_e, access_table, effort_table,
    )
    return jax.vmap(lambda i, c, a, s: probs[a, s, i, c])(
        intimacy_idx, condition_iv, action, scenario_idx,
    )


def fit_effort_full(intimacy, condition_iv, action, scenario_idx, p_action, tables, **kwargs):
    ALPHA = 1.0
    a_tab, e_tab = tables

    def loss_fn(params):
        w_v, w_d, w_e, gamma = params
        preds = predict_effort_full(
            intimacy, condition_iv, action, scenario_idx,
            ALPHA, w_v, w_d, w_e, gamma, a_tab, e_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(loss_fn, [1.0, 1.0, 1.0, 1.0], label="full", **kwargs)
    return jnp.array([ALPHA, params[0], params[1], params[2], params[3]]), nll


def fit_effort_discomfort_only(intimacy, condition_iv, action, scenario_idx, p_action, tables, **kwargs):
    ALPHA = 1.0
    a_tab, e_tab = tables

    def loss_fn(params):
        w_d, gamma = params
        preds = predict_effort_discomfort_only(
            intimacy, condition_iv, action, scenario_idx,
            ALPHA, w_d, gamma, a_tab, e_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(loss_fn, [1.0, 1.0], label="discomfort_only", **kwargs)
    return jnp.array([ALPHA, params[0], params[1]]), nll


def fit_effort_base(intimacy, condition_iv, action, scenario_idx, p_action, tables, **kwargs):
    ALPHA = 1.0
    a_tab, e_tab = tables

    def loss_fn(params):
        w_v, w_e = params
        preds = predict_effort_base(
            intimacy, condition_iv, action, scenario_idx,
            ALPHA, w_v, w_e, a_tab, e_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(loss_fn, [1.0, 1.0], label="base", **kwargs)
    return jnp.array([ALPHA, params[0], params[1]]), nll


# ==============================================================================
# Data loading
# ==============================================================================


def load_data_canonical(filepath, scenario_to_idx):
    """Load canonical 4-action × motivation forward planning data (food/nonfood)."""
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
    data["intimacy_scaled"] = data["intimacy"].map(INTIMACY_MAP)
    motivation_map = {"low": 0, "high": 1}
    data["condition_iv"] = data["motivation"].map(motivation_map)
    data["scenario_idx"] = data["scenario_label"].map(scenario_to_idx)

    intimacy = jnp.array(data["intimacy_scaled"].values)
    condition_iv = jnp.array(data["condition_iv"].values)
    action = jnp.array(data["action"].values)
    p_action = jnp.array(data["p_action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)

    print(f"Loaded {len(data)} data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")
    return data, intimacy, condition_iv, action, p_action, scenario_idx


def load_data_effort(filepath):
    """Load food_forw_intimacy_effort data (2-action × effort)."""
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
    data["intimacy_scaled"] = data["intimacy"].map(INTIMACY_MAP)
    data["condition_iv"] = data["effort"].map(EFFORT_CONDITION_TO_IDX)
    # CSV action 1/2 -> internal 0/1 (0 = non-share, 1 = saliva-share)
    data["action_idx"] = data["action"].astype(int) - 1
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    intimacy = jnp.array(data["intimacy_scaled"].values)
    condition_iv = jnp.array(data["condition_iv"].values)
    action = jnp.array(data["action_idx"].values)
    p_action = jnp.array(data["p_action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)

    print(f"Loaded {len(data)} data points")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    print(f"  Scenarios: {data['scenario_label'].nunique()}")
    return data, intimacy, condition_iv, action, p_action, scenario_idx


# ==============================================================================
# Shared fit + predict orchestration helpers (used by per-experiment scripts)
# ==============================================================================


def run_fit_and_save_results(
    *,
    experiment_slug,
    intimacy, condition_iv, action, scenario_idx, p_action,
    tables_by_variant,
    fit_funcs,
    group_cols,
    data,
):
    """Fit the 3 ablations and write outputs/<slug>/fit_results.csv.

    `fit_funcs` is a dict mapping variant_name -> (fit_fn, predict_fn, param_names).
    `tables_by_variant` is a dict mapping variant_name -> tables tuple.
    Returns the results dict (used by the predict script if combined; not used otherwise).
    """
    print("=" * 60)
    print(f"Forward planning fit: {experiment_slug}")
    print("=" * 60)

    results = {}
    param_arrays = {}
    for name, (fit_fn, _pred_fn, param_names) in fit_funcs.items():
        print("\n" + "-" * 40)
        print(f"Fitting {name.upper()} model (alpha=1 fixed)...")
        print("-" * 40)
        params, nll = fit_fn(
            intimacy, condition_iv, action, scenario_idx, p_action,
            tables_by_variant[name],
        )
        param_arrays[name] = params
        results[name] = {
            "params": {
                "alpha": float(params[0]),
                **{pn: float(params[i + 1]) for i, pn in enumerate(param_names)},
            },
            "nll": nll,
            "n_params": len(param_names),
        }

    # Compute AIC/BIC/r at the (group_cols) level
    n_obs = len(data)
    model_metrics = {}
    for name in fit_funcs.keys():
        nll = results[name]["nll"]
        n_params = results[name]["n_params"]
        aic = compute_aic(nll, n_params)
        bic = compute_bic(nll, n_params, n_obs)
        # Need to compute predictions to get r — do it with the fitted params
        _fit_fn, pred_fn, _param_names = fit_funcs[name]
        params = param_arrays[name]
        data[f"pred_{name}"] = np.array(
            pred_fn(
                intimacy, condition_iv, action, scenario_idx,
                *params, *tables_by_variant[name],
            )
        )
        r_result = compute_pearson_r_by_condition(
            data, f"pred_{name}", "p_action", group_cols
        )
        model_metrics[name] = {
            "aic": aic, "bic": bic,
            "r": r_result["r"],
            "r_ci_lower": r_result["ci_lower"],
            "r_ci_upper": r_result["ci_upper"],
        }
        print(
            f"  {name}: AIC={aic:.2f}, BIC={bic:.2f}, "
            f"r={r_result['r']:.3f} [{r_result['ci_lower']:.3f}, {r_result['ci_upper']:.3f}]"
        )

    # Save fit_results.csv
    rows = []
    for name in fit_funcs.keys():
        rows.append({
            "model": name,
            "experiment": experiment_slug,
            "nll": results[name]["nll"],
            "n_params": results[name]["n_params"],
            "aic": model_metrics[name]["aic"],
            "bic": model_metrics[name]["bic"],
            "r": model_metrics[name]["r"],
            "r_ci_lower": model_metrics[name]["r_ci_lower"],
            "r_ci_upper": model_metrics[name]["r_ci_upper"],
            **{f"param_{k}": v for k, v in results[name]["params"].items()},
        })

    output_dir = Path(__file__).resolve().parent.parent / "outputs" / experiment_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "fit_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")
    return results


def run_predict_and_save_preds(
    *,
    experiment_slug,
    cells_df,
    iv_idx_col,
    tables_by_variant,
    predict_funcs,
    fit_param_names,
):
    """Read outputs/<slug>/fit_results.csv, generate per-cell predictions, write preds.csv.

    Forward predictions are inherently per-cell: for any (scenario, action, intimacy,
    IV) cell, the model's prediction is the same regardless of which subject was
    observed in that cell. So `preds.csv` enumerates the unique cells (one row per
    cell, with `pred_<variant>` columns), not the per-trial human data. The
    per-trial data lives in `data/<slug>/main_trials_long.csv`; analysis qmds can
    join the two on the cell columns if needed.

    `cells_df` is a DataFrame with one row per (scenario, action, intimacy, IV)
    cell, including:
      - `scenario_label` (str), `scenario_idx` (int)
      - `action` (int — internal index for prediction; effort scripts pass 0/1
        even though the human-facing label is 1/2)
      - `intimacy_scaled` (float in [0, 1] — what predict_fn consumes)
      - `<iv_idx_col>` (int — the condition IV index, e.g. motivation_idx or effort_idx)
      - any additional human-readable columns to preserve in the output (e.g.
        `intimacy` as 0/50/75/100, `motivation` as "low"/"high")

    `iv_idx_col` is the name of the integer-IV column in cells_df (e.g.
    `"motivation_idx"` for canonical, `"effort_idx"` for effort).

    `predict_funcs` maps variant_name → predict_fn. `fit_param_names` maps
    variant_name → ordered list of param names after `alpha`.
    """
    output_dir = Path(__file__).resolve().parent.parent / "outputs" / experiment_slug
    fit_results_path = output_dir / "fit_results.csv"
    fit_results = pd.read_csv(fit_results_path)

    print(f"Loading fit results from {fit_results_path}...")
    print(fit_results.to_string(index=False))

    intimacy_arr = jnp.array(cells_df["intimacy_scaled"].values)
    condition_iv_arr = jnp.array(cells_df[iv_idx_col].values)
    action_arr = jnp.array(cells_df["action"].values)
    scenario_idx_arr = jnp.array(cells_df["scenario_idx"].values)

    out_df = cells_df.copy()
    for name, pred_fn in predict_funcs.items():
        row = fit_results[fit_results["model"] == name].iloc[0]
        param_names = fit_param_names[name]
        params = jnp.array(
            [row["param_alpha"]] + [row[f"param_{pn}"] for pn in param_names]
        )
        out_df[f"pred_{name}"] = np.array(
            pred_fn(
                intimacy_arr, condition_iv_arr, action_arr, scenario_idx_arr,
                *params, *tables_by_variant[name],
            )
        )

    preds_path = output_dir / "preds.csv"
    out_df.to_csv(preds_path, index=False)
    print(f"\nSaved per-cell predictions ({len(out_df)} rows) to {preds_path}")


def build_canonical_cells(scenario_labels):
    """Enumerate (scenario, action, intimacy, motivation) cells for canonical 4-action."""
    intimacy_levels = [0, 50, 75, 100]  # human-facing
    rows = []
    for s_idx, scenario_label in enumerate(scenario_labels):
        for action_int in range(4):
            for intimacy_int in intimacy_levels:
                for motivation_str, motivation_idx in (("low", 0), ("high", 1)):
                    rows.append({
                        "scenario_label": scenario_label,
                        "scenario_idx": s_idx,
                        "action": action_int,
                        "intimacy": intimacy_int,
                        "intimacy_scaled": INTIMACY_MAP[intimacy_int],
                        "motivation": motivation_str,
                        "motivation_idx": motivation_idx,
                    })
    return pd.DataFrame(rows)


def build_effort_cells(scenario_labels):
    """Enumerate (scenario, action, intimacy, effort_condition) cells for effort 2-action."""
    intimacy_levels = [0, 50, 75, 100]
    rows = []
    for s_idx, scenario_label in enumerate(scenario_labels):
        for action_int in range(2):  # internal 0/1 (CSV label 1/2)
            for intimacy_int in intimacy_levels:
                for effort_str, effort_idx in (("low", 0), ("high", 1)):
                    rows.append({
                        "scenario_label": scenario_label,
                        "scenario_idx": s_idx,
                        "action": action_int,
                        "action_csv": action_int + 1,  # 1/2 (matches data CSV)
                        "intimacy": intimacy_int,
                        "intimacy_scaled": INTIMACY_MAP[intimacy_int],
                        "effort": effort_str,
                        "effort_idx": effort_idx,
                    })
    return pd.DataFrame(rows)
