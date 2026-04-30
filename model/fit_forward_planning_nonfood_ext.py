"""
Forward-planning extensions (food and non-food). The file name keeps the
"_nonfood_ext" suffix from when this only supported non-food, but it now
accepts --domain food|nonfood — food support was added so the gamma
extension can be tested cross-domain. Neither branch perturbs the
canonical fit pipeline (`fit_forward_planning.py`); this script always
writes to its own `*_ext.csv` siblings.

Currently fits one variant:

  access_full_gamma — Full utility with a power-law intimacy modulator
      U = w_v * V - w_d * access * (1 - I)^gamma - w_e * effort

  Nests the canonical Full model (gamma = 1). 4 free params: w_v, w_d, w_e, gamma.
  alpha = 1 fixed for identifiability, mirroring the canonical pipeline.

Reuses:
  - load_data, _fit_with_adam, compute_nll, compute_aic, compute_bic,
    compute_pearson_r_by_condition  (from fit_forward_planning.py)
  - load_domain_assets, load_lm_v                          (from model_utils.py)
  - actor_forw_gamma, get_utility_gamma                    (from model_utils_nonfood_ext.py)

Outputs (in model/outputs/):
  - food:    forward_planning_fit_results_ext.csv,         forward_planning_fits_ext.csv
  - nonfood: forward_planning_fit_results_nonfood_ext.csv, forward_planning_fits_nonfood_ext.csv
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from fit_forward_planning import (
    _fit_with_adam,
    compute_aic,
    compute_bic,
    compute_nll,
    compute_pearson_r_by_condition,
    get_intimacy_index,
    load_data,
)
from model_utils import load_domain_assets, load_lm_v
from model_utils_nonfood_ext import (
    NONFOOD_SCENARIO_TYPE_IDX_TABLE,
    NONFOOD_TYPE_LABELS,
    actor_forw_gamma,
    actor_forw_gamma_vpow,
    actor_forw_typed_gamma,
)

from utils import get_project_root


def _w_d_per_scenario(w_d_per_type):
    """Broadcast a 3-vector (substance, space, privacy) to a 16-vector
    indexed by non-food scenario_idx using NONFOOD_SCENARIO_TYPE_IDX_TABLE.
    """
    return w_d_per_type[NONFOOD_SCENARIO_TYPE_IDX_TABLE]


@jax.jit
def predict_access_full_gamma(
    intimacy, reward_condition, action, scenario_idx,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_gamma(
        alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_table
    )
    return jax.vmap(lambda i, r, a, s: probs[a, s, i, r])(
        intimacy_idx, reward_condition, action, scenario_idx
    )


def fit_access_full_gamma_model(
    intimacy, reward_condition, action, scenario_idx, p_action, tables, **kwargs
):
    """tables = (access, effort, v). Init gamma = 1 (Full-model special case)."""
    ALPHA = 1.0
    a_tab, e_tab, v_tab = tables

    def loss_fn(params):
        w_v, w_d, w_e, gamma = params
        preds = predict_access_full_gamma(
            intimacy, reward_condition, action, scenario_idx,
            ALPHA, w_v, w_d, w_e, gamma, a_tab, e_tab, v_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0, 1.0, 1.0], label="access_full_gamma", **kwargs
    )
    return jnp.array([ALPHA, params[0], params[1], params[2], params[3]]), nll


@jax.jit
def predict_access_full_typed_gamma(
    intimacy, reward_condition, action, scenario_idx,
    alpha, w_v, w_d_substance, w_d_space, w_d_privacy, w_e, gamma,
    access_table, effort_table, v_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    w_d_per_type = jnp.array([w_d_substance, w_d_space, w_d_privacy])
    w_d_per_scen = _w_d_per_scenario(w_d_per_type)
    probs = actor_forw_typed_gamma(
        alpha, w_v, w_e, gamma, w_d_per_scen, access_table, effort_table, v_table
    )
    return jax.vmap(lambda i, r, a, s: probs[a, s, i, r])(
        intimacy_idx, reward_condition, action, scenario_idx
    )


def fit_access_full_gamma_alpha_model(
    intimacy, reward_condition, action, scenario_idx, p_action, tables, **kwargs
):
    """Reparameterized gamma model: w_v fixed = 1.0, α free.
    Reuses predict_access_full_gamma. 4 free params: α, w_d, w_e, γ.
    Should give identical NLL to access_full_gamma — softmax rescaling
    invariance means this is the same model in different coordinates."""
    W_V_FIXED = 1.0
    a_tab, e_tab, v_tab = tables

    def loss_fn(params):
        alpha, w_d, w_e, gamma = params
        preds = predict_access_full_gamma(
            intimacy, reward_condition, action, scenario_idx,
            alpha, W_V_FIXED, w_d, w_e, gamma, a_tab, e_tab, v_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0, 1.0, 1.0], label="access_full_gamma_alpha", **kwargs
    )
    # Pack as [alpha, w_v=1 (fixed), w_d, w_e, gamma] for main() compatibility
    return jnp.array([
        float(params[0]),  # alpha (free)
        W_V_FIXED,
        float(params[1]),  # w_d
        float(params[2]),  # w_e
        float(params[3]),  # gamma
    ]), nll


@jax.jit
def predict_access_full_gamma_vpow(
    intimacy, reward_condition, action, scenario_idx,
    alpha, w_v, w_d, w_e, gamma, beta,
    access_table, effort_table, v_table,
):
    intimacy_idx = get_intimacy_index(intimacy)
    probs = actor_forw_gamma_vpow(
        alpha, w_v, w_d, w_e, gamma, beta, access_table, effort_table, v_table
    )
    return jax.vmap(lambda i, r, a, s: probs[a, s, i, r])(
        intimacy_idx, reward_condition, action, scenario_idx
    )


def fit_access_full_gamma_vpow_model(
    intimacy, reward_condition, action, scenario_idx, p_action, tables, **kwargs
):
    """V_eff = sign(V) * |V|^β; α=1 fixed. 5 free params: w_v, w_d, w_e, γ, β.
    Init β=1 (identity — matches gamma model)."""
    ALPHA = 1.0
    a_tab, e_tab, v_tab = tables

    def loss_fn(params):
        w_v, w_d, w_e, gamma, beta = params
        preds = predict_access_full_gamma_vpow(
            intimacy, reward_condition, action, scenario_idx,
            ALPHA, w_v, w_d, w_e, gamma, beta, a_tab, e_tab, v_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0, 1.0, 1.0, 1.0],
        label="access_full_gamma_vpow", **kwargs
    )
    return jnp.array([ALPHA] + [float(p) for p in params]), nll


def fit_access_full_typed_gamma_model(
    intimacy, reward_condition, action, scenario_idx, p_action, tables, **kwargs
):
    """tables = (access, effort, v). 6 free params:
       w_v, w_d_substance, w_d_space, w_d_privacy, w_e, gamma.
       Init: all weights at 1.0 (matches the canonical pipeline)."""
    ALPHA = 1.0
    a_tab, e_tab, v_tab = tables

    def loss_fn(params):
        w_v, w_d_s, w_d_sp, w_d_p, w_e, gamma = params
        preds = predict_access_full_typed_gamma(
            intimacy, reward_condition, action, scenario_idx,
            ALPHA, w_v, w_d_s, w_d_sp, w_d_p, w_e, gamma,
            a_tab, e_tab, v_tab,
        )
        return compute_nll(preds, p_action)

    params, nll = _fit_with_adam(
        loss_fn, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        label="access_full_typed_gamma", **kwargs
    )
    return jnp.array([ALPHA] + [float(p) for p in params]), nll


def main(domain: str = "nonfood"):
    if domain not in ("food", "nonfood"):
        raise ValueError(f"Unknown domain: {domain!r} (expected 'food' or 'nonfood')")

    print("=" * 60)
    print(f"Forward-planning extensions (domain={domain})")
    print("=" * 60)

    _, scenario_to_idx, llm_tables = load_domain_assets(domain)
    v_table = load_lm_v(domain)

    if domain == "food":
        data_path = get_project_root() / "data" / "forw_plan" / "main_trials_long.csv"
        fits_filename = "forward_planning_fits_ext.csv"
        results_filename = "forward_planning_fit_results_ext.csv"
    else:
        data_path = get_project_root() / "data" / "nonfood_forw_plan" / "main_trials_long.csv"
        fits_filename = "forward_planning_fits_nonfood_ext.csv"
        results_filename = "forward_planning_fit_results_nonfood_ext.csv"

    data, intimacy, reward_condition, action, p_action, scenario_idx = load_data(
        filepath=data_path, scenario_to_idx=scenario_to_idx,
    )

    tables = (llm_tables["access"], llm_tables["effort"], v_table)

    fits = {
        "access_full_gamma": (
            fit_access_full_gamma_model,
            predict_access_full_gamma,
            ["w_v", "w_d", "w_e", "gamma"],
        ),
    }
    if domain == "nonfood":
        # Per-channel w_d (substance/space/privacy) on top of gamma.
        # Channels are non-food-specific; food has only one channel.
        fits["access_full_typed_gamma"] = (
            fit_access_full_typed_gamma_model,
            predict_access_full_typed_gamma,
            ["w_v", "w_d_substance", "w_d_space", "w_d_privacy", "w_e", "gamma"],
        )
        # Decisiveness tests:
        # gamma_alpha = reparameterization (w_v=1 fixed, α free) — should
        # give identical NLL to gamma (softmax rescaling invariance check).
        fits["access_full_gamma_alpha"] = (
            fit_access_full_gamma_alpha_model,
            predict_access_full_gamma,
            ["w_v", "w_d", "w_e", "gamma"],
        )
        # gamma_vpow = V_eff = sign(V) * |V|^β; β free. Genuine new flexibility.
        fits["access_full_gamma_vpow"] = (
            fit_access_full_gamma_vpow_model,
            predict_access_full_gamma_vpow,
            ["w_v", "w_d", "w_e", "gamma", "beta"],
        )

    results = {}
    param_arrays = {}
    for name, (fit_fn, _pred_fn, param_names) in fits.items():
        print("\n" + "-" * 40)
        print(f"Fitting {name.upper()} model (alpha=1 fixed)...")
        print("-" * 40)
        params, nll = fit_fn(
            intimacy, reward_condition, action, scenario_idx, p_action, tables,
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

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  NLL: {result['nll']:.4f}")
        print(f"  Params: {result['params']}")

    print("\n" + "-" * 40)
    print("Saving predictions...")
    print("-" * 40)
    for name, (_fit_fn, pred_fn, _param_names) in fits.items():
        params = param_arrays[name]
        data[f"pred_{name}"] = np.array(
            pred_fn(
                intimacy, reward_condition, action, scenario_idx,
                *params, *tables,
            )
        )

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / fits_filename
    data.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    print("\n" + "-" * 40)
    print("Computing model comparison metrics...")
    print("-" * 40)
    n_obs = len(data)
    group_cols = ["intimacy", "motivation", "action"]
    model_metrics = {}

    for model_name in fits.keys():
        nll = results[model_name]["nll"]
        n_params = results[model_name]["n_params"]
        aic = compute_aic(nll, n_params)
        bic = compute_bic(nll, n_params, n_obs)

        pred_col = f"pred_{model_name}"
        r_result = compute_pearson_r_by_condition(
            data, pred_col, "p_action", group_cols
        )
        model_metrics[model_name] = {
            "aic": aic,
            "bic": bic,
            "r": r_result["r"],
            "r_ci_lower": r_result["ci_lower"],
            "r_ci_upper": r_result["ci_upper"],
        }
        print(
            f"  {model_name}: AIC={aic:.2f}, BIC={bic:.2f}, "
            f"r={r_result['r']:.3f} [{r_result['ci_lower']:.3f}, {r_result['ci_upper']:.3f}]"
        )

    results_rows = []
    for model_name in fits.keys():
        row = {
            "model": model_name,
            "nll": results[model_name]["nll"],
            "n_params": results[model_name]["n_params"],
            "aic": model_metrics[model_name]["aic"],
            "bic": model_metrics[model_name]["bic"],
            "r": model_metrics[model_name]["r"],
            "r_ci_lower": model_metrics[model_name]["r_ci_lower"],
            "r_ci_upper": model_metrics[model_name]["r_ci_upper"],
            **{f"param_{k}": v for k, v in results[model_name]["params"].items()},
        }
        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)
    results_path = output_dir / results_filename
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Fit forward-planning extensions (currently: access_full_gamma) on food or non-food."
    )
    parser.add_argument(
        "--domain", choices=("food", "nonfood"), default="nonfood",
        help="Which experiment to fit: 'food' (writes *_ext.csv) or 'nonfood' (writes *_nonfood_ext.csv, default). "
             "Neither branch overwrites the canonical (non-_ext) fit outputs.",
    )
    args = parser.parse_args()
    main(domain=args.domain)
