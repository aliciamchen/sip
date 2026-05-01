"""
Fit forward-planning models to human data — unified driver for all three
forward-planning experiments.

Three access-model ablations are fit to each experiment:
  - full: w_v · V - w_d · access · (1 - I)^γ - w_e · effort
  - discomfort_only: only the access-discomfort term, V-independent
  - base: w_v · V - w_e · effort (no relational structure)

Three experiments are dispatched by `--experiment <slug>`:
  - food_forw_intimacy_desire: canonical 4-action × motivation IV (food)
  - nonfood_forw_intimacy_desire: canonical 4-action × motivation IV (nonfood)
  - food_forw_intimacy_effort: 2-action × effort IV (food, V stipulated to 1)

Uses maximum likelihood estimation with gradient descent (optax.adam). Output
filenames follow the pre-existing conventions and will be reorganized in a
later pass (Step 7 of the model/ refactor).

Shared infrastructure (NLL, AIC/BIC, Pearson r, Adam fit loop) lives at the
top of this file and is reused across both branches.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy import stats

from tables import (
    EFFORT_CONDITION_TO_IDX,
    LLM_TABLES,
    LLM_TABLES_EFFORT,
    SCENARIO_TO_IDX,
    load_domain_assets,
    load_lm_v,
)
from actors import (
    actor_forw_base,
    actor_forw_discomfort_only,
    actor_forw_effort_base,
    actor_forw_effort_discomfort_only,
    actor_forw_effort_full,
    actor_forw_full,
)

from utils import get_project_root


INTIMACY_MAP = {0: 0.0, 50: 0.5, 75: 0.75, 100: 1.0}


# ==============================================================================
# Shared infrastructure
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
# Canonical (4-action) prediction + fit functions
# ==============================================================================
# Used by food_forw_intimacy_desire and nonfood_forw_intimacy_desire. The
# "condition_iv" axis here is reward_condition (motivation: low/high).


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
    """tables = (access, effort) — V-independent. 2 free params: w_d, gamma."""
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
# Effort (2-action) prediction + fit functions
# ==============================================================================
# Used by food_forw_intimacy_effort. The "condition_iv" axis here is
# effort_condition (low/high). V is stipulated to 1 inside utility — so the
# v_table table parameter does not appear in these signatures.


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
    """tables = (access, effort). 4 free params: w_v, w_d, w_e, gamma."""
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


def load_data_canonical(filepath=None, scenario_to_idx=None):
    """Load canonical 4-action × motivation forward planning data (food/nonfood)."""
    if filepath is None:
        filepath = get_project_root() / "data" / "food_forw_intimacy_desire" / "main_trials_long.csv"
    if scenario_to_idx is None:
        scenario_to_idx = SCENARIO_TO_IDX
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


def load_data_effort(filepath=None):
    """Load food_forw_intimacy_effort data (2-action × effort)."""
    if filepath is None:
        filepath = get_project_root() / "data" / "food_forw_intimacy_effort" / "main_trials_long.csv"
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
# Experiment registry
# ==============================================================================


def _canonical_food_config():
    _, scenario_to_idx, llm_tables = load_domain_assets("food")
    v_table = load_lm_v("food")
    return {
        "data_path": get_project_root() / "data" / "food_forw_intimacy_desire" / "main_trials_long.csv",
        "data_loader": lambda fp: load_data_canonical(fp, scenario_to_idx),
        "tables_full": (llm_tables["access"], llm_tables["effort"], v_table),
        "tables_discomfort_only": (llm_tables["access"], llm_tables["effort"]),
        "tables_base": (llm_tables["access"], llm_tables["effort"], v_table),
        "fit_funcs": {
            "full": (fit_canonical_full, predict_canonical_full, ["w_v", "w_d", "w_e", "gamma"]),
            "discomfort_only": (fit_canonical_discomfort_only, predict_canonical_discomfort_only, ["w_d", "gamma"]),
            "base": (fit_canonical_base, predict_canonical_base, ["w_v", "w_e"]),
        },
        "group_cols": ["intimacy", "motivation", "action"],
        "slug": "food_forw_intimacy_desire", "fits_filename": "fits.csv",
        "results_filename": "fit_results.csv",
    }


def _canonical_nonfood_config():
    _, scenario_to_idx, llm_tables = load_domain_assets("nonfood")
    v_table = load_lm_v("nonfood")
    return {
        "data_path": get_project_root() / "data" / "nonfood_forw_intimacy_desire" / "main_trials_long.csv",
        "data_loader": lambda fp: load_data_canonical(fp, scenario_to_idx),
        "tables_full": (llm_tables["access"], llm_tables["effort"], v_table),
        "tables_discomfort_only": (llm_tables["access"], llm_tables["effort"]),
        "tables_base": (llm_tables["access"], llm_tables["effort"], v_table),
        "fit_funcs": {
            "full": (fit_canonical_full, predict_canonical_full, ["w_v", "w_d", "w_e", "gamma"]),
            "discomfort_only": (fit_canonical_discomfort_only, predict_canonical_discomfort_only, ["w_d", "gamma"]),
            "base": (fit_canonical_base, predict_canonical_base, ["w_v", "w_e"]),
        },
        "group_cols": ["intimacy", "motivation", "action"],
        "slug": "nonfood_forw_intimacy_desire", "fits_filename": "fits.csv",
        "results_filename": "fit_results.csv",
    }


def _effort_config():
    return {
        "data_path": get_project_root() / "data" / "food_forw_intimacy_effort" / "main_trials_long.csv",
        "data_loader": lambda fp: load_data_effort(fp),
        "tables_full": (LLM_TABLES_EFFORT["access"], LLM_TABLES_EFFORT["effort"]),
        "tables_discomfort_only": (LLM_TABLES_EFFORT["access"], LLM_TABLES_EFFORT["effort"]),
        "tables_base": (LLM_TABLES_EFFORT["access"], LLM_TABLES_EFFORT["effort"]),
        "fit_funcs": {
            "full": (fit_effort_full, predict_effort_full, ["w_v", "w_d", "w_e", "gamma"]),
            "discomfort_only": (fit_effort_discomfort_only, predict_effort_discomfort_only, ["w_d", "gamma"]),
            "base": (fit_effort_base, predict_effort_base, ["w_v", "w_e"]),
        },
        "group_cols": ["intimacy", "effort", "action"],
        "slug": "food_forw_intimacy_effort", "fits_filename": "fits.csv",
        "results_filename": "fit_results.csv",
    }


EXPERIMENT_CONFIGS = {
    "food_forw_intimacy_desire": _canonical_food_config,
    "nonfood_forw_intimacy_desire": _canonical_nonfood_config,
    "food_forw_intimacy_effort": _effort_config,
}


# ==============================================================================
# Main
# ==============================================================================


def main(experiment: str):
    if experiment not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment: {experiment!r}. "
            f"Choose from {list(EXPERIMENT_CONFIGS)}"
        )

    print("=" * 60)
    print(f"Forward Planning Model Fitting — experiment={experiment}")
    print("=" * 60)

    config = EXPERIMENT_CONFIGS[experiment]()
    data, intimacy, condition_iv, action, p_action, scenario_idx = config["data_loader"](config["data_path"])

    tables_by_variant = {
        "full": config["tables_full"],
        "discomfort_only": config["tables_discomfort_only"],
        "base": config["tables_base"],
    }

    results = {}
    param_arrays = {}
    for name, (fit_fn, _pred_fn, param_names) in config["fit_funcs"].items():
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

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  NLL: {result['nll']:.4f}")
        print(f"  Params: {result['params']}")

    # Per-trial predictions
    print("\n" + "-" * 40)
    print("Saving predictions...")
    print("-" * 40)
    for name, (_fit_fn, pred_fn, _param_names) in config["fit_funcs"].items():
        params = param_arrays[name]
        data[f"pred_{name}"] = np.array(
            pred_fn(
                intimacy, condition_iv, action, scenario_idx,
                *params, *tables_by_variant[name],
            )
        )

    output_dir = Path(__file__).parent / "outputs" / config["slug"]
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_path = output_dir / config["fits_filename"]
    data.to_csv(fits_path, index=False)
    print(f"Saved predictions to {fits_path}")

    # Model comparison
    print("\n" + "-" * 40)
    print("Computing model comparison metrics...")
    print("-" * 40)
    n_obs = len(data)
    model_metrics = {}
    for model_name in config["fit_funcs"].keys():
        nll = results[model_name]["nll"]
        n_params = results[model_name]["n_params"]
        aic = compute_aic(nll, n_params)
        bic = compute_bic(nll, n_params, n_obs)
        pred_col = f"pred_{model_name}"
        r_result = compute_pearson_r_by_condition(
            data, pred_col, "p_action", config["group_cols"]
        )
        model_metrics[model_name] = {
            "aic": aic, "bic": bic,
            "r": r_result["r"],
            "r_ci_lower": r_result["ci_lower"],
            "r_ci_upper": r_result["ci_upper"],
        }
        print(
            f"  {model_name}: AIC={aic:.2f}, BIC={bic:.2f}, "
            f"r={r_result['r']:.3f} [{r_result['ci_lower']:.3f}, {r_result['ci_upper']:.3f}]"
        )

    # Results CSV
    results_rows = []
    for model_name in config["fit_funcs"].keys():
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
    results_path = output_dir / config["results_filename"]
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fit results to {results_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit forward planning models.")
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENT_CONFIGS),
        default="food_forw_intimacy_desire",
        help="Which forward-planning experiment to fit.",
    )
    args = parser.parse_args()
    main(args.experiment)
