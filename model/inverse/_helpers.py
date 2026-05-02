"""
Shared infrastructure for inverse-planning fit + predict scripts.

Each experiment has its own thin fit/predict script that imports the helpers
it needs from this module. Shared concerns:

  - Loss functions (intimacy NLL, reward BCE NLL)
  - Observer fit loops (single-param alpha_observer; joint padded weights+α)
  - Data loaders (per experiment)
  - Frozen-actor-param loaders (forward fits → inverse fits)
  - Fitted-α_observer loader (alt-shown fit_results → predict scripts)

Forward-side helpers like `_fit_with_adam` are imported from
`forward/_shared.py` (available via the path setup at the top of any caller).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "forward"))

import jax
import jax.numpy as jnp
import optax
import pandas as pd

from tables import EFFORT_CONDITION_TO_IDX, SCENARIO_TO_IDX
from utils import get_project_root
from _shared import _fit_with_adam  # forward/_shared.py


# ==============================================================================
# Loss functions
# ==============================================================================


@jax.jit
def compute_intimacy_nll(posterior, response):
    """NLL = -log(P(intimacy = response/100)).

    posterior: shape (101,) over intimacy levels 0-100.
    response: integer 0-100.
    """
    epsilon = 1e-8
    response_idx = jnp.clip(jnp.round(response).astype(int), 0, 100)
    prob = posterior[response_idx]
    return -jnp.log(jnp.clip(prob, epsilon, 1.0))


@jax.jit
def compute_reward_nll(p_high, response):
    """Binary cross-entropy NLL for reward inference.

    response is 0-100 (interpreted as P(high)*100).
    """
    epsilon = 1e-8
    p_human = response / 100.0
    p_model = jnp.clip(p_high, epsilon, 1.0 - epsilon)
    return -(p_human * jnp.log(p_model) + (1 - p_human) * jnp.log(1 - p_model))


# ==============================================================================
# Frozen-param loaders
# ==============================================================================


def load_fitted_params(filepath: str = None) -> dict:
    """Load frozen actor parameters from forward planning fit results.

    Returns a dict: model_name -> dict of every `param_*` column present in that
    row (stripped of the `param_` prefix). Missing/NaN columns are omitted, so
    each model keeps only its own parameter set.

    Defaults to the canonical `food_forw_intimacy_desire/fit_results.csv`.
    """
    if filepath is None:
        filepath = (
            get_project_root()
            / "model"
            / "outputs"
            / "food_forw_intimacy_desire"
            / "fit_results.csv"
        )
    df = pd.read_csv(filepath)
    params = {}
    for _, row in df.iterrows():
        model_name = row["model"]
        model_params = {}
        for col in df.columns:
            if col.startswith("param_") and pd.notna(row[col]):
                model_params[col.replace("param_", "")] = float(row[col])
        params[model_name] = model_params
    return params


def load_food_forw_intimacy_effort_actor_params(filepath: str = None) -> dict:
    """Load frozen actor parameters from the food_forw_intimacy_effort fit."""
    if filepath is None:
        filepath = (
            get_project_root()
            / "model"
            / "outputs"
            / "food_forw_intimacy_effort"
            / "fit_results.csv"
        )
    return load_fitted_params(filepath=filepath)


def load_fitted_alpha_observer(filepath=None) -> dict:
    """Load fitted alpha_observer values from inverse planning fit_results.csv.

    `filepath` is a single path (single-experiment file). If None, reads from
    both alt-shown experiment dirs and merges (intimacy + reward).

    Returns dict with (model, experiment) -> alpha_observer. Defaults to 1.0 if NaN.
    """
    if filepath is not None:
        paths = [Path(filepath)]
    else:
        outputs_root = get_project_root() / "model" / "outputs"
        paths = [
            outputs_root / "food_inv_intimacy_desire_alt" / "fit_results.csv",
            outputs_root / "food_inv_desire_intimacy_alt" / "fit_results.csv",
        ]
    alpha_obs = {}
    for path in paths:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (row["model"], row["experiment"])
            alpha_val = row["alpha_observer"]
            alpha_obs[key] = alpha_val if pd.notna(alpha_val) else 1.0
    return alpha_obs


# ==============================================================================
# Data loaders (one per experiment)
# ==============================================================================


def load_intimacy_alt_data(filepath: str = None):
    """food_inv_intimacy_desire_alt — observer infers intimacy under known motivation."""
    if filepath is None:
        filepath = get_project_root() / "data" / "food_inv_intimacy_desire_alt" / "main_trials_long.csv"
    print("Loading food_inv_intimacy_desire_alt data...")
    data = pd.read_csv(filepath)
    data = data[data["stage"] == "posterior"].copy()
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int)
    motivation_map = {"low": 0, "high": 1}
    data["reward_condition"] = data["motivation"].map(motivation_map)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    action = jnp.array(data["action"].values)
    reward_condition = jnp.array(data["reward_condition"].values)
    response = jnp.array(data["intimacy_rating"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    print(f"Loaded {len(data)} posterior data points")
    return data, action, reward_condition, response, scenario_idx


def load_desire_alt_data(filepath: str = None):
    """food_inv_desire_intimacy_alt — observer infers desire (motivation) under known intimacy."""
    if filepath is None:
        filepath = get_project_root() / "data" / "food_inv_desire_intimacy_alt" / "main_trials_long.csv"
    print("Loading food_inv_desire_intimacy_alt data...")
    data = pd.read_csv(filepath)
    data = data[data["stage"] == "posterior"].copy()
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int)
    intimacy_map = {0: 0, 50: 1, 75: 2, 100: 3}
    data["intimacy_idx"] = data["intimacy"].map(intimacy_map)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    action = jnp.array(data["action"].values)
    intimacy_condition = jnp.array(data["intimacy_idx"].values)
    response = jnp.array(data["response"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    print(f"Loaded {len(data)} posterior data points")
    return data, action, intimacy_condition, response, scenario_idx


def load_intimacy_noalt_data(filepath=None):
    """food_inv_intimacy_desire_noalt — observer sees only the chosen action."""
    if filepath is None:
        filepath = (
            get_project_root() / "data" / "food_inv_intimacy_desire_noalt" / "main_trials_long.csv"
        )
    print("Loading food_inv_intimacy_desire_noalt data...")
    data = pd.read_csv(filepath)
    data = data[data["stage"] == "posterior"].copy()
    data["observed_action"] = data["action_condition"].str.replace("action_", "").astype(int)
    motivation_map = {"low": 0, "high": 1}
    data["reward_condition"] = data["motivation"].map(motivation_map)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    observed_action = jnp.array(data["observed_action"].values)
    reward_condition = jnp.array(data["reward_condition"].values)
    response = jnp.array(data["intimacy_rating"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    print(f"Loaded {len(data)} posterior data points")
    return data, observed_action, reward_condition, response, scenario_idx


def load_desire_noalt_data(filepath=None):
    """food_inv_desire_intimacy_noalt — observer sees only the chosen action; infers motivation."""
    if filepath is None:
        filepath = (
            get_project_root() / "data" / "food_inv_desire_intimacy_noalt" / "main_trials_long.csv"
        )
    print("Loading food_inv_desire_intimacy_noalt data...")
    data = pd.read_csv(filepath)
    data = data[data["stage"] == "posterior"].copy()
    data["observed_action"] = data["action_condition"].str.replace("action_", "").astype(int)
    intimacy_map = {0: 0, 50: 1, 75: 2, 100: 3}
    data["relationship_condition"] = data["intimacy"].map(intimacy_map)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    observed_action = jnp.array(data["observed_action"].values)
    relationship_condition = jnp.array(data["relationship_condition"].values)
    response = jnp.array(data["response"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    print(f"Loaded {len(data)} posterior data points")
    return data, observed_action, relationship_condition, response, scenario_idx


def load_intimacy_effort_data(filepath: str = None):
    """food_inv_intimacy_effort_alt — observer infers intimacy under effort manipulation."""
    if filepath is None:
        filepath = (
            get_project_root() / "data" / "food_inv_intimacy_effort_alt" / "main_trials_long.csv"
        )
    print("Loading food_inv_intimacy_effort_alt data...")
    data = pd.read_csv(filepath)
    data = data[data["stage"] == "posterior"].copy()
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int) - 1
    data["effort_condition"] = data["effort"].map(EFFORT_CONDITION_TO_IDX)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    action = jnp.array(data["action"].values)
    effort_condition = jnp.array(data["effort_condition"].values)
    response = jnp.array(data["intimacy_rating"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    print(f"Loaded {len(data)} posterior data points")
    return data, action, effort_condition, response, scenario_idx


def load_effort_intimacy_data(filepath: str = None):
    """food_inv_effort_intimacy_alt — observer infers effort under intimacy manipulation.

    Note: intimacy_idx is the index into the actor's 101-level IntimacyLevels
    axis (0..100 in 0.01 increments) — so the integer intimacy values
    {0, 50, 75, 100} double as indices.
    """
    if filepath is None:
        filepath = (
            get_project_root() / "data" / "food_inv_effort_intimacy_alt" / "main_trials_long.csv"
        )
    print("Loading food_inv_effort_intimacy_alt data...")
    data = pd.read_csv(filepath)
    data = data[data["stage"] == "posterior"].copy()
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int) - 1
    data["intimacy_idx"] = data["intimacy"].astype(int)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    action = jnp.array(data["action"].values)
    intimacy_idx = jnp.array(data["intimacy_idx"].values)
    response = jnp.array(data["response"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    print(f"Loaded {len(data)} posterior data points")
    return data, action, intimacy_idx, response, scenario_idx


# ==============================================================================
# Single-α_observer fit loop (alt-shown)
# ==============================================================================
# The full alt-shown observer takes (actor kwargs, alpha_observer, table kwargs)
# and produces a 4D table (action, scenario, intimacy_or_relationship, condition).
# This loop fits only alpha_observer with actor weights frozen.


def _fit_alpha_observer(
    observer_fn,
    actor_params: dict,
    actor_kwarg_names,
    action: jnp.ndarray,
    scenario_idx: jnp.ndarray,
    conditioning: jnp.ndarray,
    response: jnp.ndarray,
    nll_fn,
    posterior_slicer,
    table_kwargs: dict,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer by dict-keyed actor params."""
    actor_kwargs = {k: actor_params[k] for k in actor_kwarg_names}

    def observer_table(alpha_observer):
        return observer_fn(
            **actor_kwargs, alpha_observer=alpha_observer,
            **table_kwargs,
        )

    def get_nll(alpha_observer, a, s, c, resp):
        table = observer_table(alpha_observer)
        slc = posterior_slicer(table, a, s, c)
        return nll_fn(slc, resp)

    vmap_get_nll = jax.vmap(
        lambda alpha_obs, a, s, c, resp: get_nll(alpha_obs, a, s, c, resp),
        in_axes=(None, 0, 0, 0, 0),
    )

    def loss_fn(params):
        return jnp.sum(
            vmap_get_nll(params[0], action, scenario_idx, conditioning, response)
        )

    params = jnp.array([1.0])
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_loss = None
    zero_grad_count = 0
    for step in range(max_steps):
        loss, grad = grad_fn(params)
        grad_mag = float(jnp.abs(grad[0]))
        if jnp.isnan(grad[0]) or grad_mag < 1e-10:
            zero_grad_count += 1
            if zero_grad_count >= 5:
                if verbose:
                    print("  Gradient zero/NaN for 5 consecutive steps; alpha_observer=1.0")
                return 1.0, float(loss)
        else:
            zero_grad_count = 0

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 0.01, 10.0)

        if verbose and step % 200 == 0:
            print(f"  Step {step}, NLL: {loss:.4f}, alpha_observer: {params[0]:.4f}")

        if prev_loss is not None and loss > prev_loss + 1e-4:
            if verbose:
                print(f"  Loss increased at step {step}, stopping")
            break
        prev_loss = loss

    best_nll = float(loss_fn(params))
    final_alpha = float(params[0])
    if jnp.isnan(final_alpha):
        final_alpha = 1.0
    if verbose:
        print(f"  Final NLL: {best_nll:.4f}, alpha_observer: {final_alpha:.4f}")
    return final_alpha, best_nll


def fit_intimacy_observer(
    observer_fn, actor_params, actor_kwarg_names,
    action, scenario_idx, conditioning, response, table_kwargs, **kwargs,
):
    """For observers whose table is (action, scenario, intimacy, conditioning_axis)."""
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=conditioning,
        response=response,
        nll_fn=compute_intimacy_nll,
        posterior_slicer=lambda tab, a, s, c: tab[a, s, :, c],
        table_kwargs=table_kwargs,
        **kwargs,
    )


def fit_reward_observer(
    observer_fn, actor_params, actor_kwarg_names,
    action, scenario_idx, intimacy_condition, response, table_kwargs, **kwargs,
):
    """For observers whose table is (action, scenario, relationship, reward_condition)."""
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=intimacy_condition,
        response=response,
        nll_fn=compute_reward_nll,
        posterior_slicer=lambda tab, a, s, i: tab[a, s, i, 1],
        table_kwargs=table_kwargs,
        **kwargs,
    )


def fit_effort_intimacy_observer(
    observer_fn, actor_params, actor_kwarg_names,
    action, scenario_idx, intimacy_condition, response, table_kwargs, **kwargs,
):
    """For the effort-intimacy observer: table is (action, scenario, intimacy, effort) → P(effort=high)."""
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=intimacy_condition,
        response=response,
        nll_fn=compute_reward_nll,
        # Posterior over effort_condition; table[a, s, i, :] gives [P(low), P(high)]
        posterior_slicer=lambda tab, a, s, i: tab[a, s, i, 1],
        table_kwargs=table_kwargs,
        **kwargs,
    )


# ==============================================================================
# Joint padded fit (no-alt observers)
# ==============================================================================
# Used by intimacy_noalt + desire_noalt: jointly fits all utility weights + α_observer.


def fit_padded_joint_intimacy(
    observer_fn,
    utility_param_names,
    observed_action,
    scenario_idx,
    reward_condition,
    response,
    access_table,
    effort_table,
    prior_table,
    v_padded_table=None,
    lr=0.05,
    max_steps=2000,
    verbose=True,
    label="padded_joint",
):
    """Jointly fit actor utility weights + α_observer for padded intimacy observer."""
    ALPHA_ACTOR = 1.0
    n_utility = len(utility_param_names)
    table_kwargs = dict(
        access_table=access_table, effort_table=effort_table, prior_table=prior_table,
    )
    if v_padded_table is not None:
        table_kwargs["v_padded_table"] = v_padded_table

    def build_observer_table(params):
        actor_kwargs = {"alpha": ALPHA_ACTOR}
        for i, name in enumerate(utility_param_names):
            actor_kwargs[name] = params[i]
        return observer_fn(
            **actor_kwargs,
            alpha_observer=params[-1],
            **table_kwargs,
        )

    def nll_trial(table, obs_a, s, r, resp):
        post = table[0, s, obs_a, :, r]
        return compute_intimacy_nll(post, resp)

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0))

    def loss_fn(params):
        table = build_observer_table(params)
        return jnp.sum(
            vmap_nll(table, observed_action, scenario_idx, reward_condition, response)
        )

    init_params = jnp.ones(n_utility + 1)
    params, nll = _fit_with_adam(
        loss_fn, init_params, lr=lr, max_steps=max_steps, verbose=verbose, label=label,
    )
    return params, float(nll)


def fit_padded_joint_desire(
    observer_fn,
    utility_param_names,
    observed_action,
    scenario_idx,
    relationship_condition,
    response,
    access_table,
    effort_table,
    prior_table,
    v_padded_table=None,
    lr=0.05,
    max_steps=2000,
    verbose=True,
    label="padded_joint",
):
    """Jointly fit actor utility weights + α_observer for padded reward observer (relationship-keyed)."""
    ALPHA_ACTOR = 1.0
    n_utility = len(utility_param_names)
    table_kwargs = dict(
        access_table=access_table, effort_table=effort_table, prior_table=prior_table,
    )
    if v_padded_table is not None:
        table_kwargs["v_padded_table"] = v_padded_table

    def build_observer_table(params):
        actor_kwargs = {"alpha": ALPHA_ACTOR}
        for i, name in enumerate(utility_param_names):
            actor_kwargs[name] = params[i]
        return observer_fn(
            **actor_kwargs,
            alpha_observer=params[-1],
            **table_kwargs,
        )

    def nll_trial(table, obs_a, s, rel, resp):
        # Posterior over reward_condition: table[0, s, obs_a, rel, :] → P(reward=HIGH)
        p_high = table[0, s, obs_a, rel, 1]
        return compute_reward_nll(p_high, resp)

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0))

    def loss_fn(params):
        table = build_observer_table(params)
        return jnp.sum(
            vmap_nll(table, observed_action, scenario_idx, relationship_condition, response)
        )

    init_params = jnp.ones(n_utility + 1)
    params, nll = _fit_with_adam(
        loss_fn, init_params, lr=lr, max_steps=max_steps, verbose=verbose, label=label,
    )
    return params, float(nll)


# ==============================================================================
# Prediction grid helpers
# ==============================================================================
# Each "generate_*_preds" function takes fitted params and writes a long-format
# DataFrame with one row per (scenario, action, condition_axis, level).


def compute_expected_intimacy(df: pd.DataFrame) -> pd.DataFrame:
    """Expected intimacy from posterior over the 0-100 grid."""
    df = df.copy()
    df["intimacy_scaled"] = df["intimacy"] * 100
    summary = df.groupby(
        ["scenario_label", "action", "reward_condition", "model"],
        dropna=False,
    ).apply(
        lambda g: pd.Series({"expected_intimacy": (g["intimacy_scaled"] * g["density"]).sum()})
    ).reset_index()
    return summary


def compute_p_high_reward(df: pd.DataFrame) -> pd.DataFrame:
    """Extract P(high reward) for desire-inference experiments."""
    df_high = df[df["reward_condition"] == "high"].copy()
    df_high = df_high.rename(columns={"density": "p_high_reward"})
    df_high["p_high_reward"] = df_high["p_high_reward"] * 100
    df_high = df_high.drop(columns=["reward_condition"])
    return df_high


# ==============================================================================
# Variant registries (used by CV scripts that share observer functions)
# ==============================================================================
# These are imported by cv/loso_inverse_*.py scripts. Each registry pairs a
# variant name with its observer function and actor kwarg names.

# Lazy import — import here so this module doesn't have hard dependencies on
# observers/tables at top of file (they're imported anyway, but this keeps
# the registry collated with the actual observer references).
from observers import (  # noqa: E402
    observer_intimacy_base,
    observer_intimacy_discomfort_only,
    observer_intimacy_full,
    observer_reward_base,
    observer_reward_discomfort_only,
    observer_reward_full,
    observer_intimacy_base_padded,
    observer_intimacy_discomfort_only_padded,
    observer_intimacy_full_padded,
    observer_reward_base_padded_rel,
    observer_reward_discomfort_only_padded_rel,
    observer_reward_full_padded_rel,
    observer_intimacy_effort_base,
    observer_intimacy_effort_discomfort_only,
    observer_intimacy_effort_full,
    observer_effort_intimacy_base,
    observer_effort_intimacy_discomfort_only,
    observer_effort_intimacy_full,
)
from tables import LLM_TABLES, LLM_TABLES_EFFORT, load_lm_v  # noqa: E402

# Alt-shown 4-action canonical
ACCESS_VARIANTS = {
    "full": (observer_intimacy_full, observer_reward_full,
             ["alpha", "w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_intimacy_discomfort_only, observer_reward_discomfort_only,
                        ["alpha", "w_d", "gamma"], False),
    "base": (observer_intimacy_base, observer_reward_base,
             ["alpha", "w_v", "w_e"], True),
}

# Padded intimacy (no-alt motivation-keyed)
PADDED_VARIANTS_INTIMACY = {
    "full": (observer_intimacy_full_padded, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_intimacy_discomfort_only_padded, ["w_d", "gamma"], False),
    "base": (observer_intimacy_base_padded, ["w_v", "w_e"], True),
}

# Padded reward (no-alt relationship-keyed)
PADDED_VARIANTS_REWARD = {
    "full": (observer_reward_full_padded_rel, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_reward_discomfort_only_padded_rel, ["w_d", "gamma"], False),
    "base": (observer_reward_base_padded_rel, ["w_v", "w_e"], True),
}

# Effort intimacy observer (food_inv_intimacy_effort_alt)
ACCESS_VARIANTS_EFFORT = {
    "full": (observer_intimacy_effort_full, ["alpha", "w_v", "w_d", "w_e", "gamma"]),
    "discomfort_only": (observer_intimacy_effort_discomfort_only, ["alpha", "w_d", "gamma"]),
    "base": (observer_intimacy_effort_base, ["alpha", "w_v", "w_e"]),
}

# Effort intimacy observer for effort-inferred (food_inv_effort_intimacy_alt)
ACCESS_VARIANTS_EFFORT_INFERRED = {
    "full": (observer_effort_intimacy_full, ["alpha", "w_v", "w_d", "w_e", "gamma"]),
    "discomfort_only": (observer_effort_intimacy_discomfort_only, ["alpha", "w_d", "gamma"]),
    "base": (observer_effort_intimacy_base, ["alpha", "w_v", "w_e"]),
}


def alt_table_kwargs(uses_v):
    """Table kwargs for alt-shown 4-action canonical observers."""
    kw = {"access_table": LLM_TABLES["access"], "effort_table": LLM_TABLES["effort"]}
    if uses_v:
        kw["v_table"] = load_lm_v("food")
    return kw


def effort_table_kwargs():
    """Table kwargs for the food_inv_intimacy_effort_alt observer."""
    return {
        "access_table": LLM_TABLES_EFFORT["access"],
        "effort_table": LLM_TABLES_EFFORT["effort"],
    }


def effort_marginal_table_kwargs():
    """Table kwargs for the food_inv_effort_intimacy_alt observer (effort-marginal access)."""
    access_table = LLM_TABLES_EFFORT.get("access_marg", LLM_TABLES_EFFORT["access"])
    return {"access_table": access_table, "effort_table": LLM_TABLES_EFFORT["effort"]}
