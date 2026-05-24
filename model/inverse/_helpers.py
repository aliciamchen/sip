"""
Shared infrastructure for inverse-planning fit + predict scripts.

Each experiment has its own thin fit/predict script that imports the helpers
it needs from this module. Shared concerns:

  - Loss functions (intimacy NLL, reward BCE NLL)
  - Observer fit loops (single-param alpha_observer; joint padded weights+α)
  - Data loaders (per experiment)
  - Frozen-actor-param loaders (forward fits → inverse fits)

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


def load_3act_fit_results(slug: str) -> dict:
    """Load per-variant {actor utility weights + alpha_observer} for a 3-action
    inverse experiment.

    Reads `outputs/<slug>/fit_results.csv` (written by the joint
    fit_food_inv_*_3act.py scripts). Returns a dict mapping variant name (e.g.
    'full', 'discomfort_only', 'base') to a kwargs dict suitable for calling
    the observer function: `{alpha, alpha_observer, w_v?, w_d?, w_e?, gamma?}`
    (only the columns present for that variant; alpha defaults to 1.0).
    """
    path = get_project_root() / "model" / "outputs" / slug / "fit_results.csv"
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        variant = str(row["model"])
        params = {
            "alpha": float(row["param_alpha"])
            if "param_alpha" in row and pd.notna(row.get("param_alpha", None))
            else 1.0,
            "alpha_observer": float(row["alpha_observer"]),
        }
        for pn in ("w_v", "w_d", "w_e", "gamma"):
            col = f"param_{pn}"
            if col in row and pd.notna(row[col]):
                params[pn] = float(row[col])
        out[variant] = params
    return out


# ==============================================================================
# Data loaders (one per experiment)
# ==============================================================================


def load_intimacy_noalt_data(filepath=None):
    """food_inv_intimacy_desire_noalt — observer sees only the chosen action."""
    if filepath is None:
        filepath = (
            get_project_root()
            / "data"
            / "legacy"
            / "food_inv_intimacy_desire_noalt"
            / "main_trials_long.csv"
        )
    print("Loading food_inv_intimacy_desire_noalt data...")
    data = pd.read_csv(filepath)
    data = data[data["stage"] == "posterior"].copy()
    data["observed_action"] = (
        data["action_condition"].str.replace("action_", "").astype(int)
    )
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
            get_project_root()
            / "data"
            / "legacy"
            / "food_inv_desire_intimacy_noalt"
            / "main_trials_long.csv"
        )
    print("Loading food_inv_desire_intimacy_noalt data...")
    data = pd.read_csv(filepath)
    data = data[data["stage"] == "posterior"].copy()
    data["observed_action"] = (
        data["action_condition"].str.replace("action_", "").astype(int)
    )
    intimacy_map = {0: 0, 50: 1, 75: 2, 100: 3}
    data["relationship_condition"] = data["intimacy"].map(intimacy_map)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    observed_action = jnp.array(data["observed_action"].values)
    relationship_condition = jnp.array(data["relationship_condition"].values)
    response = jnp.array(data["response"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    print(f"Loaded {len(data)} posterior data points")
    return data, observed_action, relationship_condition, response, scenario_idx


# ==============================================================================
# Single-α_observer fit loop
# ==============================================================================
# Takes (actor kwargs, alpha_observer, table kwargs) and produces a 4-or-5-D
# observer table; this loop fits only alpha_observer with actor weights frozen.
# Used by the single-α 3-act wrappers below.


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
            **actor_kwargs,
            alpha_observer=alpha_observer,
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
                    print(
                        "  Gradient zero/NaN for 5 consecutive steps; alpha_observer=1.0"
                    )
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
        access_table=access_table,
        effort_table=effort_table,
        prior_table=prior_table,
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
        loss_fn,
        init_params,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label=label,
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
        access_table=access_table,
        effort_table=effort_table,
        prior_table=prior_table,
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
            vmap_nll(
                table, observed_action, scenario_idx, relationship_condition, response
            )
        )

    init_params = jnp.ones(n_utility + 1)
    params, nll = _fit_with_adam(
        loss_fn,
        init_params,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label=label,
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
    summary = (
        df.groupby(
            ["scenario_label", "action", "reward_condition", "model"],
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
    observer_intimacy_base_padded,
    observer_intimacy_discomfort_only_padded,
    observer_intimacy_full_padded,
    observer_reward_base_padded_rel,
    observer_reward_discomfort_only_padded_rel,
    observer_reward_full_padded_rel,
)

# Padded intimacy (no-alt motivation-keyed)
PADDED_VARIANTS_INTIMACY = {
    "full": (observer_intimacy_full_padded, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (
        observer_intimacy_discomfort_only_padded,
        ["w_d", "gamma"],
        False,
    ),
    "base": (observer_intimacy_base_padded, ["w_v", "w_e"], True),
}

# Padded reward (no-alt relationship-keyed)
PADDED_VARIANTS_REWARD = {
    "full": (observer_reward_full_padded_rel, ["w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (
        observer_reward_discomfort_only_padded_rel,
        ["w_d", "gamma"],
        False,
    ),
    "base": (observer_reward_base_padded_rel, ["w_v", "w_e"], True),
}


# ==============================================================================
# 3-action experiments (Studies 2, 3a, 3b, 4a, 4b)
# ==============================================================================
# Data loaders, table kwargs, and fit helpers. The single-target fits (2, 3a,
# 3b) reuse `_fit_alpha_observer` with slicers appropriate to the 5-D table
# shape (action, scenario, intimacy/rel, reward, effort). The joint fits (4a,
# 4b) marginalize the joint table to each slider judgment and sum the two NLLs.


def _load_3act_long(slug):
    """Load the posterior-stage rows of a 3-action experiment's main_trials_long.csv.

    Returns the pandas DataFrame with columns: scenario_label, scenario_idx,
    action, intimacy_idx_4, intimacy_idx_101, reward_condition (0/1),
    effort_condition (0/1), response (or two responses for joint studies).

    The exact column names in incoming CSVs may need to be normalized — this
    loader assumes:
      - `action_condition` like 'action_0' / 'action_1' / 'action_2'
      - `reward_condition` (or `motivation`) in {'low', 'high'} when present
      - `effort_condition` (or `effort`) in {'low', 'high'} when present
      - `intimacy` (or `relationship_condition`) in {0, 50, 75, 100} when present
      - `stage` filter on 'posterior'
    """
    filepath = get_project_root() / "data" / slug / "main_trials_long.csv"
    data = pd.read_csv(filepath)
    data = data[data["stage"] == "posterior"].copy()
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    motivation_map = {"low": 0, "high": 1}
    if "motivation" in data.columns:
        data["reward_condition"] = data["motivation"].map(motivation_map)
    elif (
        "reward_condition" in data.columns and data["reward_condition"].dtype == object
    ):
        data["reward_condition"] = data["reward_condition"].map(motivation_map)

    if "effort" in data.columns:
        data["effort_condition"] = data["effort"].map(EFFORT_CONDITION_TO_IDX)
    elif (
        "effort_condition" in data.columns and data["effort_condition"].dtype == object
    ):
        data["effort_condition"] = data["effort_condition"].map(EFFORT_CONDITION_TO_IDX)

    intimacy_map = {0: 0, 50: 1, 75: 2, 100: 3}
    if "intimacy" in data.columns:
        data["intimacy_idx_4"] = data["intimacy"].map(intimacy_map)
        data["intimacy_idx_101"] = data["intimacy"].astype(int)
    elif "relationship_condition" in data.columns:
        data["intimacy_idx_4"] = data["relationship_condition"].map(intimacy_map)
        data["intimacy_idx_101"] = data["relationship_condition"].astype(int)

    return data


def load_intimacy_3act_data(slug="food_inv_intimacy_3act"):
    """Study 2 — observer knows (reward, effort), infers intimacy."""
    data = _load_3act_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    reward_condition = jnp.array(data["reward_condition"].values)
    effort_condition = jnp.array(data["effort_condition"].values)
    response = jnp.array(data["intimacy_rating"].values)
    print(f"Loaded {len(data)} posterior data points")
    return data, action, scenario_idx, reward_condition, effort_condition, response


def load_effort_3act_data(slug="food_inv_effort_3act"):
    """Study 3a — observer knows (reward, intimacy), infers effort.

    Note: the observer does not see the effort paragraph; the model uses
    `lm_scenario_params_3act_marginal.csv` for access. effort_condition is the
    latent the participant infers.
    """
    data = _load_3act_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    reward_condition = jnp.array(data["reward_condition"].values)
    relationship_condition = jnp.array(data["intimacy_idx_4"].values)
    response = jnp.array(data["response"].values)  # P(effort_high) * 100
    print(f"Loaded {len(data)} posterior data points")
    return (
        data,
        action,
        scenario_idx,
        reward_condition,
        relationship_condition,
        response,
    )


def load_desire_3act_data(slug="food_inv_desire_3act"):
    """Study 3b — observer knows (effort, intimacy), infers desire (reward)."""
    data = _load_3act_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    effort_condition = jnp.array(data["effort_condition"].values)
    relationship_condition = jnp.array(data["intimacy_idx_4"].values)
    response = jnp.array(data["response"].values)  # P(high motivation) * 100
    print(f"Loaded {len(data)} posterior data points")
    return (
        data,
        action,
        scenario_idx,
        effort_condition,
        relationship_condition,
        response,
    )


def load_joint_de_3act_data(slug="food_inv_joint_de_3act"):
    """Study 4a — observer knows intimacy, jointly infers (reward, effort).

    Each posterior trial contributes two slider responses; expects columns
    `p_high_reward_rating` and `p_effort_high_rating` (0-100).
    """
    data = _load_3act_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    relationship_condition = jnp.array(data["intimacy_idx_4"].values)
    resp_reward = jnp.array(data["p_high_reward_rating"].values)
    resp_effort = jnp.array(data["p_effort_high_rating"].values)
    print(f"Loaded {len(data)} posterior data points (with 2 slider responses each)")
    return data, action, scenario_idx, relationship_condition, resp_reward, resp_effort


def load_joint_di_3act_data(slug="food_inv_joint_di_3act"):
    """Study 4b — observer knows effort, jointly infers (reward, intimacy).

    Expects columns `p_high_reward_rating` (0-100) and `intimacy_rating` (0-100).
    """
    data = _load_3act_long(slug)
    print(f"Loading {slug} data...")
    action = jnp.array(data["action"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    effort_condition = jnp.array(data["effort_condition"].values)
    resp_reward = jnp.array(data["p_high_reward_rating"].values)
    resp_intimacy = jnp.array(data["intimacy_rating"].values)
    print(f"Loaded {len(data)} posterior data points (with 2 slider responses each)")
    return data, action, scenario_idx, effort_condition, resp_reward, resp_intimacy


# ------------------------------------------------------------------------------
# Table-kwargs helpers for the new pipeline (load lazily — tables may be None
# during early development if the LM CSVs haven't been produced yet).
# ------------------------------------------------------------------------------


def _3act_tables(uses_v, effort_marginal=False, domain="food"):
    """Build the access/effort/v table kwargs for a 3-action observer."""
    from tables import LLM_TABLES_3ACT, load_lm_v_3act

    if LLM_TABLES_3ACT is None:
        raise FileNotFoundError(
            "model/outputs/lm/lm_scenario_params_3act.csv not found — "
            "run `uv run python model/lm/score_3act_features.py` first."
        )
    access_key = "access_marg" if effort_marginal else "access"
    access_table = LLM_TABLES_3ACT.get(access_key, LLM_TABLES_3ACT["access"])
    kw = {"access_table": access_table, "effort_table": LLM_TABLES_3ACT["effort"]}
    if uses_v:
        v_table = load_lm_v_3act(domain=domain)
        if v_table is None:
            raise FileNotFoundError(
                "model/outputs/lm/lm_scenario_v_3act.csv not found — "
                "run `uv run python model/lm/score_3act_v.py` first."
            )
        kw["v_table"] = v_table
    return kw


def intimacy_3act_table_kwargs(uses_v, domain="food"):
    return _3act_tables(uses_v, effort_marginal=False, domain=domain)


def effort_3act_table_kwargs(uses_v, domain="food"):
    """Study 3a uses effort-marginal access since the observer doesn't see the effort paragraph."""
    return _3act_tables(uses_v, effort_marginal=True, domain=domain)


def desire_3act_table_kwargs(uses_v, domain="food"):
    """Padded LM tables for Study 3b — observer's actor softmaxes over
    LM-generated alternatives per (scenario, observed_action, effort, intimacy)
    cell. Returns kwargs {access_table, effort_table, prior_table} plus
    `v_padded_table` when the variant uses V.

    Raises FileNotFoundError if any prerequisite LM CSV is missing — run
    `model/lm/generate_alternatives_3act.py` and
    `model/lm/score_alternatives_3act{,_v}.py` first.
    """
    from tables import load_padded_lm_tables_3act_desire

    if domain != "food":
        raise NotImplementedError(
            "Study 3b padded LM tables are only available for the food domain "
            "(no nonfood 3-act stimulus set yet)."
        )
    padded = load_padded_lm_tables_3act_desire()
    if padded is None:
        raise FileNotFoundError(
            "Padded LM tables for Study 3b not found. Run "
            "model/lm/generate_alternatives_3act.py and "
            "model/lm/score_alternatives_3act{,_v}.py to produce the "
            "lm_alternatives_*_food_inv_desire_3act.csv set."
        )
    kw = {
        "access_table": padded["access"],
        "effort_table": padded["effort"],
        "prior_table": padded["prior"],
    }
    if uses_v:
        kw["v_padded_table"] = padded["v"]
    return kw


def joint_3act_table_kwargs(uses_v, domain="food"):
    return _3act_tables(uses_v, effort_marginal=False, domain=domain)


# ------------------------------------------------------------------------------
# Single-target fits — reuse _fit_alpha_observer with the appropriate slicer.
# The 3-action observer tables are 5-D: (action, scenario, intimacy/rel, reward, effort).
# ------------------------------------------------------------------------------


def fit_intimacy_3act_observer(
    observer_fn,
    actor_params,
    actor_kwarg_names,
    action,
    scenario_idx,
    reward_condition,
    effort_condition,
    response,
    table_kwargs,
    **kwargs,
):
    """Study 2. Table: (action, scenario, intimacy_101, reward, effort)."""
    # Encode the two conditioning variables into a single integer index
    # (reward * 2 + effort) so we can reuse _fit_alpha_observer's signature.
    conditioning = reward_condition * 2 + effort_condition
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=conditioning,
        response=response,
        nll_fn=compute_intimacy_nll,
        # Slice the intimacy posterior for the right (reward, effort) cell.
        posterior_slicer=lambda tab, a, s, c: tab[a, s, :, c // 2, c % 2],
        table_kwargs=table_kwargs,
        **kwargs,
    )


def fit_effort_3act_observer(
    observer_fn,
    actor_params,
    actor_kwarg_names,
    action,
    scenario_idx,
    reward_condition,
    relationship_condition,
    response,
    table_kwargs,
    **kwargs,
):
    """Study 3a. Table: (action, scenario, relationship_4, reward, effort). Returns P(effort_high)."""
    conditioning = reward_condition * 4 + relationship_condition
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=conditioning,
        response=response,
        nll_fn=compute_reward_nll,  # binary cross-entropy
        # P(effort_high | observed) = tab[a, s, rel, reward, 1]
        posterior_slicer=lambda tab, a, s, c: tab[a, s, c % 4, c // 4, 1],
        table_kwargs=table_kwargs,
        **kwargs,
    )


def fit_desire_3act_observer(
    observer_fn,
    actor_params,
    actor_kwarg_names,
    action,
    scenario_idx,
    effort_condition,
    relationship_condition,
    response,
    table_kwargs,
    **kwargs,
):
    """Study 3b. Table: (action, scenario, relationship_4, reward, effort). Returns P(reward_high)."""
    conditioning = effort_condition * 4 + relationship_condition
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=conditioning,
        response=response,
        nll_fn=compute_reward_nll,
        # P(reward_high | observed) = tab[a, s, rel, 1, effort]
        posterior_slicer=lambda tab, a, s, c: tab[a, s, c % 4, 1, c // 4],
        table_kwargs=table_kwargs,
        **kwargs,
    )


# ------------------------------------------------------------------------------
# Joint fits (Studies 4a, 4b)
# ------------------------------------------------------------------------------


@jax.jit
def _joint_de_nll_trial(table, action, scenario_idx, rel, r_reward, r_effort):
    """Per-trial NLL for Study 4a — marginalize the joint over (reward, effort) to
    each slider, then sum the two binary cross-entropies."""
    joint = table[action, scenario_idx, rel, :, :]  # (2, 2)
    p_reward_high = joint[1, :].sum()  # marginalize effort
    p_effort_high = joint[:, 1].sum()  # marginalize reward
    return compute_reward_nll(p_reward_high, r_reward) + compute_reward_nll(
        p_effort_high, r_effort
    )


def fit_joint_de_3act_observer(
    observer_fn,
    actor_params,
    actor_kwarg_names,
    action,
    scenario_idx,
    relationship_condition,
    response_reward,
    response_effort,
    table_kwargs,
    lr=0.1,
    max_steps=1000,
    verbose=True,
):
    """Study 4a — joint over (reward, effort) given intimacy."""
    actor_kwargs = {k: actor_params[k] for k in actor_kwarg_names}

    def observer_table(alpha_observer):
        return observer_fn(
            **actor_kwargs, alpha_observer=alpha_observer, **table_kwargs
        )

    vmap_nll = jax.vmap(_joint_de_nll_trial, in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params):
        table = observer_table(params[0])
        return jnp.sum(
            vmap_nll(
                table,
                action,
                scenario_idx,
                relationship_condition,
                response_reward,
                response_effort,
            )
        )

    init = jnp.array([1.0])
    params, nll = _fit_with_adam(
        loss_fn,
        init,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="joint_de_3act",
    )
    return float(params[0]), float(nll)


@jax.jit
def _joint_di_nll_trial(
    table, action, scenario_idx, effort_condition, r_reward, r_intimacy
):
    """Per-trial NLL for Study 4b — joint over (reward, intimacy) → marginalize each."""
    joint = table[action, scenario_idx, :, :, effort_condition]  # (101, 2)
    p_intimacy = joint.sum(axis=-1)  # marginalize reward → (101,)
    p_reward_high = joint[:, 1].sum()  # marginalize intimacy → scalar
    return compute_intimacy_nll(p_intimacy, r_intimacy) + compute_reward_nll(
        p_reward_high, r_reward
    )


def fit_joint_di_3act_observer(
    observer_fn,
    actor_params,
    actor_kwarg_names,
    action,
    scenario_idx,
    effort_condition,
    response_reward,
    response_intimacy,
    table_kwargs,
    lr=0.1,
    max_steps=1000,
    verbose=True,
):
    """Study 4b — joint over (reward, intimacy) given effort."""
    actor_kwargs = {k: actor_params[k] for k in actor_kwarg_names}

    def observer_table(alpha_observer):
        return observer_fn(
            **actor_kwargs, alpha_observer=alpha_observer, **table_kwargs
        )

    vmap_nll = jax.vmap(_joint_di_nll_trial, in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params):
        table = observer_table(params[0])
        return jnp.sum(
            vmap_nll(
                table,
                action,
                scenario_idx,
                effort_condition,
                response_reward,
                response_intimacy,
            )
        )

    init = jnp.array([1.0])
    params, nll = _fit_with_adam(
        loss_fn,
        init,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="joint_di_3act",
    )
    return float(params[0]), float(nll)


# ==============================================================================
# 3-action joint fits — utility weights + α_observer (Studies 2, 3a, 3b, 4a, 4b)
# ==============================================================================
# Each inverse experiment fits its own actor utility weights jointly with
# α_observer from its own posterior data. Actor weights are NOT transferred
# from the forward-planning (Study 1a) fit — the forward and inverse tasks have
# different data and different identifiability, so transferring would conflate
# the two fits.
#
# Params layout in the returned 1-D array: [*utility_param_values, alpha_observer].
# Caller maps utility_param_names → param_<name> columns; reads params[-1] as
# alpha_observer. Actor inverse-temperature is held at α=1 (same convention as
# the legacy padded joint fits).


def _build_3act_observer_table(observer_fn, params, utility_param_names, table_kwargs):
    """Shared closure for the 5 joint fits below."""
    actor_kwargs = {"alpha": 1.0}
    for i, name in enumerate(utility_param_names):
        actor_kwargs[name] = params[i]
    return observer_fn(**actor_kwargs, alpha_observer=params[-1], **table_kwargs)


def fit_intimacy_3act_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    reward_condition,
    effort_condition,
    response,
    table_kwargs,
    lr=0.1,
    max_steps=1000,
    verbose=True,
):
    """Study 2 — joint fit of utility weights + α_observer (intimacy NLL)."""

    def nll_trial(table, a, s, r, e, resp):
        post = table[a, s, :, r, e]
        return compute_intimacy_nll(post, resp)

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params):
        table = _build_3act_observer_table(
            observer_fn, params, utility_param_names, table_kwargs
        )
        return jnp.sum(
            vmap_nll(
                table,
                action,
                scenario_idx,
                reward_condition,
                effort_condition,
                response,
            )
        )

    init = jnp.ones(len(utility_param_names) + 1)
    params, nll = _fit_with_adam(
        loss_fn,
        init,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="intimacy_3act_joint",
    )
    return params, float(nll)


def fit_effort_3act_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    reward_condition,
    relationship_condition,
    response,
    table_kwargs,
    lr=0.1,
    max_steps=1000,
    verbose=True,
):
    """Study 3a — joint fit of utility weights + α_observer (BCE for P(effort=HIGH))."""

    def nll_trial(table, a, s, r, rel, resp):
        p_high = table[a, s, rel, r, 1]
        return compute_reward_nll(p_high, resp)

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params):
        table = _build_3act_observer_table(
            observer_fn, params, utility_param_names, table_kwargs
        )
        return jnp.sum(
            vmap_nll(
                table,
                action,
                scenario_idx,
                reward_condition,
                relationship_condition,
                response,
            )
        )

    init = jnp.ones(len(utility_param_names) + 1)
    params, nll = _fit_with_adam(
        loss_fn,
        init,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="effort_3act_joint",
    )
    return params, float(nll)


def fit_desire_3act_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    effort_condition,
    relationship_condition,
    response,
    table_kwargs,
    lr=0.1,
    max_steps=1000,
    verbose=True,
):
    """Study 3b — joint fit of utility weights + α_observer (BCE for P(reward=HIGH)).

    With the LM-generated alternatives pipeline, the observer table is 6-D —
    `(padded_slot, scenario, observed_action, effort, intimacy, reward)`. The
    canonical observed action lives in slot 0 by construction, so the per-trial
    P(reward=HIGH) slice is `table[0, scenario, action, effort, intimacy, 1]`
    (where `action` is the participant-observed action index 0/1/2).
    """

    def nll_trial(table, a, s, e, rel, resp):
        p_high = table[0, s, a, e, rel, 1]
        return compute_reward_nll(p_high, resp)

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params):
        table = _build_3act_observer_table(
            observer_fn, params, utility_param_names, table_kwargs
        )
        return jnp.sum(
            vmap_nll(
                table,
                action,
                scenario_idx,
                effort_condition,
                relationship_condition,
                response,
            )
        )

    init = jnp.ones(len(utility_param_names) + 1)
    params, nll = _fit_with_adam(
        loss_fn,
        init,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="desire_3act_joint",
    )
    return params, float(nll)


def fit_joint_de_3act_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    relationship_condition,
    response_reward,
    response_effort,
    table_kwargs,
    lr=0.1,
    max_steps=1000,
    verbose=True,
):
    """Study 4a — joint fit of utility weights + α_observer (sum of two BCEs)."""

    def nll_trial(table, a, s, rel, r_reward, r_effort):
        joint = table[a, s, rel, :, :]
        p_reward_high = joint[1, :].sum()
        p_effort_high = joint[:, 1].sum()
        return compute_reward_nll(p_reward_high, r_reward) + compute_reward_nll(
            p_effort_high, r_effort
        )

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params):
        table = _build_3act_observer_table(
            observer_fn, params, utility_param_names, table_kwargs
        )
        return jnp.sum(
            vmap_nll(
                table,
                action,
                scenario_idx,
                relationship_condition,
                response_reward,
                response_effort,
            )
        )

    init = jnp.ones(len(utility_param_names) + 1)
    params, nll = _fit_with_adam(
        loss_fn,
        init,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="joint_de_3act_joint",
    )
    return params, float(nll)


def fit_joint_di_3act_observer_joint(
    observer_fn,
    utility_param_names,
    action,
    scenario_idx,
    effort_condition,
    response_reward,
    response_intimacy,
    table_kwargs,
    lr=0.1,
    max_steps=1000,
    verbose=True,
):
    """Study 4b — joint fit of utility weights + α_observer (BCE + intimacy NLL)."""

    def nll_trial(table, a, s, e, r_reward, r_intimacy):
        joint = table[a, s, :, :, e]  # (101, 2)
        p_intimacy = joint.sum(axis=-1)  # marginalize reward → (101,)
        p_reward_high = joint[:, 1].sum()  # marginalize intimacy → scalar
        return compute_intimacy_nll(p_intimacy, r_intimacy) + compute_reward_nll(
            p_reward_high, r_reward
        )

    vmap_nll = jax.vmap(nll_trial, in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params):
        table = _build_3act_observer_table(
            observer_fn, params, utility_param_names, table_kwargs
        )
        return jnp.sum(
            vmap_nll(
                table,
                action,
                scenario_idx,
                effort_condition,
                response_reward,
                response_intimacy,
            )
        )

    init = jnp.ones(len(utility_param_names) + 1)
    params, nll = _fit_with_adam(
        loss_fn,
        init,
        lr=lr,
        max_steps=max_steps,
        verbose=verbose,
        label="joint_di_3act_joint",
    )
    return params, float(nll)
