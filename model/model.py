from memo import memo
import jax
import jax.numpy as jnp

import numpy as np
import pandas as pd


scenario_labels = [
    "basketball",
    "birthday",
    "brunch",
    "conference",
    "cooking",
    "crabs",
    "dip",
    "drinks",
    "driving",
    "fair",
    "gala",
    "hike",
    "oysters",
    "social",
    "soup",
    "wedding",
]

risk_levels = [0, 1, 2, 3]
closeness_levels = [0, 1, 2, 3]

model_types = [
    "vanilla",
    "relationship",
]

# Load and fill risk, effort, closeness

risk_summary = pd.read_csv("../data/risk/risk_summary.csv")
risk_summary.insert(
    risk_summary.columns.get_loc("scenario_label") + 1,
    "scenario_idx",
    risk_summary["scenario_label"].apply(lambda x: scenario_labels.index(x)),
)
risk_matrix = (
    risk_summary.pivot(index="scenario_idx", columns="action", values="empirical_stat")
    .fillna(0)
    .values
)
assert risk_matrix.shape == (16, 4)

effort_summary = pd.read_csv("../data/effort/effort_summary.csv")
effort_summary.insert(
    effort_summary.columns.get_loc("scenario_label") + 1,
    "scenario_idx",
    effort_summary["scenario_label"].apply(lambda x: scenario_labels.index(x)),
)
effort_summary.head()

effort_matrix = (
    effort_summary.pivot(
        index="scenario_idx", columns="action", values="empirical_stat"
    )
    .fillna(0)
    .values
)
assert effort_matrix.shape == (16, 4)

discomfort_summary = pd.read_csv("../data/discomfort/discomfort_summary.csv")
discomfort_summary.insert(
    discomfort_summary.columns.get_loc("scenario_label") + 1,
    "scenario_idx",
    discomfort_summary["scenario_label"].apply(lambda x: scenario_labels.index(x)),
)

discomfort_pivot = discomfort_summary.pivot_table(
    index="scenario_idx",
    columns=["action", "closeness"],
    values="empirical_stat",
    fill_value=0,
)

discomfort_matrix = discomfort_pivot.values.reshape(
    16, 4, 4
)  # scenario_idx x action x closeness

# turn all matrices into jax arrays
risk_matrix = jnp.array(risk_matrix)
effort_matrix = jnp.array(effort_matrix)
discomfort_matrix = jnp.array(discomfort_matrix)


# define functions for the model
@jax.jit
def c_risk(scenario_idx, a):
    return risk_matrix[scenario_idx, a]


@jax.jit
def c_effort(scenario_idx, a):
    return effort_matrix[scenario_idx, a]


@jax.jit
def c_discomfort(scenario_idx, a, c):
    return discomfort_matrix[scenario_idx, a, c]


@memo
def vanilla_actor[a: risk_levels, c: closeness_levels](scenario_idx, alpha, w_r, w_c):
    cast: [actor]
    actor: knows(c)
    actor: chooses(
        a in risk_levels,
        wpp=exp(alpha * (-w_r * c_risk(scenario_idx, a))),
    )
    return Pr[actor.a == a]


@jax.jit
def get_scale(w_r, w_c, c):
    return (w_r * jnp.exp(-w_c * c))
    # return w_r / (w_c * (1 + c))


@jax.jit
def get_shape(w_c, c):
    return -c + 1
    # return w_c * (-c + 1)


@memo
def relationship_actor[a: risk_levels, c: closeness_levels](
    scenario_idx, alpha, w_r, w_c
):
    cast: [actor]
    actor: knows(c)

    actor: chooses(
        a in risk_levels,
        wpp=exp(
            alpha
            * (
                -1
                * get_scale(w_r, w_c, c)
                * log(c_risk(scenario_idx, a)) ** get_shape(w_c, c)
            )
        ),
    )
    return Pr[actor.a == a]


# define function for getting predictions for a single data point, given model type
def get_actor_predictions(
    model_type: str, scenario_idx, action, closeness, alpha, w_r, w_c
):
    """Non-JIT version for string-based model selection"""
    if model_type == "vanilla":
        return vanilla_actor(scenario_idx, alpha, w_r, w_c)[action, 0]
    elif model_type == "relationship":
        return relationship_actor(scenario_idx, alpha, w_r, w_c)[action, closeness]


# Create individual JIT-compiled prediction functions for each model type
@jax.jit
def get_vanilla_prediction(scenario_idx, action, closeness, alpha, w_r, w_c):
    """JIT-compiled prediction for risk-only model"""
    return vanilla_actor(scenario_idx, alpha, w_r, w_c)[action, 0]


@jax.jit
def get_relationship_prediction(scenario_idx, action, closeness, alpha, w_r, w_c):
    """JIT-compiled prediction for effort-only model"""
    return relationship_actor(scenario_idx, alpha, w_r, w_c)[action, closeness]


# Create vmap prediction functions for each model type
@jax.jit
def predict_vanilla_vmap(scenario_idx, action, closeness, alpha, w_r, w_c):
    """Vectorized prediction function for risk-only model"""
    return jax.vmap(
        lambda s, a, c: get_vanilla_prediction(s, a, c, alpha, w_r, w_c),
        in_axes=(0, 0, 0),
    )(scenario_idx, action, closeness)


@jax.jit
def predict_relationship_vmap(scenario_idx, action, closeness, alpha, w_r, w_c):
    """Vectorized prediction function for effort-only model"""
    return jax.vmap(
        lambda s, a, c: get_relationship_prediction(s, a, c, alpha, w_r, w_c),
        in_axes=(0, 0, 0),
    )(scenario_idx, action, closeness)


# Dictionary mapping model types to their vmap functions
vmap_predictors = {
    "vanilla": predict_vanilla_vmap,
    "relationship": predict_relationship_vmap,
}


def get_vmap_predictor(model_type: str):
    """Get the appropriate vmap prediction function for a model type"""
    if model_type not in vmap_predictors:
        raise ValueError(f"Unknown model type: {model_type}")
    return vmap_predictors[model_type]
