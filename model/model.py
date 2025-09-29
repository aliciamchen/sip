from memo import memo
import jax
import jax.numpy as jnp

import numpy as np
import pandas as pd

# constants

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
    "risk",
    "effort",
    "discomfort",
    "risk_effort",
    "risk_effort_discomfort",
]

# Load and fill risk, effort, discomfort

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
def actor_risk_only[a: risk_levels, c: closeness_levels](
    scenario_idx, alpha, w_d, w_r, w_e
):
    cast: [actor]
    actor: knows(c)
    actor: chooses(
        a in risk_levels,
        wpp=exp(alpha * (-w_r * c_risk(scenario_idx, a))),
    )
    return Pr[actor.a == a]


@memo
def actor_effort_only[a: risk_levels, c: closeness_levels](
    scenario_idx, alpha, w_d, w_r, w_e
):
    cast: [actor]
    actor: knows(c)
    actor: chooses(
        a in risk_levels,
        wpp=exp(alpha * (-w_e * c_effort(scenario_idx, a))),
    )
    return Pr[actor.a == a]


@memo
def actor_discomfort_only[a: risk_levels, c: closeness_levels](
    scenario_idx, alpha, w_d, w_r, w_e
):
    cast: [actor]
    actor: knows(c)
    actor: chooses(
        a in risk_levels,
        wpp=exp(alpha * (-w_d * c_discomfort(scenario_idx, a, c))),
    )
    return Pr[actor.a == a]


@memo
def actor_risk_effort[a: risk_levels, c: closeness_levels](
    scenario_idx, alpha, w_d, w_r, w_e
):
    cast: [actor]
    actor: knows(c)
    actor: chooses(
        a in risk_levels,
        wpp=exp(
            alpha * (-w_r * c_risk(scenario_idx, a) - w_e * c_effort(scenario_idx, a))
        ),
    )
    return Pr[actor.a == a]


@memo
def actor_risk_effort_discomfort[a: risk_levels, c: closeness_levels](
    scenario_idx, alpha, w_d, w_r, w_e
):
    cast: [actor]
    actor: knows(c)
    actor: chooses(
        a in risk_levels,
        wpp=exp(
            alpha
            * (
                -w_d * c_discomfort(scenario_idx, a, c)
                - w_r * c_risk(scenario_idx, a)
                - w_e * c_effort(scenario_idx, a)
            )
        ),
    )
    return Pr[actor.a == a]


# define function for getting predictions for a single data point, given model type
def get_actor_predictions(model_type: str, scenario_idx, action, closeness, alpha, w_d, w_r, w_e):
    """Non-JIT version for string-based model selection"""
    if model_type == "risk":
        return actor_risk_only(scenario_idx, alpha, w_d, w_r, w_e)[action, 0]
    elif model_type == "effort":
        return actor_effort_only(scenario_idx, alpha, w_d, w_r, w_e)[action, 0]
    elif model_type == "discomfort":
        return actor_discomfort_only(scenario_idx, alpha, w_d, w_r, w_e)[action, closeness]
    elif model_type == "risk_effort":
        return actor_risk_effort(scenario_idx, alpha, w_d, w_r, w_e)[action, 0]
    elif model_type == "risk_effort_discomfort":
        return actor_risk_effort_discomfort(scenario_idx, alpha, w_d, w_r, w_e)[action, closeness]

# Create individual JIT-compiled prediction functions for each model type
@jax.jit
def get_risk_prediction(scenario_idx, action, closeness, alpha, w_d, w_r, w_e):
    """JIT-compiled prediction for risk-only model"""
    return actor_risk_only(scenario_idx, alpha, w_d, w_r, w_e)[action, 0]

@jax.jit
def get_effort_prediction(scenario_idx, action, closeness, alpha, w_d, w_r, w_e):
    """JIT-compiled prediction for effort-only model"""
    return actor_effort_only(scenario_idx, alpha, w_d, w_r, w_e)[action, 0]

@jax.jit
def get_discomfort_prediction(scenario_idx, action, closeness, alpha, w_d, w_r, w_e):
    """JIT-compiled prediction for discomfort-only model"""
    return actor_discomfort_only(scenario_idx, alpha, w_d, w_r, w_e)[action, closeness]

@jax.jit
def get_risk_effort_prediction(scenario_idx, action, closeness, alpha, w_d, w_r, w_e):
    """JIT-compiled prediction for risk+effort model"""
    return actor_risk_effort(scenario_idx, alpha, w_d, w_r, w_e)[action, 0]

@jax.jit
def get_risk_effort_discomfort_prediction(scenario_idx, action, closeness, alpha, w_d, w_r, w_e):
    """JIT-compiled prediction for risk+effort+discomfort model"""
    return actor_risk_effort_discomfort(scenario_idx, alpha, w_d, w_r, w_e)[action, closeness]

# Create vmap prediction functions for each model type
@jax.jit
def predict_risk_vmap(scenario_idx, action, closeness, alpha, w_d, w_r, w_e):
    """Vectorized prediction function for risk-only model"""
    return jax.vmap(
        lambda s, a, c: get_risk_prediction(s, a, c, alpha, w_d, w_r, w_e),
        in_axes=(0, 0, 0),
    )(scenario_idx, action, closeness)

@jax.jit
def predict_effort_vmap(scenario_idx, action, closeness, alpha, w_d, w_r, w_e):
    """Vectorized prediction function for effort-only model"""
    return jax.vmap(
        lambda s, a, c: get_effort_prediction(s, a, c, alpha, w_d, w_r, w_e),
        in_axes=(0, 0, 0),
    )(scenario_idx, action, closeness)

@jax.jit
def predict_discomfort_vmap(scenario_idx, action, closeness, alpha, w_d, w_r, w_e):
    """Vectorized prediction function for discomfort-only model"""
    return jax.vmap(
        lambda s, a, c: get_discomfort_prediction(s, a, c, alpha, w_d, w_r, w_e),
        in_axes=(0, 0, 0),
    )(scenario_idx, action, closeness)

@jax.jit
def predict_risk_effort_vmap(scenario_idx, action, closeness, alpha, w_d, w_r, w_e):
    """Vectorized prediction function for risk+effort model"""
    return jax.vmap(
        lambda s, a, c: get_risk_effort_prediction(s, a, c, alpha, w_d, w_r, w_e),
        in_axes=(0, 0, 0),
    )(scenario_idx, action, closeness)

@jax.jit
def predict_risk_effort_discomfort_vmap(scenario_idx, action, closeness, alpha, w_d, w_r, w_e):
    """Vectorized prediction function for risk+effort+discomfort model"""
    return jax.vmap(
        lambda s, a, c: get_risk_effort_discomfort_prediction(s, a, c, alpha, w_d, w_r, w_e),
        in_axes=(0, 0, 0),
    )(scenario_idx, action, closeness)

# Dictionary mapping model types to their vmap functions
vmap_predictors = {
    "risk": predict_risk_vmap,
    "effort": predict_effort_vmap,
    "discomfort": predict_discomfort_vmap,
    "risk_effort": predict_risk_effort_vmap,
    "risk_effort_discomfort": predict_risk_effort_discomfort_vmap,
}

def get_vmap_predictor(model_type: str):
    """Get the appropriate vmap prediction function for a model type"""
    if model_type not in vmap_predictors:
        raise ValueError(f"Unknown model type: {model_type}")
    return vmap_predictors[model_type]
