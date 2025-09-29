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

# Load and fill risk, effort, discomfort

risk_summary = pd.read_csv("../data/risk/risk_summary.csv")
risk_summary.insert(
    risk_summary.columns.get_loc("scenario_label") + 1,
    "scenario_idx",
    risk_summary["scenario_label"].apply(lambda x: scenario_labels.index(x))
)
risk_matrix = risk_summary.pivot(index="scenario_idx", columns="action", values="empirical_stat").fillna(0).values
assert risk_matrix.shape == (16, 4)

effort_summary = pd.read_csv("../data/effort/effort_summary.csv")
effort_summary.insert(
    effort_summary.columns.get_loc("scenario_label") + 1,
    "scenario_idx",
    effort_summary["scenario_label"].apply(lambda x: scenario_labels.index(x))
)
effort_summary.head()

effort_matrix = effort_summary.pivot(index="scenario_idx", columns="action", values="empirical_stat").fillna(0).values
assert effort_matrix.shape == (16, 4)

discomfort_summary = pd.read_csv("../data/discomfort/discomfort_summary.csv")
discomfort_summary.insert(
    discomfort_summary.columns.get_loc("scenario_label") + 1,
    "scenario_idx",
    discomfort_summary["scenario_label"].apply(lambda x: scenario_labels.index(x))
)

discomfort_pivot = discomfort_summary.pivot_table(
    index="scenario_idx",
    columns=["action", "closeness"],
    values="empirical_stat",
    fill_value=0,
)

discomfort_matrix = discomfort_pivot.values.reshape(16, 4, 4) # scenario_idx x action x closeness

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
def actor_h0[a: risk_levels, c: closeness_levels](scenario_idx, alpha, w_r, w_e):
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
def actor_h1[a: risk_levels, c: closeness_levels](scenario_idx, alpha, w_d, w_r, w_e):
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