from enum import IntEnum
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from memo import memo

# ==============================================================================
# Constants
# ==============================================================================

actions = jnp.array([0, 1, 2, 3])
IntimacyLevels = jnp.arange(0, 1.01, 0.01)


class RewardConditions(IntEnum):
    LOW = 0
    HIGH = 1


class RelationshipConditions(IntEnum):
    ZERO = 0
    FIFTY = 1
    SEVENTY_FIVE = 2
    ONE_HUNDRED = 3


# Continuous intimacy values for each RelationshipConditions level — used by
# the relationship-keyed padded memos to evaluate the (1 - I) access term
# without dragging the 101-level IntimacyLevels axis into the memo.
RELATIONSHIP_LEVEL_VALUES = jnp.array([0.0, 0.5, 0.75, 1.0])


# Padded-action constants for the no-alternatives-shown inverse-planning variant.
# See load_padded_lm_tables() below for how per-cell action tables are built.
MAX_ACTIONS = 16
padded_slots = jnp.arange(MAX_ACTIONS)


class PaddedActionSlots(IntEnum):
    """Memo-friendly enum of padded action slot indices."""
    S0 = 0
    S1 = 1
    S2 = 2
    S3 = 3
    S4 = 4
    S5 = 5
    S6 = 6
    S7 = 7
    S8 = 8
    S9 = 9
    S10 = 10
    S11 = 11
    S12 = 12
    S13 = 13
    S14 = 14
    S15 = 15


class ObservedActions(IntEnum):
    """The 4 canonical observed actions (the experimental stimulus dimension)."""
    A0 = 0
    A1 = 1
    A2 = 2
    A3 = 3


# ==============================================================================
# Scenario Labels
# ==============================================================================
# Scenarios indexed alphabetically (0-15)
SCENARIO_LABELS = [
    "apples",
    "basketball",
    "birthday",
    "brunch",
    "cooking",
    "dip",
    "drinks",
    "driving",
    "fair",
    "gala",
    "hike",
    "oysters",
    "social",
    "soup",
    "takeout",
    "wedding",
]
SCENARIO_TO_IDX = {label: idx for idx, label in enumerate(SCENARIO_LABELS)}


class Scenarios(IntEnum):
    """Memo-friendly enum of the 16 scenarios (alphabetical order)."""
    APPLES = 0
    BASKETBALL = 1
    BIRTHDAY = 2
    BRUNCH = 3
    COOKING = 4
    DIP = 5
    DRINKS = 6
    DRIVING = 7
    FAIR = 8
    GALA = 9
    HIKE = 10
    OYSTERS = 11
    SOCIAL = 12
    SOUP = 13
    TAKEOUT = 14
    WEDDING = 15


# ==============================================================================
# LLM-derived scenario-specific parameters (access / effort / reward)
# ==============================================================================


def load_lm_scenario_params(filepath=None):
    """Load scenario-specific LLM parameter tables for access and effort.

    Returns a dict with:
      - "access": jnp.array of shape (16, 4)  — per (scenario, action)
      - "effort": jnp.array of shape (16, 4)  — per (scenario, action)

    Reward is NOT loaded here; it's stipulated in the utility functions as a
    binary goal-satisfaction gate (V=1 iff the action satisfies the active
    goal: sharing under HIGH motivation, or not-sharing under LOW motivation).

    Rows are ordered by SCENARIO_LABELS (alphabetical). Raises FileNotFoundError
    if the CSV is missing — run `uv run python model/lm_scenario_params.py` first.
    """
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent / "outputs" / "lm_scenario_params.csv"
        )
    df = pd.read_csv(filepath)
    access = np.zeros((len(SCENARIO_LABELS), 4), dtype=np.float32)
    effort = np.zeros((len(SCENARIO_LABELS), 4), dtype=np.float32)
    for _, row in df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        a = int(row["action"])
        access[s_idx, a] = row["access"]
        effort[s_idx, a] = row["effort"]
    return {
        "access": jnp.array(access),
        "effort": jnp.array(effort),
    }


LLM_TABLES = load_lm_scenario_params()


# ==============================================================================
# Domain-aware asset loading (food vs nonfood)
# ==============================================================================
# The memo models in this module use `Scenarios: Scenarios` as a memo dimension,
# but only the cardinality (16) is load-bearing — the IntEnum names (APPLES,
# BASKETBALL, ...) are not. So food and nonfood reuse the same memo models with
# different scenario-label↔index maps and different LLM tables.

NONFOOD_SCENARIO_LABELS = [
    "bed", "blanket", "breakup", "chapstick", "gossip", "hairbrush",
    "harmonica", "hat", "home", "locker-room", "navigation", "payment",
    "sauna", "sleeping-bag", "sunscreen", "towel",
]
NONFOOD_SCENARIO_TO_IDX = {label: idx for idx, label in enumerate(NONFOOD_SCENARIO_LABELS)}


def _load_nonfood_lm_tables():
    path = (
        Path(__file__).resolve().parent / "outputs" / "lm_scenario_params_nonfood.csv"
    )
    df = pd.read_csv(path)
    access = np.zeros((len(NONFOOD_SCENARIO_LABELS), 4), dtype=np.float32)
    effort = np.zeros((len(NONFOOD_SCENARIO_LABELS), 4), dtype=np.float32)
    for _, row in df.iterrows():
        i = NONFOOD_SCENARIO_TO_IDX[row["scenario_label"]]
        a = int(row["action"])
        access[i, a] = row["access"]
        effort[i, a] = row["effort"]
    return {"access": jnp.array(access), "effort": jnp.array(effort)}


def load_domain_assets(domain="food"):
    """Return (scenario_labels, scenario_to_idx, llm_tables) for a domain."""
    if domain == "food":
        return SCENARIO_LABELS, SCENARIO_TO_IDX, LLM_TABLES
    if domain == "nonfood":
        return NONFOOD_SCENARIO_LABELS, NONFOOD_SCENARIO_TO_IDX, _load_nonfood_lm_tables()
    raise ValueError(f"Unknown domain: {domain!r} (expected 'food' or 'nonfood')")


def load_lm_v(domain="food"):
    """Load signed-valence (V) table from lm_scenario_v{,_nonfood}.csv.

    Returns a jnp.array of shape (16, 4, 2) indexed by
    (scenario_idx, action, reward_condition), where reward_condition
    matches RewardConditions (LOW=0, HIGH=1). Values normalized to [-1, +1].

    Raises FileNotFoundError if the CSV is missing — run
    `uv run python model/lm_scenario_params.py --feature v --domain {domain}` first.
    """
    if domain == "food":
        scenario_to_idx = SCENARIO_TO_IDX
        filename = "lm_scenario_v.csv"
    elif domain == "nonfood":
        scenario_to_idx = NONFOOD_SCENARIO_TO_IDX
        filename = "lm_scenario_v_nonfood.csv"
    else:
        raise ValueError(f"Unknown domain: {domain!r}")
    path = Path(__file__).resolve().parent / "outputs" / filename
    df = pd.read_csv(path)
    motivation_to_idx = {"low": int(RewardConditions.LOW), "high": int(RewardConditions.HIGH)}
    v = np.zeros((16, 4, 2), dtype=np.float32)
    for _, row in df.iterrows():
        s = scenario_to_idx[row["scenario_label"]]
        a = int(row["action"])
        m = motivation_to_idx[row["motivation"]]
        v[s, a, m] = row["v"]
    return jnp.array(v)


def load_padded_lm_tables(
    canonical_path=None,
    canonical_v_path=None,
    alternatives_path=None,
    alternatives_features_path=None,
    alternatives_v_path=None,
):
    """Build padded (16, 4, 2, MAX_ACTIONS) tables for access, effort, v
    (signed valence), and prior.

    Slot 0 of every (scenario, observed_action, motivation) cell holds the
    observed canonical action's features (access + effort from
    lm_scenario_params.csv; V from lm_scenario_v.csv).

    Slots 1..k hold the LM-generated alternatives for that cell (access + effort
    from lm_alternatives_features.csv; V from lm_alternatives_v.csv with
    motivation_query == motivation, since the actor reasoning under a given
    motivation evaluates V under that same motivation).

    Remaining slots are null-padded with access=0, effort=0, v=0. The prior
    is uniform over valid (non-null) slots; null slots get a tiny epsilon
    (1e-8) rather than exactly 0 to keep `E[...] ** alpha_observer`
    differentiable in the observer memo, while still contributing <1e-6 of
    softmax mass.

    Returns a dict {access, effort, v, prior} (each jnp.array of shape
    (16, 4, 2, MAX_ACTIONS)), or None if any required CSV is missing.
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    canonical_path = canonical_path or outputs_dir / "lm_scenario_params.csv"
    canonical_v_path = canonical_v_path or outputs_dir / "lm_scenario_v.csv"
    alternatives_path = alternatives_path or outputs_dir / "lm_alternatives.csv"
    alternatives_features_path = (
        alternatives_features_path or outputs_dir / "lm_alternatives_features.csv"
    )
    alternatives_v_path = alternatives_v_path or outputs_dir / "lm_alternatives_v.csv"

    required = [alternatives_path, alternatives_features_path, canonical_v_path, alternatives_v_path]
    if any(not p.exists() for p in required):
        return None

    canonical_df = pd.read_csv(canonical_path)
    canonical_v_df = pd.read_csv(canonical_v_path)
    feats_df = pd.read_csv(alternatives_features_path)
    alts_v_df = pd.read_csv(alternatives_v_path)

    n_scenarios = len(SCENARIO_LABELS)
    n_observed = 4
    n_motivations = 2
    shape = (n_scenarios, n_observed, n_motivations, MAX_ACTIONS)
    access = np.zeros(shape, dtype=np.float32)
    effort = np.zeros(shape, dtype=np.float32)
    v = np.zeros(shape, dtype=np.float32)
    valid_mask = np.zeros(shape, dtype=bool)

    motivation_to_idx = {"low": int(RewardConditions.LOW), "high": int(RewardConditions.HIGH)}
    observed_str_to_idx = {f"action_{i}": i for i in range(n_observed)}

    # Canonical (slot 0) per cell — access/effort from lm_scenario_params,
    # V from lm_scenario_v indexed by (scenario, action, motivation).
    canon_ae_lookup = {}
    for _, row in canonical_df.iterrows():
        canon_ae_lookup[(row["scenario_label"], int(row["action"]))] = (
            float(row["access"]),
            float(row["effort"]),
        )
    canon_v_lookup = {}
    for _, row in canonical_v_df.iterrows():
        canon_v_lookup[(row["scenario_label"], int(row["action"]), row["motivation"])] = (
            float(row["v"])
        )
    for scenario in SCENARIO_LABELS:
        s_idx = SCENARIO_TO_IDX[scenario]
        for observed in range(n_observed):
            a_access, a_effort = canon_ae_lookup[(scenario, observed)]
            for motivation_str, m_idx in motivation_to_idx.items():
                access[s_idx, observed, m_idx, 0] = a_access
                effort[s_idx, observed, m_idx, 0] = a_effort
                v[s_idx, observed, m_idx, 0] = canon_v_lookup[(scenario, observed, motivation_str)]
                valid_mask[s_idx, observed, m_idx, 0] = True

    # LM-generated alternatives (slots 1..k) per cell. V uses the diagonal
    # of (motivation context, motivation_query) — under reward_condition=m,
    # the actor evaluates the m-context alternative's V under m.
    alt_v_lookup = {
        (r["scenario_label"], r["observed_action"], r["motivation"], int(r["alt_idx"])): float(r["v"])
        for _, r in alts_v_df.iterrows()
        if r["motivation"] == r["motivation_query"]
    }

    for _, row in feats_df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        o_idx = observed_str_to_idx[row["observed_action"]]
        m_idx = motivation_to_idx[row["motivation"]]
        alt_idx = int(row["alt_idx"])
        slot = alt_idx + 1  # +1 because slot 0 is the observed canonical action
        if slot >= MAX_ACTIONS:
            continue  # truncate silently; warn below
        access[s_idx, o_idx, m_idx, slot] = float(row["access"])
        effort[s_idx, o_idx, m_idx, slot] = float(row["effort"])
        v[s_idx, o_idx, m_idx, slot] = alt_v_lookup.get(
            (row["scenario_label"], row["observed_action"], row["motivation"], alt_idx),
            0.0,
        )
        valid_mask[s_idx, o_idx, m_idx, slot] = True

    NULL_EPSILON = 1e-8
    n_valid = valid_mask.sum(axis=-1, keepdims=True)  # (16, 4, 2, 1)
    prior_table = np.where(
        valid_mask, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON
    ).astype(np.float32)

    max_alt_count = (
        feats_df.groupby(["scenario_label", "observed_action", "motivation"])
        .size()
        .max()
    )
    if max_alt_count + 1 > MAX_ACTIONS:
        print(
            f"WARNING: largest cell has {max_alt_count} LM-generated alternatives + "
            f"1 observed = {max_alt_count + 1} actions, exceeding MAX_ACTIONS={MAX_ACTIONS}. "
            "Extra alternatives were truncated."
        )

    return {
        "access": jnp.array(access),
        "effort": jnp.array(effort),
        "v": jnp.array(v),
        "prior": jnp.array(prior_table),
    }


def load_padded_lm_tables_relationship(
    canonical_path=None,
    canonical_v_path=None,
    alternatives_path=None,
    alternatives_features_path=None,
    alternatives_v_path=None,
):
    """Build padded tables for the relationship-conditioned no-alt action space
    used by `inv_plan_desire_noalt`.

    Shapes:
      - access: (16, 4, 4, MAX_ACTIONS) — (scenario, observed_action, relationship, slot)
      - effort: (16, 4, 4, MAX_ACTIONS)
      - prior:  (16, 4, 4, MAX_ACTIONS)
      - v:      (16, 4, 4, MAX_ACTIONS, 2) — extra motivation_query axis since
        V depends on motivation but the action-space axis is now relationship.

    Slot 0 holds the observed canonical action (access/effort from
    lm_scenario_params.csv; V from lm_scenario_v.csv, broadcast across the
    relationship axis since the canonical action's V doesn't depend on
    relationship). Slots 1..k hold the LM-generated alternatives for that
    (scenario, observed, relationship) cell, from
    lm_alternatives_relationship_features.csv and lm_alternatives_relationship_v.csv.

    Returns a dict {access, effort, v, prior}, or None if any required CSV is
    missing.
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    canonical_path = canonical_path or outputs_dir / "lm_scenario_params.csv"
    canonical_v_path = canonical_v_path or outputs_dir / "lm_scenario_v.csv"
    alternatives_path = (
        alternatives_path or outputs_dir / "lm_alternatives_relationship.csv"
    )
    alternatives_features_path = (
        alternatives_features_path
        or outputs_dir / "lm_alternatives_relationship_features.csv"
    )
    alternatives_v_path = (
        alternatives_v_path or outputs_dir / "lm_alternatives_relationship_v.csv"
    )

    required = [alternatives_path, alternatives_features_path, canonical_v_path, alternatives_v_path]
    if any(not p.exists() for p in required):
        return None

    canonical_df = pd.read_csv(canonical_path)
    canonical_v_df = pd.read_csv(canonical_v_path)
    feats_df = pd.read_csv(alternatives_features_path)
    alts_v_df = pd.read_csv(alternatives_v_path)

    n_scenarios = len(SCENARIO_LABELS)
    n_observed = 4
    n_relationships = 4
    n_motivations = 2
    shape_3d = (n_scenarios, n_observed, n_relationships, MAX_ACTIONS)
    shape_4d = (n_scenarios, n_observed, n_relationships, MAX_ACTIONS, n_motivations)
    access = np.zeros(shape_3d, dtype=np.float32)
    effort = np.zeros(shape_3d, dtype=np.float32)
    v = np.zeros(shape_4d, dtype=np.float32)
    valid_mask = np.zeros(shape_3d, dtype=bool)

    relationship_to_idx = {0: 0, 50: 1, 75: 2, 100: 3}
    motivation_to_idx = {"low": int(RewardConditions.LOW), "high": int(RewardConditions.HIGH)}
    observed_str_to_idx = {f"action_{i}": i for i in range(n_observed)}

    # Canonical (slot 0): broadcast across the relationship axis. access/effort
    # don't depend on relationship; the canonical V is per (scenario, action,
    # motivation) and likewise broadcasts across relationship for slot 0.
    canon_ae_lookup = {}
    for _, row in canonical_df.iterrows():
        canon_ae_lookup[(row["scenario_label"], int(row["action"]))] = (
            float(row["access"]),
            float(row["effort"]),
        )
    canon_v_lookup = {}
    for _, row in canonical_v_df.iterrows():
        canon_v_lookup[(row["scenario_label"], int(row["action"]), row["motivation"])] = (
            float(row["v"])
        )
    for scenario in SCENARIO_LABELS:
        s_idx = SCENARIO_TO_IDX[scenario]
        for observed in range(n_observed):
            a_access, a_effort = canon_ae_lookup[(scenario, observed)]
            for r_idx in range(n_relationships):
                access[s_idx, observed, r_idx, 0] = a_access
                effort[s_idx, observed, r_idx, 0] = a_effort
                for motivation_str, m_idx in motivation_to_idx.items():
                    v[s_idx, observed, r_idx, 0, m_idx] = canon_v_lookup[
                        (scenario, observed, motivation_str)
                    ]
                valid_mask[s_idx, observed, r_idx, 0] = True

    # LM-generated alternatives (slots 1..k). V uses the full (relationship,
    # motivation_query) cross-product since alternatives are now keyed on
    # relationship and motivation is the latent the observer integrates over.
    alt_v_lookup = {
        (
            r["scenario_label"],
            r["observed_action"],
            int(r["relationship_condition"]),
            int(r["alt_idx"]),
            r["motivation_query"],
        ): float(r["v"])
        for _, r in alts_v_df.iterrows()
    }

    for _, row in feats_df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        o_idx = observed_str_to_idx[row["observed_action"]]
        r_idx = relationship_to_idx[int(row["relationship_condition"])]
        alt_idx = int(row["alt_idx"])
        slot = alt_idx + 1
        if slot >= MAX_ACTIONS:
            continue
        access[s_idx, o_idx, r_idx, slot] = float(row["access"])
        effort[s_idx, o_idx, r_idx, slot] = float(row["effort"])
        for motivation_str, m_idx in motivation_to_idx.items():
            v[s_idx, o_idx, r_idx, slot, m_idx] = alt_v_lookup.get(
                (
                    row["scenario_label"],
                    row["observed_action"],
                    int(row["relationship_condition"]),
                    alt_idx,
                    motivation_str,
                ),
                0.0,
            )
        valid_mask[s_idx, o_idx, r_idx, slot] = True

    NULL_EPSILON = 1e-8
    n_valid = valid_mask.sum(axis=-1, keepdims=True)
    prior_table = np.where(
        valid_mask, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON
    ).astype(np.float32)

    max_alt_count = (
        feats_df.groupby(["scenario_label", "observed_action", "relationship_condition"])
        .size()
        .max()
    )
    if max_alt_count + 1 > MAX_ACTIONS:
        print(
            f"WARNING: largest cell has {max_alt_count} LM-generated alternatives + "
            f"1 observed = {max_alt_count + 1} actions, exceeding MAX_ACTIONS={MAX_ACTIONS}. "
            "Extra alternatives were truncated."
        )

    return {
        "access": jnp.array(access),
        "effort": jnp.array(effort),
        "v": jnp.array(v),
        "prior": jnp.array(prior_table),
    }


# ==============================================================================
# Basic utility helpers
# ==============================================================================


@jax.jit
def get_intimacy(relationship_condition):
    """Map a discrete relationship condition to a continuous intimacy level in [0, 1]."""
    return jnp.array([0, 0.5, 0.75, 1])[relationship_condition]


# ==============================================================================
# Access-model utility functions
# ==============================================================================
# Canonical utility:
#   U(a|s, I, scen, m) =  w_v * V(a|s, m)
#                       - w_d * access[scen, a] * (1 - I)
#                       - w_e * effort[scen, a]
#
# All three components — V, access, effort — are LLM-elicited per scenario.
# V is signed in [-1, +1]: positive when an action serves the active state,
# negative when it actively works against it. access and effort use the
# canonical (16, 4) tables; V uses a (16, 4, 2) table indexed by motivation.
# Tables are passed as memo parameters so memo can JIT-compile without baking
# them into the compiled graph.
#
# Three ablations:
#   - full : full utility above (main model)
#   - discomfort_only : only the access-discomfort term (-w_d * access * (1-I))
#   - base   : base model (w_v*V - w_e*effort)


@jax.jit
def get_lm_v(action, scenario_idx, reward_condition, v_table):
    """LM-elicited signed valence: v_table[scenario_idx, action, reward_condition].

    v_table has shape (16, 4, 2). Values in [-1, +1]. Positive = action serves
    the active state; negative = action actively counterproductive.
    """
    return v_table[scenario_idx, action, reward_condition]


@jax.jit
def get_utility_full(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_table,
):
    access = access_table[scenario_idx, action]
    effort = effort_table[scenario_idx, action]
    V = get_lm_v(action, scenario_idx, reward_condition, v_table)
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * V
        - w_d * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@jax.jit
def get_utility_full_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_full(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_v, w_d, w_e, gamma,
        access_table, effort_table, v_table,
    )


@jax.jit
def get_utility_discomfort_only(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    access = access_table[scenario_idx, action]
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (-w_d * access * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_discomfort_only_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_discomfort_only(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_d, gamma,
        access_table, effort_table,
    )


@jax.jit
def get_utility_base(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table, v_table,
):
    effort = effort_table[scenario_idx, action]
    V = get_lm_v(action, scenario_idx, reward_condition, v_table)
    return alpha * (w_v * V - w_e * effort)


@jax.jit
def get_utility_base_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table, v_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_base(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_v, w_e,
        access_table, effort_table, v_table,
    )


# ==============================================================================
# Forward-planning actor models
# ==============================================================================


@memo
def actor_forw_full[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_v, w_d, w_e, gamma,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_discomfort_only(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_d, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_base[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_base(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Inverse-planning actor models (discrete relationship)
# ==============================================================================


@memo
def actor_discrete_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full_disc(
                action, scenario_idx, relationship_condition, reward_condition,
                alpha, w_v, w_d, w_e, gamma,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_discomfort_only_disc(
                action, scenario_idx, relationship_condition, reward_condition,
                alpha, w_d, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_base[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_base_disc(
                action, scenario_idx, relationship_condition, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Inverse-planning actor models (continuous intimacy)
# ==============================================================================


@memo
def actor_continuous_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full(
                action, scenario_idx, relationship, reward_condition,
                alpha, w_v, w_d, w_e, gamma,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_discomfort_only(
                action, scenario_idx, relationship, reward_condition,
                alpha, w_d, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_base[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_base(
                action, scenario_idx, relationship, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Observer inferring intimacy (alt-shown action space)
# ==============================================================================


@memo
def observer_intimacy_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_full[
                action, scenario_idx, relationship, reward_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_discomfort_only[
                action, scenario_idx, relationship, reward_condition
            ](alpha, w_d, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


@memo
def observer_intimacy_base[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_continuous_base[
                action, scenario_idx, relationship, reward_condition
            ](alpha, w_v, w_e, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# ==============================================================================
# Observer inferring reward (alt-shown action space)
# ==============================================================================


@memo
def observer_reward_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_full[
                action, scenario_idx, relationship_condition, reward_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_discomfort_only[
                action, scenario_idx, relationship_condition, reward_condition
            ](alpha, w_d, gamma, access_table, effort_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_base[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            action in actions,
            wpp=actor_discrete_base[
                action, scenario_idx, relationship_condition, reward_condition
            ](alpha, w_v, w_e, access_table, effort_table, v_table),
        ),
    ]
    observer: observes[actor.action] is action
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


# ==============================================================================
# Padded-action utility and observer (no-alternatives-shown variant, LM-generated counterfactuals)
# ==============================================================================
# The padded observer infers intimacy from a single observed action, using a
# trial-specific action space that is the union of the observed action (slot 0)
# and the LM-generated counterfactual alternatives for that cell (slots 1..k).
# Remaining slots are null-padded (access=effort=is_share=0, utility=0).


@jax.jit
def get_prior_padded(
    padded_slot, scenario_idx, observed_action, reward_condition, prior_table,
):
    """Look up the actor-prior weight for this slot. Null-padded slots have 0."""
    return prior_table[scenario_idx, observed_action, reward_condition, padded_slot]


@jax.jit
def get_lm_v_padded(
    padded_slot, scenario_idx, observed_action, reward_condition, v_padded_table,
):
    """LM-elicited signed valence for an arbitrary action in the padded action
    space.

    v_padded_table has shape (16, 4, 2, MAX_ACTIONS) — indexed by
    (scenario, observed_action, motivation, padded_slot). Slot 0 is the
    canonical action (V from lm_scenario_v.csv); slots 1..k are LM-generated
    alternatives (V from lm_alternatives_v.csv); remaining slots are
    null-padded with V=0 (no contribution after multiplying by zero prior).
    """
    return v_padded_table[scenario_idx, observed_action, reward_condition, padded_slot]


@jax.jit
def get_utility_full_padded(
    padded_slot, scenario_idx, observed_action, intimacy, reward_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_padded_table,
):
    access = access_table[scenario_idx, observed_action, reward_condition, padded_slot]
    effort = effort_table[scenario_idx, observed_action, reward_condition, padded_slot]
    V = get_lm_v_padded(
        padded_slot, scenario_idx, observed_action, reward_condition, v_padded_table,
    )
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * V
        - w_d * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@memo
def actor_continuous_full_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded(padded_slot, scenario_idx, observed_action, reward_condition, prior_table) * exp(
            get_utility_full_padded(
                padded_slot, scenario_idx, observed_action, relationship, reward_condition,
                alpha, w_v, w_d, w_e, gamma,
                access_table, effort_table, v_padded_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def observer_intimacy_full_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
    # The memo returns a posterior on relationship indexed by padded_slot as
    # the observation. Callers should evaluate at padded_slot=0, since slot 0
    # always holds the observed canonical action (see load_padded_lm_tables).
    # The actor prior_table multiplies into the softmax: wpp = prior * exp(U).
    # Null-padded slots have prior=0 so they contribute no mass.
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_full_padded[
                padded_slot, scenario_idx, observed_action, relationship, reward_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_padded_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# --- discomfort_only padded variant: V-independent (drops w_v*V and w_e*effort) ---


@jax.jit
def get_utility_discomfort_only_padded(
    padded_slot, scenario_idx, observed_action, intimacy, reward_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    access = access_table[scenario_idx, observed_action, reward_condition, padded_slot]
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (-w_d * access * jnp.power(one_minus_I, gamma))


@memo
def actor_continuous_discomfort_only_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded(padded_slot, scenario_idx, observed_action, reward_condition, prior_table) * exp(
            get_utility_discomfort_only_padded(
                padded_slot, scenario_idx, observed_action, relationship, reward_condition,
                alpha, w_d, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def observer_intimacy_discomfort_only_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ..., prior_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_discomfort_only_padded[
                padded_slot, scenario_idx, observed_action, relationship, reward_condition
            ](alpha, w_d, gamma, access_table, effort_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# --- base padded variant: drops w_d*access ---


@jax.jit
def get_utility_base_padded(
    padded_slot, scenario_idx, observed_action, intimacy, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table, v_padded_table,
):
    effort = effort_table[scenario_idx, observed_action, reward_condition, padded_slot]
    V = get_lm_v_padded(
        padded_slot, scenario_idx, observed_action, reward_condition, v_padded_table,
    )
    return alpha * (w_v * V - w_e * effort)


@memo
def actor_continuous_base_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded(padded_slot, scenario_idx, observed_action, reward_condition, prior_table) * exp(
            get_utility_base_padded(
                padded_slot, scenario_idx, observed_action, relationship, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table, v_padded_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def observer_intimacy_base_padded[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(reward_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(reward_condition),
        actor : chooses(relationship in IntimacyLevels, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_base_padded[
                padded_slot, scenario_idx, observed_action, relationship, reward_condition
            ](alpha, w_v, w_e, access_table, effort_table, v_padded_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        relationship in IntimacyLevels,
        wpp=E[actor.relationship == relationship] ** alpha_observer,
    )
    return Pr[observer.relationship == relationship]


# ==============================================================================
# Observer inferring reward (padded action space, relationship-keyed)
# ==============================================================================
# Used by `inv_plan_desire_noalt`. The observer knows scenario, observed_action,
# and relationship_condition; the latent is reward_condition. Action space is
# conditioned on (scenario, observed_action, relationship_condition) — i.e.,
# the LM alternatives are elicited per relationship level so the counterfactual
# action set matches what the observer can see.
#
# Tables are loaded by `load_padded_lm_tables_relationship`:
#   access, effort, prior:  (16, 4, 4, MAX_ACTIONS)
#   v:                       (16, 4, 4, MAX_ACTIONS, 2)  — extra motivation axis
#
# The actor's continuous-intimacy `(1 - I)` term comes from
# `RELATIONSHIP_LEVEL_VALUES[relationship_condition]`; the memo dim itself is
# the discrete `RelationshipConditions` enum (4 levels).


@jax.jit
def get_prior_padded_rel(
    padded_slot, scenario_idx, observed_action, relationship_condition, prior_table,
):
    """Look up the actor-prior weight for this slot under the relationship-keyed
    action space. Null-padded slots have ~0 (1e-8 epsilon)."""
    return prior_table[scenario_idx, observed_action, relationship_condition, padded_slot]


@jax.jit
def get_lm_v_padded_rel(
    padded_slot, scenario_idx, observed_action, relationship_condition,
    reward_condition, v_padded_table,
):
    """LM-elicited signed valence for an arbitrary action in the relationship-
    keyed padded action space. v_padded_table has shape
    (16, 4, 4, MAX_ACTIONS, 2) — indexed by
    (scenario, observed_action, relationship, padded_slot, motivation)."""
    return v_padded_table[
        scenario_idx, observed_action, relationship_condition, padded_slot, reward_condition,
    ]


@jax.jit
def get_utility_full_padded_rel(
    padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_padded_table,
):
    intimacy = RELATIONSHIP_LEVEL_VALUES[relationship_condition]
    access = access_table[scenario_idx, observed_action, relationship_condition, padded_slot]
    effort = effort_table[scenario_idx, observed_action, relationship_condition, padded_slot]
    V = get_lm_v_padded_rel(
        padded_slot, scenario_idx, observed_action, relationship_condition,
        reward_condition, v_padded_table,
    )
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * V
        - w_d * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@jax.jit
def get_utility_discomfort_only_padded_rel(
    padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    intimacy = RELATIONSHIP_LEVEL_VALUES[relationship_condition]
    access = access_table[scenario_idx, observed_action, relationship_condition, padded_slot]
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (-w_d * access * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_base_padded_rel(
    padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table, v_padded_table,
):
    effort = effort_table[scenario_idx, observed_action, relationship_condition, padded_slot]
    V = get_lm_v_padded_rel(
        padded_slot, scenario_idx, observed_action, relationship_condition,
        reward_condition, v_padded_table,
    )
    return alpha * (w_v * V - w_e * effort)


@memo
def actor_continuous_full_padded_rel[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_rel(
            padded_slot, scenario_idx, observed_action, relationship_condition, prior_table,
        ) * exp(
            get_utility_full_padded_rel(
                padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition,
                alpha, w_v, w_d, w_e, gamma,
                access_table, effort_table, v_padded_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_continuous_discomfort_only_padded_rel[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_rel(
            padded_slot, scenario_idx, observed_action, relationship_condition, prior_table,
        ) * exp(
            get_utility_discomfort_only_padded_rel(
                padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition,
                alpha, w_d, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def actor_continuous_base_padded_rel[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(observed_action)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        padded_slot in PaddedActionSlots,
        wpp=get_prior_padded_rel(
            padded_slot, scenario_idx, observed_action, relationship_condition, prior_table,
        ) * exp(
            get_utility_base_padded_rel(
                padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table, v_padded_table,
            )
        ),
    )
    return Pr[actor.padded_slot == padded_slot]


@memo
def observer_reward_full_padded_rel[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, alpha_observer, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_full_padded_rel[
                padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition
            ](alpha, w_v, w_d, w_e, gamma, access_table, effort_table, v_padded_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_discomfort_only_padded_rel[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, alpha_observer, access_table: ..., effort_table: ..., prior_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_discomfort_only_padded_rel[
                padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition
            ](alpha, w_d, gamma, access_table, effort_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]


@memo
def observer_reward_base_padded_rel[
    padded_slot: PaddedActionSlots,
    scenario_idx: Scenarios,
    observed_action: ObservedActions,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, alpha_observer, access_table: ..., effort_table: ..., v_padded_table: ..., prior_table: ...):
    cast: [actor, observer]
    observer: knows(scenario_idx)
    observer: knows(observed_action)
    observer: knows(relationship_condition)
    observer: thinks[
        actor : knows(scenario_idx),
        actor : knows(observed_action),
        actor : knows(relationship_condition),
        actor : chooses(reward_condition in RewardConditions, wpp=1),
        actor : chooses(
            padded_slot in PaddedActionSlots,
            wpp=actor_continuous_base_padded_rel[
                padded_slot, scenario_idx, observed_action, relationship_condition, reward_condition
            ](alpha, w_v, w_e, access_table, effort_table, v_padded_table, prior_table),
        ),
    ]
    observer: observes[actor.padded_slot] is padded_slot
    observer: chooses(
        reward_condition in RewardConditions,
        wpp=E[actor.reward_condition == reward_condition] ** alpha_observer,
    )
    return Pr[observer.reward_condition == reward_condition]
