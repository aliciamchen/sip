"""
Constants, enums, scenario maps, and LM-elicited table loaders shared across all
model code. Anything that names dimensions (action enums, scenario indices,
intimacy levels) or loads scenario-specific feature tables (access, effort,
signed-valence V, padded action spaces) lives here.

Dependency layer 0: imports nothing from sibling modules. `utility.py`,
`actors.py`, and `observers.py` all import from here.
"""

from enum import IntEnum
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

# ==============================================================================
# Action and intimacy axes
# ==============================================================================

actions = jnp.array([0, 1, 2, 3])
IntimacyLevels = jnp.arange(0, 1.01, 0.01)

# Effort experiment uses 2 actions instead of 4 (action_1 = non-share, action_2 = share)
actions_effort = jnp.array([0, 1])


class RewardConditions(IntEnum):
    LOW = 0
    HIGH = 1


class RelationshipConditions(IntEnum):
    ZERO = 0
    FIFTY = 1
    SEVENTY_FIVE = 2
    ONE_HUNDRED = 3


class EffortConditions(IntEnum):
    LOW = 0
    HIGH = 1


# Continuous intimacy values for each RelationshipConditions level — used by
# the relationship-keyed padded memos to evaluate the (1 - I) access term
# without dragging the 101-level IntimacyLevels axis into the memo.
RELATIONSHIP_LEVEL_VALUES = jnp.array([0.0, 0.5, 0.75, 1.0])

EFFORT_CONDITION_TO_IDX = {"low": 0, "high": 1}
N_ACTIONS_EFFORT = 2
N_EFFORT_CONDITIONS = 2


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
# Scenario labels
# ==============================================================================
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


NONFOOD_SCENARIO_LABELS = [
    "bed", "blanket", "breakup", "chapstick", "gossip", "hairbrush",
    "harmonica", "hat", "home", "locker-room", "navigation", "payment",
    "sauna", "sleeping-bag", "sunscreen", "towel",
]
NONFOOD_SCENARIO_TO_IDX = {label: idx for idx, label in enumerate(NONFOOD_SCENARIO_LABELS)}


# ==============================================================================
# Canonical (food) LM-derived scenario-specific parameters: access + effort
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
    if the CSV is missing — run `uv run python model/lm/score_canonical_features.py` first.
    """
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent / "outputs" / "lm" / "lm_scenario_params.csv"
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
# The memo models in the actors / observers modules use `Scenarios: Scenarios`
# as a memo dimension, but only the cardinality (16) is load-bearing — the
# IntEnum names (APPLES, BASKETBALL, ...) are not. So food and nonfood reuse
# the same memo models with different scenario-label↔index maps and different
# LLM tables.


def _load_nonfood_lm_tables():
    path = (
        Path(__file__).resolve().parent / "outputs" / "lm" / "lm_scenario_params_nonfood.csv"
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
    `uv run python model/lm/score_canonical_v.py --domain {domain}` first.
    """
    if domain == "food":
        scenario_to_idx = SCENARIO_TO_IDX
        filename = "lm_scenario_v.csv"
    elif domain == "nonfood":
        scenario_to_idx = NONFOOD_SCENARIO_TO_IDX
        filename = "lm_scenario_v_nonfood.csv"
    else:
        raise ValueError(f"Unknown domain: {domain!r}")
    path = Path(__file__).resolve().parent / "outputs" / "lm" / filename
    df = pd.read_csv(path)
    motivation_to_idx = {"low": int(RewardConditions.LOW), "high": int(RewardConditions.HIGH)}
    v = np.zeros((16, 4, 2), dtype=np.float32)
    for _, row in df.iterrows():
        s = scenario_to_idx[row["scenario_label"]]
        a = int(row["action"])
        m = motivation_to_idx[row["motivation"]]
        v[s, a, m] = row["v"]
    return jnp.array(v)


# ==============================================================================
# Padded LM tables for the no-alternatives-shown observer
# ==============================================================================


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
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "lm"
    canonical_path = canonical_path or outputs_dir / "lm_scenario_params.csv"
    canonical_v_path = canonical_v_path or outputs_dir / "lm_scenario_v.csv"
    alternatives_path = alternatives_path or outputs_dir / "lm_alternatives_food_inv-intimacy_desire_noalt.csv"
    alternatives_features_path = (
        alternatives_features_path or outputs_dir / "lm_alternatives_features_food_inv-intimacy_desire_noalt.csv"
    )
    alternatives_v_path = alternatives_v_path or outputs_dir / "lm_alternatives_v_food_inv-intimacy_desire_noalt.csv"

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
    used by `food_inv-desire_intimacy_noalt`.

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
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "lm"
    canonical_path = canonical_path or outputs_dir / "lm_scenario_params.csv"
    canonical_v_path = canonical_v_path or outputs_dir / "lm_scenario_v.csv"
    alternatives_path = (
        alternatives_path or outputs_dir / "lm_alternatives_food_inv-desire_intimacy_noalt.csv"
    )
    alternatives_features_path = (
        alternatives_features_path
        or outputs_dir / "lm_alternatives_features_food_inv-desire_intimacy_noalt.csv"
    )
    alternatives_v_path = (
        alternatives_v_path or outputs_dir / "lm_alternatives_v_food_inv-desire_intimacy_noalt.csv"
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
# Effort-experiment LM-derived parameter tables
# ==============================================================================
# Effort experiment uses a 2-action design with an effort_condition covariate.
# Reward is held fixed at HIGH; V(a|s) = 1 stipulated for both actions (handled
# in utility.py:get_stipulated_reward_effort).


def load_lm_scenario_params_effort(filepath=None):
    """Load access and effort tables for the effort experiment.

    Returns a dict with:
      - "access": jnp.array of shape (16, 2, 2) — (scenario, effort_condition, action)
      - "effort": jnp.array of shape (16, 2, 2)

    Raises FileNotFoundError if the CSV is missing — run
    `uv run python model/lm/score_effort_features.py` first.
    """
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent
            / "outputs"
            / "lm"
            / "lm_scenario_params_effort.csv"
        )
    df = pd.read_csv(filepath)
    shape = (len(SCENARIO_LABELS), N_EFFORT_CONDITIONS, N_ACTIONS_EFFORT)
    access = np.zeros(shape, dtype=np.float32)
    effort = np.zeros(shape, dtype=np.float32)
    for _, row in df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        e_idx = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        # CSV action 1 -> internal 0, CSV action 2 -> internal 1
        a_idx = int(row["action"]) - 1
        access[s_idx, e_idx, a_idx] = row["access"]
        effort[s_idx, e_idx, a_idx] = row["effort"]
    return {"access": jnp.array(access), "effort": jnp.array(effort)}


LLM_TABLES_EFFORT = load_lm_scenario_params_effort()


def load_lm_scenario_params_effort_marginal(filepath=None):
    """Load effort-marginal access ratings (vignette without effort paragraph).

    Used by the food_inv-effort_intimacy_alt experiment, where the observer does
    not see the effort paragraph and so cannot perceive any effort-induced
    setting differences when reasoning about action access.

    Returns a jnp.array of shape (16, 2, 2) with the marginal access value
    broadcast across the effort_condition dimension — so it slots into the
    same indexing pattern as the conditional table without changing any
    downstream code. Returns None if the CSV is missing (the conditional
    table is then used everywhere as a fallback).
    """
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent
            / "outputs"
            / "lm"
            / "lm_scenario_params_effort_marginal.csv"
        )
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    n_scen, n_act = len(SCENARIO_LABELS), N_ACTIONS_EFFORT
    flat = np.zeros((n_scen, n_act), dtype=np.float32)
    for _, row in df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        a_idx = int(row["action"]) - 1
        flat[s_idx, a_idx] = row["access"]
    # Broadcast across effort_condition so the actor utility's existing
    # indexing access_table[scenario, effort, action] returns the same
    # value for both effort conditions.
    broadcast = np.broadcast_to(flat[:, None, :], (n_scen, N_EFFORT_CONDITIONS, n_act))
    return jnp.array(broadcast)


_access_marg = load_lm_scenario_params_effort_marginal()
if _access_marg is not None:
    LLM_TABLES_EFFORT["access_marg"] = _access_marg
