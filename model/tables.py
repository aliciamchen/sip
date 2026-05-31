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

# Continuous desire latent (Studies 1a/1b infer it). ψ(d) ∈ [0, 1]: 0 = "not at
# all want the food", 1 = "extremely". Read out to the 1–7 human scale as
# 1 + 6·d. Same 101-bin grid as IntimacyLevels so the inferred-desire observers
# reuse the continuous-intimacy machinery. Enters the utility as the reward
# multiplier w_v · desire · g(a|s), where g is the desire-free goal-satisfaction
# of the action (see load_lm_g_3act / the padded g loaders below).
DesireLevels = jnp.arange(0, 1.01, 0.01)

# Effort experiment uses 2 actions instead of 4 (action_1 = non-share, action_2 = share)
actions_effort = jnp.array([0, 1])

# Three-action canonical set used by the new inverse-planning experiments
# (Studies 2, 3a, 3b, 4a, 4b in the manuscript). Action 0 = no sharing,
# action 1 = low-risk sharing, action 2 = high-risk sharing. Stimulus set is
# `experiments/scenarios.csv`; LM tables live in `outputs/lm/lm_scenario_params_3act.csv`
# (access + effort, shaped (16, 2, 3) over scenario × effort_condition × action,
# matching the effort experiment's layout) and `outputs/lm/lm_scenario_v_3act.csv`
# (V, shaped (16, 3, 2) over scenario × action × motivation).
actions_3act = jnp.array([0, 1, 2])
N_ACTIONS_3ACT = 3
# The 3 canonical actions in index order. Experiment data and LM CSVs label the
# observed action with these names (was action_0/1/2 before the May 2026 rename).
ACTION_COLS = ["no_share", "low_risk_share", "high_risk_share"]
ACTION_LABEL_TO_IDX = {label: i for i, label in enumerate(ACTION_COLS)}


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


# Padded-action constants for the 3-action inverse-planning experiments (Studies
# 2, 3a, 3b, 4a, 4b — the active roster). The 3-act canonical stimulus offers
# only 3 actions per scenario, so the LM-generated alternative set is smaller
# than for the 4-action legacy; MAX_ACTIONS_3ACT = 12 means up to 11 LM-elicited
# alternatives plus the observed canonical action in slot 0.
MAX_ACTIONS_3ACT = 12
padded_slots_3act = jnp.arange(MAX_ACTIONS_3ACT)


class PaddedActionSlots3Act(IntEnum):
    """Memo-friendly enum of padded action slot indices for the 3-action design."""

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


class ObservedActions3Act(IntEnum):
    """The 3 canonical observed actions for the 3-act design (matches actions_3act)."""

    A0 = 0
    A1 = 1
    A2 = 2


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
    "bed",
    "blanket",
    "breakup",
    "chapstick",
    "gossip",
    "hairbrush",
    "harmonica",
    "hat",
    "home",
    "locker-room",
    "navigation",
    "payment",
    "sauna",
    "sleeping-bag",
    "sunscreen",
    "towel",
]
NONFOOD_SCENARIO_TO_IDX = {
    label: idx for idx, label in enumerate(NONFOOD_SCENARIO_LABELS)
}


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
            Path(__file__).resolve().parent
            / "outputs"
            / "lm"
            / "lm_scenario_params.csv"
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
        Path(__file__).resolve().parent
        / "outputs"
        / "lm"
        / "lm_scenario_params_nonfood.csv"
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
        return (
            NONFOOD_SCENARIO_LABELS,
            NONFOOD_SCENARIO_TO_IDX,
            _load_nonfood_lm_tables(),
        )
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
    motivation_to_idx = {
        "low": int(RewardConditions.LOW),
        "high": int(RewardConditions.HIGH),
    }
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
    from lm_alternatives_features_food_inv_intimacy_desire_noalt.csv; V from lm_alternatives_v_food_inv_intimacy_desire_noalt.csv with
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
    alternatives_path = (
        alternatives_path
        or outputs_dir / "lm_alternatives_food_inv_intimacy_desire_noalt.csv"
    )
    alternatives_features_path = (
        alternatives_features_path
        or outputs_dir / "lm_alternatives_features_food_inv_intimacy_desire_noalt.csv"
    )
    alternatives_v_path = (
        alternatives_v_path
        or outputs_dir / "lm_alternatives_v_food_inv_intimacy_desire_noalt.csv"
    )

    required = [
        alternatives_path,
        alternatives_features_path,
        canonical_v_path,
        alternatives_v_path,
    ]
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

    motivation_to_idx = {
        "low": int(RewardConditions.LOW),
        "high": int(RewardConditions.HIGH),
    }
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
        canon_v_lookup[
            (row["scenario_label"], int(row["action"]), row["motivation"])
        ] = float(row["v"])
    for scenario in SCENARIO_LABELS:
        s_idx = SCENARIO_TO_IDX[scenario]
        for observed in range(n_observed):
            a_access, a_effort = canon_ae_lookup[(scenario, observed)]
            for motivation_str, m_idx in motivation_to_idx.items():
                access[s_idx, observed, m_idx, 0] = a_access
                effort[s_idx, observed, m_idx, 0] = a_effort
                v[s_idx, observed, m_idx, 0] = canon_v_lookup[
                    (scenario, observed, motivation_str)
                ]
                valid_mask[s_idx, observed, m_idx, 0] = True

    # LM-generated alternatives (slots 1..k) per cell. V uses the diagonal
    # of (motivation context, motivation_query) — under reward_condition=m,
    # the actor evaluates the m-context alternative's V under m.
    alt_v_lookup = {
        (
            r["scenario_label"],
            r["observed_action"],
            r["motivation"],
            int(r["alt_idx"]),
        ): float(r["v"])
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
    used by `food_inv_desire_intimacy_noalt`.

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
    lm_alternatives_features_food_inv_desire_intimacy_noalt.csv and lm_alternatives_v_food_inv_desire_intimacy_noalt.csv.

    Returns a dict {access, effort, v, prior}, or None if any required CSV is
    missing.
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "lm"
    canonical_path = canonical_path or outputs_dir / "lm_scenario_params.csv"
    canonical_v_path = canonical_v_path or outputs_dir / "lm_scenario_v.csv"
    alternatives_path = (
        alternatives_path
        or outputs_dir / "lm_alternatives_food_inv_desire_intimacy_noalt.csv"
    )
    alternatives_features_path = (
        alternatives_features_path
        or outputs_dir / "lm_alternatives_features_food_inv_desire_intimacy_noalt.csv"
    )
    alternatives_v_path = (
        alternatives_v_path
        or outputs_dir / "lm_alternatives_v_food_inv_desire_intimacy_noalt.csv"
    )

    required = [
        alternatives_path,
        alternatives_features_path,
        canonical_v_path,
        alternatives_v_path,
    ]
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
    motivation_to_idx = {
        "low": int(RewardConditions.LOW),
        "high": int(RewardConditions.HIGH),
    }
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
        canon_v_lookup[
            (row["scenario_label"], int(row["action"]), row["motivation"])
        ] = float(row["v"])
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
        feats_df.groupby(
            ["scenario_label", "observed_action", "relationship_condition"]
        )
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

    Originally built for observer experiments where the observer did not see
    the effort paragraph and so could not perceive any effort-induced setting
    differences when reasoning about action access.

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


# ==============================================================================
# Three-action canonical LM-derived parameter tables
# ==============================================================================
# Used by the new inverse-planning experiments (Studies 2, 3a, 3b, 4a, 4b).
# Tables are scenario × effort_condition × action for access and effort (same
# layout as the effort experiment, just 3 actions instead of 2), and
# scenario × action × motivation for V (same layout as the canonical 4-action
# pipeline, just 3 actions instead of 4).


def load_lm_scenario_params_3act(filepath=None):
    """Load access and effort tables for the 3-action design.

    Returns a dict {"access": (16, 2, 3), "effort": (16, 2, 3)} or None if the
    CSV is missing (the LM elicitation step that produces it is currently
    deferred — see plan step 4).
    """
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent
            / "outputs"
            / "lm"
            / "lm_scenario_params_3act.csv"
        )
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    shape = (len(SCENARIO_LABELS), N_EFFORT_CONDITIONS, N_ACTIONS_3ACT)
    access = np.zeros(shape, dtype=np.float32)
    effort = np.zeros(shape, dtype=np.float32)
    for _, row in df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        e_idx = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        a_idx = int(row["action"])
        access[s_idx, e_idx, a_idx] = row["access"]
        effort[s_idx, e_idx, a_idx] = row["effort"]
    return {"access": jnp.array(access), "effort": jnp.array(effort)}


def load_lm_scenario_params_3act_marginal(filepath=None):
    """Load effort-marginal access ratings for the 3-action design.

    Used by Study 3a, where the observer infers effort and so does not see the
    effort paragraph; the access table broadcast across effort_condition keeps
    the indexing pattern `access[scenario, effort, action]` unchanged for the
    actor inside the observer's `thinks[...]` block.

    Returns a jnp.array of shape (16, 2, 3) or None if the CSV is missing.
    """
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent
            / "outputs"
            / "lm"
            / "lm_scenario_params_3act_marginal.csv"
        )
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    n_scen, n_act = len(SCENARIO_LABELS), N_ACTIONS_3ACT
    flat = np.zeros((n_scen, n_act), dtype=np.float32)
    for _, row in df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        a_idx = int(row["action"])
        flat[s_idx, a_idx] = row["access"]
    broadcast = np.broadcast_to(flat[:, None, :], (n_scen, N_EFFORT_CONDITIONS, n_act))
    return jnp.array(broadcast)


def load_lm_v_3act(domain="food", filepath=None):
    """Load signed-valence V table for the 3-action design.

    Returns a jnp.array of shape (16, 3, 2) indexed by
    (scenario, action, reward_condition), or None if the CSV is missing.
    """
    if domain == "food":
        scenario_to_idx = SCENARIO_TO_IDX
        filename = "lm_scenario_v_3act.csv"
    elif domain == "nonfood":
        scenario_to_idx = NONFOOD_SCENARIO_TO_IDX
        filename = "lm_scenario_v_3act_nonfood.csv"
    else:
        raise ValueError(f"Unknown domain: {domain!r}")
    if filepath is None:
        filepath = Path(__file__).resolve().parent / "outputs" / "lm" / filename
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    motivation_to_idx = {
        "low": int(RewardConditions.LOW),
        "high": int(RewardConditions.HIGH),
    }
    v = np.zeros((len(scenario_to_idx), N_ACTIONS_3ACT, 2), dtype=np.float32)
    for _, row in df.iterrows():
        s = scenario_to_idx[row["scenario_label"]]
        a = int(row["action"])
        m = motivation_to_idx[row["motivation"]]
        v[s, a, m] = row["v"]
    return jnp.array(v)


def load_lm_g_3act(domain="food", filepath=None):
    """Load goal-satisfaction g table for the 3-action design.

    g(a|s) is the desire-free LM rating of how much an action results in the two
    people getting/eating the food, normalized to [0, 1]. It replaces the old
    per-motivation signed-valence V: desire now enters the utility as the
    continuous multiplier w_v · desire · g (see DesireLevels). Because g is
    desire-free it has NO motivation axis.

    Returns a jnp.array of shape (16, 3) indexed by (scenario, action), or None
    if the CSV is missing.
    """
    if domain == "food":
        scenario_to_idx = SCENARIO_TO_IDX
        filename = "lm_scenario_g_3act.csv"
    elif domain == "nonfood":
        scenario_to_idx = NONFOOD_SCENARIO_TO_IDX
        filename = "lm_scenario_g_3act_nonfood.csv"
    else:
        raise ValueError(f"Unknown domain: {domain!r}")
    if filepath is None:
        filepath = Path(__file__).resolve().parent / "outputs" / "lm" / filename
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    g = np.zeros((len(scenario_to_idx), N_ACTIONS_3ACT), dtype=np.float32)
    for _, row in df.iterrows():
        s = scenario_to_idx[row["scenario_label"]]
        a = int(row["action"])
        g[s, a] = row["g"]
    return jnp.array(g)


def load_lm_scenario_desire_3act(domain="food", filepath=None):
    """Load the per-condition desire scalar for the given-desire studies
    (2a `food_inv_intimacy`, 2b `food_inv_joint_ie`).

    When desire is observer-visible context, the LM reads the scenario + the
    shown desire paragraph and rates how much the two people want the food on the
    [0, 1] scale (1–7 read out as 1 + 6·d). That scalar plugs into the actor
    utility as the constant `desire` in w_v · desire · g.

    Returns a jnp.array of shape (16, 2) indexed by
    (scenario, reward_condition), or None if the CSV is missing.
    """
    if domain == "food":
        scenario_to_idx = SCENARIO_TO_IDX
        filename = "lm_scenario_desire_3act.csv"
    elif domain == "nonfood":
        scenario_to_idx = NONFOOD_SCENARIO_TO_IDX
        filename = "lm_scenario_desire_3act_nonfood.csv"
    else:
        raise ValueError(f"Unknown domain: {domain!r}")
    if filepath is None:
        filepath = Path(__file__).resolve().parent / "outputs" / "lm" / filename
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    reward_to_idx = {
        "low": int(RewardConditions.LOW),
        "high": int(RewardConditions.HIGH),
    }
    d = np.zeros((len(scenario_to_idx), 2), dtype=np.float32)
    for _, row in df.iterrows():
        s = scenario_to_idx[row["scenario_label"]]
        r = reward_to_idx[row["reward_condition"]]
        d[s, r] = row["desire"]
    return jnp.array(d)


LLM_TABLES_3ACT = load_lm_scenario_params_3act()
_access_marg_3act = load_lm_scenario_params_3act_marginal()
if LLM_TABLES_3ACT is not None and _access_marg_3act is not None:
    LLM_TABLES_3ACT["access_marg"] = _access_marg_3act


# ==============================================================================
# Padded LM tables for the 3-action desire-inference experiment (Study 3b)
# ==============================================================================
# `food_inv_desire` — observer knows effort + intimacy, infers reward.
# Alternatives are conditioned on (scenario, observed_action, effort_condition,
# intimacy_condition) — i.e. on the variables the human participant sees in the
# trial. The actor inside the observer's `thinks[...]` block evaluates V under
# both reward values (the latent), so V carries an extra motivation_query axis.


def load_padded_lm_tables_3act_desire(
    canonical_path=None,
    canonical_g_path=None,
    alternatives_path=None,
    alternatives_features_path=None,
    alternatives_g_path=None,
):
    """Build padded tables for Study 1a's LM-generated alternatives action space.

    Shapes (with S = MAX_ACTIONS_3ACT):
      - access: (16, 3, 2, 4, S) — (scenario, observed_action, effort_condition,
        intimacy_condition, slot)
      - effort: (16, 3, 2, 4, S)
      - prior:  (16, 3, 2, 4, S)
      - g:      (16, 3, 2, 4, S) — goal-satisfaction; desire-free, so NO
        motivation axis. Desire enters the utility as the continuous multiplier
        w_v · desire · g, with desire the inferred latent.

    Slot 0 of every cell holds the observed canonical action's features:
    access/effort from `lm_scenario_params_3act.csv` (depends on scenario +
    effort_condition + action but not intimacy, so broadcasts across intimacy);
    g from `lm_scenario_g_3act.csv` (depends on scenario + action, broadcasts
    across effort and intimacy).

    Slots 1..k hold the LM-generated alternatives for that cell, from
    `lm_alternatives_food_inv_desire.csv`,
    `lm_alternatives_features_food_inv_desire.csv`, and
    `lm_alternatives_g_food_inv_desire.csv`. Remaining slots are null-padded
    (access/effort/g = 0; prior = NULL_EPSILON to keep the softmax
    differentiable).

    Returns a dict {access, effort, g, prior} of jnp.arrays, or None if any
    required CSV is missing.
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "lm"
    canonical_path = canonical_path or outputs_dir / "lm_scenario_params_3act.csv"
    canonical_g_path = canonical_g_path or outputs_dir / "lm_scenario_g_3act.csv"
    alternatives_path = (
        alternatives_path or outputs_dir / "lm_alternatives_food_inv_desire.csv"
    )
    alternatives_features_path = (
        alternatives_features_path
        or outputs_dir / "lm_alternatives_features_food_inv_desire.csv"
    )
    alternatives_g_path = (
        alternatives_g_path or outputs_dir / "lm_alternatives_g_food_inv_desire.csv"
    )

    required = [
        canonical_path,
        canonical_g_path,
        alternatives_path,
        alternatives_features_path,
        alternatives_g_path,
    ]
    if any(not Path(p).exists() for p in required):
        return None

    canonical_df = pd.read_csv(canonical_path)
    canonical_g_df = pd.read_csv(canonical_g_path)
    feats_df = pd.read_csv(alternatives_features_path)
    alts_g_df = pd.read_csv(alternatives_g_path)

    n_scenarios = len(SCENARIO_LABELS)
    n_observed = N_ACTIONS_3ACT
    n_effort = N_EFFORT_CONDITIONS
    n_intimacy = 4
    shape_5d = (n_scenarios, n_observed, n_effort, n_intimacy, MAX_ACTIONS_3ACT)
    access = np.zeros(shape_5d, dtype=np.float32)
    effort = np.zeros(shape_5d, dtype=np.float32)
    g = np.zeros(shape_5d, dtype=np.float32)
    valid_mask = np.zeros(shape_5d, dtype=bool)

    intimacy_to_idx = {0: 0, 50: 1, 75: 2, 100: 3}
    observed_str_to_idx = {f"action_{i}": i for i in range(n_observed)}

    # Canonical (slot 0): access/effort depend on scenario + effort + action
    # (3-act CSV layout); broadcast across intimacy. g depends on scenario +
    # action; broadcast across effort and intimacy.
    canon_ae_lookup = {}
    for _, row in canonical_df.iterrows():
        e_idx = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        canon_ae_lookup[(row["scenario_label"], e_idx, int(row["action"]))] = (
            float(row["access"]),
            float(row["effort"]),
        )
    canon_g_lookup = {
        (row["scenario_label"], int(row["action"])): float(row["g"])
        for _, row in canonical_g_df.iterrows()
    }

    for scenario in SCENARIO_LABELS:
        s_idx = SCENARIO_TO_IDX[scenario]
        for observed in range(n_observed):
            for e_idx in range(n_effort):
                a_access, a_effort = canon_ae_lookup[(scenario, e_idx, observed)]
                for i_idx in range(n_intimacy):
                    access[s_idx, observed, e_idx, i_idx, 0] = a_access
                    effort[s_idx, observed, e_idx, i_idx, 0] = a_effort
                    g[s_idx, observed, e_idx, i_idx, 0] = canon_g_lookup[
                        (scenario, observed)
                    ]
                    valid_mask[s_idx, observed, e_idx, i_idx, 0] = True

    # LM-generated alternatives (slots 1..k). Index by (scenario,
    # observed_action, effort_condition, intimacy_condition, alt_idx). g is
    # desire-free, so no motivation_query axis.
    alt_g_lookup = {
        (
            r["scenario_label"],
            r["observed_action"],
            r["effort_condition"],
            int(r["intimacy_condition"]),
            int(r["alt_idx"]),
        ): float(r["g"])
        for _, r in alts_g_df.iterrows()
    }

    for _, row in feats_df.iterrows():
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        o_idx = observed_str_to_idx[row["observed_action"]]
        e_idx = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        i_idx = intimacy_to_idx[int(row["intimacy_condition"])]
        alt_idx = int(row["alt_idx"])
        slot = alt_idx + 1
        if slot >= MAX_ACTIONS_3ACT:
            continue
        access[s_idx, o_idx, e_idx, i_idx, slot] = float(row["access"])
        effort[s_idx, o_idx, e_idx, i_idx, slot] = float(row["effort"])
        g[s_idx, o_idx, e_idx, i_idx, slot] = alt_g_lookup.get(
            (
                row["scenario_label"],
                row["observed_action"],
                row["effort_condition"],
                int(row["intimacy_condition"]),
                alt_idx,
            ),
            0.0,
        )
        valid_mask[s_idx, o_idx, e_idx, i_idx, slot] = True

    NULL_EPSILON = 1e-8
    n_valid = valid_mask.sum(axis=-1, keepdims=True)
    prior_table = np.where(
        valid_mask, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON
    ).astype(np.float32)

    max_alt_count = (
        feats_df.groupby(
            [
                "scenario_label",
                "observed_action",
                "effort_condition",
                "intimacy_condition",
            ]
        )
        .size()
        .max()
    )
    if max_alt_count + 1 > MAX_ACTIONS_3ACT:
        print(
            f"WARNING: largest cell has {max_alt_count} LM-generated alternatives + "
            f"1 observed = {max_alt_count + 1} actions, exceeding "
            f"MAX_ACTIONS_3ACT={MAX_ACTIONS_3ACT}. Extra alternatives were truncated."
        )

    return {
        "access": jnp.array(access),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior_table),
    }


LLM_TABLES_3ACT_DESIRE_PADDED = load_padded_lm_tables_3act_desire()


# =============================================================================
# Padded LM-alternatives loaders for the migrated studies (1b, 2a, 2b)
# =============================================================================
# Each mirrors load_padded_lm_tables_3act_desire but with the cell grid and
# feature axes appropriate to which variables the observer infers (see the
# utility.py / observers.py sections for the shape rationale). All return None
# when any required CSV is missing, so imports stay clean before LM elicitation
# has been run for that study.
#
# Expected CSV schema (produced by score_3act_merged.py --study <slug>):
#   - canonical access/effort: lm_scenario_params_3act.csv
#       (scenario_label, effort_condition, action, access, effort)
#   - canonical V: lm_scenario_v_3act.csv (scenario_label, action, motivation, v)
#   - alts features: lm_alternatives_features_<slug>.csv keyed by the study's
#       generation cell + effort_condition + alt_idx, columns access, effort
#   - alts V: lm_alternatives_v_<slug>.csv keyed by gen cell + alt_idx +
#       motivation_query, column v


def _canonical_lookups(canonical_path, canonical_g_path):
    """Shared canonical (slot-0) lookups: (access,effort) per
    (scenario, effort_condition, action) and goal-satisfaction g per
    (scenario, action). g is desire-free, so it has no motivation key."""
    canonical_df = pd.read_csv(canonical_path)
    canonical_g_df = pd.read_csv(canonical_g_path)
    ae = {}
    for _, row in canonical_df.iterrows():
        e_idx = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        ae[(row["scenario_label"], e_idx, int(row["action"]))] = (
            float(row["access"]),
            float(row["effort"]),
        )
    g = {
        (row["scenario_label"], int(row["action"])): float(row["g"])
        for _, row in canonical_g_df.iterrows()
    }
    return ae, g


def load_padded_lm_tables_3act_joint_de(
    canonical_path=None,
    canonical_g_path=None,
    alternatives_features_path=None,
    alternatives_g_path=None,
):
    """Study 1b: observer knows intimacy, jointly infers (desire, effort). Cell
    grid is (scenario, observed_action, intimacy_condition). effort inferred ->
    effort table carries an effort_condition feature axis. desire enters as the
    continuous multiplier w_v · desire · g, so g is desire-free (no motivation
    axis) and matches the access shape.
      access: (16, 3, 4, S)        [scenario, obs, relationship, slot]
      effort: (16, 3, 4, 2, S)     [scenario, obs, relationship, effort_condition, slot]
      g:      (16, 3, 4, S)        [scenario, obs, relationship, slot]
      prior:  (16, 3, 4, S)
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "lm"
    canonical_path = canonical_path or outputs_dir / "lm_scenario_params_3act.csv"
    canonical_g_path = canonical_g_path or outputs_dir / "lm_scenario_g_3act.csv"
    alternatives_features_path = (
        alternatives_features_path
        or outputs_dir / "lm_alternatives_features_food_inv_joint_de.csv"
    )
    alternatives_g_path = (
        alternatives_g_path or outputs_dir / "lm_alternatives_g_food_inv_joint_de.csv"
    )
    required = [
        canonical_path,
        canonical_g_path,
        alternatives_features_path,
        alternatives_g_path,
    ]
    if any(not Path(p).exists() for p in required):
        return None

    canon_ae, canon_g = _canonical_lookups(canonical_path, canonical_g_path)
    feats_df = pd.read_csv(alternatives_features_path)
    alts_g_df = pd.read_csv(alternatives_g_path)

    n_s, n_o, n_rel, n_eff = (
        len(SCENARIO_LABELS),
        N_ACTIONS_3ACT,
        4,
        N_EFFORT_CONDITIONS,
    )
    access = np.zeros((n_s, n_o, n_rel, MAX_ACTIONS_3ACT), dtype=np.float32)
    effort = np.zeros((n_s, n_o, n_rel, n_eff, MAX_ACTIONS_3ACT), dtype=np.float32)
    g = np.zeros((n_s, n_o, n_rel, MAX_ACTIONS_3ACT), dtype=np.float32)
    valid = np.zeros((n_s, n_o, n_rel, MAX_ACTIONS_3ACT), dtype=bool)

    intimacy_to_idx = {0: 0, 50: 1, 75: 2, 100: 3}
    obs_to_idx = ACTION_LABEL_TO_IDX

    # Canonical slot 0: access/effort per (scenario, effort_condition, action),
    # broadcast across relationship; g per (scenario, action).
    for scenario in SCENARIO_LABELS:
        s = SCENARIO_TO_IDX[scenario]
        for o in range(n_o):
            for rel in range(n_rel):
                for e in range(n_eff):
                    a_access, a_effort = canon_ae[(scenario, e, o)]
                    effort[s, o, rel, e, 0] = a_effort
                    if e == 0:
                        access[s, o, rel, 0] = a_access
                g[s, o, rel, 0] = canon_g[(scenario, o)]
                valid[s, o, rel, 0] = True

    # Alternatives (slots 1..k): features keyed by (scenario, obs, intimacy,
    # effort_condition, alt_idx). access is effort-marginal (same across e); g is
    # desire-free, keyed by (scenario, obs, intimacy, alt_idx).
    alt_g_lookup = {
        (
            r["scenario_label"],
            r["observed_action"],
            int(r["intimacy_condition"]),
            int(r["alt_idx"]),
        ): float(r["g"])
        for _, r in alts_g_df.iterrows()
    }
    for _, row in feats_df.iterrows():
        s = SCENARIO_TO_IDX[row["scenario_label"]]
        o = obs_to_idx[row["observed_action"]]
        rel = intimacy_to_idx[int(row["intimacy_condition"])]
        e = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        slot = int(row["alt_idx"]) + 1
        if slot >= MAX_ACTIONS_3ACT:
            continue
        access[s, o, rel, slot] = float(row["access"])
        effort[s, o, rel, e, slot] = float(row["effort"])
        g[s, o, rel, slot] = alt_g_lookup.get(
            (
                row["scenario_label"],
                row["observed_action"],
                int(row["intimacy_condition"]),
                int(row["alt_idx"]),
            ),
            0.0,
        )
        valid[s, o, rel, slot] = True

    NULL_EPSILON = 1e-8
    n_valid = valid.sum(axis=-1, keepdims=True)
    prior = np.where(valid, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON).astype(
        np.float32
    )
    return {
        "access": jnp.array(access),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior),
    }


def load_padded_lm_tables_3act_intimacy(
    canonical_path=None,
    canonical_g_path=None,
    alternatives_features_path=None,
    alternatives_g_path=None,
):
    """Study 2a: observer knows (desire, effort), infers intimacy. Cell grid is
    (scenario, observed_action, reward_condition, effort_condition). intimacy
    inferred (continuous; access modulated by (1-I)^gamma in the utility, no
    table axis). effort observed -> effort feature taken at the cell's effort.
    Desire is given (observer-visible); it enters as w_v · desire_table[s,r] · g,
    so g is desire-free (no motivation axis) and the per-condition desire scalar
    is loaded separately via load_lm_scenario_desire_3act.
      access: (16, 3, 2, 2, S)     [scenario, obs, reward, effort, slot]
      effort: (16, 3, 2, 2, S)     [scenario, obs, reward, effort, slot]
      g:      (16, 3, 2, 2, S)     [scenario, obs, reward, effort, slot]
      prior:  (16, 3, 2, 2, S)
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "lm"
    canonical_path = canonical_path or outputs_dir / "lm_scenario_params_3act.csv"
    canonical_g_path = canonical_g_path or outputs_dir / "lm_scenario_g_3act.csv"
    alternatives_features_path = (
        alternatives_features_path
        or outputs_dir / "lm_alternatives_features_food_inv_intimacy.csv"
    )
    alternatives_g_path = (
        alternatives_g_path or outputs_dir / "lm_alternatives_g_food_inv_intimacy.csv"
    )
    required = [
        canonical_path,
        canonical_g_path,
        alternatives_features_path,
        alternatives_g_path,
    ]
    if any(not Path(p).exists() for p in required):
        return None

    canon_ae, canon_g = _canonical_lookups(canonical_path, canonical_g_path)
    feats_df = pd.read_csv(alternatives_features_path)
    alts_g_df = pd.read_csv(alternatives_g_path)

    n_s, n_o, n_rew, n_eff = (
        len(SCENARIO_LABELS),
        N_ACTIONS_3ACT,
        2,
        N_EFFORT_CONDITIONS,
    )
    access = np.zeros((n_s, n_o, n_rew, n_eff, MAX_ACTIONS_3ACT), dtype=np.float32)
    effort = np.zeros((n_s, n_o, n_rew, n_eff, MAX_ACTIONS_3ACT), dtype=np.float32)
    g = np.zeros((n_s, n_o, n_rew, n_eff, MAX_ACTIONS_3ACT), dtype=np.float32)
    valid = np.zeros((n_s, n_o, n_rew, n_eff, MAX_ACTIONS_3ACT), dtype=bool)

    rew_to_idx = {"low": int(RewardConditions.LOW), "high": int(RewardConditions.HIGH)}
    obs_to_idx = ACTION_LABEL_TO_IDX

    for scenario in SCENARIO_LABELS:
        s = SCENARIO_TO_IDX[scenario]
        for o in range(n_o):
            for rew in range(n_rew):
                for e in range(n_eff):
                    a_access, a_effort = canon_ae[(scenario, e, o)]
                    access[s, o, rew, e, 0] = a_access
                    effort[s, o, rew, e, 0] = a_effort
                    g[s, o, rew, e, 0] = canon_g[(scenario, o)]
                    valid[s, o, rew, e, 0] = True

    # Alternatives keyed by (scenario, obs, reward, effort, alt_idx). g is
    # desire-free, keyed by (scenario, obs, reward, effort, alt_idx).
    alt_g_lookup = {
        (
            r["scenario_label"],
            r["observed_action"],
            r["desire_condition"],
            r["effort_condition"],
            int(r["alt_idx"]),
        ): float(r["g"])
        for _, r in alts_g_df.iterrows()
    }
    for _, row in feats_df.iterrows():
        s = SCENARIO_TO_IDX[row["scenario_label"]]
        o = obs_to_idx[row["observed_action"]]
        rew = rew_to_idx[row["desire_condition"]]
        e = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        slot = int(row["alt_idx"]) + 1
        if slot >= MAX_ACTIONS_3ACT:
            continue
        access[s, o, rew, e, slot] = float(row["access"])
        effort[s, o, rew, e, slot] = float(row["effort"])
        g[s, o, rew, e, slot] = alt_g_lookup.get(
            (
                row["scenario_label"],
                row["observed_action"],
                row["desire_condition"],
                row["effort_condition"],
                int(row["alt_idx"]),
            ),
            0.0,
        )
        valid[s, o, rew, e, slot] = True

    NULL_EPSILON = 1e-8
    n_valid = valid.sum(axis=-1, keepdims=True)
    prior = np.where(valid, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON).astype(
        np.float32
    )
    return {
        "access": jnp.array(access),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior),
    }


def load_padded_lm_tables_3act_joint_ie(
    canonical_path=None,
    canonical_g_path=None,
    alternatives_features_path=None,
    alternatives_g_path=None,
):
    """Study 2b: observer knows desire, infers (intimacy, effort). Cell grid is
    (scenario, observed_action, reward_condition). intimacy inferred (continuous,
    no table axis); effort inferred -> effort table carries an effort_condition
    feature axis. Desire is given; it enters as w_v · desire_table[s,r] · g, so g
    is desire-free (no motivation axis) and the per-condition desire scalar is
    loaded separately via load_lm_scenario_desire_3act.
      access: (16, 3, 2, S)        [scenario, obs, reward, slot]
      effort: (16, 3, 2, 2, S)     [scenario, obs, reward, effort_condition, slot]
      g:      (16, 3, 2, S)        [scenario, obs, reward, slot]
      prior:  (16, 3, 2, S)
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "lm"
    canonical_path = canonical_path or outputs_dir / "lm_scenario_params_3act.csv"
    canonical_g_path = canonical_g_path or outputs_dir / "lm_scenario_g_3act.csv"
    alternatives_features_path = (
        alternatives_features_path
        or outputs_dir / "lm_alternatives_features_food_inv_joint_ie.csv"
    )
    alternatives_g_path = (
        alternatives_g_path or outputs_dir / "lm_alternatives_g_food_inv_joint_ie.csv"
    )
    required = [
        canonical_path,
        canonical_g_path,
        alternatives_features_path,
        alternatives_g_path,
    ]
    if any(not Path(p).exists() for p in required):
        return None

    canon_ae, canon_g = _canonical_lookups(canonical_path, canonical_g_path)
    feats_df = pd.read_csv(alternatives_features_path)
    alts_g_df = pd.read_csv(alternatives_g_path)

    n_s, n_o, n_rew, n_eff = (
        len(SCENARIO_LABELS),
        N_ACTIONS_3ACT,
        2,
        N_EFFORT_CONDITIONS,
    )
    access = np.zeros((n_s, n_o, n_rew, MAX_ACTIONS_3ACT), dtype=np.float32)
    effort = np.zeros((n_s, n_o, n_rew, n_eff, MAX_ACTIONS_3ACT), dtype=np.float32)
    g = np.zeros((n_s, n_o, n_rew, MAX_ACTIONS_3ACT), dtype=np.float32)
    valid = np.zeros((n_s, n_o, n_rew, MAX_ACTIONS_3ACT), dtype=bool)

    rew_to_idx = {"low": int(RewardConditions.LOW), "high": int(RewardConditions.HIGH)}
    obs_to_idx = ACTION_LABEL_TO_IDX

    for scenario in SCENARIO_LABELS:
        s = SCENARIO_TO_IDX[scenario]
        for o in range(n_o):
            for rew in range(n_rew):
                for e in range(n_eff):
                    a_access, a_effort = canon_ae[(scenario, e, o)]
                    effort[s, o, rew, e, 0] = a_effort
                    if e == 0:
                        access[s, o, rew, 0] = a_access
                g[s, o, rew, 0] = canon_g[(scenario, o)]
                valid[s, o, rew, 0] = True

    # Alternatives keyed by (scenario, obs, reward, effort_condition, alt_idx);
    # access effort-marginal. g is desire-free, keyed by
    # (scenario, obs, reward, alt_idx).
    alt_g_lookup = {
        (
            r["scenario_label"],
            r["observed_action"],
            r["desire_condition"],
            int(r["alt_idx"]),
        ): float(r["g"])
        for _, r in alts_g_df.iterrows()
    }
    for _, row in feats_df.iterrows():
        s = SCENARIO_TO_IDX[row["scenario_label"]]
        o = obs_to_idx[row["observed_action"]]
        rew = rew_to_idx[row["desire_condition"]]
        e = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        slot = int(row["alt_idx"]) + 1
        if slot >= MAX_ACTIONS_3ACT:
            continue
        access[s, o, rew, slot] = float(row["access"])
        effort[s, o, rew, e, slot] = float(row["effort"])
        g[s, o, rew, slot] = alt_g_lookup.get(
            (
                row["scenario_label"],
                row["observed_action"],
                row["desire_condition"],
                int(row["alt_idx"]),
            ),
            0.0,
        )
        valid[s, o, rew, slot] = True

    NULL_EPSILON = 1e-8
    n_valid = valid.sum(axis=-1, keepdims=True)
    prior = np.where(valid, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON).astype(
        np.float32
    )
    return {
        "access": jnp.array(access),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior),
    }


LLM_TABLES_3ACT_JOINT_DE_PADDED = load_padded_lm_tables_3act_joint_de()
LLM_TABLES_3ACT_INTIMACY_PADDED = load_padded_lm_tables_3act_intimacy()
LLM_TABLES_3ACT_JOINT_IE_PADDED = load_padded_lm_tables_3act_joint_ie()
