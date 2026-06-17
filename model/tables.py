"""
Constants, enums, scenario maps, and LM-elicited table loaders shared across all
model code. Anything that names dimensions (action enums, scenario indices,
intimacy/desire levels) or loads scenario-specific feature tables (risk,
effort, goal-satisfaction g, per-condition desire, padded action spaces) lives
here.

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

IntimacyLevels = jnp.arange(0, 1.01, 0.01)

# Continuous desire latent (Studies 1a/1b infer it). ψ(d) ∈ [0, 1]: 0 = "not at
# all like the food", 1 = "extremely". Matches the human desire rating, which is
# also stored on the 0–1 scale (collected 0–100, normalized in preprocessing).
# Same 101-bin grid as IntimacyLevels so the inferred-desire observers reuse the
# continuous-intimacy machinery. Enters the utility as the desire multiplier
# w_v · desire · g(a|s), where g is the desire-free goal-satisfaction of the
# action (see the padded LM-table loaders below).
DesireLevels = jnp.arange(0, 1.01, 0.01)

# Three-action canonical set used by the inverse-planning experiments. Action 0
# = no sharing, action 1 = low-risk sharing, action 2 = high-risk sharing.
# Stimulus set is `experiments/scenarios.csv`; LM tables live per study in
# `outputs/lm/<slug>/lm_scenario.csv` (canonical risk/effort/g) +
# `lm_alternatives.csv` (the alternatives' risk/effort/g).
actions = jnp.array([0, 1, 2])
N_ACTIONS = 3
# The 3 canonical actions in index order. Experiment data and LM CSVs label the
# observed action with these names (was action_0/1/2 before the May 2026 rename).
ACTION_COLS = ["no_share", "low_risk_share", "high_risk_share"]
ACTION_LABEL_TO_IDX = {label: i for i, label in enumerate(ACTION_COLS)}


class DesireConditions(IntEnum):
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


# Intimacy is a purely verbal manipulation in the experiments: the condition is
# identified by a slug (no numeric code is stored in the data). These slugs, in
# ascending order (formal -> intimate), index the RelationshipConditions axis.
INTIMACY_CONDITIONS = ["max_formal", "neither", "somewhat_intimate", "max_intimate"]
INTIMACY_CONDITION_TO_IDX = {slug: i for i, slug in enumerate(INTIMACY_CONDITIONS)}

# Continuous intimacy values for each RelationshipConditions level — used by
# the relationship-keyed padded memos to evaluate the (1 - I) risk term
# without dragging the 101-level IntimacyLevels axis into the memo. These are
# placeholder magnitudes for each verbal level (pending LM elicitation); they
# are model-internal and never saved as condition labels.
RELATIONSHIP_LEVEL_VALUES = jnp.array([0.0, 0.5, 0.75, 1.0])

EFFORT_CONDITION_TO_IDX = {"low": 0, "high": 1}
N_EFFORT_CONDITIONS = 2


# Padded-action constants for the 3-action inverse-planning experiments (the
# active roster). The canonical stimulus offers only 3 actions per scenario, so
# MAX_ACTIONS = 12 means up to 11 LM-elicited alternatives plus the
# observed canonical action in slot 0.
MAX_ACTIONS = 12
padded_slots = jnp.arange(MAX_ACTIONS)


class PaddedActionSlots(IntEnum):
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


class ObservedActions(IntEnum):
    """The 3 canonical observed actions for the 3-act design (matches actions)."""

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
# Three-action LM-derived scenario parameter tables
# ==============================================================================
# Per-study canonical risk/effort/g and the padded LM-alternatives action spaces
# are loaded by the `load_padded_lm_tables_*` functions below (each reading
# `outputs/lm/<slug>/lm_scenario.csv` + `lm_alternatives.csv`). The per-condition
# desire scalar for the given-desire studies is loaded by load_lm_scenario_desire.


def load_lm_scenario_desire(slug, filepath=None):
    """Load the per-condition desire scalar for the given-desire studies
    (2a `food_inv_intimacy`, 2b `food_inv_joint_ie`).

    When desire is observer-visible context, the LM reads the scenario + the
    shown desire paragraph and rates how much the two people would like the food
    on the [0, 1] scale (the 0–100 rating divided by 100). That scalar plugs into
    the actor utility as the constant `desire` in w_v · desire · g.

    `slug` selects the study folder `outputs/lm/<slug>/lm_scenario_desire.csv`.

    Returns a jnp.array of shape (16, 2) indexed by
    (scenario, desire_condition), or None if the CSV is missing.
    """
    scenario_to_idx = SCENARIO_TO_IDX
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent
            / "outputs"
            / "lm"
            / slug
            / "lm_scenario_desire.csv"
        )
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    desire_to_idx = {
        "low": int(DesireConditions.LOW),
        "high": int(DesireConditions.HIGH),
    }
    d = np.zeros((len(scenario_to_idx), 2), dtype=np.float32)
    for _, row in df.iterrows():
        s = scenario_to_idx[row["scenario_label"]]
        r = desire_to_idx[row["desire_condition"]]
        d[s, r] = row["desire"]
    return jnp.array(d)


# ==============================================================================
# Padded LM tables for the 3-action desire-inference experiment (Study 1a)
# ==============================================================================
# `food_inv_desire` — observer knows effort + intimacy, infers desire.
# Alternatives are conditioned on (scenario, observed_action, effort_condition,
# intimacy_condition) — i.e. on the variables the human participant sees in the
# trial. The actor inside the observer's `thinks[...]` block reasons over the
# continuous desire latent (DesireLevels); g (goal-satisfaction) is desire-free,
# so it carries no desire axis.


def load_padded_lm_tables_desire(
    canonical_path=None,
    alternatives_path=None,
):
    """Build padded tables for Study 1a's LM-generated alternatives action space.

    Shapes (with S = MAX_ACTIONS):
      - risk: (16, 3, 2, 4, S) — (scenario, observed_action, effort_condition,
        intimacy_condition, slot)
      - effort: (16, 3, 2, 4, S)
      - prior:  (16, 3, 2, 4, S)
      - g:      (16, 3, 2, 4, S) — goal-satisfaction; desire-free, so NO
        desire axis. Desire enters the utility as the continuous multiplier
        w_v · desire · g, with desire the inferred latent.

    Slot 0 of every cell holds the observed canonical action's features:
    risk/effort/g from `lm_scenario.csv` (risk/effort depend on scenario +
    effort_condition + action, broadcast across intimacy; g depends on scenario +
    action, broadcast across effort and intimacy).

    Slots 1..k hold the LM-generated alternatives for that cell, read straight
    off `outputs/lm/food_inv_desire/lm_alternatives.csv` — the one table holding
    each alternative's text plus its risk/effort/g columns (keyed by alt_idx).
    Remaining slots are null-padded (risk/effort/g = 0; prior = NULL_EPSILON to
    keep the softmax differentiable).

    Returns a dict {risk, effort, g, prior} of jnp.arrays, or None if the tables
    are missing or not yet scored.
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "lm" / "food_inv_desire"
    canonical_path = canonical_path or outputs_dir / "lm_scenario.csv"
    alternatives_path = alternatives_path or outputs_dir / "lm_alternatives.csv"

    required = [canonical_path, alternatives_path]
    if any(not Path(p).exists() for p in required):
        return None

    alts_df = pd.read_csv(alternatives_path)
    if not _alts_ready(alts_df):
        return None
    canon_ae, canon_g = _canonical_lookups(canonical_path)

    n_scenarios = len(SCENARIO_LABELS)
    n_observed = N_ACTIONS
    n_effort = N_EFFORT_CONDITIONS
    n_intimacy = 4
    shape_5d = (n_scenarios, n_observed, n_effort, n_intimacy, MAX_ACTIONS)
    risk = np.zeros(shape_5d, dtype=np.float32)
    effort = np.zeros(shape_5d, dtype=np.float32)
    g = np.zeros(shape_5d, dtype=np.float32)
    valid_mask = np.zeros(shape_5d, dtype=bool)

    intimacy_to_idx = INTIMACY_CONDITION_TO_IDX
    observed_str_to_idx = ACTION_LABEL_TO_IDX

    # Canonical (slot 0): risk/effort depend on scenario + effort + action,
    # broadcast across intimacy; g depends on scenario + action.
    for scenario in SCENARIO_LABELS:
        s_idx = SCENARIO_TO_IDX[scenario]
        for observed in range(n_observed):
            for e_idx in range(n_effort):
                a_risk, a_effort = canon_ae[(scenario, e_idx, observed)]
                for i_idx in range(n_intimacy):
                    risk[s_idx, observed, e_idx, i_idx, 0] = a_risk
                    effort[s_idx, observed, e_idx, i_idx, 0] = a_effort
                    g[s_idx, observed, e_idx, i_idx, 0] = canon_g[(scenario, observed)]
                    valid_mask[s_idx, observed, e_idx, i_idx, 0] = True

    # LM-generated alternatives (slots 1..k), keyed by (scenario, observed,
    # effort_condition, intimacy_condition, alt_idx); risk/effort/g read straight
    # off the one merged alts table. Unscored stage-1 rows (NaN risk) are skipped.
    for _, row in alts_df.iterrows():
        if pd.isna(row["risk"]):
            continue
        s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
        o_idx = observed_str_to_idx[row["observed_action"]]
        e_idx = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        i_idx = intimacy_to_idx[row["intimacy_condition"]]
        slot = int(row["alt_idx"]) + 1
        if slot >= MAX_ACTIONS:
            continue
        risk[s_idx, o_idx, e_idx, i_idx, slot] = float(row["risk"])
        effort[s_idx, o_idx, e_idx, i_idx, slot] = float(row["effort"])
        g[s_idx, o_idx, e_idx, i_idx, slot] = float(row["g"])
        valid_mask[s_idx, o_idx, e_idx, i_idx, slot] = True

    NULL_EPSILON = 1e-8
    n_valid = valid_mask.sum(axis=-1, keepdims=True)
    prior_table = np.where(
        valid_mask, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON
    ).astype(np.float32)

    scored = alts_df[alts_df["risk"].notna()]
    max_alt_count = (
        scored.groupby(
            [
                "scenario_label",
                "observed_action",
                "effort_condition",
                "intimacy_condition",
            ]
        )
        .size()
        .max()
        if len(scored)
        else 0
    )
    if max_alt_count + 1 > MAX_ACTIONS:
        print(
            f"WARNING: largest cell has {max_alt_count} LM-generated alternatives + "
            f"1 observed = {max_alt_count + 1} actions, exceeding "
            f"MAX_ACTIONS={MAX_ACTIONS}. Extra alternatives were truncated."
        )

    return {
        "risk": jnp.array(risk),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior_table),
    }


# =============================================================================
# Padded LM-alternatives loaders for the migrated studies (1b, 2a, 2b)
# =============================================================================
# Each mirrors load_padded_lm_tables_desire but with the cell grid and
# feature axes appropriate to which variables the observer infers (see the
# utility.py / observers.py sections for the shape rationale). All return None
# when any required CSV is missing, so imports stay clean before LM elicitation
# has been run for that study.
#
# Expected CSV schema (produced by score_merged.py --study <slug>, written into
# that study's folder outputs/lm/<slug>/):
#   - canonical: lm_scenario.csv
#       (scenario_label, effort_condition, action, risk, effort, g)
#   - alternatives: lm_alternatives.csv keyed by the study's generation cell +
#       effort_condition + alt_idx, with columns action_text, is_share, risk,
#       effort, g (effort is a feature axis — for effort-inferred studies each
#       alt has a row per effort_condition; risk/g repeat across them)


def _canonical_lookups(canonical_path):
    """Canonical (slot-0) lookups from lm_scenario.csv: (risk, effort) per
    (scenario, effort_condition, action) and goal-satisfaction g per
    (scenario, action). risk/effort/g all live in one row; g is desire-free
    (repeated across the effort_condition rows, so we read it once per
    (scenario, action))."""
    df = pd.read_csv(canonical_path)
    ae = {}
    g = {}
    for _, row in df.iterrows():
        e_idx = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        a_idx = ACTION_LABEL_TO_IDX[row["action"]]
        ae[(row["scenario_label"], e_idx, a_idx)] = (
            float(row["risk"]),
            float(row["effort"]),
        )
        g[(row["scenario_label"], a_idx)] = float(row["g"])
    return ae, g


def _alts_ready(alts_df):
    """An alternatives table is ready to load only once score_merged has filled
    the feature columns (risk/effort/g). Until then it's the stage-1 action list
    (texts only) and the loader returns None."""
    return (
        {"risk", "effort", "g"}.issubset(alts_df.columns)
        and not alts_df["risk"].isna().all()
        and not alts_df["g"].isna().all()
    )


def load_padded_lm_tables_joint_de(
    canonical_path=None,
    alternatives_path=None,
):
    """Study 1b: observer knows intimacy, jointly infers (desire, effort). Cell
    grid is (scenario, observed_action, intimacy_condition). effort inferred ->
    effort table carries an effort_condition feature axis. desire enters as the
    continuous multiplier w_v · desire · g, so g is desire-free (no desire
    axis) and matches the risk shape.
      risk: (16, 3, 4, S)        [scenario, obs, relationship, slot]
      effort: (16, 3, 4, 2, S)     [scenario, obs, relationship, effort_condition, slot]
      g:      (16, 3, 4, S)        [scenario, obs, relationship, slot]
      prior:  (16, 3, 4, S)
    """
    outputs_dir = (
        Path(__file__).resolve().parent / "outputs" / "lm" / "food_inv_joint_de"
    )
    canonical_path = canonical_path or outputs_dir / "lm_scenario.csv"
    alternatives_path = alternatives_path or outputs_dir / "lm_alternatives.csv"
    required = [canonical_path, alternatives_path]
    if any(not Path(p).exists() for p in required):
        return None

    canon_ae, canon_g = _canonical_lookups(canonical_path)
    alts_df = pd.read_csv(alternatives_path)
    if not _alts_ready(alts_df):
        return None

    n_s, n_o, n_rel, n_eff = (
        len(SCENARIO_LABELS),
        N_ACTIONS,
        4,
        N_EFFORT_CONDITIONS,
    )
    risk = np.zeros((n_s, n_o, n_rel, MAX_ACTIONS), dtype=np.float32)
    effort = np.zeros((n_s, n_o, n_rel, n_eff, MAX_ACTIONS), dtype=np.float32)
    g = np.zeros((n_s, n_o, n_rel, MAX_ACTIONS), dtype=np.float32)
    valid = np.zeros((n_s, n_o, n_rel, MAX_ACTIONS), dtype=bool)

    intimacy_to_idx = INTIMACY_CONDITION_TO_IDX
    obs_to_idx = ACTION_LABEL_TO_IDX

    # Canonical slot 0: risk/effort per (scenario, effort_condition, action),
    # broadcast across relationship; g per (scenario, action).
    for scenario in SCENARIO_LABELS:
        s = SCENARIO_TO_IDX[scenario]
        for o in range(n_o):
            for rel in range(n_rel):
                for e in range(n_eff):
                    a_risk, a_effort = canon_ae[(scenario, e, o)]
                    effort[s, o, rel, e, 0] = a_effort
                    if e == 0:
                        risk[s, o, rel, 0] = a_risk
                g[s, o, rel, 0] = canon_g[(scenario, o)]
                valid[s, o, rel, 0] = True

    # Alternatives (slots 1..k): risk/effort/g read straight off the merged alts
    # table. risk/g are effort-marginal/desire-free (repeated across the effort
    # rows). Unscored stage-1 rows (NaN risk) are skipped.
    for _, row in alts_df.iterrows():
        if pd.isna(row["risk"]):
            continue
        s = SCENARIO_TO_IDX[row["scenario_label"]]
        o = obs_to_idx[row["observed_action"]]
        rel = intimacy_to_idx[row["intimacy_condition"]]
        e = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        slot = int(row["alt_idx"]) + 1
        if slot >= MAX_ACTIONS:
            continue
        risk[s, o, rel, slot] = float(row["risk"])
        effort[s, o, rel, e, slot] = float(row["effort"])
        g[s, o, rel, slot] = float(row["g"])
        valid[s, o, rel, slot] = True

    NULL_EPSILON = 1e-8
    n_valid = valid.sum(axis=-1, keepdims=True)
    prior = np.where(valid, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON).astype(
        np.float32
    )
    return {
        "risk": jnp.array(risk),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior),
    }


def load_padded_lm_tables_intimacy(
    canonical_path=None,
    alternatives_path=None,
):
    """Study 2a: observer knows (desire, effort), infers intimacy. Cell grid is
    (scenario, observed_action, desire_condition, effort_condition). intimacy
    inferred (continuous; risk modulated by (1-I)^gamma in the utility, no
    table axis). effort observed -> effort feature taken at the cell's effort.
    Desire is given (observer-visible); it enters as w_v · desire_table[s,r] · g,
    so g is desire-free (no desire axis) and the per-condition desire scalar
    is loaded separately via load_lm_scenario_desire.
      risk: (16, 3, 2, 2, S)     [scenario, obs, desire, effort, slot]
      effort: (16, 3, 2, 2, S)     [scenario, obs, desire, effort, slot]
      g:      (16, 3, 2, 2, S)     [scenario, obs, desire, effort, slot]
      prior:  (16, 3, 2, 2, S)
    """
    outputs_dir = (
        Path(__file__).resolve().parent / "outputs" / "lm" / "food_inv_intimacy"
    )
    canonical_path = canonical_path or outputs_dir / "lm_scenario.csv"
    alternatives_path = alternatives_path or outputs_dir / "lm_alternatives.csv"
    required = [canonical_path, alternatives_path]
    if any(not Path(p).exists() for p in required):
        return None

    canon_ae, canon_g = _canonical_lookups(canonical_path)
    alts_df = pd.read_csv(alternatives_path)
    if not _alts_ready(alts_df):
        return None

    n_s, n_o, n_rew, n_eff = (
        len(SCENARIO_LABELS),
        N_ACTIONS,
        2,
        N_EFFORT_CONDITIONS,
    )
    risk = np.zeros((n_s, n_o, n_rew, n_eff, MAX_ACTIONS), dtype=np.float32)
    effort = np.zeros((n_s, n_o, n_rew, n_eff, MAX_ACTIONS), dtype=np.float32)
    g = np.zeros((n_s, n_o, n_rew, n_eff, MAX_ACTIONS), dtype=np.float32)
    valid = np.zeros((n_s, n_o, n_rew, n_eff, MAX_ACTIONS), dtype=bool)

    rew_to_idx = {"low": int(DesireConditions.LOW), "high": int(DesireConditions.HIGH)}
    obs_to_idx = ACTION_LABEL_TO_IDX

    for scenario in SCENARIO_LABELS:
        s = SCENARIO_TO_IDX[scenario]
        for o in range(n_o):
            for rew in range(n_rew):
                for e in range(n_eff):
                    a_risk, a_effort = canon_ae[(scenario, e, o)]
                    risk[s, o, rew, e, 0] = a_risk
                    effort[s, o, rew, e, 0] = a_effort
                    g[s, o, rew, e, 0] = canon_g[(scenario, o)]
                    valid[s, o, rew, e, 0] = True

    # Alternatives keyed by (scenario, obs, desire, effort, alt_idx); risk/effort/g
    # read straight off the merged alts table. Unscored rows (NaN risk) skipped.
    for _, row in alts_df.iterrows():
        if pd.isna(row["risk"]):
            continue
        s = SCENARIO_TO_IDX[row["scenario_label"]]
        o = obs_to_idx[row["observed_action"]]
        rew = rew_to_idx[row["desire_condition"]]
        e = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        slot = int(row["alt_idx"]) + 1
        if slot >= MAX_ACTIONS:
            continue
        risk[s, o, rew, e, slot] = float(row["risk"])
        effort[s, o, rew, e, slot] = float(row["effort"])
        g[s, o, rew, e, slot] = float(row["g"])
        valid[s, o, rew, e, slot] = True

    NULL_EPSILON = 1e-8
    n_valid = valid.sum(axis=-1, keepdims=True)
    prior = np.where(valid, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON).astype(
        np.float32
    )
    return {
        "risk": jnp.array(risk),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior),
    }


def load_padded_lm_tables_joint_ie(
    canonical_path=None,
    alternatives_path=None,
):
    """Study 2b: observer knows desire, infers (intimacy, effort). Cell grid is
    (scenario, observed_action, desire_condition). intimacy inferred (continuous,
    no table axis); effort inferred -> effort table carries an effort_condition
    feature axis. Desire is given; it enters as w_v · desire_table[s,r] · g, so g
    is desire-free (no desire axis) and the per-condition desire scalar is
    loaded separately via load_lm_scenario_desire.
      risk: (16, 3, 2, S)        [scenario, obs, desire, slot]
      effort: (16, 3, 2, 2, S)     [scenario, obs, desire, effort_condition, slot]
      g:      (16, 3, 2, S)        [scenario, obs, desire, slot]
      prior:  (16, 3, 2, S)
    """
    outputs_dir = (
        Path(__file__).resolve().parent / "outputs" / "lm" / "food_inv_joint_ie"
    )
    canonical_path = canonical_path or outputs_dir / "lm_scenario.csv"
    alternatives_path = alternatives_path or outputs_dir / "lm_alternatives.csv"
    required = [canonical_path, alternatives_path]
    if any(not Path(p).exists() for p in required):
        return None

    canon_ae, canon_g = _canonical_lookups(canonical_path)
    alts_df = pd.read_csv(alternatives_path)
    if not _alts_ready(alts_df):
        return None

    n_s, n_o, n_rew, n_eff = (
        len(SCENARIO_LABELS),
        N_ACTIONS,
        2,
        N_EFFORT_CONDITIONS,
    )
    risk = np.zeros((n_s, n_o, n_rew, MAX_ACTIONS), dtype=np.float32)
    effort = np.zeros((n_s, n_o, n_rew, n_eff, MAX_ACTIONS), dtype=np.float32)
    g = np.zeros((n_s, n_o, n_rew, MAX_ACTIONS), dtype=np.float32)
    valid = np.zeros((n_s, n_o, n_rew, MAX_ACTIONS), dtype=bool)

    rew_to_idx = {"low": int(DesireConditions.LOW), "high": int(DesireConditions.HIGH)}
    obs_to_idx = ACTION_LABEL_TO_IDX

    for scenario in SCENARIO_LABELS:
        s = SCENARIO_TO_IDX[scenario]
        for o in range(n_o):
            for rew in range(n_rew):
                for e in range(n_eff):
                    a_risk, a_effort = canon_ae[(scenario, e, o)]
                    effort[s, o, rew, e, 0] = a_effort
                    if e == 0:
                        risk[s, o, rew, 0] = a_risk
                g[s, o, rew, 0] = canon_g[(scenario, o)]
                valid[s, o, rew, 0] = True

    # Alternatives keyed by (scenario, obs, desire, effort_condition, alt_idx);
    # risk effort-marginal / g desire-free (repeated across the effort rows),
    # read straight off the merged alts table. Unscored rows (NaN risk) skipped.
    for _, row in alts_df.iterrows():
        if pd.isna(row["risk"]):
            continue
        s = SCENARIO_TO_IDX[row["scenario_label"]]
        o = obs_to_idx[row["observed_action"]]
        rew = rew_to_idx[row["desire_condition"]]
        e = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
        slot = int(row["alt_idx"]) + 1
        if slot >= MAX_ACTIONS:
            continue
        risk[s, o, rew, slot] = float(row["risk"])
        effort[s, o, rew, e, slot] = float(row["effort"])
        g[s, o, rew, slot] = float(row["g"])
        valid[s, o, rew, slot] = True

    NULL_EPSILON = 1e-8
    n_valid = valid.sum(axis=-1, keepdims=True)
    prior = np.where(valid, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON).astype(
        np.float32
    )
    return {
        "risk": jnp.array(risk),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior),
    }
