"""
Constants, enums, scenario maps, and LM-elicited table loaders shared across all
model code. Anything that names dimensions (action enums, scenario indices,
intimacy/desire levels) or loads scenario-specific feature tables (risk,
effort, goal-satisfaction g, per-condition desire, padded action spaces) lives
here.

Dependency layer 0: imports nothing from sibling modules. `utility.py`,
`actors.py`, and `observers.py` all import from here.
"""

import json
from collections import defaultdict
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

# Three-action set used by the inverse-planning experiments. Action 0
# = no sharing, action 1 = low-risk sharing, action 2 = high-risk sharing.
# Stimulus set is `experiments/scenarios.csv`; the per-study LM tables are
# `outputs/lm/<slug>/lm_runs.jsonl` (scored observed + alternative risk/effort/g
# across K runs) and `lm_alternatives.jsonl` (stage-1 texts). The legacy
# `lm_scenario.csv` + `lm_alternatives.csv` remain only as a K=1 fallback.
actions = jnp.array([0, 1, 2])
N_ACTIONS = 3
# The 3 observed actions in index order. Experiment data and LM CSVs label the
# observed action with these names (was action_0/1/2 before the May 2026 rename).
ACTION_COLS = ["no_share", "low_risk_share", "high_risk_share"]
ACTION_LABEL_TO_IDX = {label: i for i, label in enumerate(ACTION_COLS)}


class DesireConditions(IntEnum):
    LOW = 0
    HIGH = 1


class RelationshipConditions(IntEnum):
    MAX_FORMAL = 0
    SOMEWHAT_FORMAL = 1
    SOMEWHAT_INTIMATE = 2
    MAX_INTIMATE = 3


class EffortConditions(IntEnum):
    LOW = 0
    HIGH = 1


# Intimacy is a purely verbal manipulation in the experiments: the condition is
# identified by a slug (no numeric code is stored in the data). These slugs, in
# ascending order (formal -> intimate), index the RelationshipConditions axis.
INTIMACY_CONDITIONS = [
    "max_formal",
    "somewhat_formal",
    "somewhat_intimate",
    "max_intimate",
]
INTIMACY_CONDITION_TO_IDX = {slug: i for i, slug in enumerate(INTIMACY_CONDITIONS)}

# Continuous intimacy values for each RelationshipConditions level — used by
# the relationship-keyed padded memos to evaluate the (1 - I) risk term
# without dragging the 101-level IntimacyLevels axis into the memo. These are
# placeholder magnitudes for each verbal level (pending LM elicitation); they
# are model-internal and never saved as condition labels. Evenly spaced to match
# the symmetric four-level scale (max_formal / somewhat_formal / somewhat_intimate
# / max_intimate); the elicited values override these.
RELATIONSHIP_LEVEL_VALUES = jnp.array([0.0, 0.33, 0.67, 1.0])

EFFORT_CONDITION_TO_IDX = {"low": 0, "high": 1}
N_EFFORT_CONDITIONS = 2


# Padded-action constants for the 3-action inverse-planning experiments (the
# active roster). The stimulus offers only 3 actions per scenario, so
# MAX_ACTIONS = 12 means up to 11 LM-elicited alternatives plus the
# observed action in slot 0.
MAX_ACTIONS = 12


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
    """The 3 observed actions for the 3-act design (matches actions)."""

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


# The 16 nonfood scenarios (Study 3; experiments/scenarios_nonfood.csv), in
# alphabetical order like SCENARIO_LABELS. The July 2026 redesign replaced
# hat -> earbuds and payment -> salary.
NONFOOD_SCENARIO_LABELS = [
    "bed",
    "blanket",
    "breakup",
    "chapstick",
    "earbuds",
    "gossip",
    "hairbrush",
    "harmonica",
    "home",
    "locker-room",
    "navigation",
    "salary",
    "sauna",
    "sleeping-bag",
    "sunscreen",
    "towel",
]
NONFOOD_SCENARIO_TO_IDX = {
    label: idx for idx, label in enumerate(NONFOOD_SCENARIO_LABELS)
}

# Scenario-label registry keyed by study slug. The nonfood studies (3a/3b) share
# the memo enums, actors, and observers with the food studies — both stimulus
# sets have 16 scenarios and the memo `Scenarios` axis is positional — so the
# only per-domain difference the model code sees is which labels map to which
# indices. Table loaders and data loaders look labels up here by study slug.
STUDY_SCENARIO_LABELS = {
    "food_inv_desire": SCENARIO_LABELS,
    "food_inv_joint_de": SCENARIO_LABELS,
    "food_inv_intimacy": SCENARIO_LABELS,
    "food_inv_joint_ie": SCENARIO_LABELS,
    "nonfood_inv_joint_de": NONFOOD_SCENARIO_LABELS,
    "nonfood_inv_joint_ie": NONFOOD_SCENARIO_LABELS,
}


def scenario_to_idx_for_study(slug):
    """Label -> memo scenario index for one study's stimulus set. Food studies
    index by SCENARIO_LABELS, nonfood studies by NONFOOD_SCENARIO_LABELS; both
    are 16 long, so the memo axes are shared across domains."""
    return {label: idx for idx, label in enumerate(STUDY_SCENARIO_LABELS[slug])}


# ==============================================================================
# Three-action LM-derived scenario parameter tables
# ==============================================================================
# Per-study observed-action risk/effort/g and the padded LM-alternatives action spaces
# are loaded by the `load_padded_lm_tables_*` functions below (preferring the
# K-run `outputs/lm/<slug>/lm_runs.jsonl`, falling back to the legacy single-run
# `lm_scenario.csv` + `lm_alternatives.csv`). The *given-magnitude* scalars are
# now folded per-run into `lm_runs.jsonl` (a `desire` field on each record for the
# given-desire studies, an `intimacy` field for the given-relationship studies),
# loaded by load_lm_scenario_desire / load_lm_relationship_values below — both
# carrying the same leading run axis as the padded tables.


def _runs_jsonl_path(slug):
    return Path(__file__).resolve().parent / "outputs" / "lm" / slug / "lm_runs.jsonl"


def _read_runs_jsonl(path):
    """Group `lm_runs.jsonl` records by run_id. Returns a dict run_id -> [records];
    sorting its keys gives the canonical run order (k = 0..K-1) shared with the
    padded-table loaders, so per-run given magnitudes align with their run slice."""
    records_by_run = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records_by_run[int(rec["run_id"])].append(rec)
    return records_by_run


def load_lm_scenario_desire(slug, runs_filename="lm_runs.jsonl"):
    """Load the per-run, per-condition desire scalar for the given-desire studies
    (2a `food_inv_intimacy`, 2b `food_inv_joint_ie`, 3b `nonfood_inv_joint_ie`).

    When desire is observer-visible context, the LM reads the scenario + the
    shown desire paragraph and rates how much the two people would like the food
    on the [0, 1] scale. That scalar plugs into the actor utility as the constant
    `desire` in w_v · desire · g. It is scored per elicitation run (folded into
    each `lm_runs.jsonl` record's `desire` field), so it carries a leading run
    axis aligned with the padded feature tables.

    Source: the per-record `desire` field of `outputs/lm/<slug>/lm_runs.jsonl`;
    falls back to the legacy single-run `lm_scenario_desire.csv` (as K=1) so fits
    run before the JSON regeneration.

    Returns a jnp.array of shape (K, 16, 2) indexed by
    (run, scenario, desire_condition), or None if neither source is present.
    The scenario axis follows the study's own label order
    (STUDY_SCENARIO_LABELS[slug]).
    """
    scenario_to_idx = scenario_to_idx_for_study(slug)
    desire_to_idx = {
        "low": int(DesireConditions.LOW),
        "high": int(DesireConditions.HIGH),
    }
    n_scenarios = len(scenario_to_idx)

    runs_path = _runs_jsonl_path(slug).with_name(runs_filename)
    if runs_path.exists():
        records_by_run = _read_runs_jsonl(runs_path)
        run_ids = sorted(records_by_run)
        # NaN-init so a missing or failed (null) rating is caught below instead
        # of silently becoming desire = 0 (which would zero the reward term for
        # that run × scenario × condition).
        d = np.full((len(run_ids), n_scenarios, 2), np.nan, dtype=np.float32)
        for k, rid in enumerate(run_ids):
            for rec in records_by_run[rid]:
                if rec.get("desire") is None:
                    continue
                s = scenario_to_idx[rec["scenario_label"]]
                r = desire_to_idx[rec["desire_condition"]]
                _assert_no_conflicting_write(
                    slug,
                    "desire",
                    d[k, s, r],
                    float(rec["desire"]),
                    (rec["scenario_label"], rec["desire_condition"], f"run {rid}"),
                )
                d[k, s, r] = float(rec["desire"])
        _assert_no_missing_scalars(slug, "desire", d)
        return jnp.array(d)

    # Legacy CSV fallback (K=1, pre-regeneration).
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
    d = np.full((n_scenarios, 2), np.nan, dtype=np.float32)
    for _, row in df.iterrows():
        s = scenario_to_idx[row["scenario_label"]]
        r = desire_to_idx[row["desire_condition"]]
        d[s, r] = row["desire"]
    _assert_no_missing_scalars(slug, "desire", d)
    return jnp.array(d)[None, :, :]


def load_lm_relationship_values(slug, runs_filename="lm_runs.jsonl"):
    """Per-run continuous intimacy magnitude I ∈ [0, 1] for each of the four
    RelationshipConditions levels, for the given-relationship studies
    (1a `food_inv_desire`, 1b `food_inv_joint_de`).

    The LM rates the intimacy implied by each (de-anchored, verbal) relationship
    description, mirroring the desire scalar in 2a/2b. It is scored per elicitation
    run (folded into each `lm_runs.jsonl` record's `intimacy` field), so it carries
    a leading run axis aligned with the padded feature tables.

    Source: the per-record `intimacy` field of `outputs/lm/<slug>/lm_runs.jsonl`,
    placed by INTIMACY_CONDITION_TO_IDX. Falls back to the placeholder
    `RELATIONSHIP_LEVEL_VALUES` (as K=1) so 1a/1b run before the elicitation exists.

    Returns a jnp.array of shape (K, 4).
    """
    runs_path = _runs_jsonl_path(slug).with_name(runs_filename)
    if runs_path.exists():
        records_by_run = _read_runs_jsonl(runs_path)
        run_ids = sorted(records_by_run)
        # NaN-init so a missing or failed (null) rating is caught below instead
        # of silently becoming intimacy = 0 (max formal).
        out = np.full((len(run_ids), 4), np.nan, dtype=np.float32)
        for k, rid in enumerate(run_ids):
            for rec in records_by_run[rid]:
                if rec.get("intimacy") is None:
                    continue
                i = INTIMACY_CONDITION_TO_IDX[rec["intimacy_condition"]]
                _assert_no_conflicting_write(
                    slug,
                    "intimacy",
                    out[k, i],
                    float(rec["intimacy"]),
                    (rec["intimacy_condition"], f"run {rid}"),
                )
                out[k, i] = float(rec["intimacy"])
        _assert_no_missing_scalars(slug, "intimacy", out)
        return jnp.array(out)
    return RELATIONSHIP_LEVEL_VALUES[None, :]


# Per-study prior-cell condition columns (the conditions the participant sees
# BEFORE the action — the prior-stage screens) and elicited quantities. The
# nonfood studies mirror their food counterparts (same observers, own folder).
_PRIOR_STUDY_SPEC = {
    "food_inv_desire": {
        "conds": ("effort_condition", "intimacy_condition"),
        "quantities": ("prior_desire",),
    },
    "food_inv_joint_de": {
        "conds": ("intimacy_condition",),
        "quantities": ("prior_desire", "prior_effort_high"),
    },
    "food_inv_intimacy": {
        "conds": ("desire_condition", "effort_condition"),
        "quantities": ("prior_intimacy",),
    },
    "food_inv_joint_ie": {
        "conds": ("desire_condition",),
        "quantities": ("prior_intimacy", "prior_effort_high"),
    },
}
_PRIOR_STUDY_SPEC["nonfood_inv_joint_de"] = _PRIOR_STUDY_SPEC["food_inv_joint_de"]
_PRIOR_STUDY_SPEC["nonfood_inv_joint_ie"] = _PRIOR_STUDY_SPEC["food_inv_joint_ie"]

_PRIOR_COND_LEVELS = {
    "effort_condition": ["low", "high"],
    "desire_condition": ["low", "high"],
    "intimacy_condition": INTIMACY_CONDITIONS,
}
# Loader output key per elicited quantity.
_PRIOR_OUT_KEY = {
    "prior_desire": "desire_m",
    "prior_intimacy": "intimacy_m",
    "prior_effort_high": "effort_p",
}


def load_lm_priors(slug, base=False, filename=None):
    """Per-run, per-cell prior scalars for the informative-prior configs.

    Reads outputs/lm/<slug>/lm_priors{_base}.jsonl (one record per run x
    scenario x prior-visible conditions; `filename` overrides, absolute paths
    allowed for tests and the human-ceiling file). Returns None when the file
    is missing, else {out_key: (K, 16, *cond_levels)} arrays. `base=True`
    (given-relationship studies) reads the relationship-free file and
    broadcasts it across the 4-level relationship axis, mirroring the base
    alternative tables. Fail-fast: a missing cell, a duplicate with a
    conflicting value, or a scalar outside [0, 1] raises ValueError.
    """
    spec = _PRIOR_STUDY_SPEC[slug]
    conds = tuple(c for c in spec["conds"] if not (base and c == "intimacy_condition"))
    if filename is None:
        filename = f"lm_priors{'_base' if base else ''}.jsonl"
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / "outputs" / "lm" / slug / filename
    if not path.exists():
        return None

    scenario_to_idx = scenario_to_idx_for_study(slug)
    levels = [_PRIOR_COND_LEVELS[c] for c in conds]
    shape_tail = tuple(len(lv) for lv in levels)
    records_by_run = _read_runs_jsonl(path)
    run_ids = sorted(records_by_run)
    out = {
        q: np.full(
            (len(run_ids), len(scenario_to_idx), *shape_tail), np.nan, np.float32
        )
        for q in spec["quantities"]
    }
    for k, rid in enumerate(run_ids):
        for rec in records_by_run[rid]:
            s = scenario_to_idx[rec["scenario_label"]]
            idx = (k, s) + tuple(_PRIOR_COND_LEVELS[c].index(rec[c]) for c in conds)
            for q in spec["quantities"]:
                v = rec.get(q)
                if v is None:
                    continue
                if not (0.0 <= float(v) <= 1.0):
                    raise ValueError(
                        f"{slug} {path.name}: {q}={v} outside [0, 1] at "
                        f"{rec['scenario_label']} run {rid}"
                    )
                _assert_no_conflicting_write(
                    slug,
                    q,
                    out[q][idx],
                    float(v),
                    (rec["scenario_label"], conds, f"run {rid}"),
                )
                out[q][idx] = float(v)
    for q, arr in out.items():
        _assert_no_missing_scalars(slug, q, arr)
    if base and "intimacy_condition" in spec["conds"]:
        # Broadcast the relationship-free base priors across the 4-level
        # relationship axis (appended in the same position the standard file
        # has it: last).
        out = {q: np.repeat(arr[..., None], 4, axis=-1) for q, arr in out.items()}
    return {_PRIOR_OUT_KEY[q]: jnp.array(arr) for q, arr in out.items()}


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
    *,
    runs_filename="lm_runs.jsonl",
    broadcast_relationship=False,
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

    Slot 0 of every cell holds the observed action's features:
    risk/effort/g from `lm_runs.jsonl` (risk/effort depend on scenario +
    effort_condition + action, broadcast across intimacy; g depends on scenario +
    action, broadcast across effort and intimacy).

    Slots 1..k hold the LM-generated alternatives for that cell, with each
    alternative's risk/effort/g and text read from
    `outputs/lm/food_inv_desire/lm_runs.jsonl` (keyed by alt_idx; the stage-1
    texts also live in `lm_alternatives.jsonl`, which is not read here).
    (The legacy `lm_scenario.csv` / `lm_alternatives.csv` are read only as a
    K=1 fallback.)
    Remaining slots are null-padded (risk/effort/g = 0; prior = NULL_EPSILON to
    keep the softmax differentiable).

    The returned arrays carry a leading run axis K (one elicitation run per
    component of the simulated-observer mixture); K=1 on the legacy single-run
    CSVs. Returns a dict {risk, effort, g, prior, n_runs} of jnp.arrays, or None
    if the tables are missing or not yet scored.

    `runs_filename` / `broadcast_relationship` support the base-model alternative
    set: the `base` ablation has no intimacy term, so its alternatives are elicited
    without the relationship paragraph (`lm_runs_base.jsonl`, keyed by effort only).
    With `broadcast_relationship=True` the same alt set is written to every one of
    the 4 relationship indices, so the table is identical across that axis and the
    base model's predictions are relationship-invariant.
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "lm" / "food_inv_desire"
    # The base alt set drops intimacy_condition (relationship-free), so it's keyed
    # on effort only and broadcast across the relationship axis below.
    cell_cols = (
        ["effort_condition"]
        if broadcast_relationship
        else ["effort_condition", "intimacy_condition"]
    )
    runs = _run_sources(outputs_dir, cell_cols, runs_filename=runs_filename)
    if runs is None:
        return None
    K = len(runs)

    n_scenarios = len(SCENARIO_LABELS)
    n_observed = N_ACTIONS
    n_effort = N_EFFORT_CONDITIONS
    n_intimacy = 4
    # Features are NaN-initialized so an unwritten or failed (null) rating at a
    # valid slot is caught by _validate_padded_tables below; the null-padded
    # slots are zeroed after validation so their utilities are exactly 0.
    shape = (K, n_scenarios, n_observed, n_effort, n_intimacy, MAX_ACTIONS)
    risk = np.full(shape, np.nan, dtype=np.float32)
    effort = np.full(shape, np.nan, dtype=np.float32)
    g = np.full(shape, np.nan, dtype=np.float32)
    valid_mask = np.zeros(shape, dtype=bool)

    intimacy_to_idx = INTIMACY_CONDITION_TO_IDX
    observed_str_to_idx = ACTION_LABEL_TO_IDX

    for k, (obs_ae, obs_g, alts_df) in enumerate(runs):
        # Observed (slot 0): risk/effort depend on scenario + effort + action,
        # broadcast across intimacy; g depends on scenario + action.
        for scenario in SCENARIO_LABELS:
            s_idx = SCENARIO_TO_IDX[scenario]
            for observed in range(n_observed):
                for e_idx in range(n_effort):
                    a_risk, a_effort = obs_ae[(scenario, e_idx, observed)]
                    for i_idx in range(n_intimacy):
                        risk[k, s_idx, observed, e_idx, i_idx, 0] = a_risk
                        effort[k, s_idx, observed, e_idx, i_idx, 0] = a_effort
                        g[k, s_idx, observed, e_idx, i_idx, 0] = obs_g[
                            (scenario, observed)
                        ]
                        valid_mask[k, s_idx, observed, e_idx, i_idx, 0] = True

        # LM-generated alternatives (slots 1..k), keyed by (scenario, observed,
        # effort_condition, intimacy_condition, alt_idx). Unscored stage-1 rows
        # with any NaN feature are skipped.
        for _, row in alts_df.iterrows():
            if row[["risk", "effort", "g"]].isna().any():
                continue
            s_idx = SCENARIO_TO_IDX[row["scenario_label"]]
            o_idx = observed_str_to_idx[row["observed_action"]]
            e_idx = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
            slot = int(row["alt_idx"]) + 1
            if slot >= MAX_ACTIONS:
                continue
            # Base alts are relationship-free: broadcast across all intimacy indices
            # (like the observed slot 0 above); otherwise place at the row's level.
            rel_indices = (
                range(n_intimacy)
                if broadcast_relationship
                else [intimacy_to_idx[row["intimacy_condition"]]]
            )
            for i_idx in rel_indices:
                risk[k, s_idx, o_idx, e_idx, i_idx, slot] = float(row["risk"])
                effort[k, s_idx, o_idx, e_idx, i_idx, slot] = float(row["effort"])
                g[k, s_idx, o_idx, e_idx, i_idx, slot] = float(row["g"])
                valid_mask[k, s_idx, o_idx, e_idx, i_idx, slot] = True

    _validate_padded_tables(
        "food_inv_desire",
        {
            "risk": (risk, valid_mask),
            "effort": (effort, valid_mask),
            "g": (g, valid_mask),
        },
    )
    for arr in (risk, effort, g):
        arr[~valid_mask] = 0.0

    NULL_EPSILON = 1e-8
    n_valid = valid_mask.sum(axis=-1, keepdims=True)
    prior_table = np.where(
        valid_mask, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON
    ).astype(np.float32)

    _warn_truncation(runs)

    return {
        "risk": jnp.array(risk),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior_table),
        "n_runs": K,
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
# Expected schema for the legacy fallback CSVs (the K=1 back-compat path; the
# current primary output of score_merged.py --study <slug>, written into that
# study's folder outputs/lm/<slug>/, is lm_runs.jsonl):
#   - observed: lm_scenario.csv
#       (scenario_label, effort_condition, action, risk, effort, g)
#   - alternatives: lm_alternatives.csv keyed by the study's generation cell +
#       effort_condition + alt_idx, with columns action_text, risk,
#       effort, g (effort is a feature axis — for effort-inferred studies each
#       alt has a row per effort_condition; risk/g repeat across them)


def _observed_lookups(observed_path):
    """Observed-action (slot-0) lookups from lm_scenario.csv: (risk, effort) per
    (scenario, effort_condition, action) and goal-satisfaction g per
    (scenario, action). risk/effort/g all live in one row; g is desire-free
    (repeated across the effort_condition rows, so we read it once per
    (scenario, action))."""
    df = pd.read_csv(observed_path)
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


def _run_sources(outputs_dir, cell_cols, runs_filename="lm_runs.jsonl"):
    """Return a list (over elicitation runs) of `(obs_ae, obs_g, alts_df)`,
    or None if no scored LM tables exist yet.

    This is the single seam between the K-run JSON pipeline and the per-study
    padded-table loaders below: each loader just iterates the returned list and
    fills its arrays per run, reusing its existing observed/alternatives fill
    logic unchanged (each run's `(obs_ae, obs_g, alts_df)` looks exactly like
    the legacy single-run inputs).

    Source precedence:
      - `lm_runs.jsonl` (K runs) if present — one record per (run_id, cell), each
        carrying its run's scored actions (slot 0 = observed action,
        slots 1+ = alternatives).
      - else the legacy `lm_scenario.csv` + `lm_alternatives.csv` as a single run
        (K=1), the back-compat path so fits run before the JSON regeneration.

    `cell_cols` are the condition columns this study's alternatives fill reads
    (e.g. ["effort_condition", "intimacy_condition"] for 1a); the JSONL records
    must carry them so the reconstructed `alts_df` matches the legacy schema.
    """
    jsonl = outputs_dir / runs_filename
    if jsonl.exists():
        return _run_sources_jsonl(jsonl, cell_cols)

    observed_path = outputs_dir / "lm_scenario.csv"
    alternatives_path = outputs_dir / "lm_alternatives.csv"
    if not observed_path.exists() or not alternatives_path.exists():
        return None
    alts_df = pd.read_csv(alternatives_path)
    if not _alts_ready(alts_df):
        return None
    obs_ae, obs_g = _observed_lookups(observed_path)
    return [(obs_ae, obs_g, alts_df)]


def _nan_if_none(v):
    """A failed LM rating is serialized as null (see score_merged._f); load it as
    NaN rather than crashing on float(None)."""
    return float("nan") if v is None else float(v)


def _assert_no_conflicting_write(slug, name, previous, value, key):
    """Fail fast when the same lm_runs.jsonl key is written twice with a
    different value — silent last-write-wins would hide a double-appended or
    corrupted elicitation log. `previous` is the already-stored value (None or
    a NaN scalar = unwritten slot); equality is NaN-aware (a failed rating
    loads as NaN), so identical repeats pass and only conflicts raise."""
    if previous is None:
        return
    prev = np.asarray(previous, dtype=np.float32)
    new = np.asarray(value, dtype=np.float32)
    if prev.shape == () and np.isnan(prev):
        return
    if bool(np.all((prev == new) | (np.isnan(prev) & np.isnan(new)))):
        return
    raise ValueError(
        f"lm_runs.jsonl for {slug} has duplicate {name} records at {key} with "
        f"conflicting values ({previous} vs {value}) — refusing to let the "
        f"last write win. Regenerate the log with "
        f"`model/lm/score_merged.py --study {slug}`."
    )


def _assert_no_missing_scalars(slug, name, arr):
    """Fail fast if a given-magnitude scalar (desire / intimacy) is missing or
    was a failed (null) rating. The tables are NaN-initialized, so any NaN left
    after the fill marks a cell with no usable rating; letting it through would
    silently distort the utility (a zero desire kills the reward term; a zero
    intimacy means maximally formal)."""
    missing = np.argwhere(np.isnan(arr))
    if len(missing):
        raise ValueError(
            f"{len(missing)} missing {name} rating(s) in lm_runs.jsonl for "
            f"{slug} (first at index {tuple(missing[0])}). Re-run "
            f"`model/lm/score_merged.py --study {slug}` to fill them."
        )


def _validate_padded_tables(study, named_features):
    """Fail fast if any feature value at a valid slot is non-finite.

    A failed (null) LM rating is loaded as NaN. Alternatives with any NaN
    feature are skipped at fill time, but a NaN observed-action (slot-0) value
    would flow into the actor softmax and poison every trial's NLL gradient in
    the fit — NaN propagates through `jnp.sum`, it does not drop out — so the
    loaders refuse to return such a table. `named_features` maps a feature name
    to `(array, valid_mask)`, where the mask broadcasts against the array (the
    effort table gains an effort_condition axis in the effort-inferred studies).
    """
    problems = []
    for name, (arr, mask) in named_features.items():
        bad = int((~np.isfinite(arr) & mask).sum())
        if bad:
            problems.append(f"{name}: {bad} non-finite value(s) at valid slots")
    if problems:
        raise ValueError(
            f"LM tables for {study} contain non-finite feature values — "
            + "; ".join(problems)
            + ". Re-run `model/lm/score_merged.py` for the affected cells."
        )


def _run_sources_jsonl(path, cell_cols):
    """Parse `lm_runs.jsonl` into per-run `(obs_ae, obs_g, alts_df)` triples.

    Each line is one (run_id, cell) record (see the schema in score_merged.py):
      {"run_id": int, "scenario_label": str, "observed_action": str,
       <each cell_col>: str, "actions": [
           {"slot": int, "alt_idx": int|null, "is_observed": bool,
            "action_text": str,
            "risk": float, "effort": float, "g": float}, ...]}
    `obs_ae`/`obs_g` are reconstructed from the slot-0 observed actions
    (keyed (scenario, effort_idx, action) / (scenario, action), matching
    `_observed_lookups`); `alts_df` from the alternative actions.
    """
    records_by_run = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records_by_run[int(rec["run_id"])].append(rec)

    runs = []
    for run_id in sorted(records_by_run):
        obs_ae, obs_g, alt_rows = {}, {}, []
        for rec in records_by_run[run_id]:
            scenario = rec["scenario_label"]
            o_idx = ACTION_LABEL_TO_IDX[rec["observed_action"]]
            e_idx = EFFORT_CONDITION_TO_IDX[rec["effort_condition"]]
            for act in rec["actions"]:
                # Failed ratings are null -> NaN (not a crash). Alternatives with
                # any NaN feature are skipped by the padded-table loaders; a NaN
                # observed-action value leaves that (run, scenario) slot-0 feature
                # NaN, which _validate_padded_tables rejects at load time (a NaN
                # would otherwise poison the whole fit's gradients, not drop out).
                rk, ef, gg = (
                    _nan_if_none(act["risk"]),
                    _nan_if_none(act["effort"]),
                    _nan_if_none(act["g"]),
                )
                if act.get("is_observed"):
                    slug = path.parent.name
                    ae_key = (scenario, e_idx, o_idx)
                    g_key = (scenario, o_idx)
                    _assert_no_conflicting_write(
                        slug,
                        "observed (risk, effort)",
                        obs_ae.get(ae_key),
                        (rk, ef),
                        ae_key + (f"run {run_id}",),
                    )
                    _assert_no_conflicting_write(
                        slug,
                        "observed g",
                        obs_g.get(g_key),
                        gg,
                        g_key + (f"run {run_id}",),
                    )
                    obs_ae[ae_key] = (rk, ef)
                    obs_g[g_key] = gg
                else:
                    row = {
                        "scenario_label": scenario,
                        "observed_action": rec["observed_action"],
                        "alt_idx": int(act["alt_idx"]),
                        "risk": rk,
                        "effort": ef,
                        "g": gg,
                    }
                    for c in cell_cols:
                        row[c] = rec[c]
                    alt_rows.append(row)
        runs.append((obs_ae, obs_g, pd.DataFrame(alt_rows)))
    return runs


def _warn_truncation(runs):
    """Warn if any scored LM alternative was dropped by the loaders' actual
    truncation condition — slot = alt_idx + 1 >= MAX_ACTIONS. Counting alts per
    cell would miss drops when alt_idx is non-contiguous (some alts unscored),
    so the warning checks each scored row's slot directly."""
    n_dropped = 0
    max_slot = 0
    for _, _, alts_df in runs:
        if alts_df is None or len(alts_df) == 0 or "risk" not in alts_df.columns:
            continue
        scored = alts_df.dropna(subset=["risk", "effort", "g"])
        if not len(scored):
            continue
        slots = scored["alt_idx"].astype(int) + 1
        n_dropped += int((slots >= MAX_ACTIONS).sum())
        max_slot = max(max_slot, int(slots.max()))
    if n_dropped:
        print(
            f"WARNING: {n_dropped} scored LM-generated alternative(s) sit at "
            f"slot >= MAX_ACTIONS={MAX_ACTIONS} (largest slot {max_slot}) and "
            f"were truncated by the loaders."
        )


def load_padded_lm_tables_joint_de(
    *,
    slug="food_inv_joint_de",
    runs_filename="lm_runs.jsonl",
    broadcast_relationship=False,
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

    `slug` selects the study whose tables to load: Study 3a
    (`nonfood_inv_joint_de`) shares this design on the nonfood stimulus set, so
    it reuses this loader with its own outputs folder and scenario-label order
    (STUDY_SCENARIO_LABELS[slug]).

    `runs_filename` / `broadcast_relationship` support the base-model alternative
    set, exactly as in `load_padded_lm_tables_desire`: the `base` ablation has no
    intimacy term, so its alternatives are elicited without the relationship
    paragraph (`lm_runs_base.jsonl`; for 1b/3a the generation cell is scenario ×
    observed action only, with the effort feature axis coming from the scored
    records). With `broadcast_relationship=True` the same alt set is written to
    every one of the 4 relationship indices, so the base table — and the base
    model's predictions — are relationship-invariant.
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "lm" / slug
    scenario_labels = STUDY_SCENARIO_LABELS[slug]
    scenario_to_idx = scenario_to_idx_for_study(slug)
    # The base alt set drops intimacy_condition (relationship-free); the scored
    # records still carry effort_condition (the effort feature axis).
    cell_cols = (
        ["effort_condition"]
        if broadcast_relationship
        else ["intimacy_condition", "effort_condition"]
    )
    runs = _run_sources(outputs_dir, cell_cols, runs_filename=runs_filename)
    if runs is None:
        return None
    K = len(runs)

    n_s, n_o, n_rel, n_eff = (
        len(scenario_labels),
        N_ACTIONS,
        4,
        N_EFFORT_CONDITIONS,
    )
    # NaN-init + post-fill validation, as in load_padded_lm_tables_desire.
    risk = np.full((K, n_s, n_o, n_rel, MAX_ACTIONS), np.nan, dtype=np.float32)
    effort = np.full((K, n_s, n_o, n_rel, n_eff, MAX_ACTIONS), np.nan, dtype=np.float32)
    g = np.full((K, n_s, n_o, n_rel, MAX_ACTIONS), np.nan, dtype=np.float32)
    valid = np.zeros((K, n_s, n_o, n_rel, MAX_ACTIONS), dtype=bool)

    intimacy_to_idx = INTIMACY_CONDITION_TO_IDX
    obs_to_idx = ACTION_LABEL_TO_IDX

    for k, (obs_ae, obs_g, alts_df) in enumerate(runs):
        # Observed slot 0: risk/effort per (scenario, effort_condition, action),
        # broadcast across relationship; g per (scenario, action).
        for scenario in scenario_labels:
            s = scenario_to_idx[scenario]
            for o in range(n_o):
                for rel in range(n_rel):
                    for e in range(n_eff):
                        a_risk, a_effort = obs_ae[(scenario, e, o)]
                        effort[k, s, o, rel, e, 0] = a_effort
                        if e == 0:
                            risk[k, s, o, rel, 0] = a_risk
                    g[k, s, o, rel, 0] = obs_g[(scenario, o)]
                    valid[k, s, o, rel, 0] = True

        # Alternatives (slots 1..k): risk/g effort-marginal/desire-free (repeated
        # across the effort rows). Unscored stage-1 rows with any NaN feature are skipped.
        for _, row in alts_df.iterrows():
            if row[["risk", "effort", "g"]].isna().any():
                continue
            s = scenario_to_idx[row["scenario_label"]]
            o = obs_to_idx[row["observed_action"]]
            e = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
            slot = int(row["alt_idx"]) + 1
            if slot >= MAX_ACTIONS:
                continue
            # Base alts are relationship-free: broadcast across all intimacy
            # indices (like the observed slot 0); otherwise place at the row's level.
            rel_indices = (
                range(n_rel)
                if broadcast_relationship
                else [intimacy_to_idx[row["intimacy_condition"]]]
            )
            for rel in rel_indices:
                risk[k, s, o, rel, slot] = float(row["risk"])
                effort[k, s, o, rel, e, slot] = float(row["effort"])
                g[k, s, o, rel, slot] = float(row["g"])
                valid[k, s, o, rel, slot] = True

    valid_e = np.broadcast_to(valid[:, :, :, :, None, :], effort.shape)
    _validate_padded_tables(
        slug,
        {"risk": (risk, valid), "effort": (effort, valid_e), "g": (g, valid)},
    )
    risk[~valid] = 0.0
    g[~valid] = 0.0
    effort[~valid_e] = 0.0

    NULL_EPSILON = 1e-8
    n_valid = valid.sum(axis=-1, keepdims=True)
    prior = np.where(valid, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON).astype(
        np.float32
    )
    _warn_truncation(runs)
    return {
        "risk": jnp.array(risk),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior),
        "n_runs": K,
    }


def load_padded_lm_tables_intimacy(runs_filename="lm_runs.jsonl"):
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

    `runs_filename` selects the source file (default "lm_runs.jsonl"; the
    `_base` variant routes here for the relationship-free set).
    """
    outputs_dir = (
        Path(__file__).resolve().parent / "outputs" / "lm" / "food_inv_intimacy"
    )
    cell_cols = ["desire_condition", "effort_condition"]
    runs = _run_sources(outputs_dir, cell_cols, runs_filename=runs_filename)
    if runs is None:
        return None
    K = len(runs)

    n_s, n_o, n_rew, n_eff = (
        len(SCENARIO_LABELS),
        N_ACTIONS,
        2,
        N_EFFORT_CONDITIONS,
    )
    # NaN-init + post-fill validation, as in load_padded_lm_tables_desire.
    risk = np.full((K, n_s, n_o, n_rew, n_eff, MAX_ACTIONS), np.nan, dtype=np.float32)
    effort = np.full((K, n_s, n_o, n_rew, n_eff, MAX_ACTIONS), np.nan, dtype=np.float32)
    g = np.full((K, n_s, n_o, n_rew, n_eff, MAX_ACTIONS), np.nan, dtype=np.float32)
    valid = np.zeros((K, n_s, n_o, n_rew, n_eff, MAX_ACTIONS), dtype=bool)

    rew_to_idx = {"low": int(DesireConditions.LOW), "high": int(DesireConditions.HIGH)}
    obs_to_idx = ACTION_LABEL_TO_IDX

    for k, (obs_ae, obs_g, alts_df) in enumerate(runs):
        for scenario in SCENARIO_LABELS:
            s = SCENARIO_TO_IDX[scenario]
            for o in range(n_o):
                for rew in range(n_rew):
                    for e in range(n_eff):
                        a_risk, a_effort = obs_ae[(scenario, e, o)]
                        risk[k, s, o, rew, e, 0] = a_risk
                        effort[k, s, o, rew, e, 0] = a_effort
                        g[k, s, o, rew, e, 0] = obs_g[(scenario, o)]
                        valid[k, s, o, rew, e, 0] = True

        # Alternatives keyed by (scenario, obs, desire, effort, alt_idx). Unscored
        # rows with any NaN feature skipped.
        for _, row in alts_df.iterrows():
            if row[["risk", "effort", "g"]].isna().any():
                continue
            s = SCENARIO_TO_IDX[row["scenario_label"]]
            o = obs_to_idx[row["observed_action"]]
            rew = rew_to_idx[row["desire_condition"]]
            e = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
            slot = int(row["alt_idx"]) + 1
            if slot >= MAX_ACTIONS:
                continue
            risk[k, s, o, rew, e, slot] = float(row["risk"])
            effort[k, s, o, rew, e, slot] = float(row["effort"])
            g[k, s, o, rew, e, slot] = float(row["g"])
            valid[k, s, o, rew, e, slot] = True

    _validate_padded_tables(
        "food_inv_intimacy",
        {"risk": (risk, valid), "effort": (effort, valid), "g": (g, valid)},
    )
    for arr in (risk, effort, g):
        arr[~valid] = 0.0

    NULL_EPSILON = 1e-8
    n_valid = valid.sum(axis=-1, keepdims=True)
    prior = np.where(valid, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON).astype(
        np.float32
    )
    _warn_truncation(runs)
    return {
        "risk": jnp.array(risk),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior),
        "n_runs": K,
    }


def load_padded_lm_tables_joint_ie(
    *, slug="food_inv_joint_ie", runs_filename="lm_runs.jsonl"
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

    `slug` selects the study whose tables to load: Study 3b
    (`nonfood_inv_joint_ie`) shares this design on the nonfood stimulus set, so
    it reuses this loader with its own outputs folder and scenario-label order
    (STUDY_SCENARIO_LABELS[slug]). `runs_filename` selects the elicitation
    vintage (an alternate side file); default "lm_runs.jsonl" is the standard
    source.
    """
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "lm" / slug
    scenario_labels = STUDY_SCENARIO_LABELS[slug]
    scenario_to_idx = scenario_to_idx_for_study(slug)
    cell_cols = ["desire_condition", "effort_condition"]
    runs = _run_sources(outputs_dir, cell_cols, runs_filename=runs_filename)
    if runs is None:
        return None
    K = len(runs)

    n_s, n_o, n_rew, n_eff = (
        len(scenario_labels),
        N_ACTIONS,
        2,
        N_EFFORT_CONDITIONS,
    )
    # NaN-init + post-fill validation, as in load_padded_lm_tables_desire.
    risk = np.full((K, n_s, n_o, n_rew, MAX_ACTIONS), np.nan, dtype=np.float32)
    effort = np.full((K, n_s, n_o, n_rew, n_eff, MAX_ACTIONS), np.nan, dtype=np.float32)
    g = np.full((K, n_s, n_o, n_rew, MAX_ACTIONS), np.nan, dtype=np.float32)
    valid = np.zeros((K, n_s, n_o, n_rew, MAX_ACTIONS), dtype=bool)

    rew_to_idx = {"low": int(DesireConditions.LOW), "high": int(DesireConditions.HIGH)}
    obs_to_idx = ACTION_LABEL_TO_IDX

    for k, (obs_ae, obs_g, alts_df) in enumerate(runs):
        for scenario in scenario_labels:
            s = scenario_to_idx[scenario]
            for o in range(n_o):
                for rew in range(n_rew):
                    for e in range(n_eff):
                        a_risk, a_effort = obs_ae[(scenario, e, o)]
                        effort[k, s, o, rew, e, 0] = a_effort
                        if e == 0:
                            risk[k, s, o, rew, 0] = a_risk
                    g[k, s, o, rew, 0] = obs_g[(scenario, o)]
                    valid[k, s, o, rew, 0] = True

        # Alternatives keyed by (scenario, obs, desire, effort_condition, alt_idx);
        # risk effort-marginal / g desire-free (repeated across effort rows).
        # Unscored rows (any NaN feature) skipped.
        for _, row in alts_df.iterrows():
            if row[["risk", "effort", "g"]].isna().any():
                continue
            s = scenario_to_idx[row["scenario_label"]]
            o = obs_to_idx[row["observed_action"]]
            rew = rew_to_idx[row["desire_condition"]]
            e = EFFORT_CONDITION_TO_IDX[row["effort_condition"]]
            slot = int(row["alt_idx"]) + 1
            if slot >= MAX_ACTIONS:
                continue
            risk[k, s, o, rew, slot] = float(row["risk"])
            effort[k, s, o, rew, e, slot] = float(row["effort"])
            g[k, s, o, rew, slot] = float(row["g"])
            valid[k, s, o, rew, slot] = True

    valid_e = np.broadcast_to(valid[:, :, :, :, None, :], effort.shape)
    _validate_padded_tables(
        slug,
        {"risk": (risk, valid), "effort": (effort, valid_e), "g": (g, valid)},
    )
    risk[~valid] = 0.0
    g[~valid] = 0.0
    effort[~valid_e] = 0.0

    NULL_EPSILON = 1e-8
    n_valid = valid.sum(axis=-1, keepdims=True)
    prior = np.where(valid, 1.0 / np.maximum(n_valid, 1), NULL_EPSILON).astype(
        np.float32
    )
    _warn_truncation(runs)
    return {
        "risk": jnp.array(risk),
        "effort": jnp.array(effort),
        "g": jnp.array(g),
        "prior": jnp.array(prior),
        "n_runs": K,
    }
