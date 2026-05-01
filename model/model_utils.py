"""
Compatibility shim. The contents of this module have been split across:
  - tables.py:   constants, enums, scenario maps, LM table loaders
  - utility.py:  jit-compiled utility functions (get_utility_*, get_intimacy, ...)
  - actors.py:   actor memo models (forward + inverse + padded)
  - observers.py: observer memo models

Existing imports `from model_utils import X` continue to work via the
re-exports below. New code should import from the themed modules directly.

This shim will be deleted once all callers have migrated.
"""

from tables import (
    EFFORT_CONDITION_TO_IDX,
    LLM_TABLES,
    MAX_ACTIONS,
    NONFOOD_SCENARIO_LABELS,
    NONFOOD_SCENARIO_TO_IDX,
    RELATIONSHIP_LEVEL_VALUES,
    SCENARIO_LABELS,
    SCENARIO_TO_IDX,
    EffortConditions,
    IntimacyLevels,
    ObservedActions,
    PaddedActionSlots,
    RelationshipConditions,
    RewardConditions,
    Scenarios,
    actions,
    actions_effort,
    load_domain_assets,
    load_lm_scenario_params,
    load_lm_v,
    load_padded_lm_tables,
    load_padded_lm_tables_relationship,
    padded_slots,
)
from utility import (
    get_intimacy,
    get_lm_v,
    get_lm_v_padded,
    get_lm_v_padded_rel,
    get_prior_padded,
    get_prior_padded_rel,
    get_utility_base,
    get_utility_base_disc,
    get_utility_base_padded,
    get_utility_base_padded_rel,
    get_utility_discomfort_only,
    get_utility_discomfort_only_disc,
    get_utility_discomfort_only_padded,
    get_utility_discomfort_only_padded_rel,
    get_utility_full,
    get_utility_full_disc,
    get_utility_full_padded,
    get_utility_full_padded_rel,
)
from actors import (
    actor_continuous_base,
    actor_continuous_base_padded,
    actor_continuous_base_padded_rel,
    actor_continuous_discomfort_only,
    actor_continuous_discomfort_only_padded,
    actor_continuous_discomfort_only_padded_rel,
    actor_continuous_full,
    actor_continuous_full_padded,
    actor_continuous_full_padded_rel,
    actor_discrete_base,
    actor_discrete_discomfort_only,
    actor_discrete_full,
    actor_forw_base,
    actor_forw_discomfort_only,
    actor_forw_full,
)
from observers import (
    observer_intimacy_base,
    observer_intimacy_base_padded,
    observer_intimacy_discomfort_only,
    observer_intimacy_discomfort_only_padded,
    observer_intimacy_full,
    observer_intimacy_full_padded,
    observer_reward_base,
    observer_reward_base_padded_rel,
    observer_reward_discomfort_only,
    observer_reward_discomfort_only_padded_rel,
    observer_reward_full,
    observer_reward_full_padded_rel,
)
