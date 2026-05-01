"""
Compatibility shim. The contents of this module have been merged into the
shared themed modules (tables.py, utility.py, actors.py, observers.py)
alongside the canonical 4-action variants — see those files for the
effort-experiment counterparts (suffix `_effort_*`).

Existing imports `from model_utils_effort import X` continue to work via the
re-exports below. New code should import from the themed modules directly.

This shim will be deleted once all callers have migrated.
"""

from tables import (
    EFFORT_CONDITION_TO_IDX,
    LLM_TABLES_EFFORT,
    N_ACTIONS_EFFORT,
    N_EFFORT_CONDITIONS,
    EffortConditions,
    actions_effort,
    load_lm_scenario_params_effort,
    load_lm_scenario_params_effort_marginal,
)
from utility import (
    get_stipulated_reward_effort,
    get_utility_effort_base,
    get_utility_effort_discomfort_only,
    get_utility_effort_full,
)
from actors import (
    actor_continuous_effort_base,
    actor_continuous_effort_discomfort_only,
    actor_continuous_effort_full,
    actor_forw_effort_base,
    actor_forw_effort_discomfort_only,
    actor_forw_effort_full,
)
from observers import (
    observer_effort_intimacy_base,
    observer_effort_intimacy_discomfort_only,
    observer_effort_intimacy_full,
    observer_intimacy_effort_base,
    observer_intimacy_effort_discomfort_only,
    observer_intimacy_effort_full,
)
