"""Fit forward-planning models to food_forw_intimacy_effort data.

Outputs only fit_results.csv. For per-trial predictions, run
predict_food_forw_intimacy_effort.py after.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "forward"))

from _shared import (  # noqa: E402
    fit_effort_base,
    fit_effort_discomfort_only,
    fit_effort_full,
    load_data_effort,
    predict_effort_base,
    predict_effort_discomfort_only,
    predict_effort_full,
    run_fit_and_save_results,
)
from tables import LLM_TABLES_EFFORT  # noqa: E402
from utils import get_project_root  # noqa: E402

EXPERIMENT_SLUG = "food_forw_intimacy_effort"


def main():
    data_path = get_project_root() / "data" / EXPERIMENT_SLUG / "main_trials_long.csv"
    data, intimacy, condition_iv, action, p_action, scenario_idx = load_data_effort(data_path)

    tables = (LLM_TABLES_EFFORT["access"], LLM_TABLES_EFFORT["effort"])
    fit_funcs = {
        "full": (fit_effort_full, predict_effort_full, ["w_v", "w_d", "w_e", "gamma"]),
        "discomfort_only": (fit_effort_discomfort_only, predict_effort_discomfort_only, ["w_d", "gamma"]),
        "base": (fit_effort_base, predict_effort_base, ["w_v", "w_e"]),
    }
    tables_by_variant = {"full": tables, "discomfort_only": tables, "base": tables}

    run_fit_and_save_results(
        experiment_slug=EXPERIMENT_SLUG,
        intimacy=intimacy, condition_iv=condition_iv, action=action,
        scenario_idx=scenario_idx, p_action=p_action,
        tables_by_variant=tables_by_variant,
        fit_funcs=fit_funcs,
        group_cols=["intimacy", "effort", "action"],
        data=data,
    )


if __name__ == "__main__":
    main()
