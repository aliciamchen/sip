"""Generate per-trial predictions for food_forw_intimacy_effort.

Reads fit_results.csv, recomputes predictions, writes fits.csv.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "forward"))

from _shared import (  # noqa: E402
    load_data_effort,
    predict_effort_base,
    predict_effort_discomfort_only,
    predict_effort_full,
    run_predict_and_save_fits,
)
from tables import LLM_TABLES_EFFORT  # noqa: E402
from utils import get_project_root  # noqa: E402

EXPERIMENT_SLUG = "food_forw_intimacy_effort"


def main():
    data_path = get_project_root() / "data" / EXPERIMENT_SLUG / "main_trials_long.csv"
    data, intimacy, condition_iv, action, p_action, scenario_idx = load_data_effort(data_path)

    tables = (LLM_TABLES_EFFORT["access"], LLM_TABLES_EFFORT["effort"])
    predict_funcs = {
        "full": predict_effort_full,
        "discomfort_only": predict_effort_discomfort_only,
        "base": predict_effort_base,
    }
    tables_by_variant = {"full": tables, "discomfort_only": tables, "base": tables}
    fit_param_names = {
        "full": ["w_v", "w_d", "w_e", "gamma"],
        "discomfort_only": ["w_d", "gamma"],
        "base": ["w_v", "w_e"],
    }

    run_predict_and_save_fits(
        experiment_slug=EXPERIMENT_SLUG,
        intimacy=intimacy, condition_iv=condition_iv, action=action,
        scenario_idx=scenario_idx,
        tables_by_variant=tables_by_variant,
        predict_funcs=predict_funcs,
        fit_param_names=fit_param_names,
        data=data,
    )


if __name__ == "__main__":
    main()
