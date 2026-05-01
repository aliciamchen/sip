"""Fit forward-planning models to food_forw_intimacy_desire data.

Outputs only fit_results.csv (parameters + AIC/BIC/r per ablation).
For per-trial predictions, run predict_food_forw_intimacy_desire.py after.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "forward"))

from _shared import (  # noqa: E402
    fit_canonical_base,
    fit_canonical_discomfort_only,
    fit_canonical_full,
    load_data_canonical,
    predict_canonical_base,
    predict_canonical_discomfort_only,
    predict_canonical_full,
    run_fit_and_save_results,
)
from tables import LLM_TABLES, SCENARIO_TO_IDX, load_lm_v  # noqa: E402
from utils import get_project_root  # noqa: E402

EXPERIMENT_SLUG = "food_forw_intimacy_desire"


def main():
    data_path = get_project_root() / "data" / EXPERIMENT_SLUG / "main_trials_long.csv"
    data, intimacy, condition_iv, action, p_action, scenario_idx = load_data_canonical(
        data_path, SCENARIO_TO_IDX,
    )

    v_table = load_lm_v("food")
    tables_full = (LLM_TABLES["access"], LLM_TABLES["effort"], v_table)
    tables_disc = (LLM_TABLES["access"], LLM_TABLES["effort"])

    fit_funcs = {
        "full": (fit_canonical_full, predict_canonical_full, ["w_v", "w_d", "w_e", "gamma"]),
        "discomfort_only": (fit_canonical_discomfort_only, predict_canonical_discomfort_only, ["w_d", "gamma"]),
        "base": (fit_canonical_base, predict_canonical_base, ["w_v", "w_e"]),
    }
    tables_by_variant = {
        "full": tables_full,
        "discomfort_only": tables_disc,
        "base": tables_full,
    }

    run_fit_and_save_results(
        experiment_slug=EXPERIMENT_SLUG,
        intimacy=intimacy, condition_iv=condition_iv, action=action,
        scenario_idx=scenario_idx, p_action=p_action,
        tables_by_variant=tables_by_variant,
        fit_funcs=fit_funcs,
        group_cols=["intimacy", "motivation", "action"],
        data=data,
    )


if __name__ == "__main__":
    main()
