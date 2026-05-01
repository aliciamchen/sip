"""Generate per-trial predictions for food_forw_intimacy_desire.

Reads fit_results.csv, recomputes per-trial p_action under each variant's
fitted params, writes outputs/<slug>/fits.csv (data + pred_<variant> columns).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "forward"))

from _shared import (  # noqa: E402
    load_data_canonical,
    predict_canonical_base,
    predict_canonical_discomfort_only,
    predict_canonical_full,
    run_predict_and_save_fits,
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

    predict_funcs = {
        "full": predict_canonical_full,
        "discomfort_only": predict_canonical_discomfort_only,
        "base": predict_canonical_base,
    }
    tables_by_variant = {
        "full": tables_full,
        "discomfort_only": tables_disc,
        "base": tables_full,
    }
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
