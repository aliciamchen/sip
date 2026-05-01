"""Generate per-trial predictions for nonfood_forw_intimacy_desire.

Reads fit_results.csv, recomputes predictions, writes fits.csv.
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
from tables import load_domain_assets, load_lm_v  # noqa: E402
from utils import get_project_root  # noqa: E402

EXPERIMENT_SLUG = "nonfood_forw_intimacy_desire"


def main():
    _, scenario_to_idx, llm_tables = load_domain_assets("nonfood")
    v_table = load_lm_v("nonfood")
    data_path = get_project_root() / "data" / EXPERIMENT_SLUG / "main_trials_long.csv"
    data, intimacy, condition_iv, action, p_action, scenario_idx = load_data_canonical(
        data_path, scenario_to_idx,
    )

    tables_full = (llm_tables["access"], llm_tables["effort"], v_table)
    tables_disc = (llm_tables["access"], llm_tables["effort"])

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
