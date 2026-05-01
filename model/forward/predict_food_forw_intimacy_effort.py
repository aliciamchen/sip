"""Generate per-cell predictions for food_forw_intimacy_effort.

Reads fit_results.csv, computes the model's predicted action probability for
each (scenario, action, intimacy, effort_condition) cell, writes
outputs/<slug>/preds.csv.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "forward"))

from _shared import (  # noqa: E402
    build_effort_cells,
    predict_effort_base,
    predict_effort_discomfort_only,
    predict_effort_full,
    run_predict_and_save_preds,
)
from tables import LLM_TABLES_EFFORT, SCENARIO_LABELS  # noqa: E402

EXPERIMENT_SLUG = "food_forw_intimacy_effort"


def main():
    cells = build_effort_cells(SCENARIO_LABELS)

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

    run_predict_and_save_preds(
        experiment_slug=EXPERIMENT_SLUG,
        cells_df=cells,
        iv_idx_col="effort_idx",
        tables_by_variant=tables_by_variant,
        predict_funcs=predict_funcs,
        fit_param_names=fit_param_names,
    )


if __name__ == "__main__":
    main()
