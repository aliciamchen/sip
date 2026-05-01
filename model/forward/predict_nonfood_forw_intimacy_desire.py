"""Generate per-cell predictions for nonfood_forw_intimacy_desire.

Reads fit_results.csv, computes the model's predicted action probability for
each (scenario, action, intimacy, motivation) cell, writes
outputs/<slug>/preds.csv.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "forward"))

from _shared import (  # noqa: E402
    build_canonical_cells,
    predict_canonical_base,
    predict_canonical_discomfort_only,
    predict_canonical_full,
    run_predict_and_save_preds,
)
from tables import load_domain_assets, load_lm_v  # noqa: E402

EXPERIMENT_SLUG = "nonfood_forw_intimacy_desire"


def main():
    scenario_labels, _, llm_tables = load_domain_assets("nonfood")
    v_table = load_lm_v("nonfood")
    cells = build_canonical_cells(scenario_labels)

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

    run_predict_and_save_preds(
        experiment_slug=EXPERIMENT_SLUG,
        cells_df=cells,
        iv_idx_col="motivation_idx",
        tables_by_variant=tables_by_variant,
        predict_funcs=predict_funcs,
        fit_param_names=fit_param_names,
    )


if __name__ == "__main__":
    main()
