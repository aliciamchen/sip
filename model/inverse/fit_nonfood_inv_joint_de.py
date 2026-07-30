"""Fit observer + actor utility weights for nonfood_inv_joint_de.

Study 3a — joint over (desire, effort) given intimacy, on the nonfood stimulus
set (experiments/scenarios_nonfood.csv). Same design and observers as Study 1b;
only the stimulus set and LM tables differ (domain="nonfood" routes the table
loaders to outputs/lm/nonfood_inv_joint_de/). Each variant jointly fits its
utility weights, alpha_observer, and the response-noise sigma from this
experiment's belief-update data (no transfer between studies). Writes
outputs/nonfood_inv_joint_de/fit_results.json.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import _fit_dispatcher  # noqa: E402

EXPERIMENT_SLUG = "nonfood_inv_joint_de"


def main(config=None):
    _fit_dispatcher.main(EXPERIMENT_SLUG, config=config, description=__doc__)


if __name__ == "__main__":
    main()
