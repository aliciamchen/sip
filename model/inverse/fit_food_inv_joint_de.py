"""Fit observer + actor utility weights for food_inv_joint_de.

Study 1b — joint over (desire, effort) given intimacy. Each variant jointly fits
its utility weights, alpha_observer, and the response-noise sigma from this
experiment's belief-update data (no transfer between studies). Writes
outputs/food_inv_joint_de/fit_results.json.

Accepts the shared run-config flag via the dispatcher: `--no-reweighting`
runs the preregistered model and writes to alt/uniform-noreweight/ instead.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import _fit_dispatcher  # noqa: E402

EXPERIMENT_SLUG = "food_inv_joint_de"


def main(config=None):
    _fit_dispatcher.main(EXPERIMENT_SLUG, config=config, description=__doc__)


if __name__ == "__main__":
    main()
