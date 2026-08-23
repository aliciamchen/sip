"""Fit observer + actor utility weights for food_inv_joint_ie.

Study 2b — joint over (intimacy, effort) given desire. Each variant jointly fits
its utility weights, alpha_observer, and the response-noise sigma from this
experiment's belief-update data (no transfer between studies). Writes
outputs/food_inv_joint_ie/fit_results.json.

Accepts the shared run-config flag via the dispatcher: `--no-reweighting`
runs the preregistered model and writes to alt/uniform-noreweight/ instead.
"""

from model.inverse import _fit_dispatcher

EXPERIMENT_SLUG = "food_inv_joint_ie"


def main(config=None):
    _fit_dispatcher.main(EXPERIMENT_SLUG, config=config, description=__doc__)


if __name__ == "__main__":
    main()
