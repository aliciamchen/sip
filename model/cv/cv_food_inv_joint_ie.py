"""LOSO CV for food_inv_joint_ie (Study 2b).

Each fold jointly refits the actor utility weights + alpha_observer on 15
scenarios, then predicts the held-out scenario. Each held-out trial's two
belief updates (intimacy and world-state) are scored jointly under the K-run
bivariate Gaussian mixture. See `_inverse_dispatcher` for the loop body.

Accepts the shared run-config flag via the dispatcher: `--no-reweighting`
runs the preregistered model and writes to alt/uniform-noreweight/ instead.
"""

from model.cv._inverse_dispatcher import main_joint_ie


if __name__ == "__main__":
    main_joint_ie()
