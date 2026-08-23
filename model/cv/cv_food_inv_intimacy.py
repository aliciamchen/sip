"""LOSO CV for food_inv_intimacy (Study 2a).

Each fold jointly refits the actor utility weights + alpha_observer on 15
scenarios, then predicts the held-out scenario. See `_inverse_dispatcher`
for the loop body.

Accepts the shared run-config flag via the dispatcher: `--no-reweighting`
runs the preregistered model and writes to alt/uniform-noreweight/ instead.
"""

from model.cv._inverse_dispatcher import main_intimacy


if __name__ == "__main__":
    main_intimacy()
