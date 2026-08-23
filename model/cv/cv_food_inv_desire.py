"""LOSO CV for food_inv_desire (Study 1a).

Each fold jointly refits the actor utility weights + alpha_observer on 15
scenarios, then predicts the held-out scenario. See `_inverse_dispatcher`
for the loop body.

Accepts the shared run-config flag via the dispatcher: `--no-reweighting`
runs the preregistered model and writes to alt/uniform-noreweight/ instead.
"""

from model.cv._inverse_dispatcher import main_desire


if __name__ == "__main__":
    main_desire()
