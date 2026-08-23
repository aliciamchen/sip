"""LOSO CV for food_inv_desire (Study 1a).

Each fold jointly refits the actor utility weights + alpha_observer on 15
scenarios, then predicts the held-out scenario. See `_inverse_dispatcher`
for the loop body.

Accepts the shared run-config flag via the dispatcher: `--no-reweighting`
runs the preregistered model and writes to alt/uniform-noreweight/ instead.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "cv"))

from _inverse_dispatcher import main_desire  # noqa: E402


if __name__ == "__main__":
    main_desire()
