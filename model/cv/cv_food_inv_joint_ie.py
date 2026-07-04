"""LOSO CV for food_inv_joint_ie (Study 2b).

Each fold jointly refits the actor utility weights + alpha_observer on 15
scenarios, then predicts the held-out scenario. Each held-out trial's two
belief updates (intimacy and world-state) are scored jointly under the K-run
bivariate Gaussian mixture. See `_inverse_dispatcher` for the loop body.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "cv"))

from _inverse_dispatcher import main_joint_ie  # noqa: E402


if __name__ == "__main__":
    main_joint_ie()
