"""LOSO CV for food_inv_joint_de_3act (Study 4a).

Each fold jointly refits the actor utility weights + alpha_observer on 15
scenarios, then predicts the held-out scenario. Per-trial test NLL sums two
binary cross-entropies (P(reward=HIGH) and P(effort=HIGH)). See
`_inverse_3act_dispatcher` for the loop body.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "cv"))

from _inverse_3act_dispatcher import main_joint_de_3act  # noqa: E402


if __name__ == "__main__":
    main_joint_de_3act()
