"""LOSO CV for food_inv_desire (Study 3b).

Each fold jointly refits the actor utility weights + alpha_observer on 15
scenarios, then predicts the held-out scenario. See `_inverse_dispatcher`
for the loop body.
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
