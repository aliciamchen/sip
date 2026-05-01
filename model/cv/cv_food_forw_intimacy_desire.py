"""LOSO CV for food_forw_intimacy_desire."""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "cv"))

from _forward_dispatcher import main  # noqa: E402

if __name__ == "__main__":
    main("food_forw_intimacy_desire")
