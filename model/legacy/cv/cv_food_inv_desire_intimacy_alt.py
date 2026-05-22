"""LOSO CV for food_inv_desire_intimacy_alt."""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "cv"))

from _alt_dispatcher import main_desire_alt  # noqa: E402

if __name__ == "__main__":
    main_desire_alt()
