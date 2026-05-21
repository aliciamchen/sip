"""LOSO CV for food_inv_intimacy_3act.

Stub — full LOSO logic to be added once data exists. For now this just runs
the all-data fit/predict pipeline.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "cv"))


def main():
    print("LOSO CV for food_inv_intimacy_3act — TODO: implement full leave-one-scenario-out loop.")
    print("Until then, run `make fit-food_inv_intimacy_3act` followed by `make predict-food_inv_intimacy_3act`.")


if __name__ == "__main__":
    main()
