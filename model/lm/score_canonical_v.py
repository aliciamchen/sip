"""Score signed-valence V per (scenario, action, motivation).

Produces lm_scenario_v{,_nonfood}.csv depending on --domain.
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "lm"))

from _features_dispatcher import score_v_main  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=("food", "nonfood"), default="food")
    args = p.parse_args()
    score_v_main(domain=args.domain)
