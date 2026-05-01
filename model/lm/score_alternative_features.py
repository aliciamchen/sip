"""Score access + effort features for LM-generated alternatives.

--conditioning motivation (default): produces lm_alternatives_features_food_inv-intimacy_desire_noalt.csv
--conditioning relationship: produces lm_alternatives_features_food_inv-desire_intimacy_noalt.csv
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "lm"))

from _features_dispatcher import score_alternatives_main, score_alternatives_relationship_main  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=("food", "nonfood"), default="food")
    p.add_argument("--conditioning", choices=("motivation", "relationship"), default="motivation")
    args = p.parse_args()
    if args.conditioning == "motivation":
        score_alternatives_main(domain=args.domain)
    else:
        score_alternatives_relationship_main(domain=args.domain)
