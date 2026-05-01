"""Generate relationship-conditioned LM alternatives.

Produces lm_alternatives_relationship.csv (or _nonfood with --domain nonfood).
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "lm"))

from _alternatives_dispatcher import main  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=("food", "nonfood"), default="food")
    args = p.parse_args()
    main(domain=args.domain, conditioning="relationship")
