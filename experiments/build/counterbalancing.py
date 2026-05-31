"""Counterbalancing generator for all four active inverse-planning studies.

One script, one balanced design, one per-study registry — replacing the four
near-identical `experiments/<slug>/python/generate_counterbalancing.py` files.
(This lives under `experiments/build/`, which the deploy never touches — it only
pushes `_lib/` and the per-experiment dirs.)

For each study it writes `experiments/<slug>/json/full_counterbalancing.json` —
an array of N "sequences," each a 16-trial assignment of factor cells to the 16
scenarios. `experiment.js` reads a per-participant `condition_assignment` from
jsPsychPipe and selects `counterbalancing[sequence_index]`.

Balanced design (shared across all four studies):
  Each study builds `n_rounds` rounds; within a round one fixed condition list is
  rotated across the 16 scenarios (16 rotation-sequences). The cells are spread
  across the 16 slots of a round so that every factor cell ends up in the SAME
  number of trial slots overall: each round holds `16 // n_cells` copies of every
  cell plus `16 % n_cells` extra cells, and the extras are assigned cyclically so
  each cell is an extra in exactly `extra_per_cell` rounds (never twice in one
  round). Studies with more than 16 cells (1a's 24) fall out of the same formula
  with `base_count == 0` (each round carries a balanced 16-cell subset). Rounds
  are interleaved by rotation index so sequential `condition_assignment` values
  spread early participants across all rounds rather than clustering them on one
  round's cell choices.

Per study (cells = product of the manipulated factors; intimacy/effort/desire are
omitted when that variable is inferred rather than given):
  - food_inv_desire   (1a): effort x intimacy x action = 2 x 4 x 3 = 24 cells, 12 rounds -> 192 seqs
  - food_inv_joint_de (1b): intimacy x action           = 4 x 3     = 12 cells, 12 rounds -> 192 seqs
  - food_inv_intimacy (2a): desire x effort x action    = 2 x 2 x 3 = 12 cells, 12 rounds -> 192 seqs
  - food_inv_joint_ie (2b): desire x action             = 2 x 3     =  6 cells,  6 rounds ->  96 seqs

Usage:
    uv run python experiments/build/counterbalancing.py            # all studies
    uv run python experiments/build/counterbalancing.py --study food_inv_joint_ie
"""

import argparse
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from utils import get_project_root

# The 16 scenarios, shared across every study.
STORIES = [
    "basketball",
    "birthday",
    "brunch",
    "takeout",
    "cooking",
    "apples",
    "dip",
    "drinks",
    "driving",
    "fair",
    "gala",
    "hike",
    "oysters",
    "social",
    "soup",
    "wedding",
]
N_SLOTS = 16  # one trial per scenario

ACTIONS = ("no_share", "low_risk_share", "high_risk_share")

# Per-study design: the manipulated (participant-visible) factors, their levels,
# the number of rounds, and the RNG seed. `factors` and `levels` are aligned —
# a cell is `dict(zip(factors, cell_tuple))`.
STUDY_CONFIGS = {
    "food_inv_desire": {  # Study 1a — infer desire (effort + intimacy given)
        "seed": 313,
        "n_rounds": 12,
        "factors": ("effort_condition", "intimacy_condition", "action_condition"),
        "levels": (("low", "high"), (0, 50, 75, 100), ACTIONS),
    },
    "food_inv_joint_de": {  # Study 1b — joint desire + effort (intimacy given)
        "seed": 404,
        "n_rounds": 12,
        "factors": ("intimacy_condition", "action_condition"),
        "levels": ((0, 50, 75, 100), ACTIONS),
    },
    "food_inv_intimacy": {  # Study 2a — infer intimacy (desire + effort given)
        "seed": 2202,
        "n_rounds": 12,
        "factors": ("desire_condition", "effort_condition", "action_condition"),
        "levels": (("low", "high"), ("low", "high"), ACTIONS),
    },
    "food_inv_joint_ie": {  # Study 2b — joint intimacy + effort (desire given)
        "seed": 404,
        "n_rounds": 6,
        "factors": ("desire_condition", "action_condition"),
        "levels": (("low", "high"), ACTIONS),
    },
}


def assign_cells_to_rounds(cells, n_rounds, n_slots, rng):
    """Split the cell set into `n_rounds` balanced multisets of `n_slots` cells.

    Every round holds `n_slots // n_cells` copies of every cell, plus
    `n_slots % n_cells` extra cells. The extras are assigned cyclically — cell at
    index c is an extra in rounds {(c + k) mod n_rounds : k = 0..extra_per_cell-1}
    — so each cell is an extra in exactly `extra_per_cell` rounds and never twice
    in the same round. Every cell then occupies an identical number of trial slots
    across all sequences. Round labels are permuted so the cyclic structure isn't
    visible in run order.
    """
    n_cells = len(cells)
    base_count = n_slots // n_cells
    extra_per_round = n_slots - base_count * n_cells
    total_extras = n_rounds * extra_per_round
    if total_extras % n_cells != 0:
        raise ValueError(
            f"Balanced design requires n_rounds * (n_slots % n_cells) divisible by "
            f"n_cells (got {total_extras} extras over {n_cells} cells)"
        )
    extra_per_cell = total_extras // n_cells
    if extra_per_cell > n_rounds:
        raise ValueError("a cell can be an extra at most once per round")

    cells = list(cells)
    rng.shuffle(cells)
    round_buckets = [list(cells) * base_count for _ in range(n_rounds)]
    for c, cell in enumerate(cells):
        for k in range(extra_per_cell):
            round_buckets[(c + k) % n_rounds].append(cell)
    assert all(len(b) == n_slots for b in round_buckets), [
        len(b) for b in round_buckets
    ]
    order = list(range(n_rounds))
    rng.shuffle(order)
    return [round_buckets[i] for i in order]


def make_counterbalancing_round(stories, cells_for_round, factors, rng):
    """One round = `len(stories)` sequences, rotating which scenario gets cell 0
    while a single shuffled condition list stays fixed across the rotations."""
    conditions = list(cells_for_round)
    rng.shuffle(conditions)
    sequences = []
    for offset in range(len(stories)):
        rotated = stories[offset:] + stories[:offset]
        sequences.append(
            [
                {"scenario_label": s, **dict(zip(factors, c))}
                for s, c in zip(rotated, conditions)
            ]
        )
    return sequences


def build_counterbalancing(
    cells, factors, n_rounds, seed, stories=STORIES, n_slots=N_SLOTS
):
    """Build the full list of `n_rounds * n_slots` counterbalancing sequences."""
    rng = random.Random(seed)
    round_cell_sets = assign_cells_to_rounds(cells, n_rounds, n_slots, rng)
    stories = list(stories)
    round_sequences = []
    for r in range(n_rounds):
        round_sequences.append(
            make_counterbalancing_round(stories, round_cell_sets[r], factors, rng)
        )
        rng.shuffle(stories)
    # Interleave by rotation index so each block of n_rounds consecutive sequences
    # covers all rounds (and thus all cells).
    return [round_sequences[r][s] for s in range(n_slots) for r in range(n_rounds)]


def generate_for_study(slug):
    cfg = STUDY_CONFIGS[slug]
    cells = list(itertools.product(*cfg["levels"]))
    sequences = build_counterbalancing(
        cells, cfg["factors"], cfg["n_rounds"], cfg["seed"]
    )

    # Invariant check: every cell lands in the same number of trial slots.
    counts = Counter(
        tuple(sorted((k, v) for k, v in trial.items() if k != "scenario_label"))
        for seq in sequences
        for trial in seq
    )
    if len(set(counts.values())) != 1:
        raise AssertionError(
            f"{slug}: non-uniform cell coverage {min(counts.values())}-{max(counts.values())}"
        )

    out = (
        get_project_root()
        / "experiments"
        / slug
        / "json"
        / "full_counterbalancing.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(sequences, f)

    per_cell = next(iter(counts.values()))
    print(
        f"{slug}: {len(sequences)} sequences x {len(sequences[0])} trials | "
        f"{len(cells)} cells x {per_cell} slots each (uniform)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--study",
        choices=list(STUDY_CONFIGS),
        help="generate one study (default: all four)",
    )
    args = parser.parse_args()
    slugs = [args.study] if args.study else list(STUDY_CONFIGS)
    for slug in slugs:
        generate_for_study(slug)


if __name__ == "__main__":
    main()
