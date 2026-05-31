"""Counterbalancing for food_inv_intimacy.

Study 2a — Inverse intimacy (knowns = desire + effort).
Cells: desire × effort × action = 2 × 2 × 3 = 12 cells.

Each participant sees all 16 scenarios (one trial each). The 12 cells are spread
across the 16 trial slots with a balanced design: every round holds one copy of
all 12 cells plus 4 extra cells, and the extras are assigned cyclically so each
cell is an extra in exactly 4 of the 12 rounds. Every cell therefore lands in
the same number of trial slots overall (256 each), giving uniform global cell
coverage. Rounds are interleaved by rotation index so sequential DataPipe
condition assignment spreads early participants across all rounds rather than
clustering them on round 0's extra cells.

Produces 12 × 16 = 192 condition_assignment sequences.
"""

import json
import random
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root

random.seed(2202)

all_stories = [
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
N_SLOTS = 16
N_ROUNDS = 12  # × N_SLOTS rotations per round = 192 sequences

ALL_CELLS = [
    (r, e, a)
    for r in ("low", "high")
    for e in ("low", "high")
    for a in ("no_share", "low_risk_share", "high_risk_share")
]


def cell_to_dict(cell):
    desire, effort, action = cell
    return {
        "desire_condition": desire,
        "effort_condition": effort,
        "action_condition": action,
    }


def assign_cells_to_rounds(cells, n_rounds=N_ROUNDS, n_slots=N_SLOTS):
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
    assert total_extras % n_cells == 0, (
        "Balanced design requires n_rounds * (n_slots % n_cells) divisible by n_cells"
    )
    extra_per_cell = total_extras // n_cells
    assert extra_per_cell <= n_rounds, "a cell can be an extra at most once per round"

    cells = list(cells)
    random.shuffle(cells)
    round_buckets = [list(cells) * base_count for _ in range(n_rounds)]
    for c, cell in enumerate(cells):
        for k in range(extra_per_cell):
            round_buckets[(c + k) % n_rounds].append(cell)
    assert all(len(b) == n_slots for b in round_buckets), [
        len(b) for b in round_buckets
    ]
    round_order = list(range(n_rounds))
    random.shuffle(round_order)
    return [round_buckets[i] for i in round_order]


def make_trial_sequence(stories, conditions):
    assert len(stories) == len(conditions)
    return [
        {"scenario_label": s, **cell_to_dict(c)} for s, c in zip(stories, conditions)
    ]


def make_counterbalancing_round(stories, cells_for_round):
    """One round = 16 sequences obtained by rotating which scenario gets cell 0,
    keeping a single fixed condition assignment list across the 16 rotations.

    `cells_for_round` is the 16-cell multiset assigned to this round by
    `assign_cells_to_rounds`; its order is shuffled here so the cell-to-position
    mapping varies across rounds.
    """
    conditions = list(cells_for_round)
    random.shuffle(conditions)
    sequences = []
    for offset in range(len(stories)):
        rotated = stories[offset:] + stories[:offset]
        sequences.append(make_trial_sequence(rotated, conditions))
    return sequences


round_cell_sets = assign_cells_to_rounds(ALL_CELLS)
# round_sequences[r] holds the 16 rotation-sequences for round r.
round_sequences = []
for round_idx in range(N_ROUNDS):
    round_sequences.append(
        make_counterbalancing_round(all_stories, round_cell_sets[round_idx])
    )
    random.shuffle(all_stories)

# Interleave rounds by rotation index so each block of N_ROUNDS consecutive
# sequences covers all 12 rounds. DataPipe assigns condition_assignment
# sequentially, so this spreads early participants across rounds rather than
# running them all on round 0's particular extra-cell choices.
counterbalancing = [
    round_sequences[r][s] for s in range(N_SLOTS) for r in range(N_ROUNDS)
]


output_path = (
    get_project_root()
    / "experiments"
    / "food_inv_intimacy"
    / "json"
    / "full_counterbalancing.json"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(counterbalancing, f)

print(f"Generated {len(counterbalancing)} counterbalanced sequences")
print(f"Each sequence has {len(counterbalancing[0])} trials")
print(f"Cells: {len(ALL_CELLS)} (desire × effort × action = 2 × 2 × 3 = 12 cells)")
