"""Counterbalancing for food_inv_effort_3act.

Study 3a — Effort inference (knowns = reward + intimacy).
Cells: reward × intimacy × action = 2 × 4 × 3 = 24 cells.

Each participant sees all 16 scenarios (one trial each); since there are 24
cells but only 16 trial slots, each participant samples 16 of the 24 cells.
To keep global cell coverage uniform across the full set of sequences, the
24 cells are assigned to rounds via a balanced design: every cell appears in
exactly 8 of 12 rounds (so 16 cells per round, 192 cell-round picks total =
24 × 8). With 16 story-rotations per round, every cell appears in exactly
8 × 16 = 128 trial slots.

Produces 12 × 16 = 192 condition_assignment sequences (matching the legacy
food_inv_intimacy_effort_alt counterbalancing layout).
"""

import json
import random
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root

random.seed(303)

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
ROUNDS_PER_CELL = 8  # each cell appears in this many of the N_ROUNDS rounds

ALL_CELLS = [
    (r, i, a)
    for r in ("low", "high")
    for i in (0, 50, 75, 100)
    for a in ("action_0", "action_1", "action_2")
]
assert len(ALL_CELLS) * ROUNDS_PER_CELL == N_ROUNDS * N_SLOTS, (
    "Balanced design requires n_cells * rounds_per_cell == n_rounds * n_slots"
)


def cell_to_dict(cell):
    reward, intimacy, action = cell
    return {
        "reward_condition": reward,
        "intimacy_condition": intimacy,
        "action_condition": action,
    }


def assign_cells_to_rounds(cells, n_rounds=N_ROUNDS, rounds_per_cell=ROUNDS_PER_CELL):
    """Return a list of `n_rounds` cell-lists where every cell appears in exactly
    `rounds_per_cell` rounds and no round has duplicate cells.

    Construction: shuffle the cells, then for cell at index c (after shuffle)
    assign it to rounds {(c + k) mod n_rounds : k = 0..rounds_per_cell-1}. Since
    n_cells = 24 = 2 * n_rounds and rounds_per_cell = 8, each round receives
    exactly 16 distinct cells. Round labels are then permuted so the cyclic
    structure isn't visible in run order.
    """
    cells = list(cells)
    random.shuffle(cells)
    round_buckets = [[] for _ in range(n_rounds)]
    for c, cell in enumerate(cells):
        for k in range(rounds_per_cell):
            round_buckets[(c + k) % n_rounds].append(cell)
    assert all(len(b) == N_SLOTS for b in round_buckets), [
        len(b) for b in round_buckets
    ]
    assert all(len(set(b)) == N_SLOTS for b in round_buckets), (
        "duplicate cells in a round"
    )
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

    `cells_for_round` is the 16-cell subset assigned to this round by
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
# sequences covers all 12 rounds (and thus all 24 cells, since every cell is in
# 8 of 12 rounds). DataPipe assigns condition_assignment sequentially, so this
# guarantees early participants jointly cover the full cell space rather than
# all running the same round-0 cell list.
counterbalancing = [
    round_sequences[r][s] for s in range(N_SLOTS) for r in range(N_ROUNDS)
]


output_path = (
    get_project_root()
    / "experiments"
    / "food_inv_effort_3act"
    / "json"
    / "full_counterbalancing.json"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(counterbalancing, f)

print(f"Generated {len(counterbalancing)} counterbalanced sequences")
print(f"Each sequence has {len(counterbalancing[0])} trials")
print(f"Cells: {len(ALL_CELLS)} (reward × intimacy × action = 2 × 4 × 3 = 24 cells)")
