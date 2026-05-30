"""Counterbalancing for food_inv_intimacy.

Study 2 — Inverse intimacy (knowns = reward + effort).
Cells: reward × effort × action = 2 × 2 × 3 = 12 cells.

Each participant sees all 16 scenarios (one trial each). Cells are distributed
across the 16 trials as evenly as possible: each cell gets `floor(16 / n_cells)`
base slots, plus `16 - base * n_cells` random extra slots. With more than 16
cells (Studies 3a, 3b), each participant samples a 16-cell subset; across
participants the cells balance out.

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

random.seed(2202)

all_stories = [
    "basketball", "birthday", "brunch", "takeout", "cooking", "apples",
    "dip", "drinks", "driving", "fair", "gala", "hike",
    "oysters", "social", "soup", "wedding",
]
N_SLOTS = 16
N_ROUNDS = 12  # × N_SLOTS rotations per round = 192 sequences

ALL_CELLS = [(r, e, a) for r in ("low", "high") for e in ("low", "high") for a in ("action_0", "action_1", "action_2")]


def cell_to_dict(cell):
    reward, effort, action = cell
    return {"reward_condition": reward, "effort_condition": effort, "action_condition": action}


def make_all_conditions(cells, n_slots=N_SLOTS):
    """Return n_slots cells, evenly distributed across the cell set.

    floor(n_slots / n_cells) base copies of every cell, plus `n_slots % n_cells`
    extra copies chosen randomly without replacement so the extras don't
    cluster.
    """
    n_cells = len(cells)
    base_count = n_slots // n_cells
    extra = n_slots - base_count * n_cells
    result = list(cells) * base_count
    shuffled = list(cells)
    random.shuffle(shuffled)
    result += shuffled[:extra]
    random.shuffle(result)
    return result


def make_trial_sequence(stories, conditions):
    assert len(stories) == len(conditions)
    return [
        {"scenario_label": s, **cell_to_dict(c)}
        for s, c in zip(stories, conditions)
    ]


def make_counterbalancing_round(stories, cells):
    """One round = 16 sequences obtained by rotating which scenario gets cell 0,
    keeping a single fixed condition assignment list across the 16 rotations."""
    conditions = make_all_conditions(cells)
    sequences = []
    for offset in range(len(stories)):
        rotated = stories[offset:] + stories[:offset]
        sequences.append(make_trial_sequence(rotated, conditions))
    return sequences


counterbalancing = []
for round_idx in range(N_ROUNDS):
    counterbalancing.extend(make_counterbalancing_round(all_stories, ALL_CELLS))
    random.shuffle(all_stories)


output_path = (
    get_project_root() / "experiments" / "food_inv_intimacy" / "json" / "full_counterbalancing.json"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(counterbalancing, f)

print(f"Generated {len(counterbalancing)} counterbalanced sequences")
print(f"Each sequence has {len(counterbalancing[0])} trials")
print(f"Cells: {len(ALL_CELLS)} (reward × effort × action = 2 × 2 × 3 = 12 cells)")
