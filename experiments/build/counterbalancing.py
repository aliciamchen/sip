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
  rotated across the 16 scenarios (16 rotation-sequences). Two balance goals are
  served at once:

  Across participants — every factor cell ends up in the SAME number of trial
  slots overall (and so does every scenario x cell pairing): each round holds
  `16 // n_cells` copies of every cell plus `16 % n_cells` extra cells, and the
  pool of extras holds each cell exactly `extra_per_cell` times, so coverage is
  uniform no matter how the extras are arranged.

  Within a participant — a participant's marginal balance on each factor is
  entirely the balance of their round's bucket (all 16 rotations share one cell
  multiset). 16 rarely divides the factor levels cleanly, so some imbalance is
  unavoidable, but it should be (a) as small as the arithmetic allows and (b)
  spread evenly rather than concentrated. We therefore order the extra-cell pool
  with `smooth_order` (greedily keep every factor's running level counts flat)
  and chunk it across rounds, so each bucket is marginally balanced to the floor:
  binary factors split 8/8 (spread 0), 4-level intimacy 4/4/4/4 (spread 0), the
  3-level action 6/5/5 (spread 1, the unavoidable residue), and which action /
  which doubled-or-dropped cell carries that residue rotates across rounds.
  Studies with more than 16 cells (1a's 24) fall out of the same formula with
  `base_count == 0` (each round carries a balanced 16-cell subset). Rounds are
  interleaved by rotation index so sequential `condition_assignment` values spread
  early participants across all rounds rather than clustering them on one round's
  cell choices.

Sequence count scales with `n_rounds` (`n_rounds * 16` sequences). It is chosen so
the total equals each study's target N (~20 observations per scenario x cell), so
every participant in a full sample gets a distinct scenario->condition mapping and
DataPipe completes one whole balanced pass. `n_rounds` must stay a multiple of 3
for the extra-cell pool to divide evenly across cells.

Per study (cells = product of the manipulated factors; intimacy/effort/desire are
omitted when that variable is inferred rather than given):
  - food_inv_desire   (1a): effort x intimacy x action = 2 x 4 x 3 = 24 cells, 30 rounds -> 480 seqs
  - food_inv_joint_de (1b): intimacy x action           = 4 x 3     = 12 cells, 15 rounds -> 240 seqs
  - food_inv_intimacy (2a): desire x effort x action    = 2 x 2 x 3 = 12 cells, 15 rounds -> 240 seqs
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

# Intimacy levels, ordered formal -> intimate. A purely verbal manipulation: the
# condition is identified by a slug (no numeric code is stored anywhere).
INTIMACY = ("max_formal", "neither", "somewhat_intimate", "max_intimate")

# Per-study design: the manipulated (participant-visible) factors, their levels,
# the number of rounds, and the RNG seed. `factors` and `levels` are aligned —
# a cell is `dict(zip(factors, cell_tuple))`.
STUDY_CONFIGS = {
    "food_inv_desire": {  # Study 1a — infer desire (effort + intimacy given)
        "seed": 313,
        "n_rounds": 30,  # 30 * 16 = 480 seqs = target N (24 cells -> 20 obs/cell)
        "factors": ("effort_condition", "intimacy_condition", "action_condition"),
        "levels": (("low", "high"), INTIMACY, ACTIONS),
    },
    "food_inv_joint_de": {  # Study 1b — joint desire + effort (intimacy given)
        "seed": 404,
        "n_rounds": 15,  # 15 * 16 = 240 seqs = target N (12 cells -> 20 obs/cell)
        "factors": ("intimacy_condition", "action_condition"),
        "levels": (INTIMACY, ACTIONS),
    },
    "food_inv_intimacy": {  # Study 2a — infer intimacy (desire + effort given)
        "seed": 2202,
        "n_rounds": 15,  # 15 * 16 = 240 seqs = target N (12 cells -> 20 obs/cell)
        "factors": ("desire_condition", "effort_condition", "action_condition"),
        "levels": (("low", "high"), ("low", "high"), ACTIONS),
    },
    "food_inv_joint_ie": {  # Study 2b — joint intimacy + effort (desire given)
        "seed": 404,
        "n_rounds": 6,  # 6 * 16 = 96 seqs (target N=120 isn't a multiple of 16)
        "factors": ("desire_condition", "action_condition"),
        "levels": (("low", "high"), ACTIONS),
    },
}


def smooth_order(cells):
    """Order a (multi)set of cells so every factor stays marginally flat along the
    list: greedily take the cell whose levels are currently least represented.

    Cells are tuples whose positions are the factors, so "flat" means each tuple
    position's value counts stay as equal as possible. Any contiguous chunk of the
    result is then near-balanced on every factor — which is what lets us chunk the
    extra-cell pool into rounds and get marginally balanced buckets, instead of
    leaving each bucket's marginals to a shuffle. Deterministic (no RNG): ties
    break on the first cell encountered, so a fixed cell set gives a fixed order.
    """
    remaining = list(cells)
    ordered = []
    run = Counter()  # (factor_position, level) -> count placed so far

    def score(cell):
        per_factor = [run[(i, v)] for i, v in enumerate(cell)]
        return (sum(per_factor), max(per_factor))

    while remaining:
        pick = min(remaining, key=score)
        ordered.append(pick)
        remaining.remove(pick)
        for i, v in enumerate(pick):
            run[(i, v)] += 1
    return ordered


def assign_cells_to_rounds(cells, n_rounds, n_slots, rng):
    """Split the cell set into `n_rounds` marginally-balanced multisets of
    `n_slots` cells.

    Every round holds `n_slots // n_cells` copies of every cell (a marginally
    neutral base), plus a share of the `n_slots % n_cells` extra cells per round.
    The extras are drawn from a pool that holds each cell exactly `extra_per_cell`
    times, ordered by `smooth_order` and chunked across rounds: this keeps each
    bucket's per-factor marginals at the arithmetic floor while still giving every
    cell an identical number of trial slots across all sequences. Round labels are
    permuted so the chunking order isn't visible in run order.
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

    cells = list(cells)
    pool = smooth_order(cells * extra_per_cell)  # each cell extra_per_cell times
    round_buckets = []
    for r in range(n_rounds):
        bucket = list(cells) * base_count
        bucket += pool[r * extra_per_round : (r + 1) * extra_per_round]
        round_buckets.append(bucket)
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

    # Within-participant marginal check: every factor's per-sequence level counts
    # must be at the arithmetic floor (0 if 16 divides the level count, else 1) —
    # the unavoidable residue and no more.
    worst = {}
    for fi, factor in enumerate(cfg["factors"]):
        levels = cfg["levels"][fi]
        floor = 0 if N_SLOTS % len(levels) == 0 else 1
        spread = max(
            (
                lambda c: (
                    max(c.get(l, 0) for l in levels) - min(c.get(l, 0) for l in levels)
                )
            )(Counter(trial[factor] for trial in seq))
            for seq in sequences
        )
        worst[factor] = spread
        if spread > floor:
            raise AssertionError(
                f"{slug}: within-participant '{factor}' spread {spread} exceeds floor {floor}"
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
    marg = ", ".join(f"{f}={worst[f]}" for f in cfg["factors"])
    print(
        f"{slug}: {len(sequences)} sequences x {len(sequences[0])} trials | "
        f"{len(cells)} cells x {per_cell} slots each (uniform) | "
        f"within-ppt marginal spread: {marg}"
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
