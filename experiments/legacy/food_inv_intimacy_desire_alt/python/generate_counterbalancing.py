import json
import random
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root

random.seed(111)

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

all_conditions = [
    ("action_0", "low"),  # maximally formal
    ("action_0", "low"),
    ("action_0", "high"),
    ("action_0", "high"),
    ("action_1", "low"),
    ("action_1", "low"),
    ("action_1", "high"),
    ("action_1", "high"),
    ("action_2", "low"),
    ("action_2", "low"),
    ("action_2", "high"),
    ("action_2", "high"),
    ("action_3", "low"),
    ("action_3", "low"),
    ("action_3", "high"),
    ("action_3", "high"),
]


def make_trial_sequence(story_list, condition_list):
    """Create a sequence of trials with story-intimacy pairs."""
    assert len(story_list) == len(condition_list)
    return list(
        map(
            lambda story, condition: {
                "scenario_label": story,
                "action": condition[0],
                "reward": condition[1],
            },
            story_list,
            condition_list,
        )
    )


def make_counterbalancing_once(stories):
    """Create one counterbalanced sequence by rotating through all stories."""
    counterbalance_seq = []
    for trial_idx in range(len(stories)):
        # Rotate the stories list
        stories_temp = stories[trial_idx:] + stories[:trial_idx]
        this_trial_seq = make_trial_sequence(stories_temp, all_conditions)
        counterbalance_seq.append(this_trial_seq)
    return counterbalance_seq


first_sixteen = make_counterbalancing_once(all_stories)
random.shuffle(all_stories)
second_sixteen = make_counterbalancing_once(all_stories)
random.shuffle(all_stories)
third_sixteen = make_counterbalancing_once(all_stories)
random.shuffle(all_stories)
fourth_sixteen = make_counterbalancing_once(all_stories)
random.shuffle(all_stories)
fifth_sixteen = make_counterbalancing_once(all_stories)
random.shuffle(all_stories)
sixth_sixteen = make_counterbalancing_once(all_stories)
random.shuffle(all_stories)
seventh_sixteen = make_counterbalancing_once(all_stories)
random.shuffle(all_stories)
eighth_sixteen = make_counterbalancing_once(all_stories)
random.shuffle(all_stories)
ninth_sixteen = make_counterbalancing_once(all_stories)
random.shuffle(all_stories)
tenth_sixteen = make_counterbalancing_once(all_stories)
random.shuffle(all_stories)
eleventh_sixteen = make_counterbalancing_once(all_stories)
random.shuffle(all_stories)
twelfth_sixteen = make_counterbalancing_once(all_stories)

counterbalancing = (
    first_sixteen
    + second_sixteen
    + third_sixteen
    + fourth_sixteen
    + fifth_sixteen
    + sixth_sixteen
    + seventh_sixteen
    + eighth_sixteen
    + ninth_sixteen
    + tenth_sixteen
    + eleventh_sixteen
    + twelfth_sixteen
)

# Get project root and construct the output path
project_root = get_project_root()
output_path = project_root / "experiments/food_inv_intimacy_desire_alt/json/full_counterbalancing.json"

with open(output_path, "w") as f:
    json.dump(counterbalancing, f)

print(f"Generated {len(counterbalancing)} counterbalanced sequences")
print(f"Each sequence has {len(counterbalancing[0])} trials")
