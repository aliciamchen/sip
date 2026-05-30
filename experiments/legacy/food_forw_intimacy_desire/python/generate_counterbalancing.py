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

all_intimacy = [
    (0, "low"),  # maximally formal
    (0, "low"),
    (0, "high"),
    (0, "high"),
    (50, "low"),
    (50, "low"),
    (50, "high"),
    (50, "high"),
    (75, "low"),
    (75, "low"),
    (75, "high"),
    (75, "high"),
    (100, "low"),
    (100, "low"),
    (100, "high"),
    (100, "high"),  # maximally intimate
]


def make_trial_sequence(story_list, intimacy_list):
    """Create a sequence of trials with story-intimacy pairs."""
    assert len(story_list) == len(intimacy_list)
    return list(
        map(
            lambda story, intimacy: {
                "scenario_label": story,
                "intimacy": intimacy[0],
                "reward": intimacy[1],
            },
            story_list,
            intimacy_list,
        )
    )


def make_counterbalancing_once(stories):
    """Create one counterbalanced sequence by rotating through all stories."""
    counterbalance_seq = []
    for trial_idx in range(len(stories)):
        # Rotate the stories list
        stories_temp = stories[trial_idx:] + stories[:trial_idx]
        this_trial_seq = make_trial_sequence(stories_temp, all_intimacy)
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
output_path = project_root / "experiments/food_forw_intimacy_desire/json/full_counterbalancing.json"

with open(output_path, "w") as f:
    json.dump(counterbalancing, f)

print(f"Generated {len(counterbalancing)} counterbalanced sequences")
print(f"Each sequence has {len(counterbalancing[0])} trials")
