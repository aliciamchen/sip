import json
import random

random.seed(111)

all_stories = [
    "basketball", "birthday", "brunch", "conference", 
    "cooking", "crabs", "dip", "drinks",
    "driving", "fair", "gala", "hike",
    "oysters", "social", "soup", "wedding"
]

all_conditions = [
    "not_close",      # Level 1
    "not_close",      # Level 1
    "not_close",      # Level 1
    "not_close",      # Level 1
    "somewhat_close", # Level 2
    "somewhat_close", # Level 2
    "somewhat_close", # Level 2
    "somewhat_close", # Level 2
    "close",          # Level 3
    "close",          # Level 3
    "close",          # Level 3
    "close",          # Level 3
    "extremely_close", # Level 4
    "extremely_close", # Level 4
    "extremely_close", # Level 4
    "extremely_close", # Level 4
]

def make_trial_sequence(story_list, condition_list):
    """Create a sequence of trials with story-condition pairs."""
    assert len(story_list) == len(condition_list)
    return list(
        map(
            lambda story, condition: {"scenario_label": story, "closeness": condition},
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

counterbalancing = first_sixteen + second_sixteen + third_sixteen + fourth_sixteen + fifth_sixteen + sixth_sixteen + seventh_sixteen + eighth_sixteen

with open("/Users/aliciachen/Dropbox/projects/saliva-inverse-planning/experiments/discomfort/json/full_counterbalancing.json", "w") as f:
    json.dump(counterbalancing, f)

print(f"Generated {len(counterbalancing)} counterbalanced sequences")
print(f"Each sequence has {len(counterbalancing[0])} trials")

