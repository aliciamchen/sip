#!/usr/bin/env python3
"""LM scoring for the Study-1a schematic figure (the burrito running example).

This is a one-off, figure-only script (NOT part of the Make pipeline). The burrito
/ Mexican-food-truck scenario is the manuscript's illustrative example, not one of
the 16 data scenarios, so its action features can't be read from
`outputs/lm/food_inv_desire/lm_runs.jsonl`. Instead we score the four
hand-specified actions here, reusing the exact same prompts + Together client as
the real pipeline (`model/lm/`), and elicit the intimacy magnitude for the
"somewhat formal" relationship. A single deterministic run (fixed seed) is used so
the schematic is clean and reproducible.

Output: `figures/figure_data/figure_scores.json` — read by
`figures/figure_schematic_plots.py` (which never calls the API).

Requires TOGETHER_API_KEY in env or `.env`.
"""

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model" / "lm"))

from together import Together  # noqa: E402

from utils import get_project_root  # noqa: E402

# Reuse the real pipeline's LM infrastructure (model/lm/ on sys.path above).
from client import (  # noqa: E402
    MODEL_ID,
    get_ratings_concurrent,
    load_api_key,
    numeric_action_schema,
)
from prompts import (  # noqa: E402
    INTIMACY_SYSTEM_PROMPT,
    RELATIONSHIP_DESCRIPTORS,
    relationship_user_prompt,
    system_prompt,
)
from _features_dispatcher import (  # noqa: E402
    _max_tokens_for,
    format_effort_prompt_variable,
    format_g_prompt_variable,
    format_risk_prompt_variable,
    normalize_effort,
    normalize_g,
    normalize_risk,
    numeric_intimacy_schema,
    parse_action_response_variable,
    parse_intimacy_response,
)

SEED = 0  # single deterministic run
TEMPERATURE = 0.2  # scoring temperature (matches the pipeline)

# --- The burrito stimulus (from main.tex §Computational framework / the figure) -----
SCENARIO_LABEL = "burrito_figure"
RELATIONSHIP = "somewhat_formal"
DESIRE_OBJECT = "the burritos"

# Observer-visible scene for feature scoring: background + physical scene. Per the
# model, risk/effort/g are scored relationship-free (intimacy enters the utility
# separately via I), so the relationship paragraph is NOT included here.
VIGNETTE = (
    "Alice and Bob walk past Korean and Lebanese food trucks to reach a Mexican "
    "food truck. Once they arrive at the Mexican food truck, they realize that the "
    "burritos are very large and that neither of them is able to finish a whole "
    "burrito by themselves. The Mexican food truck has run out of knives. To get "
    "knives, they would have to walk to another food truck."
)

# Slot 0 is the observed action a_obs.
ACTIONS = [
    (
        "a_obs",
        "Alice and Bob walk to another food truck, ask for an extra knife, and cut "
        "the burrito in half. They each eat only from their own portion.",
    ),
    ("a_1", "Alice and Bob leave the food truck and go elsewhere."),
    ("a_2", "Alice and Bob take turns biting from the same burrito."),
    (
        "a_3",
        "Alice eats from the burrito until she gets full, and then she throws the "
        "rest away.",
    ),
]

# feature -> (system prompt, user-prompt builder, 0-6 -> [0,1] normalizer)
FEATURES = {
    "risk": (
        system_prompt("risk"),
        lambda v, acts: format_risk_prompt_variable(v, acts),
        normalize_risk,
    ),
    "effort": (
        system_prompt("effort"),
        lambda v, acts: format_effort_prompt_variable(v, acts),
        normalize_effort,
    ),
    "g": (
        system_prompt("g"),
        lambda v, acts: format_g_prompt_variable(v, acts, desire_object=DESIRE_OBJECT),
        normalize_g,
    ),
}


def score_features(client, action_texts):
    """Return {feature: [val_per_action]} normalized to [0, 1], single run."""
    n = len(action_texts)
    scores = {}
    for feat, (sys_p, build_user, normalize) in FEATURES.items():
        user_p = build_user(VIGNETTE, action_texts)
        successes, n_fail = get_ratings_concurrent(
            client,
            sys_p,
            user_p,
            lambda t: parse_action_response_variable(t, n),
            num_runs=1,
            max_tokens=_max_tokens_for(n),
            temperature=TEMPERATURE,
            response_format=numeric_action_schema(n),
            seed=SEED,
            label=f"{SCENARIO_LABEL}:{feat}",
        )
        if not successes:
            raise RuntimeError(
                f"LM returned no parseable {feat} scores ({n_fail} failures)"
            )
        raw = successes[0]  # {"action_0": v, ...}
        scores[feat] = [normalize(raw[f"action_{i}"]) for i in range(n)]
    return scores


def elicit_intimacy(client, level):
    """Return the [0, 1] intimacy magnitude for a relationship level, single run."""
    successes, n_fail = get_ratings_concurrent(
        client,
        INTIMACY_SYSTEM_PROMPT,
        relationship_user_prompt(RELATIONSHIP_DESCRIPTORS[level]),
        parse_intimacy_response,
        num_runs=1,
        max_tokens=64,
        temperature=TEMPERATURE,
        response_format=numeric_intimacy_schema(),
        seed=SEED,
        label=f"intimacy:{level}",
    )
    if not successes:
        raise RuntimeError(f"LM returned no parseable intimacy ({n_fail} failures)")
    return successes[0] / 100.0


def main():
    client = Together(api_key=load_api_key())
    keys = [k for k, _ in ACTIONS]
    texts = [t for _, t in ACTIONS]

    scores = score_features(client, texts)
    intimacy = elicit_intimacy(client, RELATIONSHIP)

    actions_out = []
    for i, (key, text) in enumerate(ACTIONS):
        actions_out.append(
            {
                "key": key,
                "action_text": text,
                "g": scores["g"][i],
                "effort": scores["effort"][i],
                "risk": scores["risk"][i],
            }
        )

    record = {
        "scenario_label": SCENARIO_LABEL,
        "vignette": VIGNETTE,
        "relationship": RELATIONSHIP,
        "intimacy": intimacy,
        "desire_object": DESIRE_OBJECT,
        "seed": SEED,
        "model_id": MODEL_ID,
        "temperature": TEMPERATURE,
        "actions": actions_out,
    }

    out_dir = get_project_root() / "figures" / "figure_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure_scores.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    # Sanity-check print.
    print(f"intimacy ({RELATIONSHIP}) = {intimacy:.3f}\n")
    print(f"{'action':6s}  {'g':>5s}  {'effort':>6s}  {'risk':>5s}   text")
    for key, a in zip(keys, actions_out):
        print(
            f"{key:6s}  {a['g']:5.2f}  {a['effort']:6.2f}  {a['risk']:5.2f}   "
            f"{a['action_text'][:54]}…"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
