"""Centralized LM prompts for all Together AI calls in this project.

All system prompts and user-prompt formatters live here so they're easy to
compare and edit together. This module has no LM-call logic — see
`lm_scenario_params*.py`, `lm_action_priors*.py`, and `lm_generate_alternatives.py`
for that.

Three rating types share the same overall structure (preamble + intro line +
rating-specific body + JSON format block):

  - **access**: bodily-channel exposure of an action (used by all access
    tables; food / saliva / contact)
  - **effort**: physical / logistical cost of executing an action
  - **prior**: how natural / default an action is in the setting (used by
    the `_prior` actor variants)

Each rating type can be requested with a fixed action count (4 for the
canonical experiments, 2 for the effort experiments) or with a variable
action count (used by the no-alt alternative-scoring path, where the LM
sees a different number of LM-generated alternatives per cell).

A fourth call type, **alternatives generation**, has its own system prompt
and user-prompt formatter at the bottom of the file.

The prompts here are byte-identical to the strings the project has been
sending to the API; the parameterization just removes the duplication
between the canonical and effort scripts. Editing any prompt here will
change all callers — re-run the relevant LM script and downstream fits.
"""

# ==============================================================================
# Shared building blocks
# ==============================================================================

_PREAMBLE_RATING = (
    "You are a participant in a human study. Respond as if you were a "
    "regular adult, just going off of your intuition."
)

# Action priors use a slightly shorter preamble (no "just"). Preserved
# byte-identical to the existing prompt rather than unifying.
_PREAMBLE_PRIOR = (
    "You are a participant in a human study. Respond as if you were a "
    "regular adult, going off of your intuition."
)

_NUMBER_WORD = {
    2: "two",
    3: "three",
    4: "four",
}

# Sample JSON values used in the system prompt's example block. Different
# rating types use different sample values (preserved from the originals).
_JSON_EXAMPLE_VALUES = {
    "access": [0.5, 1.2, 3.8, 5.5],
    "effort": [0.5, 3.2, 2.1, 1.5],
    "prior": [0.5, 1.2, 3.8, 5.5],
}

# 2-action JSON examples differ from the first 2 entries of the 4-action
# examples (different illustrative values were used in the original prompts).
_JSON_EXAMPLE_VALUES_2 = {
    "access": [0.5, 3.8],
    "effort": [0.5, 3.2],
    "prior": [0.5, 3.8],
}


def _intro_line(n_actions):
    """Build the 'For each scenario, you will read about ...' intro line.

    Pass an int for fixed-length action sets, or None for the variable-length
    case used by no-alt alternative scoring.
    """
    if n_actions is None:
        return (
            "For each scenario, you will read about a set of alternative "
            "actions the two people could take. The number of actions varies."
        )
    return (
        f"For each scenario, you will read about {_NUMBER_WORD[n_actions]} "
        "different actions the two people can take."
    )


def _json_format_block(rating_type, n_actions):
    """Build the trailing 'Respond with your numerical ratings ...' block."""
    if n_actions is None:
        # Variable-length — examples shown for 3 actions.
        example = '{"action_0": 0.5, "action_1": 1.2, "action_2": 3.8}'
        if rating_type == "effort":
            example = '{"action_0": 0.5, "action_1": 3.2, "action_2": 2.1}'
        return (
            "Respond with your numerical ratings as a JSON object whose keys "
            'are "action_0", "action_1", ... matching the number of actions '
            "given, no explanation needed. Example for 3 actions:\n"
            f"{example}"
        )
    if n_actions == 2:
        vals = _JSON_EXAMPLE_VALUES_2[rating_type]
    else:
        vals = _JSON_EXAMPLE_VALUES[rating_type][:n_actions]
    keyvals = ", ".join(f'"action_{i}": {v}' for i, v in enumerate(vals))
    return (
        "Respond with your numerical ratings in this JSON format only, "
        "no explanation needed:\n"
        f"{{{keyvals}}}"
    )


# ==============================================================================
# Rating-type-specific bodies
# ==============================================================================
#
# Each body follows the canonical prompt verbatim. They're stored as constants
# so a careful reader can see exactly what the LM is being told to do for
# each rating type.

_ACCESS_BODY = """In this survey, you will read vignettes about two people sharing food in different situations. {INTRO}

For each action, evaluate: how much does this action create a direct bodily channel between the two people — a pathway for substances from one person's body to reach the other, or for their bodies to physically contact each other?

Consider concrete things like:
- Does any substance from one person's body (saliva, breath, skin oils) reach the other person or their food?
- Does the action involve direct physical contact between the two people's bodies?
- Does the action involve one person handling food that will then enter the other person's mouth?

Simply eating in the same physical space — for example, two people at the same table with fully separate portions — does NOT by itself create such a channel, and should be rated near zero.

Rate only what the action DOES in this physical sense — not how intimate or awkward it would feel in any particular relationship.

Use this scale from 0 to 6 (continuous values allowed):
0 = No bodily channel between the two people (complete physical separation)
3 = Indirect bodily channel (e.g. eating from the same shared container with separate utensils)
6 = Direct transfer of bodily substances (e.g. sharing the same piece of food that both bite)"""


_EFFORT_BODY = """In this survey, you will read vignettes about two people in a food-sharing situation. {INTRO}

For each action, evaluate the PHYSICAL AND LOGISTICAL COST of executing the action. Consider:
- How much physical work does the action require (preparing, serving, cutting, pouring, handing over)?
- Does the action need extra items or utensils (plates, napkins, cutlery, containers)?
- Does the action add practical steps beyond simply eating?

Do NOT rate social awkwardness or interpersonal discomfort — only the physical and logistical cost.

Use this scale from 0 to 6 (continuous values allowed):
0 = No effort (acting independently, eating what you already have)
3 = Moderate effort (a few steps, some preparation)
6 = High effort (many steps, substantial setup)"""


_PRIOR_BODY_4 = """In this survey, you will read vignettes about two people in a food-sharing situation. {INTRO}

For each action, evaluate how NATURAL or EXPECTED the action is as a "default" behavior in this setting — what you'd imagine typically happening given the food, the place, and the social occasion, independent of any specific information about the two people's relationship or how much they want the food.

Consider:
- Does the action fit the social conventions of this kind of setting?
- Is it a typical way people behave here, in general?

Do NOT factor in the specific relationship closeness between the two people, or how much they individually want the food. Just rate whether the action is a natural default for the setting itself.

Use this scale from 0 to 6 (continuous values allowed):
0 = Very unusual or out of place in this setting
3 = Plausible — could happen, neither default nor unusual
6 = Very natural — the typical default behavior in this setting"""


# 2-action prior body drops the "or how much they want the food" qualifiers
# (preserved verbatim from the existing 2-action prompt).
_PRIOR_BODY_2 = """In this survey, you will read vignettes about two people in a food-sharing situation. {INTRO}

For each action, evaluate how NATURAL or EXPECTED the action is as a "default" behavior in this setting — what you'd imagine typically happening given the food, the place, and the social occasion, independent of any specific information about the two people's relationship.

Consider:
- Does the action fit the social conventions of this kind of setting?
- Is it a typical way people behave here, in general?

Do NOT factor in the specific relationship closeness between the two people. Just rate whether the action is a natural default for the setting itself.

Use this scale from 0 to 6 (continuous values allowed):
0 = Very unusual or out of place in this setting
3 = Plausible — could happen, neither default nor unusual
6 = Very natural — the typical default behavior in this setting"""


# Per-rating-type instructions used in the user prompt (the line just above
# the numbered actions).
_USER_INSTRUCTIONS = {
    "access": (
        "Rate how much each action opens each person up to the other — "
        "physically, informationally, or both (0-6 scale):"
    ),
    "effort": (
        "Rate the physical and logistical cost of executing each action — "
        "how much physical work, preparation, or extra equipment is required "
        "(0-6 scale):"
    ),
    "prior": "Rate how natural / default each action is in this setting (0-6 scale):",
}


# ==============================================================================
# Public API: rating prompts (access / effort / prior)
# ==============================================================================


def system_prompt(rating_type, n_actions=None):
    """Build the system prompt for a rating call.

    rating_type: one of "access", "effort", "prior".
    n_actions: 2 or 4 for fixed-length rating, or None for the variable-length
        case used by no-alt alternative scoring. Note that "prior" doesn't have
        a variable-length variant (the prior is only ever asked for the
        canonical 2- or 4-action set).
    """
    if rating_type not in ("access", "effort", "prior"):
        raise ValueError(f"unknown rating_type: {rating_type}")
    if rating_type == "prior" and n_actions is None:
        raise ValueError("prior does not have a variable-length variant")

    intro = _intro_line(n_actions)
    if rating_type == "access":
        body = _ACCESS_BODY.format(INTRO=intro)
    elif rating_type == "effort":
        body = _EFFORT_BODY.format(INTRO=intro)
    else:  # prior
        body = (_PRIOR_BODY_2 if n_actions == 2 else _PRIOR_BODY_4).format(INTRO=intro)

    preamble = _PREAMBLE_PRIOR if rating_type == "prior" else _PREAMBLE_RATING
    json_block = _json_format_block(rating_type, n_actions)
    return f"{preamble}\n\n{body}\n\n{json_block}"


def user_prompt(rating_type, vignette, action_texts):
    """Build the user prompt for a rating call.

    vignette is whatever scene-description text the LM should see (the caller
    is responsible for choosing whether to include condition paragraphs like
    `effort_low` / `effort_high`).
    action_texts is an ordered list of action descriptions; they're rendered
    as "Action 0: ...", "Action 1: ...", etc.
    """
    if rating_type not in ("access", "effort", "prior"):
        raise ValueError(f"unknown rating_type: {rating_type}")
    instr = _USER_INSTRUCTIONS[rating_type]
    actions_block = "\n".join(f"Action {i}: {txt}" for i, txt in enumerate(action_texts))
    return f"Scenario: {vignette}\n\n{instr}\n\n{actions_block}"


# ==============================================================================
# Public API: alternative generation
# ==============================================================================


ALTERNATIVES_SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read a vignette about two people in a food-sharing situation. You will be told what action they took in the situation.

Your job is to list the set of plausible alternative actions the two people could have taken instead. Focus specifically on different WAYS the two people could handle and consume the food together — the mechanics of sharing. The alternatives should span a range of physical closeness / saliva-transfer risk: from not consuming the food at all or one person consuming it alone, to cutting or dividing separate portions, to ways that include increasing saliva-transfer risk (e.g., double dipping or biting from the same part of the food)

Generate however many alternatives you think are plausible, but no more than 10. Only include alternatives that are plausible in the specific situation; do not pad the list with implausible options. Do not include the action they actually took.

For each alternative, tag it with is_share ∈ {0, 1}:
- is_share = 1 if both people end up consuming the same food (whether from divided portions of the same dish, shared utensils, or the same piece of food)
- is_share = 0 if only one person consumes the food, or neither does (e.g. refusing, throwing it away, one person giving it all to the other)

Respond ONLY with a JSON array in this exact format, no explanation:
[
  {"action": "short description of alternative 1", "is_share": 0 or 1},
  {"action": "short description of alternative 2", "is_share": 0 or 1}
]"""


def alternatives_user_prompt(vignette, reward_text, observed_action_text):
    """Build the user prompt for the alternative-generation call (used by
    the no-alt experiment to generate counterfactuals)."""
    return (
        f"Scenario: {vignette}\n"
        f"{reward_text}\n\n"
        f"The two people took the following action:\n"
        f"{observed_action_text}\n\n"
        f"List the set of plausible alternative ways the two people could "
        f"have handled and consumed the food instead. Vary across physical "
        f"closeness / saliva-transfer risk. Tag each with is_share ∈ {{0, 1}}. "
        f"Do not include the action they actually took."
    )
