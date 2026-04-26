"""Centralized LM prompts for all Together AI calls in this project.

All system prompts and user-prompt formatters live here so they're easy to
compare and edit together. This module has no LM-call logic — see
`lm_scenario_params*.py` and `lm_generate_alternatives.py` for that.

Two rating types share the same overall structure (preamble + intro line +
rating-specific body + JSON format block):

  - **access**: bodily, physical-contact, or informational exposure of an
    action between the two people in the scenario.
  - **effort**: physical, logistical, and time cost of executing an action.

Each rating type can be requested with a fixed action count (4 for the
canonical experiments, 2 for the effort experiments) or with a variable
action count (used by the no-alt alternative-scoring path, where the LM
sees a different number of LM-generated alternatives per cell).

A third call type, **alternatives generation**, has its own system prompt
and user-prompt formatter at the bottom of the file.

The prompts are domain-general: they cover food sharing as well as the
three non-food sub-types in `scenarios_nonfood.csv` — *substance*
(chapstick, towel, hairbrush, harmonica, sunscreen), *space* (blanket,
sleeping-bag, bed, locker-room, sauna), and *privacy* (breakup, payment,
gossip, home, navigation). The access rubric covers three channel types
(bodily-substance, direct physical-contact, informational/private-resource);
the effort rubric covers physical work, equipment/setup, time cost, and
coordination/bookkeeping. The original food-only prompts were retired in
favor of this single set after a side-by-side comparison showed the
unified prompts produced equal or slightly better fits on the food data.

Note that privacy-type scenarios tend to involve little physical or
logistical cost variation, so effort ratings on those scenarios cluster
near zero by construction.

Editing any prompt here will change all callers — re-run the relevant LM
script and downstream fits.
"""

# ==============================================================================
# Shared building blocks
# ==============================================================================

_PREAMBLE_RATING = (
    "You are a participant in a human study. Respond as if you were a "
    "regular adult, just going off of your intuition."
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
}

# 2-action JSON examples differ from the first 2 entries of the 4-action
# examples (different illustrative values were used in the original prompts).
_JSON_EXAMPLE_VALUES_2 = {
    "access": [0.5, 3.8],
    "effort": [0.5, 3.2],
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

_ACCESS_BODY = """In this survey, you will read vignettes about two people in different situations where some resource — food, an object, a physical space, or a piece of information — could be shared between them. {INTRO}

For each action, evaluate: how much does this action create a channel between the two people — a pathway for something to move from one person's side to the other, in a way that exposes one person to the other.

Consider the kinds of channels that might open up:
- Bodily-substance channels: substances from one person's body (saliva, breath, skin oils, sweat) reach the other person, or come into contact with something the other person will then use.
- Direct physical-contact channels: the two people's bodies physically touch each other, or come into very close proximity.
- Informational or private-resource channels: private information, sensitive personal details, or personal resources from one person become accessible to the other.

Simply being in the same physical space with no exchange — for example, two people each handling their own separate things, or keeping conversation to surface-level topics — does NOT by itself create such a channel, and should be rated near zero.

Rate only what the action DOES in this physical or informational sense — not how intimate or awkward it would feel in any particular relationship.

Use this scale from 0 to 6 (continuous values allowed):
0 = No channel between the two people (they remain fully separate)
3 = Indirect or limited channel (e.g. using a shared item with a barrier or after cleaning, sitting near each other without touching, sharing surface-level information)
6 = Direct channel (e.g. direct bodily-substance transfer, skin-to-skin contact, sharing private or sensitive personal details)"""


_EFFORT_BODY = """In this survey, you will read vignettes about two people in different situations where some resource — food, an object, a physical space, or a piece of information — could be shared between them. {INTRO}

For each action, evaluate the PHYSICAL, LOGISTICAL, AND TIME COST of executing the action. Consider:
- How much physical work does the action require (preparing, serving, cutting, pouring, handing over, cleaning, wiping, drying, tidying, rearranging, applying)?
- Does the action need extra items, equipment, or setup (utensils, plates, containers, sanitizing supplies, barriers, separate furniture, separate spaces)?
- How much time does the action take (waiting for something to dry, taking turns one at a time, sequential rather than simultaneous use, an extended conversation)?
- Does the action add coordination or bookkeeping steps (handing items back and forth repeatedly, tracking amounts owed, repeating a step, going somewhere else first)?

Do NOT rate social awkwardness or interpersonal discomfort — only the physical, logistical, and time cost of carrying the action out.

Use this scale from 0 to 6 (continuous values allowed):
0 = No effort (acting independently or doing the simplest direct thing, no extra steps or waiting)
3 = Moderate effort (a few steps, some preparation, some waiting, or a few extra items needed)
6 = High effort (many steps, substantial setup, significant time, or repeated coordination)"""


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
}


_BODIES = {
    "access": _ACCESS_BODY,
    "effort": _EFFORT_BODY,
}


# ==============================================================================
# Public API: rating prompts (access / effort)
# ==============================================================================


def system_prompt(rating_type, n_actions=None):
    """Build the system prompt for a rating call.

    rating_type: one of "access", "effort".
    n_actions: 2 or 4 for fixed-length rating, or None for the variable-length
        case used by no-alt alternative scoring.
    """
    if rating_type not in _BODIES:
        raise ValueError(f"unknown rating_type: {rating_type}")
    intro = _intro_line(n_actions)
    body = _BODIES[rating_type].format(INTRO=intro)
    json_block = _json_format_block(rating_type, n_actions)
    return f"{_PREAMBLE_RATING}\n\n{body}\n\n{json_block}"


def user_prompt(rating_type, vignette, action_texts):
    """Build the user prompt for a rating call.

    vignette is whatever scene-description text the LM should see (the caller
    is responsible for choosing whether to include condition paragraphs like
    `effort_low` / `effort_high`).
    action_texts is an ordered list of action descriptions; they're rendered
    as "Action 0: ...", "Action 1: ...", etc.
    """
    if rating_type not in _USER_INSTRUCTIONS:
        raise ValueError(f"unknown rating_type: {rating_type}")
    instr = _USER_INSTRUCTIONS[rating_type]
    actions_block = "\n".join(f"Action {i}: {txt}" for i, txt in enumerate(action_texts))
    return f"Scenario: {vignette}\n\n{instr}\n\n{actions_block}"


# ==============================================================================
# Public API: alternative generation
# ==============================================================================


ALTERNATIVES_SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them. You will be told what action they took in the situation.

Your job is to list the set of plausible alternative actions the two people could have taken instead. Focus specifically on different WAYS the two people could handle the situation — the mechanics of sharing or not sharing. The alternatives should span a range of exposure between the two people: from one person handling things entirely alone or no one engaging at all, through ways that involve some separation, distance, or barrier between them, to ways that bring them into closer bodily, physical, or informational contact (e.g., direct skin or saliva contact via the shared item, bodies touching, or sharing private/sensitive details).

Generate however many alternatives you think are plausible, but no more than 10. Only include alternatives that are plausible in the specific situation; do not pad the list with implausible options. Do not include the action they actually took.

For each alternative, tag it with is_share ∈ {0, 1}:
- is_share = 1 if both people end up engaging with the shared resource together (whether through divided portions, shared use of the same item, shared physical space, or mutual disclosure of information)
- is_share = 0 if only one person engages with it, or neither does (e.g. refusing, abandoning the resource, one person handling it entirely alone, or keeping the topic off-limits)

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
        "List the set of plausible alternative ways the two people could "
        "have handled the situation instead. Vary across exposure between "
        "them — bodily, physical, or informational. Tag each with "
        "is_share ∈ {0, 1}. Do not include the action they actually took."
    )
