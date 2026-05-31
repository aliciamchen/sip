"""Centralized LM prompts for all Together AI calls in this project.

All system prompts and user-prompt formatters live here so they're easy to
compare and edit together. This module has no LM-call logic — see
`lm/score_canonical_features.py` and `lm/generate_alternatives_motivation.py` for that.

Three rating types share the same overall structure (preamble + intro line +
rating-specific body + JSON format block):

  - **access**: bodily, physical-contact, or informational exposure of an
    action between the two people in the scenario.
  - **effort**: physical, logistical, and time cost of executing an action.
  - **v**: signed valence of an action with respect to the actor's
    motivational state — strongly counterproductive (-3) to strongly
    serving the state (+3). Requires a `state` paragraph (the scenario's
    `reward_high` or `reward_low` text) at call time, since the same
    action receives different valences under different states.

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
    "v": [-1.5, 0.5, 2.0, 2.8],
    "g": [0.5, 3.0, 5.5, 4.0],
}

# 2-action JSON examples differ from the first 2 entries of the 4-action
# examples (different illustrative values were used in the original prompts).
_JSON_EXAMPLE_VALUES_2 = {
    "access": [0.5, 3.8],
    "effort": [0.5, 3.2],
    "v": [-1.5, 2.0],
    "g": [0.5, 5.5],
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
        elif rating_type == "v":
            example = '{"action_0": -1.5, "action_1": 0.5, "action_2": 2.8}'
        elif rating_type == "g":
            example = '{"action_0": 0.5, "action_1": 3.0, "action_2": 5.5}'
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

# _ACCESS_BODY draws on four established literatures. The prompt body itself
# stays jargon-free (the LM is prompted "as a participant"), but the
# conceptual content of each channel is grounded as follows; cite these in
# the manuscript when defending the construct.
#
#   - Substance-transmission channels — Rozin, P. & Fallon, A. E. (1987).
#     "A perspective on disgust." Psychological Review 94(1): 23–41.
#     Establishes contamination via bodily-substance transfer as the core
#     domain of disgust; even brief contact transmits.
#
#   - Direct-contact channels — Suvilehto, J. T., Glerean, E., Dunbar, R. I. M.,
#     Hari, R., & Nummenmaa, L. (2015). "Topography of social touching depends
#     on emotional bonds between humans." PNAS 112(45): 13811–13816.
#     Body-map permissions for touch are graded by relational closeness across
#     cultures; grounds why contact extent and body region both matter.
#
#   - Informational / disclosure channels — Reis, H. T. & Shaver, P. (1988).
#     "Intimacy as an interpersonal process." In S. Duck (Ed.), Handbook of
#     Personal Relationships, Wiley. Self-disclosure + partner responsiveness
#     defines intimacy at the level of individual interactions; co-presence
#     without disclosure does not.
#
#   - Project-specific anchor — Thomas, A. J., Woo, B., Nettle, D., Spelke, E.,
#     & Saxe, R. (2022). "Early concepts of intimacy: Young humans use saliva
#     sharing to infer close relationships." Science 375(6578): 311–315.
#     Direct evidence that saliva-sharing is read as a thick-relationship cue,
#     separable from other positive social interactions.

_ACCESS_BODY = """In this survey, you will read vignettes about two people in different situations where some resource — food, an object, a physical space, or a piece of information — could be shared between them. {INTRO}

For each action, evaluate: how much does this action open one person up to the other? Three kinds of opening are possible — each is a channel through which something passes from one person's side to the other:

- Substance-transmission channels: bodily substances (saliva, breath, skin oils, sweat) from one person reach the other, either directly or via a shared vessel or item that's been on the first person's body. Even brief contact counts — the substance doesn't have to remain visible for the transmission to be real.
- Direct-contact channels: the two people's bodies physically touch each other, or come into very close proximity. The extent of contact and the body region involved both matter — brief incidental touch is a small channel; sustained skin contact, or contact with body regions normally restricted to close relationships, is a larger channel.
- Informational or private-resource channels: private information, sensitive personal details, or personal resources (a private space, a personal item, a confidential record) from one person become accessible to the other — content that someone would not share with a stranger or a passing acquaintance.

Co-presence without substance transmission, contact, or disclosure does NOT by itself create a channel — for example, two people each handling their own separate utensils, sitting in the same room without interacting, or keeping a conversation to surface-level topics. These should be rated near zero.

Rate the action's physical, contact, or informational opening — the channel itself — not how emotionally intimate or awkward it would feel. The emotional reading depends on who the two people are; here we're asking what the action does, independent of relationship.

Use this scale from 0 to 6 (continuous values allowed):
0 = No channel between the two people (they remain fully separate; the action involves no exchange of substance, contact, or disclosure)
3 = Indirect or limited channel (e.g. using a shared item after cleaning or with a barrier, sitting near each other without touching, sharing surface-level information anyone could ask about)
6 = Direct channel (e.g. direct bodily-substance transfer such as mouth-to-mouth contact or sharing a utensil that's been in one person's mouth, sustained skin-to-skin contact, or sharing private details one would not disclose to a stranger)"""


# _EFFORT_BODY is grounded in the Naïve Utility Calculus (NUC) framework
# and scoped to physical effort — motor work, equipment / preparation,
# and time. The construct does not extend to coordination or other
# cognitive cost types; this is a scope choice grounded in the physical-
# cost-only character of the empirical NUC literature, not an active
# exclusion called out in the prompt itself (the prompt simply doesn't
# list coordination as a criterion — telling the LM to ignore it would
# prime the concept). The prompt body stays jargon-free; the rating
# dimension is anchored as follows.
#
#   - Conceptual anchor (cost as trade-off quantity) — Jara-Ettinger, J.,
#     Gweon, H., Schulz, L. E., & Tenenbaum, J. B. (2016). "The naïve
#     utility calculus: Computational principles underlying commonsense
#     psychology." Trends in Cognitive Sciences 20(8): 589–604. Defines
#     cost formally as what an agent weighs against reward — the framework
#     this project's inverse-planning model instantiates.
#
#   - Single-scalar physical cost (integrating across physical sub-types)
#     — Liu, S., Ullman, T. D., Tenenbaum, J. B., & Spelke, E. S. (2017).
#     "Ten-month-old infants infer the value of goals from the costs of
#     actions." Science 358(6366): 1038–1041. Showed that observers
#     integrate distinct physical cost features (height, width, incline)
#     into one abstract cost metric — directly grounds collapsing motor,
#     equipment, and time onto one 0-6 scale and motivates restricting
#     the construct to physical cost features (rather than cognitive ones).
#
#   - Effort as a perceptible quantity separable from reward —
#     Jara-Ettinger, J., Gweon, H., Tenenbaum, J. B., & Schulz, L. E.
#     (2015). "Children's understanding of the costs and rewards underlying
#     rational action." Cognition 140: 14–23. Establishes that children at
#     4–6 can estimate action cost as distinct from goal value and agent
#     competence — grounds the assumption that an "LM-as-participant" can
#     rate physical effort with the instruction below.

_EFFORT_BODY = """In this survey, you will read vignettes about two people in different situations where some resource — food, an object, a physical space, or a piece of information — could be shared between them. {INTRO}

For each action, evaluate the *physical* cost the actor would weigh against the benefit of the action — the bodily, material, and temporal cost of carrying it out. The three cost types below all count; integrate across them into a single rating:

- Physical motor cost: how much bodily work the action requires (preparing, serving, cutting, pouring, handing over, cleaning, wiping, drying, tidying, rearranging, applying).
- Equipment and preparation cost: whether the action needs extra items or setup (utensils, plates, containers, sanitizing supplies, barriers, separate furniture, separate spaces) that someone has to obtain, set up, or take care of.
- Time cost: how long the action takes — waiting for something to dry, sequential rather than simultaneous use, an extended preparation.

Do NOT rate social awkwardness, relational discomfort, or how intimate or appropriate the action would feel — those are separate dimensions handled by other questions in this study. Here we want only the physical effort of carrying the action out.

Use this scale from 0 to 6 (continuous values allowed):
0 = No physical effort (acting independently or doing the simplest direct thing — no bodily work beyond the basic motion, no extra items, no waiting)
3 = Moderate physical effort (a few bodily steps, such as setting out a clean utensil, dividing a portion, or briefly waiting; or a small handful of extra items to obtain)
6 = High physical effort (many bodily steps, substantial setup, or significant time — for example, leaving to obtain something from far away and returning, preparing food from scratch, or cleaning and assembling many separate items)"""


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
    "v": (
        "Rate how each action affects the actor in their motivational state "
        "— from -3 (strongly counterproductive for the state) through 0 "
        "(neutral) to +3 (strongly serves the state):"
    ),
    "g": (
        "Rate how much each action results in the two people actually getting "
        "or consuming the thing at stake (0-6 scale):"
    ),
}


# _V_BODY is the reward component of the project's Bayesian inverse-planning
# model — the signed valence of an action with respect to the actor's
# motivational state. The prompt body stays jargon-free, but the rating
# dimension draws on the following established literatures:
#
#   - Formal anchor (V as reward in inverse planning) — Baker, C. L.,
#     Saxe, R., & Tenenbaum, J. B. (2009). "Action understanding as
#     inverse planning." Cognition 113(3): 329–349. The foundational
#     Bayesian model where action understanding is treated as inverse
#     inference over a (goal, reward, cost) model of the actor. V in
#     this project is literally the reward in that formalism.
#
#   - Reward as signed and separable from cost — Jara-Ettinger, J.,
#     Gweon, H., Schulz, L. E., & Tenenbaum, J. B. (2016). "The naïve
#     utility calculus: Computational principles underlying commonsense
#     psychology." Trends in Cognitive Sciences 20(8): 589–604. Grounds
#     the −3 to +3 range and the formal distinction between "irrelevant
#     action" (reward = 0) and "thwarting action" (reward < 0).
#
#   - Teleological interpretation of actions — Gergely, G. & Csibra, G.
#     (2003). "Teleological reasoning in infancy: The naïve theory of
#     rational action." Trends in Cognitive Sciences 7(7): 287–292.
#     Foundational claim that observers represent actions in relation
#     to goal-states under a principle of rational action — grounds the
#     question "does this action serve what the actor wants?" as a
#     psychologically natural one to ask.
#
#   - Diverse desires as a basic, early-emerging mental state attribution
#     — Wellman, H. M. & Liu, D. (2004). "Scaling of theory-of-mind tasks."
#     Child Development 75(2): 523–541. "Diverse Desires" is the FIRST
#     step in their ToM developmental scale — before belief understanding.
#     Grounds the claim that conditional-on-motivational-state valence
#     judgments are a basic capacity the LM-as-participant can perform.
#
#   - Observers infer reward from action choices — Liu, S., Ullman, T. D.,
#     Tenenbaum, J. B., & Spelke, E. S. (2017). "Ten-month-old infants
#     infer the value of goals from the costs of actions." Science 358:
#     1038–1041. Empirical anchor that V exists as a separable, inferable
#     quantity — observers reason about reward from choices even in infancy.

_V_BODY = """In this survey, you will read vignettes about two people in different situations where some resource — food, an object, a physical space, or a piece of information — could be shared between them. {INTRO}

For each scenario, one of the two people is in a particular motivational state — for example, wanting something (hungry, in pain, in urgent need) or wanting to avoid something (full, comfortable, wanting privacy). The state will be given to you explicitly. The same action can serve one motivational state and thwart another — your rating should be conditional on the state you are given, not on the actor's overall well-being or what you yourself would prioritize.

For each action, evaluate how that action affects the actor *given the state they are in*. Does the action serve what the actor wants in this moment? Or does it actively work against it?

Use this scale from -3 to +3 (continuous values allowed):
+3 = Strongly serves the state (the action straightforwardly fulfills what the actor needs or wants)
+1 = Mildly helps (the action partially satisfies the state)
 0 = Neutral (the action neither helps nor harms — it's irrelevant to the active state, or it neither achieves nor violates the goal)
-1 = Mildly counterproductive (the action partially works against the state, e.g. eating a small bite when full, declining a small amount of needed help)
-3 = Strongly counterproductive (the action actively makes the state worse — e.g. eating heartily when already painfully full, refusing urgently needed information when the actor is in distress)

Important: "doesn't help" and "actively harms" are different things. An action that simply fails to address the state — that's irrelevant to what the actor wants — should be rated near 0. Reserve negative ratings for actions that actively move the actor away from the state they want (eating when full, sharing when wanting privacy, etc.).

Do NOT rate how intimate or awkward the action would feel, and do NOT rate the physical effort of carrying it out — those are separate dimensions handled by other questions in this study. Here we want only how the action sits with the actor's current motivational state."""


# _G_BODY is the goal-satisfaction component of the reward term. In the
# continuous-desire model the reward enters the utility as w_v · desire · g(a|s),
# where desire is the latent magnitude (how much the dyad wants the outcome) and
# g(a|s) is this desire-free rating of how fully the action delivers the outcome.
# Splitting reward this way is what lets desire be inferred as a continuous
# latent: g is a stable, elicitable property of the action, while desire is the
# free quantity the observer recovers (or, in the given-desire studies, the
# scalar rated by `desire_user_prompt`). g replaces the old signed-valence V.
_G_BODY = """In this survey, you will read vignettes about two people in different situations where some resource — food, an object, a physical space, or a piece of information — could be shared between them. {INTRO}

For each action, evaluate how much it results in the two people actually obtaining or consuming the thing at stake in the scenario — the food they could eat, the object they could use, the space they could occupy, the information they could learn. This is about whether the action delivers the goal, NOT about how much the people want it (a separate question), and NOT about the physical effort or the interpersonal exposure the action involves (also separate questions).

An action that ends with both people getting and consuming the thing should be rated high; an action where they forgo it, abandon it, or only one person gets it should be rated low. How they get it — directly, or via a safer indirect route — does not matter here; only how fully they end up with it.

Use this scale from 0 to 6 (continuous values allowed):
0 = The thing is not obtained (the action forgoes or abandons it)
3 = Partially obtained (a reduced portion, only one person, or an incomplete version)
6 = Fully obtained (both people end up getting and consuming the thing)"""


_BODIES = {
    "access": _ACCESS_BODY,
    "effort": _EFFORT_BODY,
    "v": _V_BODY,
    "g": _G_BODY,
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


def user_prompt(rating_type, vignette, action_texts, state=None):
    """Build the user prompt for a rating call.

    vignette is whatever scene-description text the LM should see (the caller
    is responsible for choosing whether to include condition paragraphs like
    `effort_low` / `effort_high`).
    action_texts is an ordered list of action descriptions; they're rendered
    as "Action 0: ...", "Action 1: ...", etc.
    state is the actor's motivational-state paragraph (e.g. `reward_high` or
    `reward_low`). Required for rating_type="v"; ignored for access/effort.
    """
    if rating_type not in _USER_INSTRUCTIONS:
        raise ValueError(f"unknown rating_type: {rating_type}")
    if rating_type == "v" and state is None:
        raise ValueError("rating_type='v' requires a `state` paragraph")
    instr = _USER_INSTRUCTIONS[rating_type]
    actions_block = "\n".join(
        f"Action {i}: {txt}" for i, txt in enumerate(action_texts)
    )
    if rating_type == "v":
        return f"Scenario: {vignette}\n\nState: {state}\n\n{instr}\n\n{actions_block}"
    return f"Scenario: {vignette}\n\n{instr}\n\n{actions_block}"


# ==============================================================================
# Public API: alternative generation
# ==============================================================================


# ALTERNATIVES_SYSTEM_PROMPT is the methodological core of this project's
# open-world inverse-planning move: rather than reasoning over a fixed
# action set, the LM proposes a small, scenario-specific set of plausible
# counterfactual actions that then feed into the formal inverse-planning
# model with their LM-elicited utility features (access, effort, V). The
# prompt body stays jargon-free; the methodological choice is anchored
# as follows.
#
#   - Neuro-symbolic / model-synthesis approach — Wong et al. (2025).
#     "Modeling Open-World Cognition as On-Demand Synthesis of
#     Probabilistic Models." CogSci 2025 (eScholarship). The foundational
#     methodological anchor for using a language model to propose a
#     contextual symbolic model on demand, combining LM distributional
#     knowledge with formal probabilistic inference. Cited in the
#     manuscript as `wong2025modeling`.
#
#   - Frame-problem motivation — Dennett, D. C. (1984). "Cognitive wheels:
#     The frame problem of AI." In C. Hookway (Ed.), Minds, machines, and
#     evolution. Cambridge University Press. The classic philosophical
#     statement of why an observer cannot reason over all possible actions
#     but must construct a smaller, context-sensitive comparison set —
#     what this prompt operationalizes.
#
# The explicit "preserve the scenario's central goal" paragraph and its
# negative examples address an empirical failure mode: an earlier
# underspecified version of this prompt produced scenario-shifting
# alternatives ("find a different food vendor," "pay with a different
# method") that the formal inverse-planning model could not consume
# sensibly. The current wording rules those out at generation time.

ALTERNATIVES_SYSTEM_PROMPT = """You are a participant in a human study. Respond as if you were a regular adult, just going off of your intuition.

In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them. You will be told what action they took in the situation.

Your job is to list the alternative actions that would come to mind to a reasonable person in this situation — the set of options you think the two people were realistically choosing between. The alternatives should be things the two people could have done at the moment they chose the observed action — not changes to decisions they had already made earlier in the scenario.

Aim for a small, focused set — typically 3 to 5 strong alternatives. Up to 10 is allowed only if you're confident each one is salient. If you're not confident an alternative is something the people would realistically consider — not just something technically possible — leave it out. Better to return fewer strong alternatives than to pad the list. Do not include the action they actually took.

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
        "have handled the situation instead. Tag each with "
        "is_share ∈ {0, 1}. Do not include the action they actually took."
    )


# Relationship-condition descriptors used by the desire-noalt observer's
# alternative-generation pass. Mirrors the slider labels shown to participants
# in `experiments/food_inv_desire_intimacy_noalt/trials.js` and conveys the relationship
# context to the LM the same way it's conveyed to humans (numeric label +
# short qualitative descriptor).
RELATIONSHIP_DESCRIPTORS = {
    0: "0 out of 100 (maximally formal — e.g., the kind of relationship one might have with a new acquaintance, a shopkeeper, or a religious leader)",
    50: "50 out of 100 (neither formal nor intimate — e.g., the kind of relationship one might have with a casual friend or a coworker)",
    75: "75 out of 100 (somewhat intimate — e.g., the kind of relationship one might have with a close friend)",
    100: "100 out of 100 (maximally intimate — e.g., the kind of relationship one might have with a romantic partner or best friend)",
}


def alternatives_user_prompt_relationship(
    vignette, relationship_level, observed_action_text
):
    """Build the user prompt for the alternative-generation call when
    conditioning on relationship instead of motivation (desire-noalt observer).

    `relationship_level` is one of {0, 50, 75, 100}, matching the experiment's
    intimacy slider conditions.
    """
    descriptor = RELATIONSHIP_DESCRIPTORS[relationship_level]
    return (
        f"Scenario: {vignette}\n"
        f"The two people are in a relationship they would describe as "
        f"{descriptor}.\n\n"
        f"The two people took the following action:\n"
        f"{observed_action_text}\n\n"
        "List the set of plausible alternative ways the two people could "
        "have handled the situation instead. Tag each with "
        "is_share ∈ {0, 1}. Do not include the action they actually took."
    )


def alternatives_user_prompt_3act(
    vignette,
    observed_action_text,
    *,
    effort_text=None,
    intimacy_level=None,
    reward_text=None,
):
    """Build the user prompt for the alternative-generation call in the 3-action
    inverse experiments (Studies 1a, 1b, 2a, 2b).

    Composes whichever observer-visible condition paragraphs the experiment
    reveals. Each study passes only the paragraphs its observer actually sees:

      - Study 1a (`food_inv_desire`):     effort_text + intimacy_level
      - Study 1b (`food_inv_joint_de`):   intimacy_level
      - Study 2a (`food_inv_intimacy`):   reward_text + effort_text
      - Study 2b (`food_inv_joint_ie`):   reward_text

    Mirrors how the human participant sees the trial (vignette + revealed
    condition paragraphs + observed action), per `feedback_llm_as_participant`.
    `intimacy_level` is one of {0, 50, 75, 100} when provided; it's rendered
    via the shared `RELATIONSHIP_DESCRIPTORS` dict so the LM sees the same
    qualitative descriptor humans see.
    """
    parts = [f"Scenario: {vignette}"]
    if reward_text is not None:
        parts.append(reward_text)
    if effort_text is not None:
        parts.append(effort_text)
    if intimacy_level is not None:
        parts.append(
            f"The two people are in a relationship they would describe as "
            f"{RELATIONSHIP_DESCRIPTORS[intimacy_level]}."
        )
    parts.append(
        f"\nThe two people took the following action:\n{observed_action_text}\n"
    )
    parts.append(
        "List the set of plausible alternative ways the two people could "
        "have handled the situation instead. Tag each with "
        "is_share ∈ {0, 1}. Do not include the action they actually took."
    )
    return "\n".join(parts)


# ==============================================================================
# Public API: scenario-level desire scalar (given-desire studies 2a, 2b)
# ==============================================================================
# When desire is observer-visible context rather than the inferred latent, the
# actor utility needs a numeric desire magnitude. The LM reads the scenario plus
# the shown desire-state paragraph (reward_low / reward_high) and rates how much
# the two people want the thing on the same 0-100 scale the human participant
# uses. This is one rating per (scenario, desire condition) — it is NOT
# per-action (g already carries the action dependence).

DESIRE_SYSTEM_PROMPT = (
    "You are a participant in a human study. Respond as if you were a regular "
    "adult, just going off of your intuition.\n\n"
    "In this survey, you will read a vignette about two people in a situation "
    "where some resource — food, an object, a physical space, or a piece of "
    "information — could be shared between them, along with a short description "
    "of their current state. Judge how much the two people want the thing at "
    "stake in the scenario, given that state, on a scale from 0 (do not want it "
    "at all) to 100 (want it extremely). Rate only how much they want it — not "
    "what they end up doing, how much effort it takes, or how the two people are "
    "related.\n\n"
    "Respond with a JSON object in this format only, no explanation:\n"
    '{"desire": 65}'
)


def desire_user_prompt(vignette, state):
    """Build the user prompt for the scenario-level desire rating.

    `state` is the actor's motivational-state paragraph (the scenario's
    `reward_low` or `reward_high` text). Returns one 0-100 desire magnitude.
    """
    return (
        f"Scenario: {vignette}\n\n"
        f"State: {state}\n\n"
        "On a scale from 0 to 100, how much do the two people want the thing at "
        'stake in this scenario? Respond with {"desire": <number>}.'
    )
