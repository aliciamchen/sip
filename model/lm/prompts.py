"""Centralized LM prompts for all Together AI calls in this project.

All system prompts and user-prompt formatters live here so they're easy to
compare and edit together. This module has no LM-call logic — see
`lm/score_merged.py` and `lm/generate_alternatives.py` for that.

The pipeline has two elicitation steps (see the manuscript Methods):

  1. **Counterfactual action generation** (`G_LM`): given the scenario and the
     observed action, the LM proposes a small set of plausible alternative
     actions the dyad could have taken (`ALTERNATIVES_SYSTEM_PROMPT` /
     `alternatives_user_prompt`).
  2. **Utility-feature scoring** (the feature map `phi_tau`): the observed
     action and the generated alternatives are scored together, variable-length,
     on three features, each on a 0–6 scale (rescaled to [0, 1] downstream):

       - **risk**: bodily, spatial, or informational exposure an action creates
         between the two people — the relationship-independent
         interpersonal-vulnerability feature (discomfort = risk · (1−I)^γ).
       - **effort**: physical, material, and time cost of executing an action.
       - **g**: desire-free goal-satisfaction — how fully the action delivers
         the outcome at stake. The reward term is `w_v · desire · g`, so desire
         scales this stable per-action value (g replaced the old signed-valence
         rating).

Two further call types supply the magnitude of a variable that is *given*
rather than inferred in a study: a scenario-level **desire** scalar (0–100,
`DESIRE_SYSTEM_PROMPT` / `desire_user_prompt`) for the given-desire studies,
and a per-level **intimacy** scalar (0–100, `INTIMACY_SYSTEM_PROMPT` /
`relationship_user_prompt`) for the given-relationship studies. Both are at the
bottom of the file.

The prompts are domain-general: they cover food sharing as well as the
three non-food sub-types in `scenarios_nonfood.csv` — *substance*
(chapstick, towel, hairbrush, harmonica, sunscreen), *space* (blanket,
sleeping-bag, bed, locker-room, sauna), and *privacy* (breakup, payment,
gossip, home, navigation). The risk rubric covers three channel types
(bodily-substance transfer, physical-contact / shared-space, and
informational / private-resource); the effort rubric covers physical motor
work, equipment / setup, and time cost (physical cost only — not coordination
or other cognitive costs). The original food-only prompts were retired in
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


def _intro_line():
    """Build the 'You will see a set of alternative actions ...' intro line.

    Each scoring call covers a single scenario, so the wording is singular. The
    scored action set is variable-length: the LM sees the observed action plus
    however many alternatives that run generated.
    """
    return "You will see a set of alternative actions the two people could take."


def _json_format_block():
    """Build the trailing 'Respond with your numerical ratings ...' block.

    The example illustrates only the JSON shape for 3 actions; the real call has
    as many keys as actions given. The values are shown as `<number>`
    placeholders rather than concrete digits so the example cannot anchor the
    ratings.
    """
    return """Respond with your numerical ratings as a JSON object whose keys are "action_0", "action_1", ... matching the number of actions given, no explanation needed. Use whatever values your judgments warrant. The example below shows only the format (one key per action), not suggested values. Example for 3 actions:
{"action_0": <number>, "action_1": <number>, "action_2": <number>}"""


# ==============================================================================
# Rating-type-specific bodies
# ==============================================================================

# _RISK_BODY operationalizes the model's `risk(a)` feature: the interpersonal
# vulnerability an action creates — the degree to which it exposes one person to
# the other, opening them up or lowering the boundary between them. This is a
# relationship-INDEPENDENT property of the action (the model modulates it by
# intimacy separately, via (1−I)^γ), so the prompt asks for the exposure the
# action creates, not how uncomfortable it would feel given the relationship.
#
# Treating bodily, spatial, and informational/emotional exposure as one graded
# vulnerability dimension is itself theoretically motivated, not just a modeling
# convenience: across the relationship literature, closeness develops through —
# and is read from — graded self-exposure that lowers interpersonal boundaries,
# the same logic recurring across domains. Social-penetration theory frames
# relationship development as progressive, boundary-lowering self-disclosure
# (Altman, I. & Taylor, D. A. (1973). "Social Penetration: The Development of
# Interpersonal Relationships." Holt, Rinehart & Winston; see also Prager 1997
# on intimacy as mutual vulnerability); in the bodily domain the parallel is
# consubstantiation, the partial merging of selves through shared substance
# (Carsten 1995; Thomas et al. 2022). Intimacy is the graded state that lowers
# the discomfort of such vulnerable actions — exactly the role `risk(a)` plays in
# the utility. The three forms this vulnerability takes are each grounded in
# their own literature; the prompt body stays jargon-free (the LM is prompted
# "as a participant"), but the conceptual content of each is grounded as follows:
#
#   - Substance-transmission channels — Rozin, P. & Fallon, A. E. (1987).
#     "A perspective on disgust." Psychological Review 94(1): 23–41.
#     Establishes contamination via bodily-substance transfer as the core
#     domain of disgust; even brief contact transmits.
#
#   - Direct-contact / shared-space channels — Suvilehto, J. T., Glerean, E.,
#     Dunbar, R. I. M., Hari, R., & Nummenmaa, L. (2015). "Topography of social
#     touching depends on emotional bonds between humans." PNAS 112(45):
#     13811–13816 (touch permissions graded by closeness; grounds why contact
#     extent and body region matter). For the shared-space / proximity part,
#     Hall, E. T. (1966). "The Hidden Dimension." Doubleday — interpersonal
#     distance zones are governed by relational closeness.
#
#   - Informational / emotional-disclosure channels — Reis, H. T. & Shaver, P. (1988).
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

_RISK_BODY = """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them. {INTRO}

For each action, evaluate how much it makes one person interpersonally vulnerable to the other — how much it exposes them, opens them up, or lowers the boundary between them, letting something normally kept to oneself pass from one person's side to the other. This interpersonal vulnerability can take multiple forms, and a single action may involve more than one:

- Bodily / substance exposure: bodily substances (saliva, breath, skin oils, sweat) from one person reach the other, either directly or via a shared vessel or item that's been on the first person's body. Even brief contact counts — the substance doesn't have to remain visible for the exposure to be real.
- Physical contact or shared space: the two people's bodies physically touch, or they share close physical space — sustained proximity within a bounded space such as a bed, blanket, small room, or vehicle. The extent of contact or proximity and the body region involved both matter — brief incidental touch or passing nearness is a small exposure; sustained skin contact, sharing a confined space, or contact with body regions normally restricted to close relationships is a large one.
- Private or emotional disclosure: private, sensitive, or emotional information (personal details, or feelings one would not voice publicly), or access to personal resources (a private space, a personal item, a confidential record), from one person becomes accessible to the other — content or access someone would not grant a stranger or a passing acquaintance.

Co-presence without substance transfer, contact, close shared space, or disclosure does NOT by itself make one person vulnerable to the other — for example, two people each handling their own separate utensils, sitting apart in a large or public room, or keeping a conversation to surface-level topics. These should be rated near zero.

Rate the interpersonal vulnerability the action itself creates — the exposure, contact, or disclosure involved — not how intimate or awkward it would feel. Here we are asking what the action does, independent of their relationship.

Use this scale from 0 to 6 (continuous values allowed):
0 = No interpersonal vulnerability (the two people stay fully separate; no exchange of substance, no contact or shared space, no disclosure)
3 = Limited or indirect vulnerability (e.g. using a shared item after cleaning or with a barrier, sitting near each other without touching, sharing surface-level information anyone could ask about)
6 = Strong, direct vulnerability (e.g. direct bodily-substance transfer such as mouth-to-mouth contact or sharing a utensil that's been in one person's mouth, sustained skin-to-skin contact, sharing a bed or other close confined space, or disclosing private details one would not tell a stranger)"""


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
#     cost formally as what an agent weighs against desire — the framework
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
#   - Effort as a perceptible quantity separable from desire —
#     Jara-Ettinger, J., Gweon, H., Tenenbaum, J. B., & Schulz, L. E.
#     (2015). "Children's understanding of the costs and desires underlying
#     rational action." Cognition 140: 14–23. Establishes that children at
#     4–6 can estimate action cost as distinct from goal value and agent
#     competence — grounds the assumption that an "LM-as-participant" can
#     rate physical effort with the instruction below.

_EFFORT_BODY = """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them. {INTRO}

For each action, evaluate the *physical* cost the actor would weigh against the benefit of the action — the bodily, material, and temporal cost of carrying it out. The three cost types below all count; integrate across them into a single rating:

- Physical motor cost: how much bodily work the action requires (preparing, serving, cutting, pouring, handing over, cleaning, wiping, drying, tidying, rearranging, applying).
- Equipment and preparation cost: whether the action needs extra items or setup (utensils, plates, containers, sanitizing supplies, barriers, separate furniture, separate spaces) that someone has to obtain, set up, or take care of.
- Time cost: how long the action takes — waiting for something to dry, sequential rather than simultaneous use, an extended preparation.

Do NOT rate social awkwardness, relational discomfort, or how intimate or appropriate the action would feel — those are separate dimensions that we are not asking about here. Here we want only the physical effort of carrying the action out.

Use this scale from 0 to 6 (continuous values allowed):
0 = No physical effort (acting independently or doing the simplest direct thing — no bodily work beyond the basic motion, no extra items, no waiting)
3 = Moderate physical effort (a few bodily steps, such as setting out a clean utensil, dividing a portion, or briefly waiting; or a small handful of extra items to obtain)
6 = High physical effort (many bodily steps, substantial setup, or significant time — for example, leaving to obtain something from far away and returning, preparing food from scratch, or cleaning and assembling many separate items)"""


# Per-rating-type instructions used in the user prompt (the line just above
# the numbered actions).
_USER_INSTRUCTIONS = {
    "risk": (
        "Rate how much each action makes one person interpersonally vulnerable "
        "to the other — through bodily exposure, physical contact or shared "
        "space, or private disclosure (0-6 scale):"
    ),
    "effort": (
        "Rate the physical and logistical cost of executing each action — "
        "how much physical work, preparation, or extra equipment is required "
        "(0-6 scale):"
    ),
    "g": (
        "Rate how much each action results in the two people actually getting "
        "or consuming the thing at stake (0-6 scale):"
    ),
}


# _G_BODY is the goal-satisfaction component of the desire term. In the
# continuous-desire model the desire enters the utility as w_v · desire · g(a|s),
# where desire is the latent magnitude (how much the dyad wants the outcome) and
# g(a|s) is this desire-free rating of how fully the action delivers the outcome.
# Splitting desire this way is what lets desire be inferred as a continuous
# latent: g is a stable, elicitable property of the action, while desire is the
# free quantity the observer recovers (or, in the given-desire studies, the
# scalar rated by `desire_user_prompt`). g replaces the old signed-valence V.
_G_BODY = """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them. {INTRO}

For each action, evaluate how fully it results in the two people ending up with the thing at stake in the scenario — the food they could eat, the object they could use, the space they could occupy, the information they could learn. Judge only outcome attainment: whether, and how completely, the dyad ends up obtaining or consuming the thing. Do not let how much the people would like it, the physical effort involved, or how "shared" or close the action looks change this rating — those are separate dimensions that we are not asking about here, and an action can deliver the outcome fully whether it is done together or separately, directly or via a safer indirect route.

An action that ends with both people getting and consuming the thing should be rated high; an action where they forgo it, abandon it, or only one person gets it should be rated low.

Use this scale from 0 to 6 (continuous values allowed):
0 = The thing is not obtained (the action forgoes or abandons it)
3 = Partially obtained (a reduced portion, only one person, or an incomplete version)
6 = Fully obtained (both people end up getting and consuming the thing)"""


_BODIES = {
    "risk": _RISK_BODY,
    "effort": _EFFORT_BODY,
    "g": _G_BODY,
}


# ==============================================================================
# Public API: feature-scoring prompts (risk / effort / g)
# ==============================================================================


def system_prompt(rating_type):
    """Build the system prompt for a feature-scoring call.

    rating_type: one of "risk", "effort", "g". The scored action set is
    variable-length (the observed action plus that run's alternatives).
    """
    if rating_type not in _BODIES:
        raise ValueError(f"unknown rating_type: {rating_type}")
    body = _BODIES[rating_type].format(INTRO=_intro_line())
    json_block = _json_format_block()
    return f"{_PREAMBLE_RATING}\n\n{body}\n\n{json_block}"


def user_prompt(rating_type, vignette, action_texts, desire_object=None):
    """Build the user prompt for a feature-scoring call.

    vignette is whatever scene-description text the LM should see (the caller
    is responsible for choosing whether to include condition paragraphs like
    `effort_low` / `effort_high`).
    action_texts is an ordered list of action descriptions; they're rendered
    as "Action 0: ...", "Action 1: ...", etc.
    desire_object names the specific resource at stake (e.g. "the hot dog");
    when given for rating_type="g" it makes the instruction concrete instead
    of the generic "the thing at stake". Ignored for the other rating types.
    """
    if rating_type not in _USER_INSTRUCTIONS:
        raise ValueError(f"unknown rating_type: {rating_type}")
    instr = _USER_INSTRUCTIONS[rating_type]
    if rating_type == "g" and desire_object is not None:
        instr = (
            "Rate how much each action results in the two people actually "
            f"getting or consuming {desire_object} (0-6 scale):"
        )
    actions_block = "\n".join(
        f"Action {i}: {txt}" for i, txt in enumerate(action_texts)
    )
    return f"Scenario: {vignette}\n\n{instr}\n\n{actions_block}"


# ==============================================================================
# Public API: alternative generation
# ==============================================================================


# ALTERNATIVES_SYSTEM_PROMPT is the methodological core of this project's
# open-world inverse-planning move: rather than reasoning over a fixed
# action set, the LM proposes a small, scenario-specific set of plausible
# counterfactual actions that then feed into the formal inverse-planning
# model with their LM-elicited utility features (risk, effort, g). The
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
#   - Frame-problem desire — Dennett, D. C. (1984). "Cognitive wheels:
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

ALTERNATIVES_SYSTEM_PROMPT = (
    _PREAMBLE_RATING
    + "\n\n"
    + """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them. You will be told what action they took in the situation.

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
)


# Relationship-condition descriptors keyed by the verbal intimacy-condition slug
# (the experiments store intimacy as a slug, never a numeric code). These are the
# de-anchored verbal exemplars human participants see — no "X out of 100" numeric
# anchor. The same descriptors are used both to condition alternative generation
# (here, via `alternatives_user_prompt`) and to elicit the per-level intimacy
# magnitude (`relationship_user_prompt`), so the LM sees exactly what participants
# see and the intimacy rating is not circular. See `intimacy_texts` /
# `intimacyDescriptor` in `experiments/_lib/scenario.js`.
RELATIONSHIP_DESCRIPTORS = {
    "max_formal": "maximally formal — e.g., the kind of relationship one might have with a new acquaintance, a shopkeeper, or a religious leader",
    "neither": "neither formal nor intimate — e.g., the kind of relationship one might have with a casual friend or a coworker",
    "somewhat_intimate": "somewhat intimate — e.g., the kind of relationship one might have with a close friend",
    "max_intimate": "maximally intimate — e.g., the kind of relationship one might have with a romantic partner or best friend",
}


def alternatives_user_prompt(
    vignette,
    observed_action_text,
    *,
    effort_text=None,
    intimacy_level=None,
    desire_text=None,
):
    """Build the user prompt for the alternative-generation call in the 3-action
    inverse experiments (Studies 1a, 1b, 2a, 2b).

    Composes whichever observer-visible condition paragraphs the experiment
    reveals. Each study passes only the paragraphs its observer actually sees:

      - Study 1a (`food_inv_desire`):     effort_text + intimacy_level
      - Study 1b (`food_inv_joint_de`):   intimacy_level
      - Study 2a (`food_inv_intimacy`):   desire_text + effort_text
      - Study 2b (`food_inv_joint_ie`):   desire_text

    Mirrors how the human participant sees the trial (vignette + revealed
    condition paragraphs + observed action), per `feedback_llm_as_participant`.
    `intimacy_level` is one of the intimacy-condition slugs (max_formal /
    neither / somewhat_intimate / max_intimate) when provided; it's rendered
    via the shared `RELATIONSHIP_DESCRIPTORS` dict so the LM sees the same
    qualitative descriptor humans see.
    """
    parts = [f"Scenario: {vignette}"]
    if desire_text is not None:
        parts.append(desire_text)
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
# the shown desire-state paragraph (desire_low / desire_high) and rates how much
# the two people would like the thing on the same 0-100 scale the human
# participant uses. The wording mirrors the human DV ("how much would they like
# the thing" = how much obtaining/consuming it would satisfy their current
# state — its appeal). This is one rating per (scenario, desire condition) — it
# is NOT per-action (g already carries the action dependence).

DESIRE_SYSTEM_PROMPT = (
    _PREAMBLE_RATING
    + "\n\n"
    + """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them, along with a short description of their current state. Judge how much the two people would like the thing at stake in the scenario, given that state — that is, how much obtaining or consuming it would satisfy the state they are in right now (its appeal to them) — on a scale from 0 (would not like it at all) to 100 (would like it extremely). Rate only how much they would like it — not what they end up doing, how much effort it takes, or how the two people are related.

Respond with a JSON object in this exact format, no explanation. The `<number>` is a placeholder for the format only — replace it with the value your judgment warrants, anywhere on the 0–100 scale:
{"desire": <number>}"""
)


def desire_user_prompt(vignette, state, desire_object):
    """Build the user prompt for the scenario-level desire rating.

    `state` is the actor's desire-state paragraph (the scenario's
    `desire_low` or `desire_high` text). `desire_object` names the specific
    resource at stake (e.g. "the hot dog"), matching the object the human
    participant is asked about in the desire DV question
    (`experiments/_lib/scenario.js`). Returns one 0-100 desire magnitude.
    """
    return (
        f"Scenario: {vignette}\n\n"
        f"State: {state}\n\n"
        f"On a scale from 0 to 100, how much would the two people like "
        f"{desire_object}, given their state? Respond with "
        '{"desire": <number>}.'
    )


# ==============================================================================
# Public API: relationship intimacy scalar (given-relationship studies 1a, 1b)
# ==============================================================================
# When intimacy is observer-visible context (the four relationship conditions),
# the actor utility needs a numeric intimacy magnitude I ∈ [0, 1] per level. The
# LM rates the intimacy implied by each (verbal) relationship description, the
# mirror of the per-condition desire scalar in 2a/2b. The descriptions are
# scenario-independent, so this is one rating per level (4 total).
#
# The level is rated from the de-anchored verbal descriptors in
# RELATIONSHIP_DESCRIPTORS (defined above) — the same exemplars participants and
# the generation prompt see. A numeric anchor would make the rating circular (the
# LM would echo the stated number).

INTIMACY_SYSTEM_PROMPT = (
    _PREAMBLE_RATING
    + "\n\n"
    + """You will read a short description of a relationship between two people. Judge how intimate the relationship is on a scale from 0 (maximally formal) to 100 (maximally intimate). Rate only the intimacy of the relationship itself.

Respond with a JSON object in this exact format, no explanation. The `<number>` is a placeholder for the format only — replace it with the value your judgment warrants, anywhere on the 0–100 scale:
{"intimacy": <number>}"""
)


def relationship_user_prompt(descriptor):
    """User prompt for the relationship-intimacy rating (given-relationship
    studies 1a/1b). `descriptor` is a de-anchored verbal relationship descriptor
    (RELATIONSHIP_DESCRIPTORS[level]); returns one 0-100 intimacy
    magnitude for that level."""
    return (
        f"The two people are in a relationship they would describe as "
        f"{descriptor}.\n\n"
        "On a scale from 0 to 100, how intimate is this relationship? Respond "
        'with {"intimacy": <number>}.'
    )
