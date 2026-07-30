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
       - **effort**: the total executional cost of completing an action across
         the dyad — physical motor, equipment, and time cost borne by either
         person, plus (for disclosures) the production cost of producing the
         utterance.
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
(chapstick, towel, earbuds, hairbrush, harmonica, sunscreen), *space*
(blanket, sleeping-bag, bed, locker-room, sauna), and *privacy* (breakup,
salary, gossip, home, navigation). The risk rubric covers three channel types
(bodily-substance transfer, physical-contact / shared-space, and
informational / private-resource); the effort rubric covers physical motor
work, equipment / setup, time cost, and — for disclosures — the executional
cost of producing the utterance (never relational discomfort, which is the
risk dimension). The original food-only prompts were retired in
favor of this single set after a side-by-side comparison showed the
unified prompts produced equal or slightly better fits on the food data.

For disclosure actions (the privacy-type scenarios), effort is
operationalized as the executional cost of producing the account — how
long the telling takes and how much context or explaining it requires —
which the effort rubric scores as a distinct executional cost. The emotional
difficulty of revealing the content is deliberately NOT effort; that is
the risk dimension. See the invariants note in
`experiments/scenarios_nonfood.py` and the Study 3 scenario section of
the manuscript.

Editing any prompt here will change all callers — re-run the relevant LM
script and downstream fits.
"""

# ==============================================================================
# Shared building blocks
# ==============================================================================

_PREAMBLE_RATING = (
    "You are a participant in a human study. Respond as if you were a "
    "regular adult from the United States, just going off your intuition."
)

# The alternatives generator gets its own preamble: it asks for a step-by-step
# explanation before the answer, while every rating stage answers from
# intuition. Listing the comparison set is a constructive task, and this
# explain-then-answer format measurably improves coverage of high-risk sharing;
# a rating is a snap judgment for which we want the intuition-only instruction
# that mirrors the participant.
_PREAMBLE_ALTERNATIVES = (
    "You are a participant in a human study. Respond as if you were a "
    "regular adult from the United States. Think the question through step by "
    "step before answering."
)


def _intro_line():
    """Build the 'You will see a set of possible actions ...' intro line. The
    scored action set is variable-length: the LM sees the observed action plus
    however many alternatives that run generated — presented in randomized order
    with no marker of which was observed.
    """
    return "You will see a set of possible actions the two people could take."


def _json_format_block():
    """Build the trailing 'Respond with your numerical ratings ...' block.
    The example illustrates only the JSON shape for 3 actions; the real call has
    as many keys as actions given.
    """
    return """Respond with your numerical ratings as a JSON object whose keys are "action_0", "action_1", ... matching the number of actions given, no explanation needed. Example for 3 actions:
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
- Physical contact or shared space: the two people's bodies physically touch, or they share close physical space — sustained proximity within a bounded space such as a bed, blanket, small room, or vehicle. The extent of contact or proximity and the body region involved both matter — brief incidental touch or passing nearness is a small exposure; sustained skin contact, sharing a confined space, or contact with normally restricted body regions is a large one.
- Private or emotional disclosure: private, sensitive, or emotional information (personal details, or feelings one would not voice publicly), or access to personal resources (a private space, a personal item, a confidential record), from one person becomes accessible to the other.

Here we are asking what the action itself does — the interpersonal vulnerability it creates — independent of the relationship between the two people.

Co-presence without substance transfer, contact, close shared space, or disclosure does NOT by itself make one person vulnerable to the other — for example, two people each handling their own separate utensils, standing close together in a public space like an elevator, or keeping a conversation to surface-level topics. These should be rated near zero.

Use this scale from 0 to 6 (continuous values allowed):
0 = No interpersonal vulnerability (the two people stay fully separate; no exchange of substance, no contact or shared interpersonal space, no disclosure)
3 = Limited or indirect vulnerability (e.g. bodily substances reaching the other person only indirectly, through an item that has touched one person's skin; deliberate but limited physical contact, such as a hand on the shoulder; being close to each other in an open or roomy space rather than a confined one; or disclosing somewhat personal but not deeply private information)
6 = Strong, direct vulnerability (e.g. direct bodily-substance transfer such as mouth-to-mouth contact or sharing a utensil that's been in one person's mouth, sustained skin-to-skin contact, sharing a bed or other close confined space, or disclosing private details)"""


# _EFFORT_BODY is grounded in the Naïve Utility Calculus (NUC) framework and
# scoped to the total executional cost of carrying the joint action out,
# regardless of which member of the dyad performs the required work — motor
# work, equipment / preparation, time, and (for disclosures) the production
# cost of the utterance, the disclosure analogue of motor work. The construct
# does not extend to coordination or other cognitive cost types; this is a
# modeling scope choice, not a direct entailment of the cited physical-effort
# studies. The prompt body stays jargon-free; the rating dimension is anchored
# as follows.
#
#   - Conceptual anchor (cost as trade-off quantity) — Jara-Ettinger, J.,
#     Gweon, H., Schulz, L. E., & Tenenbaum, J. B. (2016). "The naïve
#     utility calculus: Computational principles underlying commonsense
#     psychology." Trends in Cognitive Sciences 20(8): 589–604. Defines
#     cost formally as what an agent weighs against desire — the framework
#     this project's inverse-planning model instantiates.
#
#   - Abstract physical-cost representation — Liu, S., Ullman,
#     T. D., Tenenbaum, J. B., & Spelke, E. S. (2017). "Ten-month-old
#     infants infer the value of goals from the costs of actions." Science
#     358(6366): 1038–1041. Provides evidence that observers represent
#     physical action costs abstractly across different physical obstacles.
#     Extending one scalar to equipment, elapsed time, dyadic contributions,
#     and utterance production is this model's operationalization. The
#     emotional vulnerability of disclosing is kept separate, in the risk
#     dimension. See the disclosure-effort rationale in
#     experiments/scenarios_nonfood.py.
#
#   - Effort as a perceptible quantity separable from desire —
#     Jara-Ettinger, J., Gweon, H., Tenenbaum, J. B., & Schulz, L. E.
#     (2015). "Children's understanding of the costs and rewards underlying
#     rational action." Cognition 140: 14–23. Establishes that children at
#     4–6 can estimate action cost as distinct from goal value and agent
#     competence — grounds the assumption that an "LM-as-participant" can
#     rate effort with the instruction below.

_EFFORT_BODY = """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them. {INTRO}

For each action, evaluate the total cost the two people would need to bear to carry it out — the physical, executional, and temporal cost of completing the action. Count required work regardless of which person performs it, including work divided between them. The cost types below all count; integrate across them into a single rating:

- Physical motor cost: how much bodily work the action requires (preparing, serving, cutting, pouring, handing over, cleaning, wiping, drying, tidying, rearranging, applying).
- Equipment and preparation cost: whether the action needs extra items or setup (utensils, plates, containers, sanitizing supplies, barriers, separate furniture, separate spaces) that either person has to obtain, set up, or take care of.
- Executional and production cost: for actions that consist of speaking, telling, or disclosing, how much work goes into producing the utterance itself — how long the account takes to deliver and how much context, backstory, or roundabout indirect phrasing the speaker must use, for it to land.
- Time cost: how long the action takes — waiting for something to dry, sequential rather than simultaneous use, an extended preparation or telling.

Do NOT rate social awkwardness, relational discomfort, or how intimate, appropriate, or emotionally hard the action would feel — those are separate dimensions that we are not asking about here. Here we want only the effort of carrying the action out.

Use this scale from 0 to 6 (continuous values allowed):
0 = No effort (neither person needs to do bodily work, obtain extra items, wait, compose, or explain)
3 = Moderate effort (a few bodily steps, such as setting out a clean utensil, dividing a portion, or briefly waiting; a small handful of extra items to obtain; or a short account that takes a little effort to produce)
6 = High effort (many bodily steps, substantial setup, or significant time — for example, leaving to obtain something from far away and returning, waiting a long time, cleaning and assembling many separate items, or producing a long account that needs extensive backstory or roundabout phrasing to convey)"""


# Per-rating-type instructions used in the user prompt (the line just above
# the numbered actions).
_USER_INSTRUCTIONS = {
    "risk": (
        "Rate how much each action makes one person interpersonally vulnerable "
        "to the other — through bodily exposure, physical contact or shared "
        "space, or private disclosure (0-6 scale):"
    ),
    "effort": (
        "Rate the total physical or executional cost of carrying out each "
        "action, counting work performed by either person — how much physical "
        "work, preparation, or equipment it takes, or, for telling or "
        "disclosing, how much explaining and roundabout phrasing producing "
        "the account takes (0-6 scale):"
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
# scalar rated by `desire_user_prompt`).
#
# g is scored on the outcome an action reaches once carried through to
# completion: a multi-step route (fetching a utensil, taking a longer path,
# acquiring something first) is credited for the end state it arrives at, not
# docked for being unfinished partway through. The cost of those steps is
# captured by `effort`, not g, so the two features stay orthogonal. Without this
# the LM oscillates run-to-run on journey-phrased actions ("go get a knife, then
# cut the hot dog" scored anywhere from 0 to 1), since the bare rubric is silent
# on whether to credit the completed outcome.
_G_BODY = """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them. {INTRO}

For each action, evaluate how fully it results in the two people ending up with the thing at stake in the scenario — the food they could eat, the object they could use, the space they could occupy, the information they could learn.

Judge only outcome attainment: whether, and how completely, the dyad ends up obtaining or consuming the thing. Judge each action by the outcome it leads to once it is carried through to completion. An action can deliver the outcome fully whether it is done together or separately, directly or via a safer indirect route. If an action involves extra steps along the way — going to fetch a utensil, taking a longer route, acquiring something first — rate it by the end state those steps arrive at, not by the fact that it is still unfinished partway through. How much work or time those steps take is a separate dimension (effort) that we are not asking about here.

An action that ends with both people getting and consuming the thing should be rated high; an action where only one person gets it, or where they end up with a reduced or incomplete version, should be rated in the middle; an action where they forgo or abandon it should be rated low.

Use this scale from 0 to 6 (continuous values allowed):
0 = The thing is not obtained (the action forgoes or abandons it)
3 = Partially obtained (a reduced portion, only one person, or an incomplete version)
6 = Fully obtained (both people end up getting or consuming the thing)"""


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
    desire_object names the specific outcome or resource at stake (e.g.
    "the hot dog", "to soothe their chapped lips");
    when given for rating_type="g" it makes the instruction concrete instead
    of the generic "the thing at stake". Ignored for the other rating types.

    Some nonfood scenarios phrase the desire object as an infinitive outcome
    rather than a noun (e.g. "to try the harmonica", "to warm up under a
    blanket"); "getting or consuming to try the harmonica" is ungrammatical, so
    those render as "actually getting to try the harmonica". Noun-phrase objects
    keep the original wording byte-identical, so the food elicitation is
    unaffected.
    """
    if rating_type not in _USER_INSTRUCTIONS:
        raise ValueError(f"unknown rating_type: {rating_type}")
    instr = _USER_INSTRUCTIONS[rating_type]
    if rating_type == "g" and desire_object is not None:
        attain = (
            f"getting {desire_object}"
            if desire_object.startswith("to ")
            else f"getting or consuming {desire_object}"
        )
        instr = (
            "Rate how much each action results in the two people actually "
            f"{attain} (0-6 scale):"
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
# Generation is framed as interpretation-driven: the observer lists the set it
# would use to read the observed action, not an exhaustive catalog of what was
# technically possible. A single bare note ("sharing can be physical ... or a
# matter of telling the other person something") keeps the disclosure/space
# modes available for the nonfood set without pushing for breadth. That note is
# IDENTICAL for every cell and condition, so it cannot produce condition
# effects; it only keeps a sharing mode from being missed. This is acknowledged
# in the SI elicitation-details section of the manuscript.
#
# The generation prompt asks for an explanation before answering (the
# explain-then-JSON close and `_PREAMBLE_ALTERNATIVES`): this format raised
# high-risk-share swing coverage from 66.5% to 75.5% on Study 1b, so it was
# adopted for the stage. There is deliberately only ONE alternatives prompt.
# The generated explanation is retained as a rationale for auditing the
# resulting comparison set; it is not treated as evidence about the model's
# internal reasoning process.

# VINTAGE MARKER (2026-07-30) --------------------------------------------------
# The current prompt source is intentionally AHEAD of the canonical LM tables,
# which the user plans to regenerate. The alternatives wording below was
# exercised in diagnostic generation runs for five studies, but it has not been
# carried through feature scoring and fitting; the Study 1b diagnostic scored
# tables predate this wording. The total-dyadic-cost effort rubric above is newer
# still and has not produced any canonical or diagnostic scored tables.
#
# New manifests record a stage-specific `prompt_sha256`; legacy manifests record
# only a whole-file `prompts_sha256`. The resume guard understands both formats
# and prevents a new run from silently appending to the old vintage.
#
# Until the full re-elicitation lands, do not mix old and new table vintages in
# fits. Regenerating `SIP_journal/si_prompts.tex` will document the intended
# rerun prompt rather than the prompt that produced the currently reported
# tables, so that discrepancy must remain explicit.
# After the re-elicitation lands, delete this marker and re-run
# `model/lm/export_prompts_latex.py`.
# ------------------------------------------------------------------------------

ALTERNATIVES_SYSTEM_PROMPT = (
    _PREAMBLE_ALTERNATIVES
    + "\n\n"
    + """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them.

The vignette omits some information about the situation, which an observer will be asked to infer.

You will be told what action they took in the situation, and which question(s) an observer is asked to answer.

Your job is to list the actions that the two people were realistically choosing between, that an observer would compare with the action they actually took, to answer the question(s).

First, briefly explain step by step which actions the two people were realistically choosing between, and what an observer would need to compare with the action they actually took to answer the question(s). Then respond with a JSON array in this exact format, with no other text after the array:
[
  {"action": "description of alternative 1"},
  {"action": "description of alternative 2"}
]
"""
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
    "max_formal": "maximally formal",
    "somewhat_formal": "somewhat formal",
    "somewhat_intimate": "somewhat intimate",
    "max_intimate": "maximally intimate",
}


def alternatives_user_prompt(
    vignette,
    observed_action_text,
    *,
    effort_text=None,
    intimacy_level=None,
    desire_text=None,
    effort_hypotheses=None,
    unknown_desire_object=None,
    unknown_intimacy=False,
):
    """Build the user prompt for the alternative-generation call in the six
    3-action inverse experiments.

    Composes whichever observer-visible condition paragraphs the experiment
    reveals. Each study passes only the paragraphs its observer actually sees:

      - Study 1a (`food_inv_desire`):        effort_text + intimacy_level
      - Study 1b (`food_inv_joint_de`):      intimacy_level
      - Study 2a (`food_inv_intimacy`):      desire_text + effort_text
      - Study 2b (`food_inv_joint_ie`):      desire_text
      - Study 3a (`nonfood_inv_joint_de`):   intimacy_level    (mirrors 1b)
      - Study 3b (`nonfood_inv_joint_ie`):   desire_text       (mirrors 2b)

    The nonfood pair differs from its food counterpart only in the stimulus set,
    so it routes through the same paragraph combination.

    Mirrors how the human participant sees the trial (vignette + revealed
    condition paragraphs + observed action), per `feedback_llm_as_participant`.
    `intimacy_level` is one of the intimacy-condition slugs (max_formal /
    somewhat_formal / somewhat_intimate / max_intimate) when provided; it's rendered
    via the shared `RELATIONSHIP_DESCRIPTORS` dict so the LM sees the same
    qualitative descriptor humans see.

    The prompt also always makes the LM epistemically aware of the latent(s) the
    study infers — the very quantities the observer-visible paragraphs above
    deliberately withhold — so the generated set spans the range those latents
    could take, mirroring the participant, who has seen the DV questions and so
    knows which quantities the trial leaves open. Inserted after the
    given-condition paragraphs and before the observed action, driven by which
    latents the study infers (the caller always supplies these):

      - `effort_hypotheses`: a `(low_text, high_text)` pair of the two effort
        paragraphs, presented as two situations one of which holds (effort-
        inferred studies 1b/2b/3a/3b).
      - `unknown_desire_object`: names the desire object as unknown-magnitude
        (desire-inferred studies 1a/1b/3a).
      - `unknown_intimacy`: flags the relationship as unknown (intimacy-inferred
        studies 2a/2b/3b).

    These are condition-independent within a study, so they shape only the
    coverage of the comparison set, not condition effects.

    The closing instruction names those same inferred latent(s) as the DV
    question(s) the listing serves ("... judge how much they would like X and
    how likely each of the two situations above is"), phrased to match the
    experiment's DVs (effort is the posterior over the two shown situations;
    intimacy is formal-vs-intimate), so the LM lists the set an observer would
    use to answer them from the observed action. It falls back to a generic
    interpretation framing when no latent kwargs are passed (the SI template
    render).
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
    if effort_hypotheses is not None:
        low_text, high_text = effort_hypotheses
        parts.append(
            "One of the following is true of the situation, but you do not "
            f"know which:\n- {low_text}\n- {high_text}\n"
            "Describe any action you list unconditionally, as something the "
            "two people could attempt in either situation — do not build "
            '"if" clauses about the unknown situation into the action '
            "description; the same action may simply turn out easy or hard "
            "depending on which situation holds."
        )
    if unknown_desire_object is not None:
        # "also" only when this follows the effort-hypotheses block (1b/3a);
        # in 1a it is the sole epistemic statement and "also" would dangle.
        also = "also " if effort_hypotheses is not None else ""
        parts.append(
            f"You {also}do not know how much the two people would like "
            f"{unknown_desire_object}."
        )
    if unknown_intimacy:
        parts.append(
            "You do not know how formal or intimate the two people's relationship is."
        )
    parts.append(
        f"\nThe two people took the following action:\n{observed_action_text}\n"
    )
    # Closing frames the listing as the comparison set for judging the study's
    # inferred latent(s) from the observed action. DV labels are built from
    # whichever latent kwargs the caller passed; a generic framing is used when
    # none are (the SI template render, which shows only given conditions).
    dv_labels = []
    if unknown_desire_object is not None:
        dv_labels.append(f"how much they would like {unknown_desire_object}")
    if unknown_intimacy:
        dv_labels.append("how formal or intimate they are")
    if effort_hypotheses is not None:
        # Effort is inferred as a posterior over the two situations shown above,
        # so the label points back to them rather than naming a magnitude. Kept
        # last so the "above" back-reference reads naturally after the others.
        dv_labels.append("how likely each of the two situations above is")
    if len(dv_labels) > 2:
        dv_phrase = ", ".join(dv_labels[:-1]) + ", and " + dv_labels[-1]
    elif len(dv_labels) == 2:
        dv_phrase = f"{dv_labels[0]} and {dv_labels[1]}"
    else:
        dv_phrase = dv_labels[0] if dv_labels else None
    if dv_phrase is not None:
        parts.append(
            "List the actions the two people were choosing between — the "
            "comparison set you would use to interpret their choice and judge "
            f"{dv_phrase}. Do not include the action they actually took."
        )
    else:
        parts.append(
            "List the actions the two people were choosing between — the "
            "comparison set you would use to interpret their choice. Do not "
            "include the action they actually took."
        )
    return "\n".join(parts)


# ==============================================================================
# Public API: scenario-level desire scalar (given-desire studies 2a, 2b)
# ==============================================================================
# When desire is observer-visible context rather than the inferred latent, the
# actor utility needs a numeric desire magnitude. The LM reads the scenario plus
# the shown desire-state paragraph (desire_low / desire_high) and rates how much
# the two people would like the thing on the same 0-100 scale the human
# participant uses. The wording mirrors the human DV question ("How much do you
# think X and Y would like <object>?") with no added construct gloss, so the LM
# answers the same question participants answer; the state-dependence is carried
# by the shown state paragraph and the "given that state" clause, and the scale
# endpoints mirror the human slider labels (Not at all / Extremely). This is one
# rating per (scenario, desire condition) — it is NOT per-action (g already
# carries the action dependence).

DESIRE_SYSTEM_PROMPT = (
    _PREAMBLE_RATING
    + "\n\n"
    + """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them, along with a short description of their current state. Judge how much the two people would like the thing at stake in the scenario, given that state, on a scale from 0 (would not like it at all) to 100 (would like it extremely). Rate only how much they would like it — not what they end up doing, how much effort it takes, or how the two people are related.

Respond with a JSON object in this exact format, no explanation:
{"desire": <number>}"""
)


def desire_user_prompt(vignette, state, desire_object):
    """Build the user prompt for the scenario-level desire rating.

    `state` is the actor's desire-state paragraph (the scenario's
    `desire_low` or `desire_high` text). `desire_object` names the specific
    outcome or resource at stake (e.g. "the hot dog", "to soothe their
    chapped lips"), matching what the human participant is asked about in
    the desire DV question
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
    + """You will read a short description of a relationship between two people. Judge how intimate the relationship is on a scale from 0 (maximally formal) to 100 (maximally intimate).

Respond with a JSON object in this exact format, no explanation:
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


# ==============================================================================
# Public API: prior scalars (for the informative-prior configs, which are
# available tooling rather than the reported model — see _priors.py)
# ==============================================================================
# Each prompt mirrors the human PRIOR-stage question exactly: the LM sees the
# vignette plus the same given-condition paragraphs the participant sees before
# the action is revealed, and answers the same question the participant's
# slider asks, with the same endpoints. One rating per (run, scenario,
# prior-visible conditions); no action text appears anywhere.

PRIOR_DESIRE_SYSTEM_PROMPT = (
    _PREAMBLE_RATING
    + "\n\n"
    + """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them. Before knowing anything about what they decide to do, judge how much the two people would like the thing at stake in the scenario, on a scale from 0 (would not like it at all) to 100 (would like it extremely). Rate only how much they would like it — not what they might do, how much effort anything takes, or how the two people are related.

Respond with a JSON object in this exact format, no explanation:
{"desire": <number>}"""
)


def prior_desire_user_prompt(vignette, desire_object, condition_texts=()):
    """Prior-desire rating (1a/1b): the participant's prior-stage screen minus
    the action. `condition_texts` are the given-condition paragraphs the study
    shows before the prior rating (1a: relationship sentence + effort
    paragraph; 1b: relationship sentence; base variants: no relationship)."""
    parts = [f"Scenario: {vignette}", *condition_texts]
    parts.append(
        f"\nBefore observing what the two people decide to do: on a scale "
        f"from 0 to 100, how much do you think they would like "
        f'{desire_object}? Respond with {{"desire": <number>}}.'
    )
    return "\n".join(parts)


PRIOR_EFFORT_SYSTEM_PROMPT = (
    _PREAMBLE_RATING
    + "\n\n"
    + """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them, followed by two descriptions of what the situation might be like. Before knowing anything about what the two people decide to do, judge which of the two situations you think is more likely, on a scale from 0 (the FIRST situation is certainly the case) to 100 (the SECOND situation is certainly the case), where 50 means the two situations are equally likely.

Respond with a JSON object in this exact format, no explanation:
{"effort": <number>}"""
)


def prior_effort_user_prompt(
    vignette, effort_low_text, effort_high_text, condition_texts=()
):
    """Prior-effort rating (1b/2b): mirrors the human effort slider, whose
    endpoints are the scenario's two effort paragraphs with "Equally likely"
    at the midpoint. The low-effort paragraph is the 0 endpoint (first), the
    high-effort paragraph the 100 endpoint (second), matching the human
    slider's left-to-right order; the response maps to P(high effort) =
    value / 100."""
    parts = [f"Scenario: {vignette}", *condition_texts]
    parts.append(f"\nFirst situation: {effort_low_text}")
    parts.append(f"Second situation: {effort_high_text}")
    parts.append(
        "\nOn a scale from 0 (certainly the first situation) to 100 "
        "(certainly the second situation), which situation do you think is "
        'more likely? Respond with {"effort": <number>}.'
    )
    return "\n".join(parts)


PRIOR_INTIMACY_SYSTEM_PROMPT = (
    _PREAMBLE_RATING
    + "\n\n"
    + """In this survey, you will read a vignette about two people in a situation where some resource — food, an object, a physical space, or a piece of information — could be shared between them. Before knowing anything about what they decide to do, judge how the two people would describe their relationship, on a scale from 0 (maximally formal) to 100 (maximally intimate), where 50 means neither formal nor intimate.

Respond with a JSON object in this exact format, no explanation:
{"intimacy": <number>}"""
)


def prior_intimacy_user_prompt(vignette, condition_texts=()):
    """Prior-intimacy rating (2a/2b): the participant's prior-stage screen
    minus the action (2a: desire + effort paragraphs; 2b: desire paragraph)."""
    parts = [f"Scenario: {vignette}", *condition_texts]
    parts.append(
        "\nBefore observing what the two people decide to do: on a scale "
        "from 0 to 100, how do you think they would describe their "
        'relationship? Respond with {"intimacy": <number>}.'
    )
    return "\n".join(parts)


# ==============================================================================
# Prompt provenance
# ==============================================================================


def prompt_fingerprint_payload(stage):
    """Return the actual prompt surfaces used by one elicitation stage.

    The provenance hash is derived from rendered system and user messages,
    rather than the bytes of this entire source file. This makes it sensitive
    to text sent by the requested stage while insulating it from comments and
    prompts used only by other stages. The fixed sentinel content exercises
    every live formatting branch; scenario-specific values are inputs to the
    template, not part of its version.
    """
    vignette = "<VIGNETTE>"
    action = "<OBSERVED_ACTION>"
    actions = ["<ACTION_0>", "<ACTION_1>"]
    condition = "<VISIBLE_CONDITION>"
    effort_hypotheses = ("<LOW_EFFORT_STATE>", "<HIGH_EFFORT_STATE>")

    if stage == "generate_alternatives":
        # Generic SI-template path, followed by study-specific formatting paths
        # across the six-study roster.
        rendered_users = [alternatives_user_prompt(vignette, action)]
        rendered_users.extend(
            alternatives_user_prompt(
                vignette,
                action,
                effort_text=condition,
                intimacy_level=level,
                unknown_desire_object="<DESIRE_OBJECT>",
            )
            for level in RELATIONSHIP_DESCRIPTORS
        )
        rendered_users.extend(
            [
                alternatives_user_prompt(
                    vignette,
                    action,
                    intimacy_level="max_formal",
                    effort_hypotheses=effort_hypotheses,
                    unknown_desire_object="<DESIRE_OBJECT>",
                ),
                alternatives_user_prompt(
                    vignette,
                    action,
                    effort_hypotheses=effort_hypotheses,
                    unknown_desire_object="<DESIRE_OBJECT>",
                ),
                alternatives_user_prompt(
                    vignette,
                    action,
                    desire_text=condition,
                    effort_text=condition,
                    unknown_intimacy=True,
                ),
                alternatives_user_prompt(
                    vignette,
                    action,
                    desire_text=condition,
                    effort_hypotheses=effort_hypotheses,
                    unknown_intimacy=True,
                ),
            ]
        )
        return {
            "system": ALTERNATIVES_SYSTEM_PROMPT,
            "users": rendered_users,
        }

    if stage == "score_merged":
        return {
            # lm_runs.jsonl embeds the generated alternatives, so its prompt
            # vintage includes the upstream generation surface as well as the
            # prompts used directly during scoring.
            "upstream_generation": prompt_fingerprint_payload(
                "generate_alternatives"
            ),
            "feature_systems": {
                rating_type: system_prompt(rating_type)
                for rating_type in ("risk", "effort", "g")
            },
            "feature_users": {
                "risk": [user_prompt("risk", vignette, actions)],
                "effort": [user_prompt("effort", vignette, actions)],
                "g": [
                    user_prompt(
                        "g", vignette, actions, desire_object="<DESIRE_OBJECT>"
                    ),
                    user_prompt(
                        "g",
                        vignette,
                        actions,
                        desire_object="to <INFINITIVE_OUTCOME>",
                    ),
                ],
            },
            "desire_system": DESIRE_SYSTEM_PROMPT,
            "desire_user": desire_user_prompt(
                vignette, condition, "<DESIRE_OBJECT>"
            ),
            "intimacy_system": INTIMACY_SYSTEM_PROMPT,
            "intimacy_users": [
                relationship_user_prompt(descriptor)
                for descriptor in RELATIONSHIP_DESCRIPTORS.values()
            ],
        }

    if stage == "priors":
        return {
            "systems": {
                "desire": PRIOR_DESIRE_SYSTEM_PROMPT,
                "effort": PRIOR_EFFORT_SYSTEM_PROMPT,
                "intimacy": PRIOR_INTIMACY_SYSTEM_PROMPT,
            },
            "users": {
                "desire": prior_desire_user_prompt(
                    vignette, "<DESIRE_OBJECT>", (condition,)
                ),
                "effort": prior_effort_user_prompt(
                    vignette,
                    "<LOW_EFFORT_STATE>",
                    "<HIGH_EFFORT_STATE>",
                    (condition,),
                ),
                "intimacy": prior_intimacy_user_prompt(vignette, (condition,)),
            },
        }

    raise ValueError(f"unknown prompt stage: {stage}")
