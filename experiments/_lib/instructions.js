// All instructions for the four active studies, in one place so the shared
// parts and the study-specific parts are easy to compare side by side. Each
// trials.js imports STUDY_INSTRUCTIONS and picks its slug's pages.

const SURVEY_TITLE = "Social interactions survey";

// Wrap a list of paragraph HTML strings in the standard container + heading
// used on every instructions page.
function instructionsPage(paragraphs) {
  return `
    <div class="instructions-container">
      <h2>${SURVEY_TITLE}</h2>
      ${paragraphs.join("\n      ")}
    </div>
  `;
}

// --- Paragraphs shared across studies ---------------------------------------

// Opening paragraph that frames the task around relationships (1a, 1b, 2b).
// Study 2a uses its own intro (see below).
const INTRO_NOTE =
  "<p>In this survey, you will read vignettes about two people in different kinds of social relationships, deciding how to eat different kinds of food in different situations.</p>";

// Formal-vs-intimate relationships explainer. Shown in every study.
const RELATIONSHIPS_NOTE =
  "<p>Some relationships are formal, like some relationships with an employee, a religious leader, a shopkeeper or a new acquaintance. Other relationships are close and intimate, like some relationships with a romantic partner, sibling or best friend.</p>";

// Some details about the characters, what they know, or their situation are not
// stated in the scenarios; we'll ask participants to evaluate some of them.
// Shown in every study.
const INFERENCE_NOTE =
  "<p>In each scenario, two people decide what to do in a situation involving food.</p>" +
  "<p>The descriptions we give you leave out some information — about the characters or about the situation. In this study, we will ask you to evaluate some of these unstated details, using your best judgment.</p>";

// Clarifies that the desire rating is about how much the two people want the
// food itself (liking, hunger, appeal), distinct from what they decide to do.
// Shown in the desire-DV studies (1a, 1b).
const DESIRE_NOTE =
  '<div class="side-note"><p>By how much the two people <strong>want the food</strong>, we mean how much they would like to eat it — for example because they like that kind of food or are hungry — separate from what they decide to do and from other aspects of the situation. For example, in some situations, people might both want the food a lot but still not end up eating it because of other reasons.</p></div>';

// Explains the desire slider: one end means "not at all", the other "extremely",
// the middle a moderate amount. Shown in the desire-DV studies (1a, 1b).
const DESIRE_SLIDER_NOTE =
  "<p>For the question about how much the two people want the food, sliding the slider all the way to one end means you think they do not want it at all, and all the way to the other end means they want it extremely. Positions in between indicate how much they want it, with the middle meaning a moderate amount.</p>";

// Explains the relationship slider: the ends mean maximally formal / maximally
// intimate, the middle means neither, and positions in between are somewhere
// along that range. Shown in the intimacy-DV studies (2a, 2b).
const RELATIONSHIP_SLIDER_NOTE =
  "<p>For the question about the relationship, sliding the slider all the way to one end means you think the two people would describe their relationship as maximally formal, and sliding it all the way to the other end means maximally intimate. The middle means their relationship is neither formal nor intimate, and positions in between indicate relationships that are somewhat formal or somewhat intimate.</p>";

// Explains the "which situation is more likely" slider: an end means that
// situation is definitely the more likely one, the middle means the two are
// equally likely, and positions in between are intermediate. Shown in the
// two-slider studies that ask it (1b, 2b).
const SITUATION_SLIDER_NOTE =
  "<p>For the question about which of the two situations is more likely, sliding the slider all the way to one end means you think that situation is definitely the more likely one. The middle means you think the two situations are equally likely, and positions in between indicate how much more likely you think one situation is than the other.</p>";

// The before/after timing, stated last (after the questions are introduced):
// each response is given once before the decision is revealed and once after.
// Singular for the one-slider studies (1a, 2a), plural for the two (1b, 2b).
const TIMING_NOTE_ONE =
  "<p>You will answer this question twice — once before and once after we show you what the two people decide to do.</p>";
const TIMING_NOTE_TWO =
  "<p>You will answer both questions twice — once before and once after we show you what the two people decide to do.</p>";

// Final page: a study-specific "read carefully" lead-in followed by the shared
// payment, don't-close, and begin lines.
function finalInstructionsPage(leadIn) {
  return instructionsPage([
    `<p>${leadIn} 🙂 You will receive $5 if you successfully complete the survey.</p>`,
    "<p>Please do not close the window until you have completed the survey. If you do so, you will lose your progress.</p>",
    "<p>Press next to begin the survey.</p>",
  ]);
}

// "Read carefully" lead-ins for the final page.
const READ_CAREFULLY_RELATIONSHIP =
  "Please pay attention to the social relationship between the two people, and read the scenarios carefully!";
const READ_CAREFULLY_PLAIN = "Please read each of the scenarios carefully!";

// The inference note as its own standalone page (shown in every study).
const INFERENCE_PAGE = instructionsPage([INFERENCE_NOTE]);

// --- Per-study instructions (study-specific pages in bold relief) -----------

export const STUDY_INSTRUCTIONS = {
  // Study 1a — infer desire (one slider).
  food_inv_desire: [
    instructionsPage([INTRO_NOTE, RELATIONSHIPS_NOTE]),
    INFERENCE_PAGE,
    instructionsPage([
      "<p>In each scenario, we will ask you to rate how much you think the two people want to eat the food.</p>",
      DESIRE_SLIDER_NOTE,
      TIMING_NOTE_ONE,
      DESIRE_NOTE,
    ]),
    finalInstructionsPage(READ_CAREFULLY_RELATIONSHIP),
  ],

  // Study 1b — jointly infer desire and effort (two sliders).
  food_inv_joint_de: [
    instructionsPage([INTRO_NOTE, RELATIONSHIPS_NOTE]),
    INFERENCE_PAGE,
    instructionsPage([
      "<p>In each scenario, we will ask you two questions. The first question asks how much you think the two people want to eat the food. The second question asks which of two situations you think is more likely. You will answer each question using a slider.</p>",
      DESIRE_SLIDER_NOTE,
      SITUATION_SLIDER_NOTE,
      TIMING_NOTE_TWO,
      DESIRE_NOTE,
    ]),
    finalInstructionsPage(READ_CAREFULLY_RELATIONSHIP),
  ],

  // Study 2a — infer intimacy (one slider).
  food_inv_intimacy: [
    instructionsPage([INTRO_NOTE, RELATIONSHIPS_NOTE]),
    INFERENCE_PAGE,
    instructionsPage([
      "<p>In each scenario, we will ask you to evaluate what kind of social relationship you think the two people are in.</p>",
      RELATIONSHIP_SLIDER_NOTE,
      TIMING_NOTE_ONE,
    ]),
    finalInstructionsPage(READ_CAREFULLY_PLAIN),
  ],

  // Study 2b — jointly infer intimacy and effort (two sliders).
  food_inv_joint_ie: [
    instructionsPage([INTRO_NOTE, RELATIONSHIPS_NOTE]),
    INFERENCE_PAGE,
    instructionsPage([
      "<p>In each scenario, we will ask you two questions. The first question asks you to evaluate what kind of social relationship you think the two people are in. The second question asks which of two situations you think is more likely. You will answer each question using a slider.</p>",
      RELATIONSHIP_SLIDER_NOTE,
      SITUATION_SLIDER_NOTE,
      TIMING_NOTE_TWO,
    ]),
    finalInstructionsPage(READ_CAREFULLY_PLAIN),
  ],
};
