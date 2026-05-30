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
  "<p>The descriptions we give you leave out some information — about the characters, what they know, or the situation. In this study, we will ask you to evaluate some of these unstated details, using your best judgment.</p>";

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
      "<p>Before observing what the two people decide to do, we will ask you to rate how much you think they want to eat the food.</p>",
      "<p>Then, we will show you what they decide to do, and ask you to re-rate how much you think they want to eat the food.</p>",
    ]),
    finalInstructionsPage(READ_CAREFULLY_RELATIONSHIP),
  ],

  // Study 1b — jointly infer desire and effort (two sliders).
  food_inv_joint_de: [
    instructionsPage([INTRO_NOTE, RELATIONSHIPS_NOTE]),
    INFERENCE_PAGE,
    instructionsPage([
      "<p>Before observing what the two people decide to do, we will ask you two questions. The first question asks how much you think they want to eat the food. The second question asks which of two situations — each described in the scenario — you think is more likely. You will answer each of the questions using a slider.</p>",
      "<p>Then, we will show you what they decide to do, and ask you to re-evaluate your answers.</p>",
    ]),
    finalInstructionsPage(READ_CAREFULLY_RELATIONSHIP),
  ],

  // Study 2a — infer intimacy (one slider).
  food_inv_intimacy: [
    instructionsPage([INTRO_NOTE, RELATIONSHIPS_NOTE]),
    INFERENCE_PAGE,
    instructionsPage([
      "<p>Before observing what the two people decide to do, we will ask you to evaluate what kind of social relationship you think they are in.</p>",
      "<p>Then, we will show you what they decide to do, and ask you to re-evaluate what kind of social relationship you think they are in.</p>",
      "<p>You will use a slider to indicate how you think the two people would describe their relationship, from maximally formal to maximally intimate.</p>",
    ]),
    finalInstructionsPage(READ_CAREFULLY_PLAIN),
  ],

  // Study 2b — jointly infer intimacy and effort (two sliders).
  food_inv_joint_ie: [
    instructionsPage([INTRO_NOTE, RELATIONSHIPS_NOTE]),
    INFERENCE_PAGE,
    instructionsPage([
      "<p>Before observing what the two people decide to do, we will ask you two questions. The first question asks you to evaluate what kind of social relationship you think they are in. The second question asks which of two situations — each described in the scenario — you think is more likely. You will answer each of the questions using a slider.</p>",
      "<p>Then, we will show you what they decide to do, and ask you to re-evaluate your answers.</p>",
    ]),
    finalInstructionsPage(READ_CAREFULLY_PLAIN),
  ],
};
