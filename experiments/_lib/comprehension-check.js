// Comprehension check shown right after the instructions, before the trials.
// Organized like instructions.js: shared question blocks composed into a
// per-study array in STUDY_COMPREHENSION_CHECKS. Each study's questions capture
// the handful of things a participant must internalize from the instructions —
// that they're inferring something unstated, that they rate before and after
// seeing the decision, and what the study's slider(s) actually mean.
//
// makeComprehensionGate(jsPsych, {...}) turns those questions into a gated
// nested timeline: the participant gets a fixed number of attempts (default 3),
// re-reading the instructions and retrying on each miss. If they never get all
// questions right, the experiment ends (jsPsych.abortExperiment) on a message
// asking them to return the study on Prolific — so no data is saved for them.

import { makeInstructionsScreen } from "./timeline.js";

export const COMPREHENSION_MAX_ATTEMPTS = 3;

// --- Shared questions -------------------------------------------------------
// Each question is { prompt, name, options, correct }, where `correct` is the
// exact option string. makeComprehensionGate strips `correct` before handing the
// questions to the plugin and uses it to grade in on_finish.

// The task is inference, not recall (tracks INFERENCE_NOTE in instructions.js).
const INFERENCE_Q = {
  prompt:
    "In each scenario, two people decide what to do in a situation involving food. What will we ask you to do?",
  name: "inference",
  options: [
    "Use the situation and what they decide to do, to judge something the scenario doesn't directly tell us.",
    "Type out everything the scenario said, word for word.",
    "Decide what the two people should have done.",
    "Rate how realistic or well-written the scenario is.",
  ],
  correct:
    "Use the situation and what they decide to do, to judge something the scenario doesn't directly tell us.",
};

// Each rating is given before and after the decision is revealed (tracks
// TIMING_NOTE_ONE / TIMING_NOTE_TWO).
const TIMING_Q = {
  prompt: "How many times will you give each rating?",
  name: "timing",
  options: [
    "Twice — once before and once after we show you what the two people decide to do.",
    "Once, before we show you what they decide to do.",
    "Once, after we show you what they decide to do.",
    "Only when I am unsure of the answer.",
  ],
  correct:
    "Twice — once before and once after we show you what the two people decide to do.",
};

// "Liking the food" is about the food's appeal / their current motivational
// state (hunger, interest), separate from what they decide to do and from how
// hard the food is to get (tracks DESIRE_NOTE). Shown in the desire studies (1a, 1b).
const DESIRE_MEANING_Q = {
  prompt:
    "When we ask how much the two people would like the food, what do we mean?",
  name: "desire_meaning",
  options: [
    "How much they would enjoy eating it right now — for example because they are hungry or like that kind of food — separate from what they decide to do.",
    "Whether they actually end up eating the food.",
    "How easy or hard it would be for them to get or eat the food.",
    "How close or formal the two people's relationship is.",
  ],
  correct:
    "How much they would enjoy eating it right now — for example because they are hungry or like that kind of food — separate from what they decide to do.",
};

// The relationship scale runs formal -> intimate (tracks RELATIONSHIP_SLIDER_NOTE
// / RELATIONSHIPS_NOTE). Shown in the intimacy studies (2a, 2b).
const INTIMACY_SCALE_Q = {
  prompt:
    "When we ask about the two people's relationship, what does the scale measure?",
  name: "intimacy_scale",
  options: [
    "How formal or intimate the relationship is — from maximally formal to maximally intimate.",
    "How much the two people would like the food.",
    "How much physical effort their action would take.",
    "How long the two people have known each other, in years.",
  ],
  correct:
    "How formal or intimate the relationship is — from maximally formal to maximally intimate.",
};

// --- Nonfood (Study 3) variants -----------------------------------------------
// Mirror the food questions with the domain framing swapped (tracks the
// NONFOOD_* paragraphs in instructions.js).

const NONFOOD_INFERENCE_Q = {
  prompt:
    "In each scenario, two people decide what to do in a situation where something — an object, a physical space, or a piece of information — could be shared between them. What will we ask you to do?",
  name: "inference",
  options: [
    "Use the situation and what they decide to do, to judge something the scenario doesn't directly tell us.",
    "Type out everything the scenario said, word for word.",
    "Decide what the two people should have done.",
    "Rate how realistic or well-written the scenario is.",
  ],
  correct:
    "Use the situation and what they decide to do, to judge something the scenario doesn't directly tell us.",
};

const NONFOOD_DESIRE_MEANING_Q = {
  prompt:
    "When we ask how much the two people would like the thing at stake in the scenario, what do we mean?",
  name: "desire_meaning",
  options: [
    "How much they would enjoy getting or doing it right now — for example because they need it or are interested in it — separate from what they decide to do.",
    "Whether they actually end up getting it.",
    "How easy or hard it would be for them to get it.",
    "How close or formal the two people's relationship is.",
  ],
  correct:
    "How much they would enjoy getting or doing it right now — for example because they need it or are interested in it — separate from what they decide to do.",
};

const NONFOOD_INTIMACY_SCALE_Q = {
  prompt:
    "When we ask about the two people's relationship, what does the scale measure?",
  name: "intimacy_scale",
  options: [
    "How formal or intimate the relationship is — from maximally formal to maximally intimate.",
    "How much the two people would like the thing at stake in the scenario.",
    "How much physical effort their action would take.",
    "How long the two people have known each other, in years.",
  ],
  correct:
    "How formal or intimate the relationship is — from maximally formal to maximally intimate.",
};

// Negative-format item: three things the joint task really asks for (the
// "which situation is more likely" slider, the before/after timing, and using
// judgment about unstated details) plus one thing it never asks for, which is
// the answer. Shown in the joint studies (1b, 2b, 3a, 3b); the "which situation
// is more likely" option is only a real task action in these studies.
const SECOND_QUESTION_Q = {
  prompt: "Which of these will you NOT be asked to do in this task?",
  name: "not_asked",
  options: [
    "Judge which of two possible situations is more likely.",
    "Predict what the two people will do next.",
    "Give your answers both before and after seeing what the two people decide.",
    "Base your answers on your best judgment, since some details are not stated.",
  ],
  correct: "Predict what the two people will do next.",
};

// --- Per-study comprehension checks -----------------------------------------
// Each study's array mirrors the structure of its instructions: the shared
// inference + timing items, plus the slider-meaning item(s) specific to it.

export const STUDY_COMPREHENSION_CHECKS = {
  food_inv_desire: [INFERENCE_Q, DESIRE_MEANING_Q, TIMING_Q],
  food_inv_joint_de: [
    INFERENCE_Q,
    DESIRE_MEANING_Q,
    SECOND_QUESTION_Q,
    TIMING_Q,
  ],
  food_inv_intimacy: [INFERENCE_Q, INTIMACY_SCALE_Q, TIMING_Q],
  food_inv_joint_ie: [
    INFERENCE_Q,
    INTIMACY_SCALE_Q,
    SECOND_QUESTION_Q,
    TIMING_Q,
  ],
  nonfood_inv_joint_de: [
    NONFOOD_INFERENCE_Q,
    NONFOOD_DESIRE_MEANING_Q,
    SECOND_QUESTION_Q,
    TIMING_Q,
  ],
  nonfood_inv_joint_ie: [
    NONFOOD_INFERENCE_Q,
    NONFOOD_INTIMACY_SCALE_Q,
    SECOND_QUESTION_Q,
    TIMING_Q,
  ],
};

// --- The gate ---------------------------------------------------------------

const INTRO_PREAMBLE = `
  <div>
    <h3>Comprehension check</h3>
    <p>Please answer the following questions about the task. You must answer all of them correctly to begin the study.</p>
  </div>
`;

// Full-page notice shown at the start of each retry (attempts > 0), before the
// instructions are re-shown, so the participant knows why they're back and how
// many attempts remain.
function retryNoticeHtml(remaining) {
  const tries = remaining === 1 ? "attempt" : "attempts";
  return `
    <div class="instructions-container">
      <h2>Not quite — let's try again</h2>
      <p>One or more of your answers was incorrect. Please read the instructions again carefully. You have <strong>${remaining} ${tries}</strong> remaining.</p>
      <p>Press any key to review the instructions.</p>
    </div>
  `;
}

function failHtml(maxAttempts) {
  return `
    <div class="instructions-container">
      <h2>Unable to continue</h2>
      <p>Unfortunately, you did not answer the comprehension questions correctly within ${maxAttempts} attempts, so you are not able to continue with this study.</p>
      <p>Please return your submission on Prolific. You will not be penalized for returning the study.</p>
      <p>Thank you for your time.</p>
    </div>
  `;
}

// Build the comprehension gate: a nested timeline of [instructions, check] that
// loops while the participant has attempts left and hasn't passed, and ends the
// experiment on a return-to-Prolific message once attempts run out. Replaces the
// standalone instructions screen in the timeline, so the instructions are shown
// (and re-shown on each retry) as part of the gate.
export function makeComprehensionGate(
  jsPsych,
  { instructionsPages, questions, maxAttempts = COMPREHENSION_MAX_ATTEMPTS },
) {
  // Closure state shared between the retry notice, the check trial, and the
  // loop_function.
  let attempts = 0;
  let passed = false;

  // Shown only on retries (attempts > 0), at the top of the loop before the
  // instructions are re-shown. conditional_function is re-evaluated each loop.
  const retryNotice = {
    timeline: [
      {
        type: jsPsychHtmlKeyboardResponse,
        choices: "ALL_KEYS",
        stimulus: () => retryNoticeHtml(maxAttempts - attempts),
      },
    ],
    conditional_function: () => attempts > 0,
  };

  const checkTrial = {
    type: jsPsychSurveyMultiChoice,
    preamble: INTRO_PREAMBLE,
    questions: questions.map((q) => ({
      prompt: q.prompt,
      name: q.name,
      options: q.options,
      required: true,
    })),
    button_label: "Continue",
    data: { response_type: "comprehension_check" },
    on_finish: function (data) {
      const responses = data.response || {};
      let correctCount = 0;
      questions.forEach((q) => {
        const ok = responses[q.name] === q.correct ? 1 : 0;
        data[`comprehension_${q.name}`] = ok;
        correctCount += ok;
      });
      attempts += 1;
      passed = correctCount === questions.length;
      data.comprehension_correct_count = correctCount;
      data.comprehension_passed = passed;
      data.comprehension_attempt = attempts;
      // Out of attempts and still not passed: end the experiment. abortExperiment
      // skips the remaining timeline (trials, exit survey, and the DataPipe save),
      // so failed participants leave no saved data and are told to return the study.
      if (!passed && attempts >= maxAttempts) {
        jsPsych.abortExperiment(failHtml(maxAttempts));
      }
    },
  };

  return {
    timeline: [
      retryNotice,
      makeInstructionsScreen(instructionsPages),
      checkTrial,
    ],
    loop_function: () => !passed && attempts < maxAttempts,
  };
}

// Confirmation shown once after the participant passes the gate, just before the
// trials begin. Goes in the timeline right after the gate — participants who fail
// are ended inside the gate (abortExperiment) and never reach this page.
export function makeComprehensionPassPage() {
  return {
    type: jsPsychHtmlKeyboardResponse,
    choices: "ALL_KEYS",
    stimulus: `
      <div class="instructions-container">
        <h2>Congrats, you have passed the comprehension check!</h2>
        <p>Press any key to begin the experiment.</p>
      </div>
    `,
  };
}
