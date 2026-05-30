// Study 2 — Inverse intimacy under known desire + effort.
// Design: 2 (desire: low/high) × 2 (effort: low/high) × 3 (observed action).
// Follows the noalt pattern from food_inv_intimacy_desire_noalt: no candidate
// action list shown; the participant sees only the single observed action at
// the posterior stage.

import {
  makeConsentScreen,
  makeInstructionsScreen,
  makeInterTrialBlank,
  makeExitSurvey,
  makeSaveData,
  makeThankYou,
} from "../_lib/timeline.js";
import { makeAttentionCheckSingleSlider } from "../_lib/attention-check.js";
import { makeMemoryCheckForStimulus } from "../_lib/memory-checks.js";
import { makeConfig } from "../_lib/config.js";

export const CONFIG = makeConfig("food_inv_intimacy");

const getDesireText = (stim) =>
  stim.desire_condition === "low" ? stim.desire_low : stim.desire_high;
const getEffortText = (stim) =>
  stim.effort_condition === "low" ? stim.effort_low : stim.effort_high;

const INTIMACY_LABELS = [
  "0<br>Maximally formal",
  "50<br>Neither formal nor intimate",
  "100<br>Maximally intimate",
];

const INSTRUCTIONS_PAGES = [
  `
    <div class="instructions-container">
        <h2>Social interactions survey</h2>
        <p>In this survey, you will read vignettes about two people deciding how to eat different kinds of food in different situations.</p>
        <p>Before observing what the two people decide to do, we will ask you to evaluate what kind of social relationship you think they are in.</p>
        <p>Then, we will show you what they decide to do, and ask you to re-evaluate what kind of social relationship you think they are in.</p>
    </div>
  `,
  `
    <div class="instructions-container">
        <h2>Social interactions survey</h2>
        <p>Some relationships are formal, like some relationships with an employee, a religious leader, a shopkeeper or a new acquaintance. Other relationships are close and intimate, like some relationships with a romantic partner, sibling or best friend.</p>
        <p>You will use sliders to indicate how you think the two people would describe their relationship, from a scale of 0 (maximally formal) to 100 (maximally intimate).</p>
    </div>
  `,
  `
    <div class="instructions-container">
      <h2>Social interactions survey</h2>
        <p>Please read each of the scenarios and ways of eating food carefully! 🙂 You will receive $5 if you successfully complete the survey.</p>
        <p>Please do not close the window until you have completed the survey. If you do so, you will lose your progress.</p>
        <p>Press next to begin the survey.</p>
    </div>
  `,
];

function makeStimulusTrials(jsPsych, stimuli) {
  const trials = [];

  stimuli.forEach((stimulus, stimulusIndex) => {
    if (stimulusIndex === CONFIG.ATTENTION_CHECK_INDEX) {
      trials.push(makeAttentionCheckSingleSlider(CONFIG.ATTENTION_TOLERANCE));
    }

    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          <div class="vignette-text">
            <p>${stimulus.vignette}</p>
            <p><strong>${getDesireText(stimulus)}</strong></p>
            <p><strong>${getEffortText(stimulus)}</strong></p>
          </div>
          <p><strong>Before observing what they decide to do, how do you think ${stimulus.name_0} and ${stimulus.name_1} would describe their relationship, on a scale from 0 (maximally formal) to 100 (maximally intimate)?</strong></p>
        </div>
      `,
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: INTIMACY_LABELS,
      button_label: "Continue",
      data: {
        response_type: "response",
        stage: "prior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        desire_condition: stimulus.desire_condition,
        effort_condition: stimulus.effort_condition,
      },
    });

    trials.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: "",
      choices: "NO_KEYS",
      trial_duration: 1000,
    });

    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          <div class="vignette-text">
            <p>${stimulus.vignette}</p>
            <p><strong>${getDesireText(stimulus)}</strong></p>
            <p><strong>${getEffortText(stimulus)}</strong></p>
          </div>
          <div class="vignette-text vignette-observed">
            <p><em>They decide to take the following action:</em></p>
            <p>${stimulus[`${stimulus.action_condition}`]}</p>
          </div>
          <p><strong>Now that you have observed what they decide to do, how do you think ${stimulus.name_0} and ${stimulus.name_1} would describe their relationship, on a scale from 0 (maximally formal) to 100 (maximally intimate)?</strong></p>
        </div>
      `,
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: INTIMACY_LABELS,
      button_label: "Continue",
      data: {
        response_type: "response",
        stage: "posterior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        desire_condition: stimulus.desire_condition,
        effort_condition: stimulus.effort_condition,
      },
    });

    const memoryCheck = makeMemoryCheckForStimulus(stimulus);
    if (memoryCheck) trials.push(memoryCheck);

    trials.push(makeInterTrialBlank(jsPsych, CONFIG.INTER_TRIAL_DURATIONS));
  });

  return trials;
}

export function makeTimeline(
  jsPsych,
  stimuli,
  consentHtml,
  exitSurveyHtml,
  subjectId,
) {
  return [
    makeConsentScreen(consentHtml),
    makeInstructionsScreen(INSTRUCTIONS_PAGES),
    ...makeStimulusTrials(jsPsych, stimuli),
    makeExitSurvey(jsPsych, exitSurveyHtml),
    makeSaveData(jsPsych, CONFIG.PIPE_EXPERIMENT_ID, subjectId),
    makeThankYou(CONFIG.PROLIFIC_COMPLETION_URL),
  ];
}
