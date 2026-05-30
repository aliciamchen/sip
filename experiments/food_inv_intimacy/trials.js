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
import { STUDY_INSTRUCTIONS } from "../_lib/instructions.js";

export const CONFIG = makeConfig("food_inv_intimacy");

const getDesireText = (stim) =>
  stim.desire_condition === "low" ? stim.desire_low : stim.desire_high;
const getEffortText = (stim) =>
  stim.effort_condition === "low" ? stim.low_risk_share_effort_low : stim.low_risk_share_effort_high;

const INTIMACY_LABELS = [
  "Maximally formal",
  "Neither formal nor intimate",
  "Maximally intimate",
];

const INSTRUCTIONS_PAGES = STUDY_INSTRUCTIONS.food_inv_intimacy;

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
            <p>${getDesireText(stimulus)}</p>
            <p>${getEffortText(stimulus)}</p>
          </div>
          <p><strong>Before observing what they decide to do, how do you think ${stimulus.name_0} and ${stimulus.name_1} would describe their relationship?</strong></p>
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
            <p>${getDesireText(stimulus)}</p>
            <p>${getEffortText(stimulus)}</p>
          </div>
          <div class="vignette-text vignette-observed">
            <p><em>They decide to take the following action:</em></p>
            <p>${stimulus[`${stimulus.action_condition}`]}</p>
          </div>
          <p><strong>Now that you have observed what they decide to do, how do you think ${stimulus.name_0} and ${stimulus.name_1} would describe their relationship?</strong></p>
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
