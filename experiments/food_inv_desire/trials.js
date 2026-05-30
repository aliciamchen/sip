// Study 1a — Desire inference under known effort + intimacy.
// Design: 2 (effort) × 4 (intimacy: 0/50/75/100) × 3 (observed action).
// Follows the noalt pattern: intimacy descriptor preamble, then vignette +
// effort paragraph + a continuous 0-100 slider ("how much do {name_0} and
// {name_1} want to eat the food?", endpoints not-at-all/extremely) on which
// participants rate desire before and after observing the single action. The
// desire paragraph is NOT shown (desire is the target of inference). No
// candidate action list.

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

const intimacy_texts = {
  0: "maximally formal",
  50: "neither formal nor intimate",
  75: "somewhat intimate",
  100: "maximally intimate",
};

export const CONFIG = makeConfig("food_inv_desire");

const getEffortText = (stim) =>
  stim.effort_condition === "low" ? stim.low_risk_share_effort_low : stim.low_risk_share_effort_high;

const INSTRUCTIONS_PAGES = STUDY_INSTRUCTIONS.food_inv_desire;

// Continuous 0-100 desire slider: endpoints labeled "not at all" / "extremely"
// (no numbers shown).
const DESIRE_SLIDER_LABELS = ["Not at all", "Extremely"];

function makeStimulusTrials(jsPsych, stimuli) {
  const trials = [];

  stimuli.forEach((stimulus, stimulusIndex) => {
    if (stimulusIndex === CONFIG.ATTENTION_CHECK_INDEX) {
      trials.push(makeAttentionCheckSingleSlider(CONFIG.ATTENTION_TOLERANCE));
    }

    trials.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          <div class="vignette-text">
            <p>Consider ${stimulus.name_0} and ${stimulus.name_1}, who would describe their relationship as <strong>${intimacy_texts[stimulus.intimacy_condition]}</strong>.</p>
          </div>
          <p style="text-align: center;"><em>Press any key to see the scenario.</em></p>
        </div>
      `,
      choices: "ALL_KEYS",
    });

    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          <div class="vignette-text">
            <p>Consider ${stimulus.name_0} and ${stimulus.name_1}, who would describe their relationship as <strong>${intimacy_texts[stimulus.intimacy_condition]}</strong>.</p>
            <p>${stimulus.vignette}</p>
            <p><strong>${getEffortText(stimulus)}</strong></p>
          </div>
          <p><strong>Before observing what they decide to do, how much do you think ${stimulus.name_0} and ${stimulus.name_1} want to eat the food?</strong></p>
        </div>
      `,
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: DESIRE_SLIDER_LABELS,
      button_label: "Continue",
      data: {
        response_type: "response",
        stage: "prior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        intimacy_condition: stimulus.intimacy_condition,
        effort_condition: stimulus.effort_condition,
        desire_low: stimulus.desire_low,
        desire_high: stimulus.desire_high,
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
            <p>Consider ${stimulus.name_0} and ${stimulus.name_1}, who would describe their relationship as <strong>${intimacy_texts[stimulus.intimacy_condition]}</strong>.</p>
            <p>${stimulus.vignette}</p>
            <p><strong>${getEffortText(stimulus)}</strong></p>
          </div>
          <div class="vignette-text vignette-observed">
            <p><em>They decide to take the following action:</em></p>
            <p>${stimulus[`${stimulus.action_condition}`]}</p>
          </div>
          <p><strong>Now that you have observed what they decide to do, how much do you think ${stimulus.name_0} and ${stimulus.name_1} want to eat the food?</strong></p>
        </div>
      `,
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: DESIRE_SLIDER_LABELS,
      button_label: "Continue",
      data: {
        response_type: "response",
        stage: "posterior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        intimacy_condition: stimulus.intimacy_condition,
        effort_condition: stimulus.effort_condition,
        desire_low: stimulus.desire_low,
        desire_high: stimulus.desire_high,
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
