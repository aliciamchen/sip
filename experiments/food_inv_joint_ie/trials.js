// Study 2b — Joint inference over intimacy and effort, given desire.
// Design: 2 (desire) × 3 (observed action). Follows the noalt pattern with a
// desire paragraph as preamble; both sliders on one page per phase
// (one for intimacy 0–100 numeric, one for effort with paragraph endpoints).
// No candidate action list.

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
import { makeTwoSliderForm } from "../_lib/two-slider.js";
import { STUDY_INSTRUCTIONS } from "../_lib/instructions.js";

export const CONFIG = makeConfig("food_inv_joint_ie");

const INSTRUCTIONS_PAGES = STUDY_INSTRUCTIONS.food_inv_joint_ie;

const getDesireText = (stim) =>
  stim.desire_condition === "low" ? stim.desire_low : stim.desire_high;

const INTIMACY_LABELS = [
  "Maximally formal",
  "Neither formal nor intimate",
  "Maximally intimate",
];

const effortLabels = (stim) => [
  stim.low_risk_share_effort_low,
  "Equally likely",
  stim.low_risk_share_effort_high,
];

const desirePreamble = (stim) => `<p>${getDesireText(stim)}</p>`;

function priorSliderStimulus(stimulus, stimulusIndex, stimuliLength) {
  return `
    <div>
      <h2>Scenario ${stimulusIndex + 1} of ${stimuliLength}</h2>
      <div class="vignette-text">
        <p>${stimulus.vignette}</p>
        ${desirePreamble(stimulus)}
      </div>
      <p><strong>Before observing what they decide to do, please answer the questions below.</strong></p>
    </div>
  `;
}

function posteriorSliderStimulus(stimulus, stimulusIndex, stimuliLength) {
  return `
    <div>
      <h2>Scenario ${stimulusIndex + 1} of ${stimuliLength}</h2>
      <div class="vignette-text">
        <p>${stimulus.vignette}</p>
        ${desirePreamble(stimulus)}
      </div>
      <div class="vignette-text vignette-observed">
        <p><em>They decide to take the following action:</em></p>
        <p>${stimulus[`${stimulus.action_condition}`]}</p>
      </div>
      <p><strong>Now that you have observed what they decide to do, please re-evaluate.</strong></p>
    </div>
  `;
}

function makeStimulusTrials(jsPsych, stimuli) {
  const trials = [];

  stimuli.forEach((stimulus, stimulusIndex) => {
    if (stimulusIndex === CONFIG.ATTENTION_CHECK_INDEX) {
      trials.push(makeAttentionCheckSingleSlider(CONFIG.ATTENTION_TOLERANCE));
    }

    const jointSliders = [
      {
        name: "intimacy",
        prompt: `How would ${stimulus.name_0} and ${stimulus.name_1} describe their relationship?`,
        labels: INTIMACY_LABELS,
      },
      {
        name: "effort",
        prompt: "Which situation do you think is more likely?",
        labels: effortLabels(stimulus),
      },
    ];
    const jointData = {
      response_type: "response",
      stimulus_index: stimulusIndex,
      scenario_label: stimulus.scenario_label,
      action_condition: stimulus.action_condition,
      desire_condition: stimulus.desire_condition,
      low_risk_share_effort_low: stimulus.low_risk_share_effort_low,
      low_risk_share_effort_high: stimulus.low_risk_share_effort_high,
    };

    // Prior — intimacy + effort sliders on one page
    trials.push(
      makeTwoSliderForm({
        preamble: priorSliderStimulus(stimulus, stimulusIndex, stimuli.length),
        sliders: jointSliders,
        data: { ...jointData, stage: "prior" },
      }),
    );

    trials.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: "",
      choices: "NO_KEYS",
      trial_duration: 1000,
    });

    // Posterior — intimacy + effort sliders on one page
    trials.push(
      makeTwoSliderForm({
        preamble: posteriorSliderStimulus(
          stimulus,
          stimulusIndex,
          stimuli.length,
        ),
        sliders: jointSliders,
        data: { ...jointData, stage: "posterior" },
      }),
    );

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
