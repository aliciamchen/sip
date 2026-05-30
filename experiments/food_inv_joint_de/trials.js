// Study 1b — Joint inference over desire and effort, given intimacy.
// Design: 4 (intimacy) × 3 (observed action). Follows the noalt pattern with
// an intimacy descriptor preamble; both sliders on one page per phase. The
// desire slider is a continuous 0-100 rating ("how much do {name_0} and
// {name_1} want to eat the food?", endpoints not-at-all/extremely); the effort
// slider is a continuous 0-100 rating between the two effort paragraphs. No
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
import { makeTwoSliderForm } from "../_lib/two-slider.js";
import { STUDY_INSTRUCTIONS } from "../_lib/instructions.js";

const intimacy_texts = {
  0: "maximally formal",
  50: "neither formal nor intimate",
  75: "somewhat intimate",
  100: "maximally intimate",
};

export const CONFIG = makeConfig("food_inv_joint_de");

const INSTRUCTIONS_PAGES = STUDY_INSTRUCTIONS.food_inv_joint_de;

// Desire DV is a continuous 0-100 slider (a direct desire rating, not a
// two-states probability slider): endpoints "not at all" / "extremely".
const DESIRE_SLIDER_LABELS = ["Not at all", "Extremely"];

const effortLabels = (stim) => [
  stim.low_risk_share_effort_low,
  "Equally likely",
  stim.low_risk_share_effort_high,
];

const intimacyPreamble = (stim) =>
  `<p>Consider ${stim.name_0} and ${stim.name_1}, who would describe their relationship as <strong>${intimacy_texts[stim.intimacy_condition]}</strong>.</p>`;

function priorSliderStimulus(stimulus, stimulusIndex, stimuliLength) {
  return `
    <div>
      <h2>Scenario ${stimulusIndex + 1} of ${stimuliLength}</h2>
      <div class="vignette-text">
        ${intimacyPreamble(stimulus)}
        <p>${stimulus.vignette}</p>
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
        ${intimacyPreamble(stimulus)}
        <p>${stimulus.vignette}</p>
      </div>
      <div class="vignette-text vignette-observed">
        <p><em>They decide to take the following action:</em></p>
        <p>${stimulus[`${stimulus.action_condition}`]}</p>
      </div>
      <p><strong>Now that you have observed what they decide to do, please re-evaluate.</strong></p>
    </div>
  `;
}

const desirePrompt = (stim) =>
  `How much do you think ${stim.name_0} and ${stim.name_1} want to eat the food?`;
const EFFORT_PROMPT = "Which situation do you think is more likely?";

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
            ${intimacyPreamble(stimulus)}
          </div>
          <p style="text-align: center;"><em>Press any key to see the scenario.</em></p>
        </div>
      `,
      choices: "ALL_KEYS",
    });

    const jointSliders = [
      {
        name: "desire",
        prompt: desirePrompt(stimulus),
        labels: DESIRE_SLIDER_LABELS,
      },
      {
        name: "effort",
        prompt: EFFORT_PROMPT,
        labels: effortLabels(stimulus),
      },
    ];
    const jointData = {
      response_type: "response",
      stimulus_index: stimulusIndex,
      scenario_label: stimulus.scenario_label,
      action_condition: stimulus.action_condition,
      intimacy_condition: stimulus.intimacy_condition,
      desire_low: stimulus.desire_low,
      desire_high: stimulus.desire_high,
      low_risk_share_effort_low: stimulus.low_risk_share_effort_low,
      low_risk_share_effort_high: stimulus.low_risk_share_effort_high,
    };

    // Prior — desire + effort sliders on one page
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

    // Posterior — desire + effort sliders on one page
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
