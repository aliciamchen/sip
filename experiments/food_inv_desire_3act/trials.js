// Study 3b — Desire inference under known effort + intimacy.
// Design: 2 (effort) × 4 (intimacy: 0/50/75/100) × 3 (observed action).
// Follows the noalt pattern from food_inv_desire_intimacy_noalt: intimacy
// descriptor preamble, then vignette + effort paragraph + slider with the two
// reward paragraphs as endpoints. No candidate action list.

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

const intimacy_texts = {
  0: "0 (maximally formal)",
  50: "50 (neither formal nor intimate)",
  75: "75 (somewhat intimate)",
  100: "100 (maximally intimate)",
};

export const CONFIG = {
  ATTENTION_CHECK_INDEX: 14,
  ATTENTION_TOLERANCE: 0.02,
  INTER_TRIAL_DURATIONS: [1500, 1750, 2000],
  PIPE_EXPERIMENT_ID: "ixxsoCvjY9kH",
  PROLIFIC_COMPLETION_URL:
    "https://app.prolific.com/submissions/complete?cc=C1A889GX",
};

const getEffortText = (stim) =>
  stim.effort_condition === "low" ? stim.effort_low : stim.effort_high;

const INSTRUCTIONS_PAGES = [
  `
    <div class="instructions-container">
        <h2>Social interactions survey</h2>
        <p>In this survey, you will read vignettes about two people in different kinds of social relationships, deciding how to eat different kinds of food in different situations.</p>
        <p>Some relationships are formal, like some relationships with an employee, a religious leader, a shopkeeper or a new acquaintance. Other relationships are close and intimate, like some relationships with a romantic partner, sibling or best friend.</p>
    </div>
  `,
  `
    <div class="instructions-container">
        <h2>Social interactions survey</h2>
        <p>Before observing what action the two people decide to take, we will ask you to evaluate how likely you think two possible situations are.</p>
        <p>You will use a slider to evaluate the likelihood of the two situations. The slider will be labeled with the two situations. The farther you move it toward one side, the more likely you think that situation is, compared to the other.</p>
        <p>Then, we will show you what they decide to do, and ask you to re-evaluate how likely you think the two possible situations are.</p>
    </div>
  `,
  `
    <div class="instructions-container">
      <h2>Social interactions survey</h2>
        <p>Please pay attention to the social relationship between the two people, and read each of the scenarios and ways of eating food carefully! 🙂 You will receive $5 if you successfully complete the survey.</p>
        <p>Please do not close the window until you have completed the survey. If you do so, you will lose your progress.</p>
        <p>Press next to begin the survey.</p>
    </div>
  `,
];

function rewardSliderLabels(stimulus) {
  return [
    `<div class="slider-endpoint">${stimulus.reward_low}</div>`,
    `<div class="slider-endpoint">Equally likely</div>`,
    `<div class="slider-endpoint">${stimulus.reward_high}</div>`,
  ];
}

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
            <p>On a scale from 0 (maximally formal) to 100 (maximally intimate), ${stimulus.name_0} and ${stimulus.name_1} are in a relationship they would describe as <strong>${intimacy_texts[stimulus.intimacy_condition]}</strong>.</p>
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
            <p>On a scale from 0 (maximally formal) to 100 (maximally intimate), ${stimulus.name_0} and ${stimulus.name_1} are in a relationship they would describe as <strong>${intimacy_texts[stimulus.intimacy_condition]}</strong>.</p>
            <p>${stimulus.vignette}</p>
            <p><strong>${getEffortText(stimulus)}</strong></p>
          </div>
          <p><strong>Before observing what they decide to do, which situation do you think is more likely?</strong></p>
        </div>
      `,
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: rewardSliderLabels(stimulus),
      button_label: "Continue",
      data: {
        response_type: "response",
        stage: "prior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        intimacy_condition: stimulus.intimacy_condition,
        effort_condition: stimulus.effort_condition,
        reward_low: stimulus.reward_low,
        reward_high: stimulus.reward_high,
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
            <p>On a scale from 0 (maximally formal) to 100 (maximally intimate), ${stimulus.name_0} and ${stimulus.name_1} are in a relationship they would describe as <strong>${intimacy_texts[stimulus.intimacy_condition]}</strong>.</p>
            <p>${stimulus.vignette}</p>
            <p><strong>${getEffortText(stimulus)}</strong></p>
          </div>
          <div class="vignette-text vignette-observed">
            <p><em>They decide to take the following action:</em></p>
            <p>${stimulus[`${stimulus.action_condition}`]}</p>
          </div>
          <p><strong>Now that you have observed what they decide to do, which situation do you think is more likely?</strong></p>
        </div>
      `,
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: rewardSliderLabels(stimulus),
      button_label: "Continue",
      data: {
        response_type: "response",
        stage: "posterior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        intimacy_condition: stimulus.intimacy_condition,
        effort_condition: stimulus.effort_condition,
        reward_low: stimulus.reward_low,
        reward_high: stimulus.reward_high,
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
