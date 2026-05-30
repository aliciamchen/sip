// Study 2b — Joint inference over intimacy and effort, given desire.
// Design: 2 (desire) × 3 (observed action). Follows the noalt pattern with a
// desire/reward paragraph as preamble; two sliders per prior/posterior phase
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

export const CONFIG = makeConfig("food_inv_joint_ie");

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
        <p>Before observing what action the two people decide to take, we will ask you to evaluate, using two sliders, two things about their situation. The first slider asks about the social relationship between them. The second slider asks about their physical situation.</p>
        <p>Then, we will show you what they decide to do, and ask you to re-evaluate both sliders.</p>
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

const getRewardText = (stim) =>
  stim.reward_condition === "low" ? stim.reward_low : stim.reward_high;

const INTIMACY_LABELS = [
  "0<br>Maximally formal",
  "50<br>Neither formal nor intimate",
  "100<br>Maximally intimate",
];

const effortLabels = (stim) => [
  `<div class="slider-endpoint">${stim.effort_low}</div>`,
  `<div class="slider-endpoint">Equally likely</div>`,
  `<div class="slider-endpoint">${stim.effort_high}</div>`,
];

const rewardPreamble = (stim) => `<p>${getRewardText(stim)}</p>`;

function priorSliderStimulus(stimulus, stimulusIndex, stimuliLength) {
  return `
    <div>
      <h2>Scenario ${stimulusIndex + 1} of ${stimuliLength}</h2>
      <div class="vignette-text">
        ${rewardPreamble(stimulus)}
        <p>${stimulus.vignette}</p>
      </div>
      <p><strong>Before observing what they decide to do, please answer the question below.</strong></p>
    </div>
  `;
}

function posteriorSliderStimulus(stimulus, stimulusIndex, stimuliLength) {
  return `
    <div>
      <h2>Scenario ${stimulusIndex + 1} of ${stimuliLength}</h2>
      <div class="vignette-text">
        ${rewardPreamble(stimulus)}
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

function makeStimulusTrials(jsPsych, stimuli) {
  const trials = [];

  stimuli.forEach((stimulus, stimulusIndex) => {
    if (stimulusIndex === CONFIG.ATTENTION_CHECK_INDEX) {
      trials.push(makeAttentionCheckSingleSlider(CONFIG.ATTENTION_TOLERANCE));
    }

    // Prior — intimacy slider (0–100 numeric)
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: priorSliderStimulus(stimulus, stimulusIndex, stimuli.length),
      prompt: "<p>How would they describe their relationship?</p>",
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: INTIMACY_LABELS,
      button_label: "Continue",
      data: {
        response_type: "response",
        response_target: "intimacy",
        stage: "prior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        reward_condition: stimulus.reward_condition,
      },
    });

    // Prior — effort slider (paragraph endpoints)
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: priorSliderStimulus(stimulus, stimulusIndex, stimuli.length),
      prompt: "<p>Which physical situation do you think is more likely?</p>",
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: effortLabels(stimulus),
      button_label: "Continue",
      data: {
        response_type: "response",
        response_target: "effort",
        stage: "prior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        reward_condition: stimulus.reward_condition,
        effort_low: stimulus.effort_low,
        effort_high: stimulus.effort_high,
      },
    });

    trials.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: "",
      choices: "NO_KEYS",
      trial_duration: 1000,
    });

    // Posterior — intimacy slider (0–100 numeric)
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: posteriorSliderStimulus(stimulus, stimulusIndex, stimuli.length),
      prompt: "<p>How would they describe their relationship?</p>",
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: INTIMACY_LABELS,
      button_label: "Continue",
      data: {
        response_type: "response",
        response_target: "intimacy",
        stage: "posterior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        reward_condition: stimulus.reward_condition,
      },
    });

    // Posterior — effort slider (paragraph endpoints)
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: posteriorSliderStimulus(stimulus, stimulusIndex, stimuli.length),
      prompt: "<p>Which physical situation do you think is more likely?</p>",
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: effortLabels(stimulus),
      button_label: "Continue",
      data: {
        response_type: "response",
        response_target: "effort",
        stage: "posterior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        reward_condition: stimulus.reward_condition,
        effort_low: stimulus.effort_low,
        effort_high: stimulus.effort_high,
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
