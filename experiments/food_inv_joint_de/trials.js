// Study 1b — Joint inference over desire and effort, given intimacy.
// Design: 4 (intimacy) × 3 (observed action). Follows the noalt pattern with
// an intimacy descriptor preamble; two sliders per prior/posterior phase. The
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

const intimacy_texts = {
  0: "maximally formal",
  50: "neither formal nor intimate",
  75: "somewhat intimate",
  100: "maximally intimate",
};

export const CONFIG = makeConfig("food_inv_joint_de");

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
        <p>Before observing what action the two people decide to take, we will ask you two questions using two sliders. The first asks how much you think they want to eat the food, on a scale from 0 ("not at all") to 100 ("extremely"). The second asks which of two physical situations you think is more likely.</p>
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

// Desire DV is a continuous 0-100 slider (a direct desire rating, not a
// two-states probability slider): endpoints "not at all" / "extremely".
const DESIRE_SLIDER_LABELS = ["0<br>not at all", "100<br>extremely"];

const effortLabels = (stim) => [
  `<div class="slider-endpoint">${stim.effort_low}</div>`,
  `<div class="slider-endpoint">Equally likely</div>`,
  `<div class="slider-endpoint">${stim.effort_high}</div>`,
];

const intimacyPreamble = (stim) =>
  `<p>${stim.name_0} and ${stim.name_1} would describe their relationship as <strong>${intimacy_texts[stim.intimacy_condition]}</strong>.</p>`;

function priorSliderStimulus(stimulus, stimulusIndex, stimuliLength) {
  return `
    <div>
      <h2>Scenario ${stimulusIndex + 1} of ${stimuliLength}</h2>
      <div class="vignette-text">
        ${intimacyPreamble(stimulus)}
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
  `<p>How much do you think ${stim.name_0} and ${stim.name_1} want to eat the food?</p>`;
const EFFORT_PROMPT = "<p>Which physical situation do you think is more likely?</p>";

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

    // Prior — desire (reward) slider, continuous 0-100
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: priorSliderStimulus(stimulus, stimulusIndex, stimuli.length),
      prompt: desirePrompt(stimulus),
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: DESIRE_SLIDER_LABELS,
      button_label: "Continue",
      data: {
        response_type: "response",
        response_target: "reward",
        stage: "prior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        intimacy_condition: stimulus.intimacy_condition,
        reward_low: stimulus.reward_low,
        reward_high: stimulus.reward_high,
      },
    });

    // Prior — effort slider (continuous 0-100 rating between two states)
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: priorSliderStimulus(stimulus, stimulusIndex, stimuli.length),
      prompt: EFFORT_PROMPT,
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
        intimacy_condition: stimulus.intimacy_condition,
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

    // Posterior — desire (reward) slider, continuous 0-100
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: posteriorSliderStimulus(stimulus, stimulusIndex, stimuli.length),
      prompt: desirePrompt(stimulus),
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: DESIRE_SLIDER_LABELS,
      button_label: "Continue",
      data: {
        response_type: "response",
        response_target: "reward",
        stage: "posterior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        intimacy_condition: stimulus.intimacy_condition,
        reward_low: stimulus.reward_low,
        reward_high: stimulus.reward_high,
      },
    });

    // Posterior — effort slider (continuous 0-100 rating between two states)
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: posteriorSliderStimulus(stimulus, stimulusIndex, stimuli.length),
      prompt: EFFORT_PROMPT,
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
        intimacy_condition: stimulus.intimacy_condition,
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
