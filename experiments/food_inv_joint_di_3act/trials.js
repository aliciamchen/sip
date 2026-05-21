// Study 4b — Joint inference over desire and intimacy, given effort.
// Design: 2 (effort) × 3 (observed action). Follows the noalt pattern: no
// candidate action list, only the single observed action at the posterior
// stage. No intimacy preamble (intimacy is one of the inferred variables).
// Two sliders per prior/posterior phase: desire (with reward paragraph
// endpoints) and intimacy (0-100, maximally formal -> maximally intimate).

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

export const CONFIG = {
  ATTENTION_CHECK_INDEX: 14,
  ATTENTION_TOLERANCE: 0.02,
  INTER_TRIAL_DURATIONS: [1500, 1750, 2000],
  PIPE_EXPERIMENT_ID: "w3Hb64KR0DwT",
  PROLIFIC_COMPLETION_URL:
    "https://app.prolific.com/submissions/complete?cc=TODO_FILL_IN",
};

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
        <p>Some relationships are formal, like some relationships with an employee, a religious leader, a shopkeeper or a new acquaintance. Other relationships are close and intimate, like some relationships with a romantic partner, sibling or best friend.</p>
    </div>
  `,
  `
    <div class="instructions-container">
        <h2>Social interactions survey</h2>
        <p>Before observing what action the two people decide to take, we will ask you to use two sliders. The first slider asks which of two motivational states is more likely. The second slider asks how you would describe their relationship, from 0 (maximally formal) to 100 (maximally intimate).</p>
        <p>Then, we will show you what they decide to do, and ask you to re-evaluate both sliders.</p>
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

const rewardLabels = (stim) => [
  `<div class="slider-endpoint">${stim.reward_low}</div>`,
  `<div class="slider-endpoint">Equally likely</div>`,
  `<div class="slider-endpoint">${stim.reward_high}</div>`,
];

function vignetteBlock(stimulus) {
  return `
    <div class="vignette-text">
      <p>${stimulus.vignette}</p>
      <p><strong>${getEffortText(stimulus)}</strong></p>
    </div>
  `;
}

function observedActionBlock(stimulus) {
  return `
    <div class="vignette-text vignette-observed">
      <p><em>They decide to take the following action:</em></p>
      <p>${stimulus[`${stimulus.action_condition}`]}</p>
    </div>
  `;
}

function makeStimulusTrials(jsPsych, stimuli) {
  const trials = [];

  stimuli.forEach((stimulus, stimulusIndex) => {
    if (stimulusIndex === CONFIG.ATTENTION_CHECK_INDEX) {
      trials.push(makeAttentionCheckSingleSlider(CONFIG.ATTENTION_TOLERANCE));
    }

    // Prior — desire slider
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          ${vignetteBlock(stimulus)}
          <p><strong>Before observing what they decide to do, which situation do you think is more likely?</strong></p>
        </div>
      `,
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: rewardLabels(stimulus),
      button_label: "Continue",
      data: {
        response_type: "response",
        response_target: "reward",
        stage: "prior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        effort_condition: stimulus.effort_condition,
        reward_low: stimulus.reward_low,
        reward_high: stimulus.reward_high,
      },
    });

    // Prior — intimacy slider
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          ${vignetteBlock(stimulus)}
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
        response_target: "intimacy",
        stage: "prior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        effort_condition: stimulus.effort_condition,
      },
    });

    trials.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: "",
      choices: "NO_KEYS",
      trial_duration: 1000,
    });

    // Posterior — desire slider
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          ${vignetteBlock(stimulus)}
          ${observedActionBlock(stimulus)}
          <p><strong>Now that you have observed what they decide to do, which situation do you think is more likely?</strong></p>
        </div>
      `,
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: rewardLabels(stimulus),
      button_label: "Continue",
      data: {
        response_type: "response",
        response_target: "reward",
        stage: "posterior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        effort_condition: stimulus.effort_condition,
        reward_low: stimulus.reward_low,
        reward_high: stimulus.reward_high,
      },
    });

    // Posterior — intimacy slider
    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          ${vignetteBlock(stimulus)}
          ${observedActionBlock(stimulus)}
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
        response_target: "intimacy",
        stage: "posterior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
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
