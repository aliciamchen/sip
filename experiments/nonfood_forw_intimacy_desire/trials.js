import {
  makeConsentScreen,
  makeInstructionsScreen,
  makeInterTrialBlank,
  makeExitSurvey,
  makeSaveData,
  makeThankYou,
} from "../_lib/timeline.js";
import { makeAttentionCheckProbabilitySliders } from "../_lib/attention-check.js";
import {
  makeMemoryCheckForStimulus,
  NONFOOD_MEMORY_CHECKS,
} from "../_lib/memory-checks.js";

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
  PIPE_EXPERIMENT_ID: "yPiVa0yxh01E",
  PROLIFIC_COMPLETION_URL:
    "https://app.prolific.com/submissions/complete?cc=C1A889GX",
};

const getRewardText = (stimulus) =>
  stimulus.reward_condition === "low"
    ? stimulus.reward_low
    : stimulus.reward_high;

const INSTRUCTIONS_PAGES = [
  `
    <div class="instructions-container">
        <h2>Social interactions survey</h2>
        <p>In this survey, you will read vignettes about two people in different kinds of social relationships, in different kinds of social situations.</p>
        <p>Some relationships are formal, like some relationships with an employee, a religious leader, a shopkeeper or a new acquaintance. Other relationships are close and intimate, like some relationships with a romantic partner, sibling or best friend.</p>
    </div>
  `,
  `
    <div class="instructions-container">
        <h2>Social interactions survey</h2>
        <p>For each scenario, you will read about four different actions the two people can take. You will use sliders to indicate the probability that the two people will choose each action. The probabilities must sum to 100%. You can move sliders freely, and when you release a slider, all values will be automatically adjusted to sum to 100%.</p>
        <p>(Note that this means that sometimes you might have to move the sliders multiple times to get to the probabilities you want.)</p>
    </div>
  `,
  `
    <div class="instructions-container">
      <h2>Social interactions survey</h2>
        <p>Please pay attention to the social relationship between the two people, and read each of the scenarios and the actions carefully! 🙂 You will receive $6.25 if you successfully complete the survey. </p>
        <p>Please do not close the window until you have completed the survey. If you do so, you will lose your progress.</p>
        <p>Press next to begin the survey.</p>
    </div>
  `,
];

const ATTENTION_CHECK_TARGETS = [0, 0, 0.25, 0.75];

function makeStimulusTrials(jsPsych, stimuli) {
  const trials = [];

  stimuli.forEach((stimulus, stimulusIndex) => {
    if (stimulusIndex === CONFIG.ATTENTION_CHECK_INDEX) {
      trials.push(
        makeAttentionCheckProbabilitySliders(
          CONFIG.ATTENTION_TOLERANCE,
          ATTENTION_CHECK_TARGETS,
        )
      );
    }

    trials.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          <div class="vignette-text">
            <p>On a scale from 0 (maximally formal) to 100 (maximally intimate), ${stimulus.name_0} and ${stimulus.name_1} are in a relationship they would describe as <strong>${intimacy_texts[stimulus.intimacy_condition]}</strong>.</p>
            <p>${stimulus.vignette}</p>
            <p>${getRewardText(stimulus)}</p>
          </div>
          <p style="text-align: center;"><em>Press any key to see the actions.</em></p>
        </div>
      `,
      choices: "ALL_KEYS",
    });

    const actionLabels = [
      stimulus.action_0,
      stimulus.action_1,
      stimulus.action_2,
      stimulus.action_3,
    ];

    trials.push({
      type: jsPsychProbabilitySliders,
      labels: actionLabels,
      start: [0.25, 0.25, 0.25, 0.25],
      button_label: "Continue",
      show_reset: true,
      show_chips: true,
      instruction_html: `
        <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
        <div class="vignette-text vignette-text-wide">
          <p>On a scale from 0 (maximally formal) to 100 (maximally intimate), ${stimulus.name_0} and ${stimulus.name_1} are in a relationship they would describe as <strong>${intimacy_texts[stimulus.intimacy_condition]}</strong>.</p>
          <p>${stimulus.vignette}</p>
          <p>${getRewardText(stimulus)}</p>
        </div>
        <p><strong>Please indicate the probability that the two people will choose each action.</strong></p>
      `,
      precision: 3,
      require_total_exact: true,
      data: {
        response_type: "response",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        vignette: stimulus.vignette,
        intimacy_condition: stimulus.intimacy_condition,
        reward_condition: stimulus.reward_condition,
      },
    });

    const memoryCheck = makeMemoryCheckForStimulus(stimulus, NONFOOD_MEMORY_CHECKS);
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
