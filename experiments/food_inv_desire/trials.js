// Study 1a — Desire inference under known effort + intimacy.
// Design: 2 (effort) × 4 (intimacy: max_formal/neither/somewhat_intimate/max_intimate) × 3 (observed action).
// Intimacy descriptor preamble, then vignette + effort paragraph + a continuous
// 0-100 desire slider (scenario-specific question, e.g. "how much do you think
// Carissa and Josh would like the hot dog?" — see `desire_object` in scenarios.csv;
// labels Not at all / Moderately / Extremely) rated before and after the single observed
// action. The desire paragraph is NOT shown (desire is inferred).

import { makeInterTrialBlank } from "../_lib/timeline.js";
import { makeAttentionCheckSingleSlider } from "../_lib/attention-check.js";
import { makeMemoryCheckForStimulus } from "../_lib/memory-checks.js";
import { makeConfig } from "../_lib/config.js";
import { STUDY_INSTRUCTIONS } from "../_lib/instructions.js";
import { STUDY_COMPREHENSION_CHECKS } from "../_lib/comprehension-check.js";
import {
  intimacyDescriptor,
  getEffortText,
  desireQuestion,
  DESIRE_SLIDER_LABELS,
  singleSliderTrial,
  pressAnyKeyPage,
  blankPause,
} from "../_lib/scenario.js";

export const CONFIG = makeConfig("food_inv_desire");
export const INSTRUCTIONS_PAGES = STUDY_INSTRUCTIONS.food_inv_desire;
export const COMPREHENSION_QUESTIONS = STUDY_COMPREHENSION_CHECKS.food_inv_desire;

function desireSlider(stimulus, index, total, stage, observedAction) {
  return singleSliderTrial({
    stimulus,
    index,
    total,
    stage,
    observedAction,
    paragraphs: [
      intimacyDescriptor(stimulus),
      `<p>${stimulus.vignette}</p>`,
      `<p>${getEffortText(stimulus)}</p>`,
    ],
    labels: DESIRE_SLIDER_LABELS,
    leadInQuestion: desireQuestion(stimulus, { lowercase: true }),
    data: {
      intimacy_condition: stimulus.intimacy_condition,
      effort_condition: stimulus.effort_condition,
    },
  });
}

export function makeStimulusTrials(jsPsych, stimuli) {
  const trials = [];
  const total = stimuli.length;

  stimuli.forEach((stimulus, index) => {
    if (index === CONFIG.ATTENTION_CHECK_INDEX) {
      trials.push(makeAttentionCheckSingleSlider(CONFIG.ATTENTION_TOLERANCE));
    }
    trials.push(pressAnyKeyPage(intimacyDescriptor(stimulus), index, total));
    trials.push(desireSlider(stimulus, index, total, "prior", null));
    trials.push(blankPause());
    trials.push(
      desireSlider(
        stimulus,
        index,
        total,
        "posterior",
        stimulus[stimulus.action_condition],
      ),
    );
    const memoryCheck = makeMemoryCheckForStimulus(stimulus);
    if (memoryCheck) trials.push(memoryCheck);
    trials.push(makeInterTrialBlank(jsPsych, CONFIG.INTER_TRIAL_DURATIONS));
  });

  return trials;
}
