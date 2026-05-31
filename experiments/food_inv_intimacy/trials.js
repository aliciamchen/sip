// Study 2a — Inverse intimacy under known desire + effort.
// Design: 2 (desire) × 2 (effort) × 3 (observed action). No preamble page; the
// participant sees the vignette + desire + effort paragraphs and rates the
// relationship on a single 0-100 slider before and after the observed action.

import { makeInterTrialBlank } from "../_lib/timeline.js";
import { makeAttentionCheckSingleSlider } from "../_lib/attention-check.js";
import { makeMemoryCheckForStimulus } from "../_lib/memory-checks.js";
import { makeConfig } from "../_lib/config.js";
import { STUDY_INSTRUCTIONS } from "../_lib/instructions.js";
import {
  getDesireText,
  getEffortText,
  INTIMACY_SLIDER_LABELS,
  singleSliderTrial,
  blankPause,
} from "../_lib/scenario.js";

export const CONFIG = makeConfig("food_inv_intimacy");
export const INSTRUCTIONS_PAGES = STUDY_INSTRUCTIONS.food_inv_intimacy;

function intimacySlider(stimulus, index, total, stage, observedAction) {
  return singleSliderTrial({
    stimulus,
    index,
    total,
    stage,
    observedAction,
    paragraphs: [
      `<p>${stimulus.vignette}</p>`,
      `<p>${getDesireText(stimulus)}</p>`,
      `<p>${getEffortText(stimulus)}</p>`,
    ],
    labels: INTIMACY_SLIDER_LABELS,
    leadInQuestion: `how do you think ${stimulus.name_0} and ${stimulus.name_1} would describe their relationship?`,
    data: {
      desire_condition: stimulus.desire_condition,
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
    trials.push(intimacySlider(stimulus, index, total, "prior", null));
    trials.push(blankPause());
    trials.push(
      intimacySlider(
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
