// Study 1a — Desire inference under known effort + intimacy.
// Design: 2 (effort) × 4 (intimacy: 0/50/75/100) × 3 (observed action).
// Intimacy descriptor preamble, then vignette + effort paragraph + a continuous
// 0-100 desire slider (scenario-specific question, e.g. "how much do Carissa and
// Josh both want to eat the hot dog?" — see `desire_phrase` in scenarios.csv;
// labels Not at all / Moderately / Extremely) rated before and after the single observed
// action. The desire paragraph is NOT shown (desire is inferred).

import { makeInterTrialBlank } from "../_lib/timeline.js";
import { makeAttentionCheckSingleSlider } from "../_lib/attention-check.js";
import { makeMemoryCheckForStimulus } from "../_lib/memory-checks.js";
import { makeConfig } from "../_lib/config.js";
import { STUDY_INSTRUCTIONS } from "../_lib/instructions.js";
import {
  intimacyDescriptor,
  getEffortText,
  desireQuestion,
  DESIRE_SLIDER_LABELS,
  pressAnyKeyPage,
  blankPause,
  scenarioStimulus,
} from "../_lib/scenario.js";

export const CONFIG = makeConfig("food_inv_desire");
export const INSTRUCTIONS_PAGES = STUDY_INSTRUCTIONS.food_inv_desire;

function desireSlider(stimulus, index, total, stage, observedAction) {
  const lead =
    stage === "prior"
      ? "Before observing what they decide to do"
      : "Now that you have observed what they decide to do";
  return {
    type: jsPsychHtmlSliderResponse,
    stimulus: scenarioStimulus({
      index,
      total,
      paragraphs: [
        intimacyDescriptor(stimulus),
        `<p>${stimulus.vignette}</p>`,
        `<p><strong>${getEffortText(stimulus)}</strong></p>`,
      ],
      observedAction,
      leadIn: `${lead}, ${desireQuestion(stimulus, { lowercase: true })}`,
    }),
    slider_width: 900,
    slider_min: 0,
    slider_max: 100,
    step: 1,
    require_movement: true,
    labels: DESIRE_SLIDER_LABELS,
    button_label: "Continue",
    data: {
      response_type: "response",
      stage,
      stimulus_index: index,
      scenario_label: stimulus.scenario_label,
      action_condition: stimulus.action_condition,
      intimacy_condition: stimulus.intimacy_condition,
      effort_condition: stimulus.effort_condition,
      desire_low: stimulus.desire_low,
      desire_high: stimulus.desire_high,
    },
  };
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
