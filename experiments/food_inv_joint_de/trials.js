// Study 1b — Joint inference over desire and effort, given intimacy.
// Design: 4 (intimacy) × 3 (observed action). Intimacy descriptor preamble;
// both sliders on one page per phase — a continuous 0-100 desire slider and an
// effort slider between the two effort paragraphs.

import { makeInterTrialBlank } from "../_lib/timeline.js";
import { makeAttentionCheckSingleSlider } from "../_lib/attention-check.js";
import { makeMemoryCheckForStimulus } from "../_lib/memory-checks.js";
import { makeConfig } from "../_lib/config.js";
import { makeTwoSliderForm } from "../_lib/two-slider.js";
import { STUDY_INSTRUCTIONS } from "../_lib/instructions.js";
import { STUDY_COMPREHENSION_CHECKS } from "../_lib/comprehension-check.js";
import {
  intimacyDescriptor,
  desireQuestion,
  DESIRE_SLIDER_LABELS,
  effortLabels,
  pressAnyKeyPage,
  blankPause,
  scenarioStimulus,
} from "../_lib/scenario.js";

export const CONFIG = makeConfig("food_inv_joint_de");
export const INSTRUCTIONS_PAGES = STUDY_INSTRUCTIONS.food_inv_joint_de;
export const COMPREHENSION_QUESTIONS = STUDY_COMPREHENSION_CHECKS.food_inv_joint_de;

function preamble(stimulus, index, total, stage, observedAction) {
  const leadIn =
    stage === "prior"
      ? "Before observing what they decide to do, please answer the questions below."
      : "Now that you have observed what they decide to do, please answer the questions again.";
  return scenarioStimulus({
    index,
    total,
    paragraphs: [intimacyDescriptor(stimulus), `<p>${stimulus.vignette}</p>`],
    observedAction,
    leadIn,
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

    const sliders = [
      {
        name: "desire",
        prompt: desireQuestion(stimulus),
        labels: DESIRE_SLIDER_LABELS,
      },
      {
        name: "effort",
        prompt: "Which situation do you think is more likely?",
        labels: effortLabels(stimulus),
      },
    ];
    const data = {
      response_type: "response",
      stimulus_index: index,
      scenario_label: stimulus.scenario_label,
      action_condition: stimulus.action_condition,
      intimacy_condition: stimulus.intimacy_condition,
      low_risk_share_effort_low: stimulus.low_risk_share_effort_low,
      low_risk_share_effort_high: stimulus.low_risk_share_effort_high,
    };

    trials.push(
      makeTwoSliderForm({
        preamble: preamble(stimulus, index, total, "prior", null),
        sliders,
        data: { ...data, stage: "prior" },
      }),
    );
    trials.push(blankPause());
    trials.push(
      makeTwoSliderForm({
        preamble: preamble(
          stimulus,
          index,
          total,
          "posterior",
          stimulus[stimulus.action_condition],
        ),
        sliders,
        data: { ...data, stage: "posterior" },
      }),
    );

    const memoryCheck = makeMemoryCheckForStimulus(stimulus);
    if (memoryCheck) trials.push(memoryCheck);
    trials.push(makeInterTrialBlank(jsPsych, CONFIG.INTER_TRIAL_DURATIONS));
  });

  return trials;
}
