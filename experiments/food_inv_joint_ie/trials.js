// Study 2b — Joint inference over intimacy and effort, given desire.
// Design: 2 (desire) × 3 (observed action). Desire paragraph shown after the
// vignette; both sliders on one page per phase — an intimacy 0-100 slider and
// an effort slider between the two effort paragraphs. No preamble page.

import { makeInterTrialBlank } from "../_lib/timeline.js";
import { makeAttentionCheckSingleSlider } from "../_lib/attention-check.js";
import { makeMemoryCheckForStimulus } from "../_lib/memory-checks.js";
import { makeConfig } from "../_lib/config.js";
import { makeTwoSliderForm } from "../_lib/two-slider.js";
import { STUDY_INSTRUCTIONS } from "../_lib/instructions.js";
import {
  getDesireText,
  INTIMACY_SLIDER_LABELS,
  effortLabels,
  blankPause,
  scenarioStimulus,
} from "../_lib/scenario.js";

export const CONFIG = makeConfig("food_inv_joint_ie");
export const INSTRUCTIONS_PAGES = STUDY_INSTRUCTIONS.food_inv_joint_ie;

function preamble(stimulus, index, total, stage, observedAction) {
  const leadIn =
    stage === "prior"
      ? "Before observing what they decide to do, please answer the questions below."
      : "Now that you have observed what they decide to do, please re-evaluate.";
  return scenarioStimulus({
    index,
    total,
    paragraphs: [
      `<p>${stimulus.vignette}</p>`,
      `<p>${getDesireText(stimulus)}</p>`,
    ],
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

    const sliders = [
      {
        name: "intimacy",
        prompt: `How would ${stimulus.name_0} and ${stimulus.name_1} describe their relationship?`,
        labels: INTIMACY_SLIDER_LABELS,
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
      desire_condition: stimulus.desire_condition,
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
