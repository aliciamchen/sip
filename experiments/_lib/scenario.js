// Shared per-trial scenario pieces for the inverse-planning experiments:
// condition-paragraph getters, slider labels, the "press any key" page, the
// pause between the prior and posterior ratings, and the scenario-stimulus HTML
// builder. Each trials.js composes these into its study-specific trials.

// Intimacy is a purely verbal manipulation: the condition is identified by a
// slug, and only the verbal descriptor is ever shown or saved (no numeric code).
export const intimacy_texts = {
  max_formal: "maximally formal",
  somewhat_formal: "somewhat formal",
  somewhat_intimate: "somewhat intimate",
  max_intimate: "maximally intimate",
};

// Condition paragraphs (shown when the variable is given to the participant).
export const getDesireText = (stim) =>
  stim.desire_condition === "low" ? stim.desire_low : stim.desire_high;
export const getEffortText = (stim) =>
  stim.effort_condition === "low"
    ? stim.low_risk_share_effort_low
    : stim.low_risk_share_effort_high;

// The desire-elicitation question, naming both characters and the scenario-
// specific food (e.g. "How much do you think Carissa and Josh would like the hot
// dog?"). The object is the per-scenario `desire_object` from scenarios.csv (e.g.
// "the hot dog", "the coffee", "all the oyster types"). Asked in Studies 1a and
// 1b; 1a embeds it (lowercased) after a prior/posterior framing clause, so it
// takes a `lowercase` option.
export const desireQuestion = (stim, { lowercase = false } = {}) =>
  `${lowercase ? "how much" : "How much"} do you think ${stim.name_0} and ${stim.name_1} would like ${stim.desire_object}?`;

// "Consider X and Y, who would describe their relationship as Z." — shown when
// intimacy is given (Studies 1a, 1b).
export const intimacyDescriptor = (stim) =>
  `<p>Consider ${stim.name_0} and ${stim.name_1}, who would describe their relationship as <strong>${intimacy_texts[stim.intimacy_condition]}</strong>.</p>`;

// Slider labels.
export const DESIRE_SLIDER_LABELS = ["Not at all", "Moderately", "Extremely"];
export const INTIMACY_SLIDER_LABELS = [
  "Maximally formal",
  "Neither formal nor intimate",
  "Maximally intimate",
];
// Effort slider endpoints: the two effort paragraphs, with "Equally likely" mid.
export const effortLabels = (stim) => [
  stim.low_risk_share_effort_low,
  "Equally likely",
  stim.low_risk_share_effort_high,
];

// Shared pixel width for every rating slider (single-slider 1a/2a and the
// two-slider 1b/2b form), so the width lives in one place. Kept a bit narrower
// than the 720px text-content width (.vignette-text) so the slider sits inside
// the text column rather than overhanging it.
export const SLIDER_WIDTH = 680;

// A "press any key to continue" page showing the given preamble HTML.
export function pressAnyKeyPage(preambleHtml, index, total) {
  return {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
      <div>
        <h2>Scenario ${index + 1} of ${total}</h2>
        <div class="vignette-text">
          ${preambleHtml}
        </div>
        <p style="text-align: center;"><em>Press any key to continue.</em></p>
      </div>
    `,
    choices: "ALL_KEYS",
  };
}

// Blank pause between the prior and posterior ratings of a scenario.
export function blankPause(durationMs = 1000) {
  return {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: "",
    choices: "NO_KEYS",
    trial_duration: durationMs,
  };
}

// Build a scenario stimulus: heading, the vignette block (the given paragraphs,
// in order), optionally the observed action, then a bold lead-in line. Used as
// the single-slider stimulus (1a, 2a) and as the two-slider form preamble
// (1b, 2b). `paragraphs` is a list of <p> HTML strings (the caller controls
// content, order, and any bolding).
export function scenarioStimulus({
  index,
  total,
  paragraphs,
  observedAction = null,
  leadIn,
}) {
  const observed = observedAction
    ? `
      <div class="vignette-text vignette-observed">
        <p><em>They decide to take the following action:</em></p>
        <p>${observedAction}</p>
      </div>`
    : "";
  return `
    <div>
      <h2>Scenario ${index + 1} of ${total}</h2>
      <div class="vignette-text">
        ${paragraphs.join("\n        ")}
      </div>${observed}
      <p><strong>${leadIn}</strong></p>
    </div>
  `;
}

// A single-slider rating trial (Studies 1a, 2a): the scenario stimulus with a
// prior/posterior lead-in, plus a continuous 0-100 html-slider-response. The
// caller supplies the paragraphs, the slider labels, the lead-in question clause
// (appended after the "Before/Now that..." framing), and any study-specific
// condition fields to merge into the saved trial data.
export function singleSliderTrial({
  stimulus,
  index,
  total,
  stage,
  observedAction,
  paragraphs,
  labels,
  leadInQuestion,
  data = {},
}) {
  const lead =
    stage === "prior"
      ? "Before observing what they decide to do"
      : "Now that you have observed what they decide to do";
  return {
    type: jsPsychHtmlSliderResponse,
    stimulus: scenarioStimulus({
      index,
      total,
      paragraphs,
      observedAction,
      leadIn: `${lead}, ${leadInQuestion}`,
    }),
    slider_width: SLIDER_WIDTH,
    slider_min: 0,
    slider_max: 100,
    step: 1,
    require_movement: true,
    labels,
    button_label: "Continue",
    data: {
      response_type: "response",
      stage,
      stimulus_index: index,
      scenario_label: stimulus.scenario_label,
      action_condition: stimulus.action_condition,
      ...data,
    },
  };
}
