// Shared per-trial scenario pieces for the inverse-planning experiments:
// condition-paragraph getters, slider labels, the "press any key" page, the
// pause between the prior and posterior ratings, and the scenario-stimulus HTML
// builder. Each trials.js composes these into its study-specific trials.

export const intimacy_texts = {
  0: "maximally formal",
  50: "neither formal nor intimate",
  75: "somewhat intimate",
  100: "maximally intimate",
};

// Condition paragraphs (shown when the variable is given to the participant).
export const getDesireText = (stim) =>
  stim.desire_condition === "low" ? stim.desire_low : stim.desire_high;
export const getEffortText = (stim) =>
  stim.effort_condition === "low"
    ? stim.low_risk_share_effort_low
    : stim.low_risk_share_effort_high;

// "Consider X and Y, who would describe their relationship as Z." — shown when
// intimacy is given (Studies 1a, 1b).
export const intimacyDescriptor = (stim) =>
  `<p>Consider ${stim.name_0} and ${stim.name_1}, who would describe their relationship as <strong>${intimacy_texts[stim.intimacy_condition]}</strong>.</p>`;

// Slider labels.
export const DESIRE_SLIDER_LABELS = ["Not at all", "Extremely"];
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

// A "press any key to see the scenario" page showing the given preamble HTML.
export function pressAnyKeyPage(preambleHtml, index, total) {
  return {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
      <div>
        <h2>Scenario ${index + 1} of ${total}</h2>
        <div class="vignette-text">
          ${preambleHtml}
        </div>
        <p style="text-align: center;"><em>Press any key to see the scenario.</em></p>
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
