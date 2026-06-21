// Render one page with two sliders, via jsPsych's survey-html-form plugin
// (the html-slider-response plugin only supports a single slider per trial).
// Used by the joint-inference studies (1b, 2b) so both ratings are collected on
// a single page per prior/posterior phase.
//
// makeTwoSliderForm({ preamble, sliders, data, buttonLabel }):
//   - preamble:    scenario HTML shown above the sliders.
//   - sliders:     array of { name, prompt, labels: [html, ...], min=0, max=100,
//                  step=1, start=50, width=SLIDER_WIDTH }. The trial's response is an
//                  object keyed by each slider's `name`, e.g. {desire: "60",
//                  effort: "30"} (stored in `data.response`).
//   - data:        jsPsych data attached to the trial.
//   - buttonLabel: submit-button text (default "Continue").
//
// require_movement is emulated to match the single-slider trials: the Continue
// button stays disabled until every slider on the page has been moved.

import { SLIDER_WIDTH, revealRatingOnKeypress } from "./scenario.js";

function oneSliderHtml({
  name,
  prompt,
  labels,
  min = 0,
  max = 100,
  step = 1,
  start = 50,
  width = SLIDER_WIDTH,
}) {
  const cells = labels
    .map((label) => `<div class="two-slider-label">${label}</div>`)
    .join("");
  return `
    <div class="two-slider-block">
      <div class="two-slider-prompt">${prompt}</div>
      <div class="two-slider-track" style="width: ${width}px;">
        <input type="range" class="jspsych-slider" name="${name}"
               min="${min}" max="${max}" step="${step}" value="${start}" />
        <div class="two-slider-labels">${cells}</div>
      </div>
    </div>
  `;
}

export function makeTwoSliderForm({
  preamble,
  sliders,
  data,
  buttonLabel = "Continue",
}) {
  return {
    type: jsPsychSurveyHtmlForm,
    preamble,
    html: sliders.map(oneSliderHtml).join(""),
    button_label: buttonLabel,
    data,
    on_load: function () {
      const next = document.querySelector("#jspsych-survey-html-form-next");
      const ranges = document.querySelectorAll(
        "#jspsych-survey-html-form input[type='range']",
      );
      if (!next || ranges.length === 0) return;
      next.disabled = true;
      const moved = new Set();
      ranges.forEach((range) => {
        range.addEventListener("input", () => {
          moved.add(range.getAttribute("name"));
          if (moved.size === ranges.length) next.disabled = false;
        });
      });
      // Build the screen up gradually: hide the rating UI until the participant
      // has read the scenario — and, on the posterior, the observed action — and
      // pressed a key.
      revealRatingOnKeypress({
        hideSelectors: [
          "#lead-in",
          ".two-slider-block",
          "#jspsych-survey-html-form-next",
        ],
      });
    },
  };
}
