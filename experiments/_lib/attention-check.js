// Attention check factories.
//
// Inverse experiments use a single jsPsychHtmlSliderResponse with target 0.
// Forward experiments use jsPsychProbabilitySliders parameterized by the
// expected target probabilities (e.g. [0, 0, 0.25, 0.75] for 4-action
// experiments, [0.25, 0.75] for 2-action experiments).
// Each factory returns the jsPsych trial object that goes into the timeline
// at CONFIG.ATTENTION_CHECK_INDEX in the per-experiment trial loop.

export function makeAttentionCheckSingleSlider(tolerance) {
  return {
    type: jsPsychHtmlSliderResponse,
    labels: ["0", "50", "100"],
    slider_min: 0,
    slider_max: 100,
    step: 1,
    require_movement: true,
    button_label: "Continue",
    stimulus: `
      <div>
        <p>This is an attention check to make sure you're not a bot and that we can award you your pay for the study.</p>
        <p><strong>Please set the slider all the way to the left (0).</strong></p>
      </div>
    `,
    data: { response_type: "attention_check" },
    on_finish: function (data) {
      data.attention_passed = Math.abs(data.response - 0) < tolerance;
    },
  };
}

export function makeAttentionCheckProbabilitySliders(tolerance, targets) {
  const n = targets.length;
  const labels = targets.map((t) => `Please set this slider to ${Math.round(t * 100)}%`);
  const start = new Array(n).fill(1 / n);
  return {
    type: jsPsychProbabilitySliders,
    labels,
    start,
    button_label: "Continue",
    show_reset: true,
    show_chips: false,
    instruction_html: `
      <div>
        <p>This is an attention check to make sure you're not a bot and that we can award you your pay for the study.</p>
        <p><strong>Please set each slider to the exact percentage requested below.</strong></p>
      </div>
    `,
    precision: 3,
    require_total_exact: true,
    data: { response_type: "attention_check" },
    on_finish: function (data) {
      const probs = data.probs || [];
      data.attention_passed = targets.every(
        (t, i) => Math.abs((probs[i] ?? NaN) - t) < tolerance
      );
    },
  };
}
