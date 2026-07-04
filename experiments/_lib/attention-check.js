// Attention check factory.
//
// Inverse experiments use a single jsPsychHtmlSliderResponse with target 0.
// The factory returns the jsPsych trial object that goes into the timeline
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
      // `tolerance` is in raw slider units (0-100, integer steps), so
      // tolerance 0 means only an exact 0 passes — the criterion applied to
      // every participant collected so far. (An earlier value of 0.02 was a
      // normalized-scale leftover that, compared against the raw integer
      // response, was equivalent to requiring exactly 0.)
      data.attention_passed = Math.abs(data.response - 0) <= tolerance;
    },
  };
}
