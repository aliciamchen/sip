// Shared timeline-screen factories used by every active experiment.
// Each function returns a jsPsych trial object; callers compose them into a
// timeline alongside their experiment-specific prior/posterior trial loop.

export function makeConsentScreen(consentHtml) {
  return {
    type: jsPsychInstructions,
    pages: [`<div>${consentHtml}</div>`],
    show_clickable_nav: true,
    show_page_number: true,
  };
}

export function makeInstructionsScreen(pages) {
  return {
    type: jsPsychInstructions,
    pages,
    show_clickable_nav: true,
    show_page_number: true,
  };
}

export function makeInterTrialBlank(jsPsych, durations) {
  return {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: "Next scenario",
    choices: "NO_KEYS",
    trial_duration: () =>
      jsPsych.randomization.sampleWithoutReplacement(durations, 1)[0],
  };
}

export function makeExitSurvey(jsPsych, exitSurveyHtml) {
  return {
    type: jsPsychSurveyHtmlForm,
    preamble: `
      <div>
        <h2>Exit Survey</h2>
        <p>You have reached the end of the survey. To collect your pay, please complete the following questions. Your answer to these questions will not affect your pay or whether your submission is approved for payment, so please answer honestly.</p>
      </div>
    `,
    html: exitSurveyHtml,
    on_finish: function (data) {
      data.attention_passed = jsPsych.data
        .get()
        .filter({ response_type: "attention_check" })
        .select("attention_passed").values[0];
      data.memory_correct_count = jsPsych.data
        .get()
        .filter({ response_type: "memory_check" })
        .select("memory_correct_count")
        .sum();
      // Only participants who passed the comprehension check reach the exit
      // survey, so this records which attempt they passed on (1..max) as a
      // quality signal, not an exclusion field.
      const comprehensionAttempts = jsPsych.data
        .get()
        .filter({ response_type: "comprehension_check" })
        .select("comprehension_attempt").values;
      data.comprehension_attempt =
        comprehensionAttempts[comprehensionAttempts.length - 1];
      data.response_type = "exit_survey";
    },
  };
}

export function makeSaveData(jsPsych, pipeExperimentId, subjectId) {
  return {
    type: jsPsychPipe,
    action: "save",
    experiment_id: pipeExperimentId,
    filename: `${subjectId}.json`,
    data_string: () => jsPsych.data.get().json(),
  };
}

export function makeThankYou(prolificCompletionUrl) {
  return {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `<p>Thanks for participating in the experiment!</p>
                  <p><a href="${prolificCompletionUrl}">Click here to return to Prolific and complete the study</a>.</p>
                  <p>It is now safe to close the window. Your pay will be delivered within a few days.</p>
                  `,
    choices: "NO_KEYS",
  };
}
