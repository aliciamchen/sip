// Shared experiment entry point. Replaces the per-experiment experiment.js
// boilerplate (asset fetch, jsPsych init, URL-param extraction, condition
// assignment, sequence-to-stimuli mapping, shuffle, run).
//
// Each experiment's experiment.js calls runExperiment({ config, makeTimeline,
// consentTemplate }), where:
//   - config: the CONFIG object exported from trials.js (built there with
//     makeConfig("<slug>") from _lib/config.js), holding PIPE_EXPERIMENT_ID,
//     PROLIFIC_COMPLETION_URL, ATTENTION_CHECK_INDEX, ATTENTION_TOLERANCE,
//     INTER_TRIAL_DURATIONS
//   - makeTimeline(jsPsych, stimuli, consentHtml, exitSurveyHtml, subjectId):
//     returns the full ordered timeline array
//   - consentTemplate: filename (without extension) under _lib/consent/;
//     the active experiments all use "food-inverse"

export function runExperiment({ config, makeTimeline, consentTemplate }) {
  Promise.all([
    fetch("json/stimuli.json").then((r) => r.json()),
    fetch("json/full_counterbalancing.json").then((r) => r.json()),
    fetch(`../_lib/consent/${consentTemplate}.html`).then((r) => r.text()),
    fetch("../_lib/exit-survey.html").then((r) => r.text()),
  ])
    .then(([stimuli, counterbalancing, consentHtml, exitSurveyHtml]) =>
      createExperiment({
        config,
        makeTimeline,
        stimuli,
        counterbalancing,
        consentHtml,
        exitSurveyHtml,
      })
    )
    .catch((error) => {
      console.error("Error loading experiment files:", error);
      alert("Error loading experiment data. Please refresh the page.");
    });
}

async function createExperiment({
  config,
  makeTimeline,
  stimuli,
  counterbalancing,
  consentHtml,
  exitSurveyHtml,
}) {
  const jsPsych = initJsPsych({ show_progress_bar: true });

  const subject_id =
    jsPsych.data.getURLVariable("PROLIFIC_PID") == undefined
      ? jsPsych.randomization.randomID(12)
      : jsPsych.data.getURLVariable("PROLIFIC_PID");
  const study_id = jsPsych.data.getURLVariable("STUDY_ID");
  const session_id = jsPsych.data.getURLVariable("SESSION_ID");

  const condition_assignment = await jsPsychPipe.getCondition(
    config.PIPE_EXPERIMENT_ID
  );
  const assignedSequence = counterbalancing[condition_assignment];

  jsPsych.data.addProperties({
    study_id,
    session_id,
    subject_id,
    url: window.location.href,
    condition_assignment,
  });

  // Spread the assigned sequence's per-scenario factor fields onto each
  // stimulus. The counterbalancing file determines which factor fields exist
  // (action_condition, desire_condition, effort_condition, intimacy_condition
  // — whichever subset applies to this study).
  const stimuliWithConditions = stimuli.map((stimulus) => {
    const sequenceItem = assignedSequence.find(
      (item) => item.scenario_label === stimulus.scenario_label
    );
    return sequenceItem ? { ...stimulus, ...sequenceItem } : stimulus;
  });

  const shuffledStimuli = jsPsych.randomization.shuffle(stimuliWithConditions);

  const timeline = makeTimeline(
    jsPsych,
    shuffledStimuli,
    consentHtml,
    exitSurveyHtml,
    subject_id
  );

  jsPsych.run(timeline);
}
