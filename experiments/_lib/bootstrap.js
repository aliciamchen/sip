// Shared experiment entry point: asset fetch, jsPsych init, URL-param
// extraction, condition assignment, sequence-to-stimuli mapping, shuffle, and
// timeline assembly. The full timeline (consent → instructions → stimulus
// trials → exit survey → save → thank-you) is built here, so each experiment's
// trials.js only supplies the study-specific stimulus trials.
//
// Each experiment's experiment.js calls runExperiment({ config,
// makeStimulusTrials, instructionsPages, consentTemplate }), where:
//   - config: the CONFIG object from trials.js (built with makeConfig("<slug>")),
//     holding PIPE_EXPERIMENT_ID, PROLIFIC_COMPLETION_URL, ATTENTION_CHECK_INDEX,
//     ATTENTION_TOLERANCE, INTER_TRIAL_DURATIONS
//   - makeStimulusTrials(jsPsych, stimuli): returns the study's per-scenario
//     trial array (slid in between the instructions and the exit survey)
//   - instructionsPages: the study's instructions pages (from STUDY_INSTRUCTIONS)
//   - consentTemplate: filename (without extension) under _lib/consent/;
//     the active experiments all use "food-inverse"

import {
  makeConsentScreen,
  makeInstructionsScreen,
  makeExitSurvey,
  makeSaveData,
  makeThankYou,
} from "./timeline.js";

export function runExperiment({
  config,
  makeStimulusTrials,
  instructionsPages,
  consentTemplate,
}) {
  Promise.all([
    fetch("json/stimuli.json").then((r) => r.json()),
    fetch("json/full_counterbalancing.json").then((r) => r.json()),
    fetch(`../_lib/consent/${consentTemplate}.html`).then((r) => r.text()),
    fetch("../_lib/exit-survey.html").then((r) => r.text()),
  ])
    .then(([stimuli, counterbalancing, consentHtml, exitSurveyHtml]) =>
      createExperiment({
        config,
        makeStimulusTrials,
        instructionsPages,
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
  makeStimulusTrials,
  instructionsPages,
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
  // DataPipe returns an index in [0, N) where N is the condition count set on
  // the DataPipe dashboard. If that count is ever larger than the number of
  // counterbalancing sequences we actually have, wrap with modulo so the
  // participant still gets a valid sequence instead of `undefined`.
  // `sequence_index` is the index actually used — map it to
  // full_counterbalancing.json in analysis (condition_assignment is kept raw).
  const n_sequences = counterbalancing.length;
  const sequence_index = Number.isInteger(condition_assignment)
    ? ((condition_assignment % n_sequences) + n_sequences) % n_sequences
    : 0;
  const assignedSequence = counterbalancing[sequence_index];

  jsPsych.data.addProperties({
    study_id,
    session_id,
    subject_id,
    url: window.location.href,
    condition_assignment,
    sequence_index,
  });

  // Spread the assigned sequence's per-scenario factor fields onto each
  // stimulus. The counterbalancing file determines which factor fields exist
  // (action_condition, desire_condition, effort_condition, intimacy_condition
  // — whichever subset applies to this study). stimuli.json and
  // full_counterbalancing.json are generated separately, so guard against drift:
  // every stimulus must match exactly one sequence entry (and vice versa), or we
  // would silently run trials with undefined conditions / a missing observed
  // action. Fail loudly instead.
  const stimuliWithConditions = stimuli.map((stimulus) => {
    const sequenceItem = assignedSequence.find(
      (item) => item.scenario_label === stimulus.scenario_label
    );
    if (!sequenceItem) {
      throw new Error(
        `Counterbalancing drift: no sequence entry for scenario "${stimulus.scenario_label}" ` +
          `(sequence_index ${sequence_index}). stimuli.json and full_counterbalancing.json are out of sync.`
      );
    }
    return { ...stimulus, ...sequenceItem };
  });

  const stimulusLabels = new Set(stimuli.map((s) => s.scenario_label));
  const orphanLabels = [
    ...new Set(
      assignedSequence
        .map((item) => item.scenario_label)
        .filter((label) => !stimulusLabels.has(label))
    ),
  ];
  if (orphanLabels.length > 0) {
    throw new Error(
      `Counterbalancing drift: sequence_index ${sequence_index} references scenarios ` +
        `not present in stimuli.json: ${orphanLabels.join(", ")}.`
    );
  }

  const shuffledStimuli = jsPsych.randomization.shuffle(stimuliWithConditions);

  const timeline = [
    makeConsentScreen(consentHtml),
    makeInstructionsScreen(instructionsPages),
    ...makeStimulusTrials(jsPsych, shuffledStimuli),
    makeExitSurvey(jsPsych, exitSurveyHtml),
    makeSaveData(jsPsych, config.PIPE_EXPERIMENT_ID, subject_id),
    makeThankYou(config.PROLIFIC_COMPLETION_URL),
  ];

  jsPsych.run(timeline);
}
