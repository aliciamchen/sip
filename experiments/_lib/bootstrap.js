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
//     ATTENTION_TOLERANCE, INTER_TRIAL_DURATIONS, PAYMENT, DURATION_MINUTES
//   - makeStimulusTrials(jsPsych, stimuli): returns the study's per-scenario
//     trial array (slid in between the instructions and the exit survey)
//   - instructionsPages: the study's instructions pages (from STUDY_INSTRUCTIONS)
//   - comprehensionQuestions: the study's comprehension-check questions (from
//     STUDY_COMPREHENSION_CHECKS); when present, the instructions screen is
//     replaced by a gated instructions + comprehension check (3 attempts, then
//     the experiment ends asking the participant to return the study)
//   - consentTemplate: filename (without extension) under _lib/consent/;
//     the active experiments all use "food-inverse"

import {
  makeConsentScreen,
  makeInstructionsScreen,
  makeExitSurvey,
  makeSaveData,
  makeThankYou,
} from "./timeline.js";
import {
  makeComprehensionGate,
  makeComprehensionPassPage,
} from "./comprehension-check.js";

export function runExperiment({
  config,
  makeStimulusTrials,
  instructionsPages,
  comprehensionQuestions,
  consentTemplate,
}) {
  // Refuse to start a study with no Prolific completion code configured (no
  // entry in PROLIFIC_COMPLETION_CODES in config.js): the codes are
  // study-specific, so running without one would silently send every
  // participant to another study's completion URL. Failing loudly at startup
  // makes launching an unconfigured study impossible rather than quietly wrong.
  if (!config.PROLIFIC_COMPLETION_URL) {
    const message =
      "This study has no Prolific completion code configured. Add its code to " +
      "PROLIFIC_COMPLETION_CODES in experiments/_lib/config.js before launching.";
    alert(message);
    throw new Error(message);
  }
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
        comprehensionQuestions,
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
  comprehensionQuestions,
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
  //
  // If DataPipe resolves to a non-integer (API hiccup, misconfigured
  // experiment ID), fall back to a RANDOM sequence rather than pinning every
  // affected participant to sequence 0, which would quietly skew the
  // counterbalance for the whole outage window. condition_assignment_source
  // records the provenance ("datapipe" vs "random_fallback") in the saved
  // data so affected sessions are identifiable in analysis.
  const n_sequences = counterbalancing.length;
  const datapipeAssignmentOk = Number.isInteger(condition_assignment);
  if (!datapipeAssignmentOk) {
    console.warn(
      "jsPsychPipe.getCondition did not return an integer " +
        `(got ${JSON.stringify(condition_assignment)}); ` +
        "assigning a RANDOM counterbalancing sequence instead. " +
        "Check the DataPipe experiment ID / condition-assignment settings."
    );
  }
  const sequence_index = datapipeAssignmentOk
    ? ((condition_assignment % n_sequences) + n_sequences) % n_sequences
    : Math.floor(Math.random() * n_sequences);
  const condition_assignment_source = datapipeAssignmentOk
    ? "datapipe"
    : "random_fallback";
  const assignedSequence = counterbalancing[sequence_index];

  jsPsych.data.addProperties({
    study_id,
    session_id,
    subject_id,
    url: window.location.href,
    condition_assignment,
    condition_assignment_source,
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

  // Participant-facing boilerplate (the consent form and the instructions
  // pages) is authored with {{PAYMENT}} / {{DURATION_MINUTES}} placeholders so
  // the per-study numbers come from CONFIG rather than being hardcoded in text
  // shared by every study. Fill them here, before the pages reach the timeline
  // (the comprehension gate re-shows the instructions, so it gets the filled
  // pages too).
  const filledConsentHtml = fillConfigPlaceholders(consentHtml, config);
  const filledInstructionsPages = instructionsPages.map((page) =>
    fillConfigPlaceholders(page, config)
  );

  // When the study supplies comprehension questions, gate entry on them: the
  // instructions are shown inside the gate (and re-shown on each retry), so this
  // replaces the standalone instructions screen. Passing the gate leads into a
  // confirmation page; failing it ends the experiment before that page is reached.
  const introStages = comprehensionQuestions
    ? [
        makeComprehensionGate(jsPsych, {
          instructionsPages: filledInstructionsPages,
          questions: comprehensionQuestions,
        }),
        makeComprehensionPassPage(),
      ]
    : [makeInstructionsScreen(filledInstructionsPages)];

  // Save under PROLIFIC_PID_SESSION_ID.json when Prolific supplies a session
  // (falling back to the bare subject_id otherwise, e.g. local test runs): a
  // participant who completes the study twice does so under different
  // SESSION_IDs, so the re-completion surfaces as a second file for the same
  // subject_id instead of colliding on one filename.
  const saveFilename = session_id
    ? `${subject_id}_${session_id}.json`
    : `${subject_id}.json`;

  const timeline = [
    makeConsentScreen(filledConsentHtml),
    ...introStages,
    ...makeStimulusTrials(jsPsych, shuffledStimuli),
    makeExitSurvey(jsPsych, exitSurveyHtml),
    makeSaveData(jsPsych, config.PIPE_EXPERIMENT_ID, saveFilename),
    makeThankYou(config.PROLIFIC_COMPLETION_URL),
  ];

  jsPsych.run(timeline);
}

// Fill {{KEY}} placeholders in participant-facing HTML with values from the
// study CONFIG (e.g. {{PAYMENT}}, {{DURATION_MINUTES}}), so per-study numbers
// are configured in _lib/config.js instead of hardcoded in shared text. A
// placeholder with no CONFIG value throws rather than silently showing a
// literal "{{...}}" to a participant.
function fillConfigPlaceholders(html, config) {
  return html.replace(/\{\{([A-Z_]+)\}\}/g, (placeholder, key) => {
    if (config[key] === undefined || config[key] === null) {
      throw new Error(
        `Participant-facing text references ${placeholder} but CONFIG has no ` +
          `${key}; set it in _lib/config.js (SHARED_CONFIG or a per-study override).`
      );
    }
    return String(config[key]);
  });
}
