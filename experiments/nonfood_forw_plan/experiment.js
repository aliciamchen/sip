import { makeTimeline, CONFIG } from "./trials.js";

let stimuli = [];
let consentHtml = "";
let exitSurveyHtml = "";
let counterbalancing = [];

Promise.all([
  fetch("json/stimuli.json").then((response) => response.json()),
  fetch("json/full_counterbalancing.json").then((response) => response.json()),
  fetch("html/consent.html").then((response) => response.text()),
  fetch("html/exit-survey.html").then((response) => response.text()),
])
  .then(
    ([
      stimuliData,
      counterbalancingData,
      consentContent,
      exitSurveyContent,
    ]) => {
      stimuli = stimuliData;
      counterbalancing = counterbalancingData;
      consentHtml = consentContent;
      exitSurveyHtml = exitSurveyContent;
      initExperiment();
    }
  )
  .catch((error) => {
    console.error("Error loading experiment files:", error);
    alert("Error loading experiment data. Please refresh the page.");
  });

async function createExperiment() {
  const jsPsych = initJsPsych({
    show_progress_bar: true,
  });

  var subject_id =
    jsPsych.data.getURLVariable("PROLIFIC_PID") == undefined
      ? jsPsych.randomization.randomID(12)
      : jsPsych.data.getURLVariable("PROLIFIC_PID");
  var study_id = jsPsych.data.getURLVariable("STUDY_ID");
  var session_id = jsPsych.data.getURLVariable("SESSION_ID");

  jsPsych.data.addProperties({
    study_id: study_id,
    session_id: session_id,
    subject_id: subject_id,
    url: window.location.href,
  });

  const condition_assignment = await jsPsychPipe.getCondition(CONFIG.PIPE_EXPERIMENT_ID);
  const assignedSequence = counterbalancing[condition_assignment];

  const stimuliWithIntimacyReward = stimuli.map((stimulus) => {
    const sequenceItem = assignedSequence.find(
      (item) => item.scenario_label === stimulus.scenario_label
    );
    return {
      ...stimulus,
      intimacy_condition: sequenceItem ? sequenceItem.intimacy : "",
      reward_condition: sequenceItem ? sequenceItem.reward : "",
    };
  });

  const shuffledStimuli = jsPsych.randomization.shuffle(stimuliWithIntimacyReward);

  let timeline = makeTimeline(
    jsPsych,
    shuffledStimuli,
    consentHtml,
    exitSurveyHtml,
    subject_id
  );

  jsPsych.run(timeline);
}

function initExperiment() {
  createExperiment();
}
