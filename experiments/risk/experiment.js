import { makeTimeline } from "./trials.js";

let stimuli = [];
let consentHtml = "";
let exitSurveyHtml = "";

// Load all required files
Promise.all([
  fetch("json/stimuli.json").then((response) => response.json()),
  fetch("html/consent.html").then((response) => response.text()),
  fetch("html/exit-survey.html").then((response) => response.text()),
])
  .then(([stimuliData, consentContent, exitSurveyContent]) => {
    stimuli = stimuliData;
    consentHtml = consentContent;
    exitSurveyHtml = exitSurveyContent;
    initExperiment();
  })
  .catch((error) => {
    console.error("Error loading experiment files:", error);
    alert("Error loading experiment data. Please refresh the page.");
  });

function initExperiment() {
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

  // Randomize the order of scenarios before building the timeline
  const shuffledStimuli = jsPsych.randomization.shuffle(stimuli);

  let timeline = makeTimeline(
    jsPsych,
    shuffledStimuli,
    consentHtml,
    exitSurveyHtml,
    subject_id
  );

  jsPsych.run(timeline);
}
