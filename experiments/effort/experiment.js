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
  var study_id = jsPsych.data.getURLVariable('STUDY_ID');
  var session_id = jsPsych.data.getURLVariable('SESSION_ID');

  jsPsych.data.addProperties({
    study_id: study_id,
    session_id: session_id,
    subject_id: subject_id,
    url: window.location.href,
  });

  let timeline = [];

  timeline.push({
    type: jsPsychInstructions,
    pages: [`<div>${consentHtml}</div>`],
    show_clickable_nav: true,
    show_page_number: true,
  });

  timeline.push({
    type: jsPsychInstructions,
    pages: [
      `
        <div>
            <h2>Social interactions survey</h2>
            <p>In this survey, you will read vignettes about two people sharing different kinds of food in different situations. For each scenario, you will read about four different actions the two people can take.</p>
            <p>For each action, we will ask you to evaluate how much effort that action takes, in the context of the scenario. Please consider each option independently.</p>
            <p>Please read each of the scenarios and ways of sharing food carefully! 🙂 You will receive $2.50 if you successfully complete the survey. </p>
            <p>Please do not close the window until you have completed the survey. If you do so, you will lose your progress.</p>
            <p>Press next to begin the survey.</p>
        </div>
        `,
    ],
    show_clickable_nav: true,
    show_page_number: true,
  });

  stimuli.forEach((stimulus, stimulusIndex) => {
    timeline.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: `
                  <div>
                      <h2>Scenario ${stimulusIndex + 1} of ${
        stimuli.length
      }</h2>
                      <div class="vignette-text">
                          <p>${stimulus.vignette}</p>
                      </div>
                      <p><em>Press any key to see the actions.</em></p>
                  </div>
              `,
      choices: "ALL_KEYS",
    });

    const surveyQuestions = [];

    for (let i = 0; i < 4; i++) {
      surveyQuestions.push({
        prompt: `<div class="action-text">${stimulus[`action_${i}`]}</div>`,
        name: `action_${i}`,
        labels: [
          "No effort at all",
          "Very little effort",
          "Little effort",
          "Moderate effort",
          "Considerable effort",
          "High effort",
          "Extremely high effort",
        ],
        required: true,
      });
    }

    timeline.push({
      type: jsPsychSurveyLikert,
      questions: surveyQuestions,
      preamble: `
                  <div>
                      <p>${stimulus.vignette}</p>
                      <p><strong>Please rate how much effort each action requires.</strong></p>
                  </div>
              `,
      randomize_question_order: false,
      button_label: "Continue",
      scale_width: 950,
    });
  });

  timeline.push({
    type: jsPsychSurveyHtmlForm,
    preamble: `
      <div>
        <h2>Exit Survey</h2>
        <p>To collect your pay, please complete the following questions. Your answer to these questions will not affect your pay, so please answer honestly.</p>
      </div>
    `,
    html: exitSurveyHtml,
  });


  timeline.push({
    type: jsPsychPipe,
    action: "save",
    experiment_id: "aqa8eVvU3qSu",
    filename: `${subject_id}.json`,
    data_string: () => jsPsych.data.get().json(),
  });

  timeline.push({
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `<p>Thanks for participating in the experiment!</p>
                  <p><a href="https://app.prolific.com/submissions/complete?cc=C1E1PWV8">Click here to return to Prolific and complete the study</a>.</p>
                  <p>It is now safe to close the window. Your pay will be delivered within a few days.</p>
                  `,
    choices: "NO_KEYS",
  });

  jsPsych.run(timeline);
}
