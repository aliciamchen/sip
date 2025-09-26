export function makeTimeline(
  jsPsych,
  stimuli,
  consentHtml,
  exitSurveyHtml,
  subjectId
) {
  const consent = {
    type: jsPsychInstructions,
    pages: [`<div>${consentHtml}</div>`],
    show_clickable_nav: true,
    show_page_number: true,
  };

  const instructions = {
    type: jsPsychInstructions,
    pages: [
      `
            <div class="instructions-container">
                <h2>Social interactions survey</h2>
                <p>In this survey, you will read vignettes about two people sharing different kinds of food in different situations. For each scenario, you will read about four different actions the two people can take.</p>
                <p>For each action, we will ask you to evaluate how much effort that action takes, in the context of the scenario. Please consider each option independently.</p>
                <p>Please read each of the scenarios and ways of sharing food carefully! 🙂 You will receive $5 if you successfully complete the survey. </p>
                <p>Please do not close the window until you have completed the survey. If you do so, you will lose your progress.</p>
                <p>Press next to begin the survey.</p>
            </div>
            `,
    ],
    show_clickable_nav: true,
    show_page_number: true,
  };

  const trials = [];

  stimuli.forEach((stimulus, stimulusIndex) => {
    trials.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: `
                    <div>
                        <h2>Scenario ${stimulusIndex + 1} of ${
        stimuli.length
      }</h2>
                        <div class="vignette-text">
                            <p>${stimulus.vignette}</p>
                        </div>
                        <p style="text-align: center;"><em>Press any key to see the actions.</em></p>
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

    trials.push({
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
      data: {
        trial_type: "response",
        scenario_label: stimulus.scenario_label,
        vignette: stimulus.vignette,
      },
    });

    // Attention check for the "hike" scenario
    if (stimulus.scenario_label === "hike") {
      trials.push({
        type: jsPsychSurveyMultiChoice,
        preamble: `
          <div>
            <h3>Attention Check</h3>
            <p>This is an attention check to make sure you're not a bot and that we can award you your pay for the study.</p>
            <p>Please answer the following questions about the previous scenario.</p>
          </div>
        `,
        questions: [
          {
            prompt: "What were the names of the people in the scenario?",
            name: "names",
            options: [
              "Alvin and Allen",
              "Tony and Kevin",
              "Tony and Alvin",
              "Kevin and Alvin",
            ],
            required: true,
          },
          {
            prompt: "What food did Alvin bring?",
            name: "food",
            options: [
              "Snacks and energy bars",
              "Peanut butter and jelly sandwiches",
            ],
            required: true,
          },
        ],
        button_label: "Continue",
        on_finish: function (data) {
          const responses = data.response || {};
          const correctNames = responses.names === "Tony and Alvin" ? 1 : 0;
          const correctFood =
            responses.food === "Snacks and energy bars" ? 1 : 0;
          const totalCorrect = correctNames + correctFood;
          data.trial_type = "attention_check";
          data.attention_correct_count = totalCorrect;
          data.attention_correct_names = correctNames;
          data.attention_correct_food = correctFood;
        },
      });
    }


    // Attention check for the "wedding" scenario
    if (stimulus.scenario_label === "wedding") {
        trials.push({
          type: jsPsychSurveyMultiChoice,
          preamble: `
            <div>
              <h3>Attention Check</h3>
              <p>This is an attention check to make sure you're not a bot and that we can award you your pay for the study.</p>
              <p>Please answer the following questions about the previous scenario.</p>
            </div>
          `,
          questions: [
            {
              prompt: "Where were the people in the scenario?",
              name: "location",
              options: [
                "A wedding",
                "A darty",
                "A birthday party",
                "A religious organization",
              ],
              required: true,
            },
            {
              prompt: "What food did the people in the scenario order?",
              name: "food",
              options: [
                "Ralph ordered the mushroom risotto, and Maxwell ordered the coconut curry salmon",
                "Maxwell ordered the mushroom risotto, and Ralph ordered the coconut curry salmon",
              ],
              required: true,
            },
          ],
          button_label: "Continue",
          on_finish: function (data) {
            const responses = data.response || {};
            const correctLocation = responses.location === "A wedding" ? 1 : 0;
            const correctFood =
              responses.food === "Maxwell ordered the mushroom risotto, and Ralph ordered the coconut curry salmon" ? 1 : 0;
            const totalCorrect = correctLocation + correctFood;
            data.trial_type = "attention_check";
            data.attention_correct_count = totalCorrect;
            data.attention_correct_location = correctLocation;
            data.attention_correct_food = correctFood;
          },
        });
      }

    trials.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: "Next scenario",
      choices: "NO_KEYS",
      trial_duration: function () {
        return jsPsych.randomization.sampleWithoutReplacement(
          [1500, 1750, 2000, 2300],
          1
        )[0];
      },
    });
  });

  const exitSurvey = {
    type: jsPsychSurveyHtmlForm,
    preamble: `
      <div>
        <h2>Exit Survey</h2>
        <p>You have reached the end of the survey. To collect your pay, please complete the following questions. Your answer to these questions will not affect your pay, so please answer honestly.</p>
      </div>
    `,
    html: exitSurveyHtml,
  };

  const saveData = {
    type: jsPsychPipe,
    action: "save",
    experiment_id: "pnRwHvJ3SpWg",
    filename: `${subjectId}.json`,
    data_string: () => jsPsych.data.get().json(),
  };

  const thankYou = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `<p>Thanks for participating in the experiment!</p>
                  <p><a href="https://app.prolific.com/submissions/complete?cc=C1A889GX">Click here to return to Prolific and complete the study</a>.</p>
                  <p>It is now safe to close the window. Your pay will be delivered within a few days.</p>
                  `,
    choices: "NO_KEYS",
  };

  let timeline = [];

  timeline.push(consent);
  timeline.push(instructions);
  timeline.push(...trials);
  timeline.push(exitSurvey);
  timeline.push(saveData);
  timeline.push(thankYou);

  return timeline;
}
