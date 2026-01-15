const likert_labels = [
  "Extremely unlikely",
  "Unlikely",
  "Somewhat unlikely",
  "Neither likely nor unlikely",
  "Somewhat likely",
  "Likely",
  "Extremely likely",
];

const closeness_texts = {
  not_close: "not close",
  somewhat_close: "somewhat close",
  close: "close",
  extremely_close: "extremely close",
};

export const CONFIG = {
  PIPE_EXPERIMENT_ID: "oOYpWkStTX9B",
  PROLIFIC_COMPLETION_URL:
    "https://app.prolific.com/submissions/complete?cc=C1A889GX",
};

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
                <p>In this survey, you will read vignettes about two people in different kinds of social relationships, sharing different kinds of food in different situations.</p>
                <p>The two people in each scenario are trying to choose an action to <strong>establish, maintain, or communicate about</strong> their social relationship.</p>
                <p>For each scenario, you will read about four different actions the two people can take. For each action, we will ask you to evaluate the likelihood of the two people choosing that action. Please consider each option independently.</p>
                <p>Please pay attention to the social relationship between the two people, and read each of the scenarios and ways of sharing food carefully! 🙂 You will receive $5 if you successfully complete the survey. </p>
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
    // add attention check after the 14th scenario
    if (stimulusIndex === 14) {
      const attentionCheckQuestions = [];
      const attentionCheckPrompts = [
        `Please select "Extremely unlikely."`,
        `Please select "Extremely likely."`,
        `Please select "Neither likely nor unlikely."`,
        `Please select "Somewhat unlikely."`,
      ];
      for (let i = 0; i < 4; i++) {
        attentionCheckQuestions.push({
          prompt: `<div class="action-text">${attentionCheckPrompts[i]}</div>`,
          name: `attention_check_${i}`,
          labels: likert_labels,
          required: true,
        });
      }

      trials.push({
        type: jsPsychSurveyLikert,
        questions: attentionCheckQuestions,
        preamble: `
          <div>
            <p>This is an attention check to make sure you're not a bot and that we can award you your pay for the study.</p>
            <p><strong>Please answer the following questions.</strong></p>
          </div>
        `,
        randomize_question_order: false,
        button_label: "Continue",
        scale_width: 950,
        data: {
          response_type: "attention_check",
        },
        on_finish: function (data) {
          const responses = data.response || {};
          data.attention_passed =
            responses.attention_check_0 == 0 &&
            responses.attention_check_1 == 6 &&
            responses.attention_check_2 == 3 &&
            responses.attention_check_3 == 2;
        },
      });
    }

    trials.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: `
                    <div>
                        <h2>Scenario ${stimulusIndex + 1} of ${
        stimuli.length
      }</h2>
                        <div class="vignette-text">
                        <p class="closeness-info">Consider ${
                          stimulus.name_0
                        } and ${
        stimulus.name_1
      }, who would describe their relationship as <strong>${
        closeness_texts[stimulus.closeness_condition]
      }</strong>.</p>
                            <p>${stimulus.vignette}</p>
                                    <p>${stimulus.name_0} and ${
        stimulus.name_1
      } want to establish, maintain, or communicate that their relationship is <strong>${
        closeness_texts[stimulus.closeness_condition]
      }</strong>.</p>
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
        labels: likert_labels,
        required: true,
      });
    }

    trials.push({
      type: jsPsychSurveyLikert,
      questions: surveyQuestions,
      preamble: `
        <div>
          <p class="closeness-info">Consider ${stimulus.name_0} and ${
        stimulus.name_1
      }, who would describe their relationship as <strong>${
        closeness_texts[stimulus.closeness_condition]
      }</strong>.</p>
          <p>${stimulus.vignette}</p>
        <p>${stimulus.name_0} and ${
        stimulus.name_1
      } want to establish, maintain, or communicate that their relationship is <strong>${
        closeness_texts[stimulus.closeness_condition]
      }</strong>.</p>
          <p><strong>For each action, please rate how likely the two people are to choose that action.</strong></p>
        </div>
      `,
      randomize_question_order: false,
      button_label: "Continue",
      scale_width: 950,
      data: {
        response_type: "response",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        vignette: stimulus.vignette,
        closeness_condition: stimulus.closeness_condition,
      },
    });

    // Memory check for the "hike" scenario
    if (stimulus.scenario_label === "hike") {
      trials.push({
        type: jsPsychSurveyMultiChoice,
        preamble: `
          <div>
            <h3>Memory Check</h3>
            <p>This is a memory check to make sure you're not a bot and that we can incorporate your responses into our study. Your responses on the memory check will not affect your pay or whether your submission is approved for payment.</p>
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
          data.response_type = "memory_check";
          data.memory_correct_count = totalCorrect;
          data.memory_correct_names = correctNames;
          data.memory_correct_food = correctFood;
        },
      });
    }

    // Memory check for the "wedding" scenario
    if (stimulus.scenario_label === "wedding") {
      trials.push({
        type: jsPsychSurveyMultiChoice,
        preamble: `
            <div>
              <h3>Memory Check</h3>
              <p>This is a memory check to make sure you're not a bot and that we can incorporate your responses into our study. Your responses on the memory check will not affect your pay or whether your submission is approved for payment.</p>
              <p>Please answer the following question about the previous scenario.</p>
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
        ],
        button_label: "Continue",
        on_finish: function (data) {
          const responses = data.response || {};
          const correctLocation = responses.location === "A wedding" ? 1 : 0;
          const totalCorrect = correctLocation;
          data.response_type = "memory_check";
          data.memory_correct_count = totalCorrect;
          data.memory_correct_location = correctLocation;
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
      data.response_type = "exit_survey";
    },
  };

  const saveData = {
    type: jsPsychPipe,
    action: "save",
    experiment_id: CONFIG.PIPE_EXPERIMENT_ID,
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
