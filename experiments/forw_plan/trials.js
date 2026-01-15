const intimacy_texts = {
  0: "0 (maximally formal)",
  50: "50 (neither formal nor intimate)",
  75: "75 (somewhat intimate)",
  100: "100 (maximally intimate)",
};

export const CONFIG = {
  ATTENTION_CHECK_INDEX: 14,
  ATTENTION_TOLERANCE: 0.02,
  INTER_TRIAL_DURATIONS: [1500, 1750, 2000],
  PIPE_EXPERIMENT_ID: "vYTA0MgXDpyR",
  PROLIFIC_COMPLETION_URL:
    "https://app.prolific.com/submissions/complete?cc=C1A889GX",
};

const getRewardText = (stimulus) =>
  stimulus.reward_condition === "low"
    ? stimulus.reward_low
    : stimulus.reward_high;

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
                <p>Some relationships are formal, like some relationships with an employee, a religious leader, a shopkeeper or a new acquaintance. Other relationships are close and intimate, like some relationships with a romantic partner, sibling or best friend.</p> 
            </div>
            `,
      `
            <div class="instructions-container">
                <h2>Social interactions survey</h2>
                <p>For each scenario, you will read about four different actions the two people can take. You will use sliders to indicate the probability that the two people will choose each action. The probabilities must sum to 100%. You can move sliders freely, and when you release a slider, all values will be automatically adjusted to sum to 100%.</p>
                <p>(Note that this means that sometimes you might have to move the sliders multiple times to get to the probabilities you want.)</p>
            </div>
            `,
      `
            <div class="instructions-container">
              <h2>Social interactions survey</h2>
                <p>Please pay attention to the social relationship between the two people, and read each of the scenarios and ways of sharing food carefully! 🙂 You will receive $6.25 if you successfully complete the survey. </p>
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
    if (stimulusIndex === CONFIG.ATTENTION_CHECK_INDEX) {
      const attentionCheckLabels = [
        "Please set this slider to 0%",
        "Please set this slider to 0%",
        "Please set this slider to 25%",
        "Please set this slider to 75%",
      ];

      trials.push({
        type: jsPsychProbabilitySliders,
        labels: attentionCheckLabels,
        start: [0.25, 0.25, 0.25, 0.25], // Start with equal probabilities
        button_label: "Continue",
        show_reset: true,
        show_chips: false,
        instruction_html: `
          <div>
            <p>This is an attention check to make sure you're not a bot and that we can award you your pay for the study.</p>
            <p><strong>Please set each slider to the exact percentage requested below.</strong></p>
          </div>
        `,
        precision: 3,
        require_total_exact: true, // Allow non-100% totals for attention check
        data: {
          response_type: "attention_check",
        },
        on_finish: function (data) {
          const probs = data.probs || [];
          const tol = CONFIG.ATTENTION_TOLERANCE;
          data.attention_passed =
            Math.abs(probs[0] - 0.0) < tol &&
            Math.abs(probs[1] - 0.0) < tol &&
            Math.abs(probs[2] - 0.25) < tol &&
            Math.abs(probs[3] - 0.75) < tol;
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
                        <p>On a scale from 0 (maximally formal) to 100 (maximally intimate), ${
                          stimulus.name_0
                        } and ${
        stimulus.name_1
      } are in a relationship they would describe as <strong>${
        intimacy_texts[stimulus.intimacy_condition]
      }</strong>.</p>
                            <p>${stimulus.vignette}</p>
                            <p>${getRewardText(stimulus)}</p>
                        </div>
                        <p style="text-align: center;"><em>Press any key to see the actions.</em></p>
                    </div>
                `,
      choices: "ALL_KEYS",
    });

    const actionLabels = [];
    for (let i = 0; i < 4; i++) {
      actionLabels.push(stimulus[`action_${i}`]);
    }

    trials.push({
      type: jsPsychProbabilitySliders,
      labels: actionLabels,
      start: [0.25, 0.25, 0.25, 0.25], 
      button_label: "Continue",
      show_reset: true,
      show_chips: true,
      instruction_html: `
        <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
        <div class="vignette-text vignette-text-wide">
          <p>On a scale from 0 (maximally formal) to 100 (maximally intimate), ${
            stimulus.name_0
          } and ${
        stimulus.name_1
      } are in a relationship they would describe as <strong>${
        intimacy_texts[stimulus.intimacy_condition]
      }</strong>.</p>
          <p>${stimulus.vignette}</p>
          <p>${getRewardText(stimulus)}</p>
        </div>
        <p><strong>Please indicate the probability that the two people will choose each action.</strong></p>
      `,
      precision: 3,
      require_total_exact: true,
      data: {
        response_type: "response",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        vignette: stimulus.vignette,
        intimacy_condition: stimulus.intimacy_condition,
        reward_condition: stimulus.reward_condition,
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
          CONFIG.INTER_TRIAL_DURATIONS,
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
                  <p><a href="${CONFIG.PROLIFIC_COMPLETION_URL}">Click here to return to Prolific and complete the study</a>.</p>
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
