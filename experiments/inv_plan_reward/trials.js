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
  PIPE_EXPERIMENT_ID: "1p56Xsptu0tX",
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
                <p>Some relationships are formal, like some relationships with an employee, a religious leader, a shopkeeper or a new acquaintance. Other relationships are close and intimate, like some relationships with a romantic partner, sibling or best friend.</p> 
            </div>
            `,
      `
            <div class="instructions-container">
                <h2>Social interactions survey</h2>
                <p>For each scenario, you will read about four different actions you might expect the two people to take.</p>
                <p>Before observing what action they decide to take, we will ask you to evaluate how likely you think two possible situations are, for the two people in the scenario.</p>
                <p>You will use a slider to evaluate the likelihood of the two situations. The slider will be labeled with the two situations. The farther you move it toward one side, the more likely you think that situation is, compared to the other.</p>
                <p>Then, we will show you what action they take, and ask you to re-evaluate how likely you think the two possible situations are.</p>
            </div>
            `,
      `
            <div class="instructions-container">
              <h2>Social interactions survey</h2>
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
    if (stimulusIndex === CONFIG.ATTENTION_CHECK_INDEX) {
      trials.push({
        type: jsPsychHtmlSliderResponse,
        labels: ["0", "50", "100"],
        slider_min: 0,
        slider_max: 100,
        step: 1,
        require_movement: true,
        button_label: "Continue",
        stimulus: `
          <div>
            <p>This is an attention check to make sure you're not a bot and that we can award you your pay for the study.</p>
            <p><strong>Please set the slider all the way to the left (0).</strong></p>
          </div>
        `,
        data: {
          response_type: "attention_check",
        },
        on_finish: function (data) {
          data.attention_passed =
            Math.abs(data.response - 0) < CONFIG.ATTENTION_TOLERANCE;
        },
      });
    }

    // Create action labels for probability sliders
    const actionLabels = [];
    for (let i = 0; i < 4; i++) {
      actionLabels.push(stimulus[`action_${i}`]);
    }

    trials.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          <div class="vignette-text">
            <p>On a scale from 0 (maximally formal) to 100 (maximally intimate), ${
              stimulus.name_0
            } and ${
        stimulus.name_1
      } are in a relationship they would describe as <strong>${
        intimacy_texts[stimulus.intimacy_condition]
      }</strong>.</p>
          </div>
          <p style="text-align: center;"><em>Press any key to see the scenario.</em></p>
        </div>
      `,
      choices: "ALL_KEYS",
    });

    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          <div class="vignette-text">
            <p>On a scale from 0 (maximally formal) to 100 (maximally intimate), ${
              stimulus.name_0
            } and ${
        stimulus.name_1
      } are in a relationship they would describe as <strong>${
        intimacy_texts[stimulus.intimacy_condition]
      }</strong>.</p>
            <p>${stimulus.vignette}</p>
            <p><em>You expect that ${stimulus.name_0} and ${
        stimulus.name_1
      } will take one of the following actions:</em></p>
            <ul>
              <li>${stimulus[`action_0`]}</li>
              <li>${stimulus[`action_1`]}</li>
              <li>${stimulus[`action_2`]}</li>
              <li>${stimulus[`action_3`]}</li>
            </ul>
          </div>
          <p><strong>Before observing what they decide to do, which situation do you think is more likely?</strong></p>
        </div>
      `,
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: [
        `<div style="width: 70%;margin: 0 auto;">${stimulus.reward_low}</div>`,
        `<div style="width: 70%;margin: 0 auto;">Equally likely</div>`,
        `<div style="width: 70%;margin: 0 auto;">${stimulus.reward_high}</div>`,
      ],
      button_label: "Continue",
      data: {
        response_type: "response",
        stage: "prior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        intimacy_condition: stimulus.intimacy_condition,
        reward_low: stimulus.reward_low,
        reward_high: stimulus.reward_high,
      },
    });

    trials.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: "",
      choices: "NO_KEYS",
      trial_duration: 1000,
    });

    trials.push({
      type: jsPsychHtmlSliderResponse,
      stimulus: `
        <div>
          <h2>Scenario ${stimulusIndex + 1} of ${stimuli.length}</h2>
          <div class="vignette-text">
            <p>On a scale from 0 (maximally formal) to 100 (maximally intimate), ${
              stimulus.name_0
            } and ${
        stimulus.name_1
      } are in a relationship they would describe as <strong>${
        intimacy_texts[stimulus.intimacy_condition]
      }</strong>.</p>
            <p>${stimulus.vignette}</p>
            <p><em>You expect that ${stimulus.name_0} and ${
        stimulus.name_1
      } will take one of the following actions:</em></p>
            <ul>
              <li>${stimulus[`action_0`]}</li>
              <li>${stimulus[`action_1`]}</li>
              <li>${stimulus[`action_2`]}</li>
              <li>${stimulus[`action_3`]}</li>
            </ul>
          </div>
          <div class="vignette-text vignette-observed">
            <p><em>They decide to take the following action:</em></p>
            <p>${stimulus[`${stimulus.action_condition}`]}</p>
          </div>
          <p><strong>Now that you have observed what they decide to do, which situation do you think is more likely?</strong></p>
        </div>
      `,
      slider_width: 900,
      slider_min: 0,
      slider_max: 100,
      step: 1,
      require_movement: true,
      labels: [
        `<div style="width: 70%;margin: 0 auto;">${stimulus.reward_low}</div>`,
        `<div style="width: 70%;margin: 0 auto;">Equally likely</div>`,
        `<div style="width: 70%;margin: 0 auto;">${stimulus.reward_high}</div>`,
      ],
      button_label: "Continue",
      data: {
        response_type: "response",
        stage: "posterior",
        stimulus_index: stimulusIndex,
        scenario_label: stimulus.scenario_label,
        action_condition: stimulus.action_condition,
        intimacy_condition: stimulus.intimacy_condition,
        reward_low: stimulus.reward_low,
        reward_high: stimulus.reward_high,
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
