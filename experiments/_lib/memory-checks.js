// Per-scenario memory checks. Each experiment passes a lookup of the memory
// checks that apply to its scenario set. Food experiments use the hike + wedding
// scenarios (FOOD_MEMORY_CHECKS); the nonfood experiments use sleeping-bag +
// salary (NONFOOD_MEMORY_CHECKS). Both sets ask three questions total across
// the two checks (the exclusion rules count questions, not checks).
// makeMemoryCheckForStimulus returns the trial object keyed by scenario_label,
// or null if no memory check applies.

const MEMORY_CHECK_PREAMBLE_MULTI = `
  <div>
    <h3>Memory Check</h3>
    <p>This is a memory check to make sure you're not a bot and that we can incorporate your responses into our study. Your responses on the memory check will not affect your pay or whether your submission is approved for payment.</p>
    <p>Please answer the following questions about the previous scenario.</p>
  </div>
`;

const MEMORY_CHECK_PREAMBLE_SINGLE = `
  <div>
    <h3>Memory Check</h3>
    <p>This is a memory check to make sure you're not a bot and that we can incorporate your responses into our study. Your responses on the memory check will not affect your pay or whether your submission is approved for payment.</p>
    <p>Please answer the following question about the previous scenario.</p>
  </div>
`;

export const FOOD_MEMORY_CHECKS = {
  hike: {
    type: jsPsychSurveyMultiChoice,
    preamble: MEMORY_CHECK_PREAMBLE_MULTI,
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
      const correctFood = responses.food === "Snacks and energy bars" ? 1 : 0;
      data.response_type = "memory_check";
      data.memory_correct_count = correctNames + correctFood;
      data.memory_correct_names = correctNames;
      data.memory_correct_food = correctFood;
    },
  },
  wedding: {
    type: jsPsychSurveyMultiChoice,
    preamble: MEMORY_CHECK_PREAMBLE_SINGLE,
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
      data.response_type = "memory_check";
      data.memory_correct_count = correctLocation;
      data.memory_correct_location = correctLocation;
    },
  },
};

export const NONFOOD_MEMORY_CHECKS = {
  "sleeping-bag": {
    type: jsPsychSurveyMultiChoice,
    preamble: MEMORY_CHECK_PREAMBLE_MULTI,
    questions: [
      {
        prompt: "What were the names of the people in the scenario?",
        name: "names",
        options: [
          "Henry and Gabriel",
          "Henry and Gabe",
          "Hugo and Gabriel",
          "Henry and Daniel",
        ],
        required: true,
      },
      {
        prompt: "What happened to Gabriel's sleeping bag?",
        name: "incident",
        options: [
          "He dropped it in a stream",
          "He forgot to pack it",
          "It was torn by an animal",
          "He left it at the trailhead",
        ],
        required: true,
      },
    ],
    button_label: "Continue",
    on_finish: function (data) {
      const responses = data.response || {};
      const correctNames = responses.names === "Henry and Gabriel" ? 1 : 0;
      const correctIncident =
        responses.incident === "He dropped it in a stream" ? 1 : 0;
      data.response_type = "memory_check";
      data.memory_correct_count = correctNames + correctIncident;
      data.memory_correct_names = correctNames;
      data.memory_correct_incident = correctIncident;
    },
  },
  salary: {
    type: jsPsychSurveyMultiChoice,
    preamble: MEMORY_CHECK_PREAMBLE_SINGLE,
    questions: [
      {
        prompt: "Where were the people in the scenario?",
        name: "location",
        options: [
          "At a dinner after an industry event",
          "In their office break room",
          "At a company holiday party",
          "On a video call",
        ],
        required: true,
      },
    ],
    button_label: "Continue",
    on_finish: function (data) {
      const responses = data.response || {};
      const correctLocation =
        responses.location === "At a dinner after an industry event" ? 1 : 0;
      data.response_type = "memory_check";
      data.memory_correct_count = correctLocation;
      data.memory_correct_location = correctLocation;
    },
  },
};

export function makeMemoryCheckForStimulus(stimulus, memoryChecks = FOOD_MEMORY_CHECKS) {
  return memoryChecks[stimulus.scenario_label] ?? null;
}
