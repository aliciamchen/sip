// Shared experiment configuration, collected in one place so the values that are
// repeated across every experiment live in a single file. Each experiment's
// trials.js does
//
//   export const CONFIG = makeConfig("food_inv_desire");
//
// and gets the shared defaults plus its own DataPipe ID. To change a default for
// one experiment, pass overrides as a second argument, e.g.
//
//   export const CONFIG = makeConfig("food_inv_desire", { ATTENTION_CHECK_INDEX: 10 });
//
// The shared _lib/ is deployed alongside every experiment, so this file ships too.

// DataPipe experiment IDs, keyed by experiment slug. To get one, create the
// experiment on https://pipe.jspsych.org and paste its "Experiment ID" here.
// "TODO_FILL_IN_DATAPIPE_ID" marks one not yet created — that experiment will not
// save data or assign a condition until it has a real ID.
export const DATAPIPE_IDS = {
  food_inv_desire: "ixxsoCvjY9kH", // Study 1a
  food_inv_joint_de: "TODO_FILL_IN_DATAPIPE_ID", // Study 1b
  food_inv_intimacy: "TODO_FILL_IN_DATAPIPE_ID", // Study 2a
  food_inv_joint_ie: "TODO_FILL_IN_DATAPIPE_ID", // Study 2b
};

// Settings identical across all active experiments.
export const SHARED_CONFIG = {
  ATTENTION_CHECK_INDEX: 14,
  ATTENTION_TOLERANCE: 0.02,
  INTER_TRIAL_DURATIONS: [1500, 1750, 2000],
  PROLIFIC_COMPLETION_URL:
    "https://app.prolific.com/submissions/complete?cc=C1A889GX",
};

// Build an experiment's CONFIG: the shared defaults plus its DataPipe ID, with
// any per-experiment overrides applied last.
export function makeConfig(slug, overrides = {}) {
  return {
    ...SHARED_CONFIG,
    PIPE_EXPERIMENT_ID: DATAPIPE_IDS[slug],
    ...overrides,
  };
}
