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
  food_inv_joint_de: "zxPavRt21FCr", // Study 1b
  food_inv_intimacy: "wX18fBV6FFV1", // Study 2a
  food_inv_joint_ie: "Z3EeK2XpEZPr", // Study 2b
  nonfood_inv_joint_de: "TODO_FILL_IN_DATAPIPE_ID", // Study 3a
  nonfood_inv_joint_ie: "TODO_FILL_IN_DATAPIPE_ID", // Study 3b
};

// Prolific completion codes, keyed by experiment slug. Each Prolific study
// issues its own completion code (the `cc` parameter of its completion URL),
// so a newly created study must get its own entry here — bootstrap.js refuses
// to start (alert + throw) when the current study has no code, so an
// unconfigured study cannot silently send participants to another study's
// completion URL. The four food studies share one code because they ran under
// a single Prolific project.
export const PROLIFIC_COMPLETION_CODES = {
  food_inv_desire: "C1A889GX", // Study 1a
  food_inv_joint_de: "C1A889GX", // Study 1b
  food_inv_intimacy: "C1A889GX", // Study 2a
  food_inv_joint_ie: "C1A889GX", // Study 2b
  // nonfood_inv_joint_de / nonfood_inv_joint_ie (Studies 3a/3b) deliberately
  // have no entry yet: add each study's code when it is created on Prolific.
};

// Settings identical across all active experiments.
export const SHARED_CONFIG = {
  ATTENTION_CHECK_INDEX: 14,
  // Raw slider units (0-100, integer steps): 0 = only an exact 0 passes, which
  // is the criterion every participant so far has been scored under. Keep at 0
  // mid-collection so the criterion stays constant within a study's sample.
  ATTENTION_TOLERANCE: 0,
  INTER_TRIAL_DURATIONS: [1500, 1750, 2000],
  // Payment and expected duration, shown to participants in the consent form
  // and the final instructions page (as {{PAYMENT}} / {{DURATION_MINUTES}}
  // placeholders that bootstrap.js fills from CONFIG). A study that pays
  // differently or runs longer overrides these via makeConfig's second
  // argument rather than editing the shared participant-facing text.
  PAYMENT: "$5",
  DURATION_MINUTES: 20,
};

// Build an experiment's CONFIG: the shared defaults plus its DataPipe ID and
// Prolific completion URL, with any per-experiment overrides applied last.
// A slug with no completion code gets PROLIFIC_COMPLETION_URL undefined, which
// bootstrap.js turns into a hard startup failure.
export function makeConfig(slug, overrides = {}) {
  const completionCode = PROLIFIC_COMPLETION_CODES[slug];
  return {
    ...SHARED_CONFIG,
    PIPE_EXPERIMENT_ID: DATAPIPE_IDS[slug],
    PROLIFIC_COMPLETION_URL:
      completionCode === undefined
        ? undefined
        : `https://app.prolific.com/submissions/complete?cc=${completionCode}`,
    ...overrides,
  };
}
