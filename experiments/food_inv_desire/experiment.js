import { runExperiment } from "../_lib/bootstrap.js";
import { CONFIG, makeStimulusTrials, INSTRUCTIONS_PAGES } from "./trials.js";

runExperiment({
  config: CONFIG,
  makeStimulusTrials,
  instructionsPages: INSTRUCTIONS_PAGES,
  consentTemplate: "food-inverse",
});
