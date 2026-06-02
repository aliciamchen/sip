import { runExperiment } from "../_lib/bootstrap.js";
import {
  CONFIG,
  makeStimulusTrials,
  INSTRUCTIONS_PAGES,
  COMPREHENSION_QUESTIONS,
} from "./trials.js";

runExperiment({
  config: CONFIG,
  makeStimulusTrials,
  instructionsPages: INSTRUCTIONS_PAGES,
  comprehensionQuestions: COMPREHENSION_QUESTIONS,
  consentTemplate: "food-inverse",
});
