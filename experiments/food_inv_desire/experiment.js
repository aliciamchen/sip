import { runExperiment } from "../_lib/bootstrap.js";
import { CONFIG, makeTimeline } from "./trials.js";

runExperiment({
  config: CONFIG,
  makeTimeline,
  consentTemplate: "food-inverse",
});
