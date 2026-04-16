---
paths:
  - "experiments/**/*"
---

# Experiment structure

Each experiment folder contains:
- `index.html` - Entry point
- `experiment.js` - jsPsych 8.x experiment logic
- `trials.js` - Trial configuration

Experiments collect data via jsPsych-contrib/pipe plugin to `data/<experiment_name>/raw_data/`.
