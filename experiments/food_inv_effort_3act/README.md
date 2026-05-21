# food_inv_effort_3act

Study 3a — Effort inference.

## Design

**Factor crossing**: 2 (desire: low/high) × 4 (intimacy: 0/50/75/100) × 3 (observed action)

**Known to participant**: reward_condition, intimacy

**Inferred by participant**: effort_condition (one slider). Slider endpoints are the two effort paragraphs (effort_low / effort_high). NOTE: the observer should NOT see either effort paragraph as part of the vignette; they appear as slider endpoints only.

## Trial structure

Each trial shows: vignette + reward_low/high paragraph + intimacy descriptor + observed action (NO effort paragraph in the vignette)

The participant gives prior and posterior ratings on 1 slider(s):
P(effort_high) (0-100). Endpoints = effort_low / effort_high paragraphs.

Each participant sees 16 trials total (one per scenario). Cell assignment is rotated across participants so cells are balanced in aggregate.

## Stimulus source

Loads stimuli from `experiments/scenarios_3act.csv` via the routing in `experiments/csv_to_json.py`. Regenerate the stimuli JSON with:

```bash
uv run python experiments/csv_to_json.py
```

## Files in this directory

- `index.html` — entry point
- `experiment.js` — jsPsych 8.x boilerplate
- `trials.js` — **TODO**: adapt from the cloned `food_inv_intimacy_effort_alt` template to match this study's design (different paragraphs shown, different slider count, 3 actions instead of 2). See `## Trial structure` above for the spec.
- `python/generate_counterbalancing.py` — **TODO**: write a counterbalancing script that assigns each participant 16 cells covering all scenarios, with cell-balanced rotation across participants.
- `json/stimuli.json` — generated from `scenarios_3act.csv`; do not edit by hand.

## Status

Scaffolding only. The `trials.js` here is a verbatim clone of the 2-action effort-experiment template and **will not run correctly** until adapted to this study's factor structure. See the manuscript's Methods section for the exact wording of the inference instructions.
