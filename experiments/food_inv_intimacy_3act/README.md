# food_inv_intimacy_3act

Study 2 — Inverse planning of intimacy.

## Design

**Factor crossing**: 2 (desire: low/high) × 2 (effort: low/high) × 3 (observed action: action_0/1/2)

**Known to participant**: reward_condition, effort_condition

**Inferred by participant**: intimacy (one slider, 0-100 = maximally formal -> maximally intimate)

## Trial structure

Each trial shows: vignette + reward_low/high paragraph + effort_low/high paragraph + observed action

The participant gives prior and posterior ratings on 1 slider(s):
intimacy (0 = maximally formal, 100 = maximally intimate)

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
