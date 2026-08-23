# Experiments

## Stimulus sources

There are two scenario CSVs, each generated from a Python source of truth: `scenarios.py` → `scenarios.csv` (the food set, Studies 1a/1b/2a/2b) and `scenarios_nonfood.py` → `scenarios_nonfood.csv` (the non-food set, Studies 3a/3b). Edit the `.py` file and regenerate with `uv run python experiments/<file>.py` — never edit the CSVs directly, since the next regeneration will overwrite the edits. After regenerating, run `uv run python experiments/build/csv_to_json.py` to propagate the changes into each experiment's `json/stimuli.json`.

### `scenarios.csv` and `scenarios_nonfood.csv` — 3-action sets

`scenarios.csv` is the food stimulus set (Studies 1a, 1b, 2a, 2b) and `scenarios_nonfood.csv` the non-food one (Studies 3a, 3b; 16 scenarios spanning bodily-substance transfer, shared physical exposure, and private access, marked by an extra `scenario_type` column). Both carry the effort paragraphs alongside the desire and intimacy framing so all three latent variables (desire, effort, intimacy) can be manipulated alongside the observed action.

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier used in data files |
| `name_0`, `name_1` | Character names in the vignette |
| `vignette` | Base scenario description |
| `desire_object` | Scenario-specific object of the desire question "How much do you think {name_0} and {name_1} would like ___?" (e.g. `the hot dog`, `to try the harmonica`), so the question names the actual thing at stake rather than "the food". Used in the desire-DV studies (1a, 1b, 3a) |
| `desire_low`, `desire_high` | Two desire-state paragraphs |
| `low_risk_share_effort_low`, `low_risk_share_effort_high` | Two effort-state paragraphs — the low version makes the resource that `low_risk_share` relies on easy to obtain, the high version makes it costly |
| `no_share` | No sharing |
| `low_risk_share` | Low-risk (non-saliva) sharing — the effort cost lives in the `low_risk_share_effort_*` paragraph, not the action text |
| `high_risk_share` | High-risk (saliva) sharing |

## Build scripts

The generators that turn source files into the artifacts each experiment loads are in [`experiments/build/`](build/) (never deployed — the deploy only pushes `_lib/` and the per-experiment dirs). The authored scenario data stays at `experiments/scenarios.py`.

- `csv_to_json.py` — scenario CSV → each experiment's `json/stimuli.json` (routing in its `SOURCES` table).
- `counterbalancing.py` — per-study `json/full_counterbalancing.json` (designs in its `STUDY_CONFIGS` table; `--study <slug>` for one).
- `sync_entry_files.py` — writes every active experiment's `index.html` + `experiment.js` from one template (identical across studies, except that `experiment.js` names the study's consent template).
- `_scenario_io.py` — shared `write_scenarios_csv()` that the `scenarios*.py` source files call.

The `Makefile` wraps these: `make stimuli`, `make counterbalancing`, `make entry-files`, or `make experiments` for all three.

## Active experiments

### Inverse planning (3-action set)

In all six, the participant sees a single observed action per trial (never a list of candidate actions), with prior and posterior slider responses. The DV scales are desire as a continuous 0–100 rating with a scenario-specific question that names the characters and the thing at stake ("How much do you think Carissa and Josh would like the hot dog?", not at all → moderately → extremely), effort as a continuous 0–100 rating between the two effort paragraphs as endpoints, and intimacy as a continuous 0–100 rating from maximally formal to maximally intimate. When intimacy is a given condition rather than a DV, it is a verbal level (`max_formal` / `somewhat_formal` / `somewhat_intimate` / `max_intimate`) shown as a relationship descriptor, never a number. Each experiment dir's `README.md` documents the design spec.

Each entry links to that study's design spec; the slug ↔ study mapping is in the [root README](../README.md#experiments), and the full per-study designs are specified in the manuscript.

- [food_inv_desire](food_inv_desire/README.md) — **Study 1a**: infer desire under known effort + intimacy.
- [food_inv_joint_de](food_inv_joint_de/README.md) — **Study 1b**: joint inference over desire and effort given intimacy (two sliders per trial).
- [food_inv_intimacy](food_inv_intimacy/README.md) — **Study 2a**: infer intimacy under known desire + effort.
- [food_inv_joint_ie](food_inv_joint_ie/README.md) — **Study 2b**: joint inference over intimacy and effort given desire (two sliders per trial).
- [nonfood_inv_joint_de](nonfood_inv_joint_de/README.md) — **Study 3a**: Study 1b's design on the non-food scenario set.
- [nonfood_inv_joint_ie](nonfood_inv_joint_ie/README.md) — **Study 3b**: Study 2b's design on the non-food scenario set.

## Counterbalancing

[`build/counterbalancing.py`](build/counterbalancing.py) generates every experiment's `json/full_counterbalancing.json`: an array of pre-built sequences, each assigning one condition cell to each of the 16 scenarios. When a participant starts, `experiment.js` uses their jsPsychPipe `condition_assignment` to pick a sequence, so each participant sees 16 trials (one per scenario) with the scenario-to-condition assignment balanced in both directions: across participants, every condition cell (and every scenario × cell pairing) appears equally often; within a participant, the 16 trials are split as evenly across each factor's levels as 16 trials allow. The number of sequences matches each study's target sample size, so every participant in a full sample gets a distinct scenario → condition mapping. Each study's design, sequence count, and seed are in the script's `STUDY_CONFIGS` table, and the balancing scheme itself is implemented and commented in the script.

## Shared experiment code

Most of each experiment lives in shared modules under [`_lib/`](_lib/); each `trials.js` is reduced to its study-specific stimulus trials. The shared modules are:

- `bootstrap.js` — `runExperiment({ config, makeStimulusTrials, instructionsPages, comprehensionQuestions, consentTemplate })`: fetches assets, initializes jsPsych, assigns the counterbalancing condition, and **assembles the whole timeline** (consent → instructions + comprehension gate → stimulus trials → exit survey → save → thank-you). The per-study `makeStimulusTrials` is slotted in; the rest is identical for every experiment, so timeline-wide changes happen here.
- `config.js` — the `DATAPIPE_IDS` and `PROLIFIC_COMPLETION_CODES` maps (both keyed by slug; a study with no completion code refuses to start, so a new study cannot silently reuse another's code) plus the settings shared by every experiment (attention-check index/tolerance, inter-trial durations, and the payment and duration shown in the consent form and instructions). Each `trials.js` builds its `CONFIG` with `makeConfig("<slug>")` (pass overrides as a second argument to depart from a default, e.g. a study that pays differently).
- `instructions.js` — `STUDY_INSTRUCTIONS`, every study's instruction pages in one place (shared notes + per-study pages), easy to compare; the non-food studies swap the food-specific paragraphs for domain-general variants.
- `comprehension-check.js` — `STUDY_COMPREHENSION_CHECKS` (each study's quiz questions, built from shared question blocks like `instructions.js`) and `makeComprehensionGate(...)`. The gate is shown right after the instructions: the participant must answer every question correctly to start the study, gets three attempts (the instructions are re-shown on each miss), and if they never pass the experiment ends asking them to return the study on Prolific, so no data is saved for them.
- `scenario.js` — the per-trial building blocks: condition-paragraph getters (`getDesireText`/`getEffortText`), the intimacy descriptor, slider labels, the "press any key" page, the prior/posterior pause, and `scenarioStimulus(...)` which renders the vignette block + observed action + lead-in.
- `two-slider.js` — `makeTwoSliderForm(...)` renders two sliders on one page (Studies 1b/2b) via `survey-html-form`.
- `timeline.js`, `attention-check.js`, `memory-checks.js`, `style.css`, and the consent templates in `consent/` round out the boilerplate.

So a `trials.js` only defines that study's `makeStimulusTrials` (composing the `_lib` pieces) and exports `CONFIG`, `INSTRUCTIONS_PAGES`, `COMPREHENSION_QUESTIONS`, and `makeStimulusTrials`. The per-experiment `index.html` and `experiment.js` are generated from a single source by `build/sync_entry_files.py` (identical across studies except for the consent template named in `experiment.js`; run it after changing the entry template, e.g. a jsPsych version bump).

This shared layout means the experiments are not standalone folders: each one references `../_lib/` via relative paths. Deploys (see below) need to push `_lib/` to the server alongside the experiment.

## Deploying experiments

Experiments are hosted on an MIT athena web locker and pushed with [`bin/deploy-experiment`](../bin/deploy-experiment), which rsyncs the experiment directory plus the shared `_lib/` (run it with `--help`-style bare invocation to see the options; `--all` / `make deploy-all` pushes everything in one SSH session). Every deploy first regenerates the generated assets from their sources and aborts if anything changed, so stale stimuli can never ship; run just that check with `make check-experiments`. Launching a study also requires its DataPipe experiment ID in the `DATAPIPE_IDS` map and its Prolific completion code in the `PROLIFIC_COMPLETION_CODES` map, both in [`_lib/config.js`](_lib/config.js).

## Previewing trials

[`experiments/preview/`](preview/) is a standalone page that renders any study × scenario × condition exactly as a participant would see it (it calls the same `makeStimulusTrials` functions as the live experiments, without running jsPsych or recording anything). Serve it locally with `make preview` and open `http://localhost:8000/preview/`, or deploy it with `bin/deploy-experiment preview`.
