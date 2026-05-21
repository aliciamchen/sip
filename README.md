# Inverse planning in the context of sociological structure

Cognitive science research on social inference and food sharing: how people decide to share saliva-transferring food with someone given the relationship between them, and how observers infer the relationship, desire, or physical effort from observed actions.

The current manuscript organizes the work into five studies: two forward-planning experiments (Studies 1a, 1b) with desire and effort manipulations on the canonical 4-action and 2-action food stimulus sets respectively, and a family of inverse-planning experiments (Studies 2, 3a, 3b, 4a, 4b) built on a new 3-action set (no sharing / low-risk sharing / high-risk sharing) that lets the three latent variables — desire, effort, intimacy — be crossed and inferred in different combinations. Study 5 generalizes the inverse-inference findings to non-food domains and is still being collected.

Six older 4-action and 2-action inverse experiments are archived under `data/legacy/` and described in [data/legacy/README.md](data/legacy/README.md). The model scripts and analysis qmds for those legacy experiments remain in place and runnable, but they are no longer part of the default `make all` pipeline.

## Quick start

```bash
# Python deps (uses uv: https://github.com/astral-sh/uv)
uv sync

# R deps (renv): open R from the project root, then
# renv::restore()

# Quarto (https://quarto.org/docs/get-started/) is needed to render the analysis docs

make all                # full pipeline: fit → predict → CV → render qmds
make help               # list all targets
```

The processed CSVs are checked into the repo, so `make all` works without raw JSON. Rendered analysis documents land in `_output/analysis/`; figures land in `figures/`.

The `Makefile` exposes per-stage and per-experiment targets (`make fit`, `make cv`, `make analysis-food-forw-intimacy-desire-analysis`, etc.); see `make help`. Underlying script invocations are documented in [Manual pipeline steps](#manual-pipeline-steps) below.

## Pipeline at a glance

```
jsPsych experiments (experiments/) → JSON → json_to_csv.py → processed CSVs (data/)
                                                              ↓
LM elicitation (lm/score_*.py) → scenario tables (model/outputs/lm/)
                                                              ↓
                                  fit (forward / inverse) → predict → CV
                                                              ↓
                                  R/Quarto analysis (analysis/) → figures
```

## Experiments

The stable identifier for each experiment is its directory slug in `data/` and `experiments/` (paper-level numbers change as the writeup evolves).

### Study 1 — Forward planning (actors choose actions)

- **Study 1a: Desire manipulation** (`food_forw_intimacy_desire/`) — actors choose among four candidate actions given intimacy (4 levels) × desire (2 levels). Uses the 4-action canonical stimulus set in `experiments/scenarios.csv`.
- **Study 1b: Effort manipulation** (`food_forw_intimacy_effort/`) — actors choose between two actions (a non-saliva alternative and a saliva-sharing one) given intimacy (4 levels) × relative effort (2 levels). Reward is held fixed at high; an effort paragraph in the vignette makes the resource that the non-saliva action relies on either easy or costly to obtain. Uses `experiments/scenarios_effort.csv`.

### Studies 2–4 — Inverse planning on the 3-action set

The inverse-planning experiments use a new 3-action stimulus set (no sharing / low-risk sharing / high-risk sharing) defined in `experiments/scenarios_3act.csv`, which merges effort paragraphs into the canonical scenarios so that all three latent variables — desire, effort, intimacy — can be manipulated alongside the observed action. On each trial the participant sees a vignette plus whichever condition paragraphs are revealed by the design, then a single observed action, then gives prior/posterior ratings on one or two sliders.

- **Study 2 — Inverse intimacy** (`food_inv_intimacy_3act/`) — known: desire + effort. Inferred: intimacy. Design: 2 × 2 × 3.
- **Study 3a — Effort inference** (`food_inv_effort_3act/`) — known: desire + intimacy. Inferred: effort. Design: 2 × 4 × 3. The observer does not see the effort paragraph; the model uses an effort-marginal access table.
- **Study 3b — Desire inference** (`food_inv_desire_3act/`) — known: effort + intimacy. Inferred: desire. Design: 2 × 4 × 3.
- **Study 4a — Joint inference (desire + effort)** (`food_inv_joint_de_3act/`) — known: intimacy. Inferred jointly: desire and effort. Design: 4 × 3. Two sliders per trial.
- **Study 4b — Joint inference (desire + intimacy)** (`food_inv_joint_di_3act/`) — known: effort. Inferred jointly: desire and intimacy. Design: 2 × 3. Two sliders per trial.

### Study 5 — Generalization beyond food sharing

A non-food stimulus set in `experiments/scenarios_nonfood.csv` parallels the 4-action canonical food set across substance sharing (chapstick, hairbrush), shared physical space (blanket, bed), and informational/situational privacy (a breakup conversation, a phone passcode). A 3-action variant of the non-food scenarios, with effort paragraphs added, is still pending. The plan covers non-food replications of Studies 2, 3b, and 4b; the four scaffolded but data-less `nonfood_inv_*` dirs in `experiments/` are leftover 4-action stubs and should be retired during the next round of cleanup.

Currently only `nonfood_forw_intimacy_desire` has data on the non-food side.

### Scenarios

Three Python scripts in `experiments/` are the source of truth for the stimuli; each writes a `.csv` artifact next to it. Edit the `.py` file and regenerate.

- [`experiments/scenarios.py`](experiments/scenarios.py) → `scenarios.csv` (4-action canonical, Study 1a).
- [`experiments/scenarios_effort.py`](experiments/scenarios_effort.py) → `scenarios_effort.csv` (2-action effort, Study 1b).
- [`experiments/scenarios_3act.py`](experiments/scenarios_3act.py) → `scenarios_3act.csv` (3-action, Studies 2–4).
- [`experiments/scenarios_nonfood.py`](experiments/scenarios_nonfood.py) → `scenarios_nonfood.csv` (4-action non-food).

See the [experiments README](experiments/README.md) for the column schema.

## Utility model

A joint actor chooses an action by softmaxing over the utility:

```
P(a | s, I) ∝ exp( U(a | s, I) )

U(a | s, I) = w_v · V(a|s)
            − w_d · access(a) · (1 − I)
            − w_e · effort(a)
```

`V(a|s, m)` is the signed valence of the action with respect to the actor's motivational state (in [-1, +1]; positive = serves the state, negative = actively counterproductive). `access(a)` is a graded measure of how much an action opens the actor up to the other person — their body, private information, and physical space. `effort(a)` is the physical effort of executing the action. Intimacy `I` scales the access-discomfort term: at high intimacy the `−w_d · access · (1 − I)` penalty shrinks toward zero, so higher-access actions become relatively more attractive. At low intimacy the penalty is at full strength, so higher-access actions are costly.

Three ablations of this utility are fit and compared for both the forward-planning (actor) and inverse-planning (observer) models:

- **Full** (`full`) — the full utility above: V, the access-discomfort term, and effort (the main model).
- **Discomfort-only** (`discomfort_only`) — only the access-discomfort term `−w_d · access · (1 − I)`; drops V and effort to ask whether the access signal alone can account for behavior.
- **Base** (`base`) — `w_v · V − w_e · effort`; no relational structure.

### Where the utility values come from

All three components — V, access, effort — are **LLM-generated per scenario** because they genuinely vary by scenario (a wedding meal differs from a basketball hot dog in bodily exposure, logistical effort, and how strongly each available action serves or counters the actor's motivational state). The LLM is Llama-3.3-70B via Together AI (10 runs averaged, mean ± std saved):

- `V`: signed valence per (scenario, action, motivation) on a -3 to +3 scale, normalized to [-1, +1]. -3 = action is strongly counterproductive for the actor's current state (e.g., eating heartily when full); 0 = neutral; +3 = action strongly serves the state. Produced by `model/lm/score_canonical_v.py` → `model/outputs/lm/lm_scenario_v.csv`.
- `access`: how much each action opens each person up to the other — physically (bodily substance transfer, skin contact, spatial proximity), informationally, or both — on a 0-6 scale normalized to [0, 2]. Produced by `model/lm/score_canonical_features.py` → `model/outputs/lm/lm_scenario_params.csv`.
- `effort`: physical, logistical, and time cost of executing the action, on a 0-6 scale normalized to [0, 1]. Produced by the same script.

A single prompt set in `model/lm/prompts.py` is used for both the food and non-food pipelines. The access rubric covers three channel types — bodily-substance transfer, direct physical contact, and informational/private-resource disclosure — so the same prompt works for food sharing, shared objects, shared physical space, and privacy or information-disclosure scenarios.

The Together AI calls go through `model/lm/client.py`, which fans the 10 runs across a thread pool, constrains the output to a JSON schema via Together's `response_format`, retries transient errors, and checkpoints to disk after each scenario. Output rows record both `n_runs_*` (successful runs) and `n_failures_*` so it's clear how many of the 10 runs returned parseable output.

The no-alt observer experiments need V scored not just for the four canonical actions but also for LM-generated counterfactual alternatives. The two no-alt experiments use different conditioning axes for those alternatives, following the design rule that alternatives should be conditioned on what the observer can see (not on the latent the observer infers):

- `food_inv_intimacy_desire_noalt`: observer sees motivation, infers intimacy. Alternatives are conditioned on (scenario, observed action, motivation). Generated by `model/lm/generate_alternatives_motivation.py`; features and V scored by `score_alternative_features.py` and `score_alternative_v.py`.
- `food_inv_desire_intimacy_noalt`: observer sees relationship, infers motivation. Alternatives are conditioned on (scenario, observed action, relationship). Generated by `model/lm/generate_alternatives_relationship.py`; same scripts handle features and V via `--conditioning relationship`.

Generating LM tables requires `TOGETHER_API_KEY` in `.env`. The fitting and prediction scripts index into these tables by `scenario_idx`; they require the relevant CSVs to exist.

## Repository structure

```
analysis/          R/Quarto analysis scripts and data processing
data/              Processed experiment data (one folder per experiment slug)
experiments/       jsPsych experiment code + scenario definitions
model/             Computational models
  forward/         Per-experiment forward-planning fit/predict scripts
  inverse/         Per-experiment inverse-planning fit/predict scripts
  cv/              Per-experiment LOSO cross-validation scripts
  lm/              LM-elicitation scripts (Together AI)
  outputs/         Fitted parameters, predictions, CV results
    lm/            LM-elicited scenario tables
    <slug>/        Per-experiment outputs
figures/           Generated figures used in the paper
LM_evals/          Language-model evaluation code
```

See the [data codebook](data/README.md), [experiments README](experiments/README.md), [model README](model/README.md), and [model outputs README](model/outputs/README.md) for details on each directory.

## Dependencies

### Python (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if needed
uv sync                                            # install deps + create .venv
```

To use pip instead: `python3 -m venv .venv && source .venv/bin/activate && pip install .`. Run scripts with plain `python` rather than `uv run python` in that case.

### R (renv)

R version 4.5.2 is required (specified in the lockfile). Open R from the project root — `.Rprofile` will bootstrap renv automatically, then run `renv::restore()` to install all packages. If the auto-bootstrap fails: `install.packages("renv"); renv::activate(); renv::restore()`.

### Quarto

[Quarto](https://quarto.org/docs/get-started/) is needed to render the analysis documents.

## Manual pipeline steps

The Makefile is the recommended way to run the pipeline; the steps below show the underlying invocations for reference.

Convert raw data (not included in the repository) to CSV with anonymized participant IDs:

```bash
uv run python analysis/json_to_csv.py <experiment_slug>
```

Generate LLM-derived scenario parameters (prerequisite for all model fits; requires `TOGETHER_API_KEY`):

```bash
uv run python model/lm/score_canonical_features.py                  # food access + effort (Study 1a, legacy inverse)
uv run python model/lm/score_canonical_features.py --domain nonfood # non-food access + effort
uv run python model/lm/score_canonical_v.py                         # food signed-valence V (Study 1a, legacy inverse)
uv run python model/lm/score_canonical_v.py --domain nonfood        # non-food V
uv run python model/lm/score_effort_features.py                     # Study 1b: access + effort + effort-marginal access on the 2-action effort set
uv run python model/lm/score_3act_features.py                       # Studies 2/3/4: access + effort + effort-marginal access on the 3-action set
uv run python model/lm/score_3act_v.py                              # Studies 2/3/4: signed-valence V on the 3-action set
# Legacy no-alt observers (under data/legacy/) additionally need LM-generated alternatives:
uv run python model/lm/generate_alternatives_motivation.py
uv run python model/lm/score_alternative_features.py
uv run python model/lm/score_alternative_v.py
uv run python model/lm/generate_alternatives_relationship.py
uv run python model/lm/score_alternative_features.py --conditioning relationship
uv run python model/lm/score_alternative_v.py --conditioning relationship
```

Each fit/predict/CV script is named after the experiment it serves and lives in `model/forward/`, `model/inverse/`, or `model/cv/`. Run `fit_<slug>.py`, then `predict_<slug>.py`; for cross-validation run `cv_<slug>.py`.

```bash
# forward (Studies 1a, 1b)
uv run python model/forward/fit_food_forw_intimacy_desire.py
uv run python model/forward/predict_food_forw_intimacy_desire.py
# inverse (Studies 2, 3a, 3b, 4a, 4b)
uv run python model/inverse/fit_food_inv_intimacy_3act.py
uv run python model/inverse/predict_food_inv_intimacy_3act.py
# cross-validation
uv run python model/cv/cv_food_forw_intimacy_desire.py
uv run python model/cv/cv_food_inv_intimacy_3act.py
```

The new 3-action inverse experiments freeze actor weights from `food_forw_intimacy_desire` and fit only `α_observer`. The joint Studies 4a/4b sum two binary cross-entropy NLLs across the two slider responses per trial. The legacy inverse experiments (under `data/legacy/`) freeze actor weights from `food_forw_intimacy_desire` or `food_forw_intimacy_effort` depending on the action set; their full per-slug commands are documented in `.claude/rules/model.md`. `food_inv_effort_3act` (Study 3a) uses **effort-marginal access** because that observer doesn't see the effort paragraph.

All reported model-vs-human correlations in the analysis qmds are out-of-sample, pooled across 16 LOSO folds. Forward CV refits actor weights ($w_v, w_d, w_e, \gamma$) per fold; alt-shown inverse CV refits only $\alpha_\mathrm{obs}$; no-alt inverse CV refits all weights jointly per fold.

Render analysis qmds:

```bash
quarto render analysis/<qmd-name>.qmd
# e.g. analysis/food-forw-intimacy-desire-analysis.qmd
# see analysis/ for the full list of 12 qmds
```

Plots are saved in the `figures/` directory; rendered docs in `_output/analysis/`.
