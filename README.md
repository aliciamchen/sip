# Inverse planning in the context of sociological structure

Cognitive science research on social inference and food sharing: how people decide to share saliva-transferring food with someone given the relationship between them, and how observers infer either the relationship or the actor's desire from the observed action. The project comprises two forward-planning actor experiments and six inverse-planning observer experiments on the canonical food set, plus a generalization set beyond food sharing that is currently in progress (one experiment collected so far; the full set is TBD).

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

### Forward planning (actors choose actions)

- **Desire manipulation** (`food_forw_intimacy_desire/`) — actors choose among four candidate actions given intimacy (4 levels) × desire (2 levels).
- **Effort manipulation** (`food_forw_intimacy_effort/`) — actors choose between two actions (a non-saliva alternative and a saliva-sharing one) given intimacy (4 levels) × relative effort (2 levels). Reward is held fixed at high; an effort paragraph in the vignette makes the resource that the non-saliva action relies on either easy or costly to obtain.

### Inverse planning (observers infer latents from actions)

The first four use the canonical 4-action food-sharing set:

- **Intimacy inference, alternatives shown** (`food_inv_intimacy_desire_alt/`) — observers see all four candidate actions (with the actor's desire known) and infer intimacy.
- **Desire inference, alternatives shown** (`food_inv_desire_intimacy_alt/`) — observers see all four candidate actions (with intimacy known) and infer desire.
- **Intimacy inference, no alternatives shown** (`food_inv_intimacy_desire_noalt/`) — observers see only the single observed action; on the model side, counterfactual alternatives are LM-generated.
- **Desire inference, no alternatives shown** (`food_inv_desire_intimacy_noalt/`) — same noalt setup applied to the desire-inference direction; the slider endpoints are the scenario's `reward_low` / `reward_high` paragraphs.

The other two use the effort 2-action set:

- **Intimacy inference, effort manipulation** (`food_inv_intimacy_effort_alt/`) — observers see observed action (2 levels) × effort (2 levels) and infer intimacy. Both candidate actions are shown; prior/posterior intimacy sliders.
- **Effort inference** (`food_inv_effort_intimacy_alt/`) — flips the inverse direction: observers see observed action (2 levels) × intimacy (4 levels) and infer the effort context. The vignette no longer reveals the effort paragraph; instead the two effort paragraphs become the slider endpoints, and participants give prior and posterior probability ratings for which effort situation is more likely.

### Generalization beyond food sharing

A non-food stimulus set parallels the canonical food set: 16 scenarios covering substance sharing (chapstick, hairbrush), shared physical space (blanket, bed), and informational/situational privacy (a breakup conversation, a phone passcode). The non-food experiments mirror the canonical food set one-to-one and live alongside them under `experiments/`: `nonfood_forw_intimacy_desire/`, `nonfood_inv_intimacy_desire_alt/`, `nonfood_inv_desire_intimacy_alt/`, `nonfood_inv_intimacy_desire_noalt/`, and `nonfood_inv_desire_intimacy_noalt/`. The modeling pipeline reuses the canonical scripts via a `--domain food|nonfood` flag.

Currently only `nonfood_forw_intimacy_desire` has data; the four non-food inverse experiments are pending. The full set of generalization experiments is TBD.

### Scenarios

The scenarios used in the experiments are in [`experiments/scenarios.csv`](experiments/scenarios.csv), generated from [`experiments/scenarios.py`](experiments/scenarios.py) (the source of truth — edit the Python file and regenerate with `uv run python experiments/scenarios.py`). Parallel sets are in [`experiments/scenarios_nonfood.csv`](experiments/scenarios_nonfood.csv) and [`experiments/scenarios_effort.csv`](experiments/scenarios_effort.csv), generated the same way. See the [experiments README](experiments/README.md) for the column schema.

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
uv run python model/lm/score_canonical_features.py                  # food access + effort
uv run python model/lm/score_canonical_features.py --domain nonfood # non-food access + effort
uv run python model/lm/score_canonical_v.py                         # food signed-valence V
uv run python model/lm/score_canonical_v.py --domain nonfood        # non-food V
uv run python model/lm/score_effort_features.py                     # effort-experiment access + effort + effort-marginal access
# no-alt observers also need LM-generated alternatives + their features and V:
uv run python model/lm/generate_alternatives_motivation.py
uv run python model/lm/score_alternative_features.py
uv run python model/lm/score_alternative_v.py
uv run python model/lm/generate_alternatives_relationship.py
uv run python model/lm/score_alternative_features.py --conditioning relationship
uv run python model/lm/score_alternative_v.py --conditioning relationship
```

Each fit/predict/CV script is named after the experiment it serves and lives in `model/forward/`, `model/inverse/`, or `model/cv/`. Run `fit_<slug>.py`, then `predict_<slug>.py`; for cross-validation run `cv_<slug>.py`.

```bash
# forward
uv run python model/forward/fit_food_forw_intimacy_desire.py
uv run python model/forward/predict_food_forw_intimacy_desire.py
# inverse
uv run python model/inverse/fit_food_inv_intimacy_desire_alt.py
uv run python model/inverse/predict_food_inv_intimacy_desire_alt.py
# cross-validation
uv run python model/cv/cv_food_forw_intimacy_desire.py
uv run python model/cv/cv_food_inv_intimacy_desire_alt.py
```

The alt-shown inverse experiments freeze actor weights from their corresponding forward fit and fit only `α_observer`; the no-alt experiments refit all actor weights jointly with `α_observer` because the padded observer's variable-length action space doesn't accept the alt-shown weights as a transplant. The two effort inverse observers freeze actor weights from `food_forw_intimacy_effort`. `food_inv_effort_intimacy_alt` flips the inference direction (infers effort from observed action, intimacy) and uses **effort-marginal access** because that observer doesn't see the effort paragraph.

All reported model-vs-human correlations in the analysis qmds are out-of-sample, pooled across 16 LOSO folds. Forward CV refits actor weights ($w_v, w_d, w_e, \gamma$) per fold; alt-shown inverse CV refits only $\alpha_\mathrm{obs}$; no-alt inverse CV refits all weights jointly per fold.

Render analysis qmds:

```bash
quarto render analysis/<qmd-name>.qmd
# e.g. analysis/food-forw-intimacy-desire-analysis.qmd
# see analysis/ for the full list of 12 qmds
```

Plots are saved in the `figures/` directory; rendered docs in `_output/analysis/`.
