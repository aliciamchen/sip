# Inverse planning in the context of sociological structure

Cognitive science research on social inference and food sharing: how people decide to share saliva-transferring food with someone given the relationship between them, and how observers infer the relationship, desire, or physical effort from observed actions.

The current manuscript organizes the work into four inverse-planning studies, all built on a 3-action food stimulus set (no sharing / low-risk sharing / high-risk sharing) that lets the three latent variables — desire, effort, and intimacy — be crossed and inferred in different combinations. In each study the observer sees a single action and infers one or two of the latent variables: Study 1a infers desire, Study 1b jointly infers desire and effort, Study 2a infers intimacy, and Study 2b jointly infers intimacy and effort. A planned Study 3 will generalize these findings to non-food domains; its exact design is still being decided.

Earlier work is archived under `data/legacy/` and described in [data/legacy/README.md](data/legacy/README.md): three forward-planning experiments (the original Studies 1a/1b plus a non-food forward experiment), and six pre-3-action inverse experiments on the older 4-action and 2-action sets. Their model scripts and analysis qmds remain runnable via per-slug Makefile targets, but they are no longer part of the default `make all` pipeline.

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

### Active studies — inverse planning on the 3-action set

The active experiments use a 3-action stimulus set (no sharing / low-risk sharing / high-risk sharing) defined in `experiments/scenarios_3act.csv`, which merges effort paragraphs into the canonical scenarios so that all three latent variables — desire, effort, intimacy — can be manipulated alongside the observed action. On each trial the participant sees a vignette plus whichever condition paragraphs are revealed by the design, then a single observed action, then gives prior/posterior ratings on one or two sliders. The dependent-variable scales are: desire on a 1–7 Likert ("how much do they want the food?"), effort as a continuous 0–100 rating between two physical states, and intimacy on a 0–100 numeric scale (maximally formal → maximally intimate).

- **Study 1a — Desire inference** (`food_inv_desire/`) — known: effort + intimacy. Inferred: desire. Design: 2 × 4 × 3. The choice set the actor reasons over is not the fixed 3-action canonical set: for each (scenario, observed_action, effort, intimacy) cell the LM generates plausible counterfactual alternatives, and the observer's actor softmaxes over `{observed_action} ∪ generated_alts`, padded to 12 slots with the observed action in slot 0. See [LM-generated alternatives](#lm-generated-alternatives-and-merged-scoring) below.
- **Study 1b — Joint inference (desire + effort)** (`food_inv_joint_de/`) — known: intimacy. Inferred jointly: desire and effort. Design: 4 × 3. Two sliders per trial.
- **Study 2a — Inverse intimacy** (`food_inv_intimacy/`) — known: desire + effort. Inferred: intimacy. Design: 2 × 2 × 3.
- **Study 2b — Joint inference (intimacy + effort)** (`food_inv_joint_ie/`) — known: desire. Inferred jointly: intimacy and effort. Design: 2 × 3. Two sliders per trial.

Study 1a uses the LM-generated-alternatives padded-action pipeline; the other three are being migrated to it (they currently use the fixed 3-action choice set). None of the experiments show the alternative actions to participants — the alternatives exist only inside the observer's model of how the actor chose.

A planned **Study 3** will generalize to non-food domains (substance/contact, shared space, privacy); its design is not yet finalized, so no Study 3 experiment is in the active roster.

### Legacy experiments

Archived under `data/legacy/` and described in [data/legacy/README.md](data/legacy/README.md); runnable via per-slug Makefile targets but not part of `make all`:

- **Forward-planning experiments** (real data): `food_forw_intimacy_desire` (4-action, desire × intimacy), `food_forw_intimacy_effort` (2-action effort), and `nonfood_forw_intimacy_desire` (non-food 4-action). These were the manuscript's earlier Studies 1a/1b plus a non-food forward experiment, archived in May 2026 when the manuscript reframed around four inverse studies.
- **Pre-3-action inverse experiments**: six older 4-action and 2-action inverse experiments, two of which (`food_inv_intimacy_desire_noalt`, `food_inv_desire_intimacy_noalt`) keep runnable model + analysis code.
- **Study 1a pilot** (`food_inv_desire_pilot`): the original 1a pilot, collected with a 0–100 "probability of two states" slider for desire rather than the 1–7 Likert the manuscript now specifies. Retained for reference; Study 1a will be re-collected on the corrected DV.

### Scenarios

Python scripts in `experiments/` are the source of truth for the stimuli; each writes a `.csv` artifact next to it. Edit the `.py` file and regenerate.

- [`experiments/scenarios_3act.py`](experiments/scenarios_3act.py) → `scenarios_3act.csv` (3-action, active Studies 1a/1b/2a/2b).
- [`experiments/scenarios.py`](experiments/scenarios.py) → `scenarios.csv` (4-action canonical, legacy forward + legacy inverse).
- [`experiments/scenarios_effort.py`](experiments/scenarios_effort.py) → `scenarios_effort.csv` (2-action effort, legacy forward).
- [`experiments/scenarios_nonfood.py`](experiments/scenarios_nonfood.py) → `scenarios_nonfood.csv` (4-action non-food, legacy + planned Study 3).

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

Generating LM tables requires `TOGETHER_API_KEY` in `.env`. The fitting and prediction scripts index into these tables by `scenario_idx`; they require the relevant CSVs to exist.

### LM-generated alternatives and merged scoring

Study 1a (`food_inv_desire`) goes further than the fixed-action design: rather than scoring V/access/effort on the 3 canonical actions, the LM **generates** plausible counterfactual actions per (scenario, observed_action, effort, intimacy) cell — 16 × 3 × 2 × 4 = 384 cells — and then canonical actions + generated alternatives are scored together on V/access/effort. The observer's actor softmaxes over `{observed_action} ∪ generated_alts`, padded to a fixed slot count (`MAX_ACTIONS_3ACT = 12`) with the observed canonical action in slot 0 and null padding on unused slots (epsilon-weighted prior so the softmax stays differentiable). The alternatives are model-internal only — participants never see them; they exist solely inside the observer's model of how the actor chose.

The pipeline is two stages. First, alternatives are generated per cell. Second, a single merged scoring script rates the canonical actions and the unique alts together — putting slot 0 and slots 1..k on the same comparative scale by construction:

```bash
uv run python model/lm/generate_alternatives_3act.py --study food_inv_desire
uv run python model/lm/score_3act_merged.py          --study food_inv_desire
```

The alternative-generation prompt (`prompts.py:alternatives_user_prompt_3act`) mirrors what the human participant sees in the trial — vignette + effort paragraph + relationship descriptor + observed action — per the principle that the LM should be prompted with one condition at a time. The merged scoring script makes three design choices, each chosen to align with what the formal model treats each feature as varying with:

- **Canonical and alternative actions are scored together** in the same prompt per scenario, so slot 0 and slots 1..k are calibrated against a shared comparative reference frame rather than being scored in separate prompts with potentially mismatched anchors.
- **Access is effort-marginal**: the access scoring prompt omits the effort paragraph and a single access rating per action is broadcast across both effort conditions in the output CSVs. The model already modulates access by intimacy via `(1−I)^γ` in the utility, so access(a|s) is formally intimacy- and effort-independent; eliciting access without the effort paragraph avoids double-counting context that the utility formula handles separately. Effort is still elicited per (scenario, effort_condition), and V is elicited per (scenario, motivation_query).
- **V is LM-elicited**, not derived from the `is_share` flag. Each alternative gets a continuous LM-rated V ∈ [−1, +1] per motivation_query (`reward_low` and `reward_high`). The `is_share` field is preserved in the alternatives CSV as diagnostic metadata.

The merged scoring writes five CSVs at once (preserving the existing loader schemas): `lm_scenario_params_3act{,_marginal}.csv` (canonical access + effort), `lm_scenario_v_3act.csv` (canonical V), `lm_alternatives_features_food_inv_desire.csv` (alts access + effort), and `lm_alternatives_v_food_inv_desire.csv` (alts V). `tables.py:load_padded_lm_tables_3act_desire` assembles these into the 5-D access/effort/prior tables and 6-D V table the Study 1a memo observer indexes into.

Studies 1b, 2a, and 2b currently use the fixed-3-action pipeline (the `score_3act_features.py` and `score_3act_v.py` scripts writing to the canonical CSVs without alts), and are being migrated to the padded-alts pipeline. The required changes for each migrated study follow the Study 1a template: new padded actor and observer memos, new utility and prior helpers, a new padded table loader sized for that study's conditioning structure (the access/effort/V table shapes depend on which variables the observer sees), updated joint-fit helpers, updated fit/predict/CV scripts, and registry additions in `generate_alternatives_3act.py` and `score_3act_merged.py`. The merged scoring's per-scenario logic generalizes — only the `_STUDY_CONFIG` entries need to be added for new studies.

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
preregs/           AsPredicted-format preregistration documents (one per active experiment slug)
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
# active 3-action pipeline (Studies 1a/1b/2a/2b)
uv run python model/lm/score_3act_features.py                       # access + effort
uv run python model/lm/score_3act_v.py                              # signed-valence V
uv run python model/lm/generate_alternatives_3act.py --study food_inv_desire  # Study 1a: LM-generated alternatives per cell
uv run python model/lm/score_3act_merged.py          --study food_inv_desire  # Study 1a: merged canonical + alts scoring (access-marginal, effort-conditional, V LM-elicited)
# legacy tables (back the legacy forward experiments)
uv run python model/lm/score_canonical_features.py                  # 4-action canonical access + effort
uv run python model/lm/score_canonical_features.py --domain nonfood # non-food access + effort
uv run python model/lm/score_canonical_v.py                         # 4-action canonical V
uv run python model/lm/score_canonical_v.py --domain nonfood        # non-food V
uv run python model/lm/score_effort_features.py                     # 2-action effort set
```

Each fit/predict/CV script is named after the experiment it serves and lives in `model/inverse/` (active studies) or `model/forward/` (legacy forwards), with CV in `model/cv/`. Run `fit_<slug>.py`, then `predict_<slug>.py`; for cross-validation run `cv_<slug>.py`.

```bash
# active inverse (Studies 1a/1b/2a/2b)
uv run python model/inverse/fit_food_inv_desire.py
uv run python model/inverse/predict_food_inv_desire.py
uv run python model/cv/cv_food_inv_desire.py
# legacy forward (per-slug only)
uv run python model/forward/fit_food_forw_intimacy_desire.py
uv run python model/cv/cv_food_forw_intimacy_desire.py
```

Each inverse experiment jointly fits its own actor utility weights ($w_v, w_d, w_e, \gamma$) and $\alpha_\mathrm{observer}$ from its own posterior data, rather than transferring actor weights between studies. The joint Studies 1b/2b sum two per-slider losses (squared error on the 1–7 Likert desire/intimacy rating plus binary cross-entropy on the 0–100 effort slider) per trial.

All reported model-vs-human correlations in the analysis qmds are out-of-sample, pooled across 16 LOSO folds. CV refits the actor utility weights ($w_v, w_d, w_e, \gamma$) and $\alpha_\mathrm{obs}$ per fold.

Render analysis qmds:

```bash
quarto render analysis/<qmd-name>.qmd
# e.g. analysis/food-inv-desire-analysis.qmd
# see analysis/ for the full list of qmds
```

Plots are saved in the `figures/` directory; rendered docs in `_output/analysis/`.
