# Inverse planning in the context of sociological structure

Cognitive science research on social inference and food sharing: how people decide to share saliva-transferring food with someone given the relationship between them, and how observers infer the relationship, desire, or physical effort from observed actions.

The current manuscript organizes the work into four inverse-planning studies, all built on a 3-action food stimulus set (no sharing / low-risk sharing / high-risk sharing) that lets the three latent variables — desire, effort, and intimacy — be crossed and inferred in different combinations. In each study the observer sees a single action and infers one or two of the latent variables: Study 1a infers desire, Study 1b jointly infers desire and effort, Study 2a infers intimacy, and Study 2b jointly infers intimacy and effort. A planned Study 3 will generalize these findings to non-food domains; its exact design is still being decided.

## Quick start

```bash
# Python deps (uses uv: https://github.com/astral-sh/uv)
uv sync

# R deps (renv): open R from the project root, then
# renv::restore()

# Quarto (https://quarto.org/docs/get-started/) is needed to render the analysis docs

make all                # full pipeline: fit → CV → render qmds
make help               # list all targets
```

The processed CSVs are checked into the repo, so `make all` works without raw JSON. Rendered analysis documents land in `_output/analysis/`; figures land in `figures/`.

The `Makefile` exposes per-stage and per-experiment targets (`make fit`, `make cv`, `make analysis-food-inv-desire-analysis`, etc.); see `make help`. Underlying script invocations are documented in [Manual pipeline steps](#manual-pipeline-steps) below.

## Pipeline at a glance

```
jsPsych experiments (experiments/) → JSON → json_to_csv.py → processed CSVs (data/)
                                                              ↓
LM elicitation (model/lm/generate_alternatives.py → score_*.py) → scenario tables (model/outputs/lm/<slug>/)
                                                              ↓
                                  fit (inverse) → CV
                                                              ↓
                                  R/Quarto analysis (analysis/) → figures
```

## Experiments

The stable identifier for each experiment is its directory slug in `data/` and `experiments/` (paper-level numbers change as the writeup evolves).

### Active studies — inverse planning on the 3-action set

The active experiments use a 3-action stimulus set (no sharing / low-risk sharing / high-risk sharing) defined in `experiments/scenarios.csv`, which holds the effort paragraphs alongside the desire and intimacy framing so that all three latent variables — desire, effort, intimacy — can be manipulated alongside the observed action. On each trial the participant sees a vignette plus whichever condition paragraphs are revealed by the design, then a single observed action, then gives prior/posterior ratings on one or two sliders. The dependent-variable scales are: desire as a continuous 0–100 rating with a scenario-specific question naming the characters and the food ("how much do you think Carissa and Josh would like the hot dog?", not at all → moderately → extremely), effort as a continuous 0–100 rating between two physical states, and intimacy on a 0–100 numeric scale (maximally formal → maximally intimate).

- **Study 1a — Desire inference** (`food_inv_desire/`) — known: effort + intimacy. Inferred: desire. Design: 2 × 4 × 3. The choice set the actor reasons over is not the fixed 3-action set: for each (scenario, observed_action, effort, intimacy) cell the LM generates plausible counterfactual alternatives, and the observer's actor softmaxes over `{observed_action} ∪ generated_alts`, padded to 12 slots with the observed action in slot 0. See [LM-generated alternatives](#lm-generated-alternatives-and-merged-scoring) below.
- **Study 1b — Joint inference (desire + effort)** (`food_inv_joint_de/`) — known: intimacy. Inferred jointly: desire and effort. Design: 4 × 3. Two sliders per trial.
- **Study 2a — Inverse intimacy** (`food_inv_intimacy/`) — known: desire + effort. Inferred: intimacy. Design: 2 × 2 × 3.
- **Study 2b — Joint inference (intimacy + effort)** (`food_inv_joint_ie/`) — known: desire. Inferred jointly: intimacy and effort. Design: 2 × 3. Two sliders per trial.

All four studies use the LM-generated-alternatives padded-action pipeline. None of the experiments show the alternative actions to participants — the alternatives exist only inside the observer's model of how the actor chose.

A planned **Study 3** will generalize to non-food domains (substance/contact, shared space, privacy); its design is not yet finalized, so no Study 3 experiment is in the active roster.

To see what any of these trials looks like to a participant — for sharing scenarios and manipulations with collaborators — there is a standalone preview page at `experiments/preview/`. Run `make preview` and open `http://localhost:8000/preview/`, or use the deployed version. See the [experiments README](experiments/README.md#previewing-trials) for details.

### Legacy experiments

Data from earlier work — three forward-planning experiments, the pre-3-action inverse studies, and the original Study 1a pilot — is archived under `data/legacy/` and described in [data/legacy/README.md](data/legacy/README.md). Only the data is kept; the legacy model and analysis code is in git history.

### Scenarios

Python scripts in `experiments/` are the source of truth for the stimuli; each writes a `.csv` artifact next to it. Edit the `.py` file and regenerate.

- [`experiments/scenarios.py`](experiments/scenarios.py) → `scenarios.csv` (3-action, the active Studies 1a/1b/2a/2b).
- [`experiments/scenarios_nonfood.py`](experiments/scenarios_nonfood.py) → `scenarios_nonfood.csv` (the non-food set).

See the [experiments README](experiments/README.md) for the column schema.

The scenario tables reproduced in the paper's supplementary material are generated from these CSVs by [`experiments/export_scenarios_latex.py`](experiments/export_scenarios_latex.py), which renders the food and non-food sets as LaTeX tables. This is a reporting step that does not feed back into the modeling pipeline, so the source of truth for the stimuli remains the `.py` files.

## Utility model

A joint actor chooses an action by softmaxing over the utility:

```
P(a | s, I, d) ∝ exp( U(a | s, I, d) )

U(a | s, I, d) = w_v · d · g(a)
               − w_d · risk(a) · (1 − I)^γ
               − w_e · effort(a)
```

The reward term is `w_v · d · g(a)`. `d` is **desire** — how much the dyad wants the outcome — on a [0, 1] scale (the 0–100 human rating, divided by 100), and `g(a|s) ∈ [0, 1]` is the **goal-satisfaction** of the action: how fully it delivers the outcome (e.g. whether the two people end up getting and eating the food), independent of how much they want it. Desire scales a stable per-action value rather than selecting between two desire states. `risk(a)` is a graded measure of how much an action opens the actor up to the other person — their body, private information, and physical space. `effort(a)` is the physical effort of executing the action. Intimacy `I` scales the risk-discomfort term through a power-law modulator `(1 − I)^γ`, whose exponent γ is a free parameter (initialized at 1.0; γ = 1 is the linear special case): at high intimacy the `−w_d · risk · (1 − I)^γ` penalty shrinks toward zero, so higher-risk actions become relatively more attractive. At low intimacy the penalty is at full strength, so higher-risk actions are costly.

Desire is the **inferred latent** in the desire studies (recovered as a continuous posterior over a 101-bin grid, the same way intimacy is inferred) and **observer-visible context** in the studies where desire is given (where its magnitude is an LM-rated scalar per scenario and desire condition). Three ablations of this utility are fit and compared for the inverse-planning (observer) models:

- **Full** (`full`) — the full utility above: the reward term, the risk-discomfort term, and effort (the main model).
- **Discomfort-only** (`discomfort_only`) — only the risk-discomfort term `−w_d · risk · (1 − I)^γ`; drops the reward term and effort to ask whether the risk signal alone can account for behavior.
- **Base** (`base`) — `w_v · d · g − w_e · effort`; no relational structure.

### Where the utility values come from

The action-level utility components — goal-satisfaction, risk, effort — are **LLM-generated per scenario** because they genuinely vary by scenario (a wedding meal differs from a basketball hot dog in bodily exposure, logistical effort, and how fully each available action delivers the food). The LLM is Llama-3.3-70B via Together AI. The full two-step elicitation (generate alternatives, then score every action) is repeated for K independent **runs** per cell; each run is scored once (there is no inner rating-averaging), and the run-to-run spread of both the alternatives and their feature scores becomes the spread of the simulated-observer mixture the model is fit against.

- `g` (goal-satisfaction): how fully each action results in the two people getting/consuming the thing at stake, on a 0-6 scale normalized to [0, 1]. Desire-free — desire enters the reward term separately as the multiplier `w_v · d · g`. Produced by `model/lm/score_merged.py` into each study's folder, `model/outputs/lm/<slug>/lm_runs.jsonl` (one record per run × cell, holding every action's `risk`/`effort`/`g`).
- `risk`: how much each action opens each person up to the other — physically (bodily substance transfer, skin contact, spatial proximity), informationally, or both — on a 0-6 scale normalized to [0, 1], the same scale as effort and g. (The absolute scale of any feature is not separately identifiable — the freely-fitted weight absorbs a constant factor — so all three features share [0, 1].) Elicited by `model/lm/score_merged.py` into the same `lm_runs.jsonl` records.
- `effort`: physical, logistical, and time cost of executing the action, on a 0-6 scale normalized to [0, 1]. Elicited by the same script.

The desire magnitude `d` is not an action feature. In the desire-inference studies it is the latent the observer recovers; in the studies where desire is given context, the LM rates it per (scenario, desire condition) on the 0–100 scale. The given-relationship studies (1a/1b) likewise have the LM rate the intimacy implied by each of the four (de-anchored) relationship descriptions. Both are rated per elicitation run — so they vary run-to-run alongside the action features — and folded into each `lm_runs.jsonl` record (`desire` for the given-desire studies, `intimacy` for the given-relationship studies).

A single prompt set in `model/lm/prompts.py` is used for both the food and non-food pipelines. The risk rubric covers three channel types — bodily-substance transfer, direct physical contact, and informational/private-resource disclosure — so the same prompt works for food sharing, shared objects, shared physical space, and privacy or information-disclosure scenarios. For the paper, [`model/lm/export_prompts_latex.py`](model/lm/export_prompts_latex.py) renders these templates verbatim into the prompt boxes reproduced in the supplementary material, so the prompts shown there are generated from `prompts.py` rather than transcribed by hand.

The Together AI calls go through `model/lm/client.py`, which constrains the output to a JSON schema via Together's `response_format`, retries transient errors, and checkpoints to disk as `(scenario, run)` units complete. The number of runs (`K_RUNS`, default 20) and the generation temperature (`ALT_T`) are environment-overridable; a cheap `make lm-alternatives K_RUNS=1` smoke test is worth running before the full paid K=20 regeneration.

Generating LM tables requires `TOGETHER_API_KEY` in `.env`. The fitting and CV scripts index into these tables by `scenario_idx`; they require a study's `lm_runs.jsonl` to exist first.

### LM-generated alternatives and merged scoring

Study 1a (`food_inv_desire`) goes further than the fixed-action design: rather than scoring goal-satisfaction/risk/effort on the 3 observed actions, the LM **generates** plausible counterfactual actions per (scenario, observed_action, effort, intimacy) cell — 16 × 3 × 2 × 4 = 384 cells — and then observed actions + generated alternatives are scored together on goal-satisfaction/risk/effort. The observer's actor softmaxes over `{observed_action} ∪ generated_alts`, padded to a fixed slot count (`MAX_ACTIONS = 12`) with the observed action in slot 0 and null padding on unused slots (epsilon-weighted prior so the softmax stays differentiable). The alternatives are model-internal only — participants never see them; they exist solely inside the observer's model of how the actor chose.

The pipeline is two stages. First, alternatives are generated per cell. Second, a single merged scoring script rates the observed actions and the unique alts together — putting slot 0 and slots 1..k on the same comparative scale by construction:

```bash
uv run python model/lm/generate_alternatives.py --study food_inv_desire
uv run python model/lm/score_merged.py          --study food_inv_desire
```

The alternative-generation prompt (`prompts.py:alternatives_user_prompt`) mirrors what the human participant sees in the trial — vignette + effort paragraph + relationship descriptor + observed action — per the principle that the LM should be prompted with one condition at a time. The merged scoring script makes three design choices, each chosen to align with what the formal model treats each feature as varying with:

- **Observed and alternative actions are scored together** in the same prompt per scenario, so slot 0 and slots 1..k are calibrated against a shared comparative reference frame rather than being scored in separate prompts with potentially mismatched anchors.
- **Risk is effort-marginal**: the risk scoring prompt omits the effort paragraph and a single risk rating per action is broadcast across both effort conditions in the output CSVs. The model already modulates risk by intimacy via `(1−I)^γ` in the utility, so risk(a|s) is formally intimacy- and effort-independent; eliciting risk without the effort paragraph avoids double-counting context that the utility formula handles separately. Effort is still elicited per (scenario, effort_condition).
- **Goal-satisfaction `g` is LM-elicited desire-free**, not derived from the `is_share` flag. Each action gets one continuous LM-rated `g ∈ [0, 1]` ("how fully does this deliver the outcome"), with no desire axis — desire enters the reward term separately as the multiplier `w_v · d · g`. For the studies where desire is given context, the LM additionally rates the desire magnitude `d` per (scenario, desire condition) in each run. The `is_share` field is preserved in the alternatives records as diagnostic metadata.

The merged scoring writes `lm_runs.jsonl` into the study's own folder, `model/outputs/lm/<slug>/` — one record per `(run_id, cell)`, holding that run's scored actions (slot 0 = the observed action, slots 1..k = the run's alternatives, each with its `risk`/`effort`/`g`/`is_share`) plus that run's given-magnitude scalar for the cell (the per-condition `desire` for the given-desire studies, the per-level `intimacy` for the given-relationship studies). Keeping each study's tables in its own folder means the observed actions, which are re-scored per run in the comparative frame of that run's own alternative set, are never overwritten by another study's run. `tables.py:load_padded_lm_tables_desire` assembles these into the padded risk/effort/g/prior tables — now carrying a leading run axis — that the Study 1a memo observer indexes into (per run), and `load_lm_scenario_desire` / `load_lm_relationship_values` read the per-run given magnitudes from the same records.

For the base ablation, which has no intimacy term in its utility, the alternatives are elicited **without** the relationship descriptor (via `generate_alternatives.py --base` / `make lm-base`), so the base model's choice set — and therefore its predictions — do not depend on the relationship. This relationship-free set is written to `lm_runs_base.jsonl` (96 cells for Study 1a, dropping the intimacy axis) and is broadcast across the four relationship levels by the loader; the fit and cross-validation route only the base variant to it, while the full and discomfort-only models keep the relationship-conditioned set above. This applies to the given-relationship studies (1a/1b) only, since Studies 2a/2b infer intimacy and never show a relationship descriptor during generation.

Studies 1b, 2a, and 2b use the same merged padded-alts pipeline as Study 1a; each follows the Study 1a template: padded actor and observer memos, utility and prior helpers, a padded table loader sized for that study's conditioning structure (the risk/effort/g table shapes depend on which variables the observer sees), joint-fit helpers, fit/CV scripts, and registry entries in `generate_alternatives.py` and `score_merged.py`. The merged scoring's per-scenario logic generalizes — only the `_STUDY_CONFIG` entries differ between studies.

## Repository structure

```
analysis/          R/Quarto analysis scripts and data processing
data/              Processed experiment data (one folder per experiment slug)
experiments/       jsPsych experiment code + scenario definitions
model/             Computational models
  inverse/         Per-experiment inverse-planning fit scripts
  cv/              Per-experiment LOSO cross-validation scripts (the model's predictions)
  lm/              LM-elicitation scripts (Together AI)
  outputs/         Fitted parameters and CV results (predictions, all out-of-sample)
    lm/            LM-elicited scenario tables
    <slug>/        Per-experiment outputs
preregs/           AsPredicted-format preregistration documents (Study 1a present; 1b/2a/2b pending)
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
# active 3-action pipeline (Studies 1a/1b/2a/2b), per study slug:
uv run python model/lm/generate_alternatives.py --study food_inv_desire  # Study 1a: LM-generated alternatives per cell
uv run python model/lm/score_merged.py          --study food_inv_desire  # Study 1a: merged observed + alts scoring (risk-marginal, effort-conditional, g LM-elicited)
```

Each fit/CV script is named after the experiment it serves: fits live in `model/inverse/`, CV in `model/cv/`. Run `fit_<slug>.py`, then `cv_<slug>.py` (CV produces the out-of-sample predictions — there is no separate in-sample predict step).

```bash
# active inverse (Studies 1a/1b/2a/2b)
uv run python model/inverse/fit_food_inv_desire.py
uv run python model/cv/cv_food_inv_desire.py
```

Each inverse experiment jointly fits its own actor utility weights ($w_v, w_d, w_e, \gamma$), $\alpha_\mathrm{observer}$, and a response-noise scale $\sigma$ from its own data, rather than transferring actor weights between studies. The dependent measure is the **belief update** (posterior − prior rating), scored against the model's belief update (posterior mean − prior mean) under the K-component simulated-observer Gaussian mixture $\frac{1}{K}\sum_k \mathcal{N}(u \mid \delta_k, \sigma^2)$. The joint Studies 1b/2b score the (desire, effort) / (intimacy, effort) updates jointly with a bivariate Gaussian per component (a single isotropic $\sigma$, with the cross-dimension correlation carried by the spread of the runs' joint $\delta_k$), the effort update being $P(\text{effort}=\text{high}) - 0.5$. The primary model-comparison metric is per-trial held-out log-likelihood.

All reported model-vs-human correlations in the analysis qmds are out-of-sample, pooled across 16 LOSO folds. CV refits the actor utility weights ($w_v, w_d, w_e, \gamma$) and $\alpha_\mathrm{obs}$ per fold.

Render analysis qmds:

```bash
quarto render analysis/<qmd-name>.qmd
# e.g. analysis/food-inv-desire-analysis.qmd
# see analysis/ for the full list of qmds
```

Plots are saved in the `figures/` directory; rendered docs in `_output/analysis/`.

### Manuscript figures

The analysis qmds save publication PDFs into `figures/` with the shared `save_figure()` helper in `analysis/utils.R`. It renders through the `cairo_pdf` device, which embeds the theme font and renders the `ggpattern` fills cleanly (the base `pdf()` device does neither reliably), and it takes its width and height from the chunk's `fig-width`/`fig-height`, so the saved file matches the displayed plot.

The journal manuscript is a separate Overleaf-synced git repository (`SIP_journal/`, which is not part of this repo), and Overleaf needs the figure files committed inside it rather than referenced from here — symlinks do not sync. The `make sync-journal-figures` target copies a curated set of these PDFs into `SIP_journal/figures/`, renaming each to the name the manuscript uses; the source-to-paper-name mapping is the `JOURNAL_FIGURES` list in the `Makefile`. After syncing, commit and push `SIP_journal/` to Overleaf.
