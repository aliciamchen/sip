---
paths:
  - "model/**/*"
---

# Model structure

Models are built using the `memo` DSL with a JAX backend. The roster is four inverse-planning **observer** models (an actor reasoning inside an observer's `thinks[...]` block). They use the risk utility below with a uniform action prior.

## Utility

```
P(a | s, I, d) ∝ exp( U(a|s, I, d) )

U(a|s, I, d) = w_v · d · g(a)
             − w_d · risk(a) · (1 − I)^γ
             − w_e · effort(a)
```

Intimacy `I` scales the risk-discomfort term (bodily/spatial/informational exposure) through a power-law modulator `(1 − I)^γ`: at high intimacy the penalty shrinks toward zero, so higher-risk actions become relatively more attractive. The exponent γ is a free parameter (initialized at 1.0; γ = 1 reproduces the linear-intimacy special case). Empirically food prefers γ < 1 (late relaxation) and non-food prefers γ > 1 (early relaxation).

The reward term is `w_v · d · g(a)`: `d` is **desire** — how much the dyad wants the outcome — on a [0, 1] scale (read out to the 0–100 human rating as `100·d`), and `g(a|s) ∈ [0, 1]` is the **goal-satisfaction** of the action (how fully it delivers the outcome; desire-free, LM-elicited). Desire *scales* this stable per-action value. `d` is the **inferred latent** in Studies 1a/1b (over the 101-bin `DesireLevels` grid) and observer-visible **given context** in 2a/2b (an LM-rated scalar per (scenario, desire_condition); see `load_lm_scenario_desire`). Three ablations are fit and compared:

- **full** — the full utility above (the main Full model)
- **discomfort_only** — only the risk-discomfort term (`−w_d · risk · (1 − I)^γ`); drops the reward term and effort (Discomfort-only)
- **base** — `w_v · d · g − w_e · effort`; no relational structure (Base model). Has no intimacy term, so γ does not apply.

Parameters: `w_v` (reward weight), `w_d` (risk-discomfort weight), `w_e` (effort weight), `gamma` (intimacy power-law exponent, free, init 1.0, clipped ≥ 1e-6 by the optimizer's clip), plus `alpha` (actor softmax temperature, fixed to 1) and `alpha_observer` (observer softmax temperature). Each ablation uses only the subset of weights its utility requires; full and discomfort_only fit γ, base does not.

## Active roster (four inverse studies, padded LM-alternatives pipeline)

All four active studies are on the **3-action** stimulus set (`scenarios.csv`) and use the LM-generated-alternatives padded-action pipeline: the observer's actor softmaxes over `{observed action} ∪ LM-generated alternatives`, padded to `MAX_ACTIONS = 12` with the participant-observed action in slot 0. Each observer comes in `_full` / `_discomfort_only` / `_base`. The fit and CV slice **slot 0** (the observed action) of the observer table.

| Slug | Study | Observer family | Actor | Infers | Observer-table dims |
|---|---|---|---|---|---|
| `food_inv_desire`   | 1a | `observer_desire_*`    | `actor_discrete_*_padded_desire`     | desire (101-bin)       | (padded_slot, scenario, observed_action, effort, relationship, desire) |
| `food_inv_joint_de` | 1b | `observer_joint_de_*`  | `actor_discrete_*_padded_joint_de`   | desire + effort        | (padded_slot, scenario, observed_action, relationship, desire, effort) |
| `food_inv_intimacy` | 2a | `observer_intimacy_*`  | `actor_continuous_*_padded_intimacy` | intimacy (101-bin)     | (padded_slot, scenario, observed_action, reward, effort, relationship) |
| `food_inv_joint_ie` | 2b | `observer_joint_ie_*`  | `actor_continuous_*_padded_joint_ie` | intimacy + effort      | (padded_slot, scenario, observed_action, reward, relationship, effort) |

Desire is inferred as a continuous latent in 1a/1b (over the 101-bin `DesireLevels` grid, just like intimacy in 2a/2b); in 2a/2b it is given context, so the actor reads its magnitude from `desire_table[scenario, desire_condition]` and `desire_condition` stays an observed 2-level axis. Studies 1a/1b take intimacy as **observed** (discrete `RelationshipConditions`, 4 levels); Studies 2a/2b **infer** intimacy as a continuous latent (`IntimacyLevels`, 101 bins). The joint observers use memo's `chooses(x in X, y in Y, ...)` multi-choice syntax and return `Pr[a, b]`; downstream code marginalizes for the per-slider predictions.

### Per-study padded table shapes

The alternative set is indexed by the **cell grid** = (scenario, observed_action, + the variables the observer sees). A feature gains an extra axis when the variable it depends on is *inferred* (the alt set is shared across that variable's hypotheses, but the feature value differs): effort gains an `effort_condition` axis when effort is inferred. `g` (goal-satisfaction) is desire-free, so it carries **no** desire axis and has the same shape as `risk` (risk is intimacy- and effort-independent so it's only indexed by the cell grid + slot). With `S = MAX_ACTIONS`:

- **1a desire** — risk (16,3,2,4,S), effort (16,3,2,4,S), g (16,3,2,4,S), prior (16,3,2,4,S)
- **1b joint_de** — risk (16,3,4,S), effort (16,3,4,2,S), g (16,3,4,S), prior (16,3,4,S)
- **2a intimacy** — risk (16,3,2,2,S), effort (16,3,2,2,S), g (16,3,2,2,S), prior (16,3,2,2,S); + `desire_table` (16,2)
- **2b joint_ie** — risk (16,3,2,S), effort (16,3,2,2,S), g (16,3,2,S), prior (16,3,2,S); + `desire_table` (16,2)

### DV likelihoods

- **desire** (1a, 1b) → continuous 0–100 rating. `compute_desire_nll`: NLL over the 101-bin `DesireLevels` posterior at the response bin (an exact parallel of `compute_intimacy_nll`).
- **effort** (1b, 2b) → 0–100 continuous rating. `compute_effort_nll`: binary cross-entropy on `P(effort=HIGH)`.
- **intimacy** (2a, 2b) → 0–100 numeric. `compute_intimacy_nll`: NLL over the 101-bin posterior at the response bin.

Joint studies sum the two appropriate per-slider losses. Each study fits its own actor utility weights + `alpha_observer` from its own data — no transfer between studies.

## Stimulus sets and LM table families

The studies use the **3-action** set (`scenarios.csv`). Fixed-action tables: `LLM_TABLES` (risk/effort, (16, 2, 3)) and `load_lm_g()` (goal-satisfaction g, (16, 3)). The given-desire studies (2a/2b) also load `load_lm_scenario_desire()` (per-condition desire scalar, (16, 2)). Each study also has a padded LM-alternatives table family loaded by `load_padded_lm_tables_{desire,joint_de,intimacy,joint_ie}` (built from per-study `lm_alternatives_{,features_,g_}<slug>.csv` + the shared canonical CSVs).

All LM table loaders return `None` when their CSV is missing, so imports stay clean before elicitation has been run.

The LM call infrastructure goes through `model/lm/client.py`, which fans NUM_RUNS calls across a thread pool, constrains output to a JSON schema via `response_format`, retries transient errors, and checkpoints per-scenario; new LM call sites should reuse `get_ratings_concurrent` + the schema helpers rather than calling Together directly. CSV outputs include both `n_runs_*` and `n_failures_*` columns.

## Layout

`model/` is organized so that every script's name tells you what it does. Per-experiment scripts live in `inverse/`, with CV in `cv/`. LM-elicitation scripts live in `lm/`. Shared math is in the core modules at the top of `model/`.

### Core math (one copy, shared across all experiments)

- `tables.py` — enums (`Scenarios`, `DesireConditions`, `RelationshipConditions`, `EffortConditions`, `IntimacyLevels`, `DesireLevels`, `PaddedActionSlots`, `ObservedActions`), the `actions` array, `SCENARIO_LABELS`, the fixed-action loaders (`LLM_TABLES`, `load_lm_g`, `load_lm_scenario_desire`), and the per-study padded loaders (`load_padded_lm_tables_{desire,joint_de,intimacy,joint_ie}`).
- `utility.py` — jit-compiled utility functions. Active: the padded families `get_utility_{full,discomfort_only,base}_padded_{desire,joint_de,intimacy,joint_ie}` plus their `get_prior_padded_*` and `get_lm_g_padded_*` helpers (the full/base reward term is `w_v · desire · g`; the given-desire studies also take a `desire_table`).
- `actors.py` — actor memos. Active: the padded inverse actors `actor_discrete_*_padded_{desire,joint_de}` (discrete observed intimacy) and `actor_continuous_*_padded_{intimacy,joint_ie}` (continuous inferred intimacy), used inside the observers' `thinks[...]` blocks.
- `observers.py` — observer memos, one family per active study (`observer_desire_*`, `observer_joint_de_*`, `observer_intimacy_*`, `observer_joint_ie_*`), each in `_full` / `_discomfort_only` / `_base`.
- `test_model_compliance.py` — validation tests (desire-utility ablation algebra + the `observer_desire_full` posterior-normalization check).

### LM elicitation (`model/lm/`)

- `client.py` — shared LM-call infrastructure (`get_ratings_concurrent`, schema helpers, JSON parsing, `load_api_key`).
- `prompts.py` — prompt templates; `alternatives_user_prompt` composes only the observer-visible condition paragraphs per study.
- `score_features.py` — fixed-action risk + effort (16 × 2 effort × 3 actions) plus a marginal risk table. → `outputs/lm/lm_scenario_params{,_marginal}.csv`.
- `generate_alternatives.py --study <slug>` — LM-generated counterfactual alternatives per cell. → `outputs/lm/lm_alternatives_<slug>.csv`. The `_STUDY_CONFIG` registry covers all four active studies; each iterates scenario × observed_action over only the observer-visible axes (cell counts: 1a 384, 1b 192, 2a 192, 2b 96).
- `score_merged.py --study <slug>` — scores the unified [canonical + unique alts] list per scenario on risk (effort-marginal), effort (per effort_condition), and goal-satisfaction g (one desire-free prompt), so slot 0 and slots 1..k share one comparative frame. For the given-desire studies (2a, 2b) it additionally rates the per-(scenario, desire_condition) desire scalar. Writes the shared canonical CSVs (`lm_scenario_g.csv`, and `lm_scenario_desire.csv` for 2a/2b) + per-study `lm_alternatives_features_<slug>.csv` and `lm_alternatives_g_<slug>.csv`. For studies whose observer **infers** effort (1b, 2b), each alt's effort feature is emitted for both effort conditions (effort is a feature axis, not a generation axis).
- `_features_dispatcher.py` — internal multi-mode helper for the canonical scorers.

**Three design choices in merged scoring:** (1) canonical + alts scored together (shared comparative frame); (2) risk is effort-marginal — risk(a|s) is formally intimacy- and effort-independent (modulated by `(1-I)^γ` in the utility), so it's elicited without the effort paragraph and broadcast; (3) the reward term is `w_v · desire · g`, where g (goal-satisfaction) is LM-elicited desire-free per action (one prompt, no desire axis) and `desire` is the inferred latent (1a/1b) or an LM-rated per-condition scalar (2a/2b); `is_share` is preserved only as diagnostic metadata. (If the journal manuscript at `SIP_journal/main.tex` still describes a signed-valence `V`, the code is ahead of it — the code uses `w_v · desire · g`; see `docs/continuous-desire-model.md`.)

### Inverse planning (`model/inverse/`)

Four active experiments, each with its own `fit_<slug>.py` + `predict_<slug>.py` (thin wrappers that define the three variants and call the shared helpers).

- `_helpers.py` — NLLs (`compute_intimacy_nll`, `compute_effort_nll`, `compute_desire_nll`); per-study data loaders (`load_{desire,joint_de,intimacy,joint_ie}_data`); padded table-kwargs builders (`desire_table_kwargs`, `joint_de_table_kwargs`, `intimacy_table_kwargs`, `joint_ie_table_kwargs`, all raising a clear `FileNotFoundError` until the study's LM CSVs exist); and the joint-fit helpers (`fit_{desire,joint_de,intimacy,joint_ie}_observer_joint`) that build the observer table from `{utility weights, α_observer}`, slice slot 0, and minimize the study's DV loss with Adam.
- `fit_<slug>.py` — for each ablation, jointly fits the actor utility weights + `α_observer` from this experiment's posterior data. Writes `outputs/<slug>/fit_results.csv`.
- `predict_<slug>.py` — reads its own `fit_results.csv` via `load_fit_results`, runs the observer, writes `outputs/<slug>/preds_<variant>.npy` + a summary CSV.

### Cross-validation (`model/cv/`)

All model-vs-human correlations in the analysis qmds are out-of-sample, from leave-one-scenario-out (LOSO) CV (`outputs/<slug>/cv_preds_summary.csv`).

- `_inverse_dispatcher.py` — LOSO logic for the four inverse studies. Exports `main_{desire,joint_de,intimacy,joint_ie}`. Each loops over 16 scenarios, refits utility weights + `alpha_observer` on the 15-scenario training set via the matching `fit_*_observer_joint` helper, slices slot 0 of the held-out scenario, and computes the per-trial test loss (the study's DV loss: desire NLL over the 101-bin posterior, BCE on the 0–100 effort slider, intimacy NLL over the 101-bin posterior; joint studies sum the two).
- `cv_<slug>.py` — one per experiment, a thin wrapper around the dispatcher main.

### Outputs (`model/outputs/`)

Per `outputs/<slug>/`:
- `fit_results.csv` — fitted parameters + AIC/BIC/r per ablation.
- `preds_<variant>.npy`, `preds_summary.csv` — per-variant prediction arrays and a summary CSV.
- `cv_folds.csv` — per-fold fit results from LOSO CV.
- `cv_preds_summary.csv` — held-out per-condition summary.

LM-elicited tables live in `outputs/lm/` (`lm_scenario_*`, `lm_alternatives_*`). Preregistration documents are in `preregs/` at the repo root. Sandboxed/experimental code is in `model/sandbox/`.

### Terminology

`desire_condition` is the observed 2-level desire **condition** for the given-desire studies (2a/2b), indexing `desire_table`; in 1a/1b desire is the inferred continuous latent (`DesireLevels`). The fitted reward-term weight is `w_v` / `param_w_v` (not `w_d`, the risk weight) — keep it named `w_v`. The per-action discomfort feature is **risk** (weight `w_d`).

## Commands

LM tables (require `TOGETHER_API_KEY` in `.env`; Llama-3.3-70B via Together AI, 10 runs averaged). Active 3-action pipeline:

```bash
uv run python model/lm/score_features.py    # → lm_scenario_params{,_marginal}.csv (fixed-action)
# per-study LM-generated alternatives + merged scoring (one of the 4 slugs):
uv run python model/lm/generate_alternatives.py --study food_inv_desire
uv run python model/lm/score_merged.py          --study food_inv_desire
```


Active inverse fits + predictions + CV:

```bash
uv run python model/inverse/fit_food_inv_desire.py      # Study 1a (or joint_de / intimacy / joint_ie)
uv run python model/inverse/predict_food_inv_desire.py
uv run python model/cv/cv_food_inv_desire.py
```


Tests:

```bash
uv run python model/test_model_compliance.py
```
