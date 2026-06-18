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

Parameters: `w_v` (reward weight), `w_d` (risk-discomfort weight), `w_e` (effort weight), `gamma` (intimacy power-law exponent, free, init 1.0, clipped ≥ 1e-6 by the optimizer's clip), plus `alpha` (actor softmax temperature, fixed to 1), `alpha_observer` (observer softmax temperature), and `sigma` (response-noise scale, see DV likelihoods). Each ablation uses only the subset of weights its utility requires; full and discomfort_only fit γ, base does not. The fit param vector is `[*utility weights, alpha_observer, sigma]`.

Intimacy magnitude `I` for the four relationship levels in the given-relationship studies (1a/1b) is **LM-elicited** (mirroring the per-condition desire scalar in 2a/2b) — a global 4-vector loaded by `load_lm_relationship_values` from `lm_given.json` and passed into the desire/joint_de observer memos as `relationship_values`, falling back to the placeholder `RELATIONSHIP_LEVEL_VALUES` until the elicitation has been run.

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

The alternative set is indexed by the **cell grid** = (scenario, observed_action, + the variables the observer sees). A feature gains an extra axis when the variable it depends on is *inferred* (the alt set is shared across that variable's hypotheses, but the feature value differs): effort gains an `effort_condition` axis when effort is inferred. `g` (goal-satisfaction) is desire-free, so it carries **no** desire axis and has the same shape as `risk` (risk is intimacy- and effort-independent so it's only indexed by the cell grid + slot). Every table carries a **leading run axis** `K` (one elicitation run per simulated-observer mixture component; `K=1` on the legacy single-run CSVs). With `S = MAX_ACTIONS`:

- **1a desire** — risk (K,16,3,2,4,S), effort (K,16,3,2,4,S), g (K,16,3,2,4,S), prior (K,16,3,2,4,S)
- **1b joint_de** — risk (K,16,3,4,S), effort (K,16,3,4,2,S), g (K,16,3,4,S), prior (K,16,3,4,S)
- **2a intimacy** — risk (K,16,3,2,2,S), effort (K,16,3,2,2,S), g (K,16,3,2,2,S), prior (K,16,3,2,2,S); + `desire_table` (16,2)
- **2b joint_ie** — risk (K,16,3,2,S), effort (K,16,3,2,2,S), g (K,16,3,2,S), prior (K,16,3,2,S); + `desire_table` (16,2)

The run axis is sliced per-run in the fit loop (`_build_observer_tables_runs`) and the observer memo is run once per run — the run axis is a likelihood-side construct, not a memo axis. `desire_table` / `relationship_values` are run-independent.

### DV likelihoods (belief-update Gaussian mixture)

The DV is the **belief update** `u = posterior rating − prior rating` (per participant per trial), scored against the model's belief update. Each elicitation run k gives a model update `δ_k = posterior mean − prior mean` for the inferred latent (the posterior mean is `post @ grid`; the prior mean is computed from the uniform prior, = 0.5). A participant's update is scored under the K-component mixture `(1/K) Σ_k N(u | δ_k, σ²)`:

- **desire** (1a) / **intimacy** (2a) → `mixture_nll_1d(u, deltas, sigma)`.
- **joint** (1b desire+effort, 2b intimacy+effort) → `mixture_nll_2d`, a bivariate Gaussian per component with a **single isotropic σ** (covariance σ²·I₂); the cross-dimension correlation comes from the spread of the runs' joint `δ_k`. Effort reduces to `P(effort=HIGH)`, with `δ_effort = P(HIGH) − 0.5`.

`σ` (response-noise scale) is fitted jointly with the weights + `alpha_observer`, per study. Each study fits its own actor utility weights — no transfer between studies. The fit/CV slice **slot 0** (observed action) across all K runs.

## Stimulus sets and LM table families

The studies use the **3-action** set (`scenarios.csv`). All LM `risk`/`effort`/`g` ratings are on a 0-6 scale normalized to `[0, 1]` (the absolute scale is absorbed by the freely-fitted weight, so all three share one range). Each study's LM tables live in **its own folder**, `outputs/lm/<slug>/`. The padded LM-alternatives table family is loaded by `load_padded_lm_tables_{desire,joint_de,intimacy,joint_ie}`, which prefer `lm_runs.jsonl` — one record per `(run_id, cell)` holding that run's scored actions (slot 0 = observed canonical action, slots 1..k = alternatives) — and fall back to the legacy single-run `lm_scenario.csv` + `lm_alternatives.csv` (as `K=1`) so fits run before the JSON regeneration. The given-desire studies (2a/2b) load the per-condition desire scalar (16, 2) via `load_lm_scenario_desire(slug)`, and the given-relationship studies (1a/1b) load the per-level intimacy 4-vector via `load_lm_relationship_values(slug)` — both from `lm_given.json` (CSV/placeholder fallback).

All LM table loaders return `None` when their source is missing, so imports stay clean before elicitation has been run.

The LM call infrastructure goes through `model/lm/client.py` (`get_ratings_concurrent` + schema helpers, JSON-schema-constrained output, transient-error retries); new LM call sites should reuse it rather than calling Together directly. The K-run pipeline scores each `(scenario, run)` **once** (`num_runs=1`, no inner averaging) — the K runs are the variation axis (both alternatives and feature scores vary run-to-run), which becomes the mixture's spread.

## Layout

`model/` is organized so that every script's name tells you what it does. Per-experiment scripts live in `inverse/`, with CV in `cv/`. LM-elicitation scripts live in `lm/`. Shared math is in the core modules at the top of `model/`.

### Core math (one copy, shared across all experiments)

- `tables.py` — enums (`Scenarios`, `DesireConditions`, `RelationshipConditions`, `EffortConditions`, `IntimacyLevels`, `DesireLevels`, `PaddedActionSlots`, `ObservedActions`), the `actions` array, `SCENARIO_LABELS`, the given-magnitude scalar loaders `load_lm_scenario_desire` (2a/2b) / `load_lm_relationship_values` (1a/1b) from `lm_given.json`, and the per-study padded loaders (`load_padded_lm_tables_{desire,joint_de,intimacy,joint_ie}`, each reading `lm_runs.jsonl` with a leading run axis, falling back to the legacy single-run CSVs at K=1).
- `utility.py` — jit-compiled utility functions. Active: the padded families `get_utility_{full,discomfort_only,base}_padded_{desire,joint_de,intimacy,joint_ie}` plus their `get_prior_padded_*` and `get_lm_g_padded_*` helpers (the full/base reward term is `w_v · desire · g`; the given-desire studies also take a `desire_table`).
- `actors.py` — actor memos. Active: the padded inverse actors `actor_discrete_*_padded_{desire,joint_de}` (discrete observed intimacy) and `actor_continuous_*_padded_{intimacy,joint_ie}` (continuous inferred intimacy), used inside the observers' `thinks[...]` blocks.
- `observers.py` — observer memos, one family per active study (`observer_desire_*`, `observer_joint_de_*`, `observer_intimacy_*`, `observer_joint_ie_*`), each in `_full` / `_discomfort_only` / `_base`.
- `test_model_compliance.py` — validation tests (desire-utility ablation algebra + the `observer_desire_full` posterior-normalization check).

### LM elicitation (`model/lm/`)

- `client.py` — shared LM-call infrastructure (`get_ratings_concurrent`, schema helpers, JSON parsing, `load_api_key`).
- `prompts.py` — prompt templates; `alternatives_user_prompt` composes only the observer-visible condition paragraphs per study.
- `generate_alternatives.py --study <slug>` — LM-generated counterfactual alternatives per cell, repeated for **K runs** (`K_RUNS` env, default 20; nonzero `ALT_T` so runs differ; deterministic per-(cell, run) seed). → `outputs/lm/<slug>/lm_alternatives.csv` (stage-1 texts, now carrying a `run_id` column). The `_STUDY_CONFIG` registry covers all four active studies; each iterates scenario × observed_action over only the observer-visible axes (cells: 1a 384, 1b 192, 2a 192, 2b 96 — ×K runs).
- `score_merged.py --study <slug>` — for each `(scenario, run)`, scores that run's unified [canonical + unique alts] list **once** (no inner averaging) on risk (effort-marginal), effort (per effort_condition), and desire-free goal-satisfaction g, so slot 0 and slots 1..k share one comparative frame. Writes `lm_runs.jsonl` (one record per `(run_id, cell)` with the run's scored actions) + `lm_given.json` (the study's given-magnitude scalars: per-(scenario, desire_condition) desire for 2a/2b, and/or the per-level intimacy 4-vector rated from the **de-anchored** relationship descriptors for 1a/1b — both run-independent, scored once).
- `_features_dispatcher.py` — internal helper for `score_merged.py` (prompt formatters, 0-6 → [0,1] normalizers, response parsers incl. desire/intimacy scalar parsers).

**Three design choices in merged scoring:** (1) canonical + alts scored together (shared comparative frame); (2) risk is effort-marginal — risk(a|s) is formally intimacy- and effort-independent (modulated by `(1-I)^γ` in the utility), so it's elicited without the effort paragraph and broadcast; (3) the reward term is `w_v · desire · g`, where g (goal-satisfaction) is LM-elicited desire-free per action and `desire` is the inferred latent (1a/1b) or an LM-rated per-condition scalar (2a/2b); `is_share` is preserved only as diagnostic metadata.

### Inverse planning (`model/inverse/`)

Four active experiments, each with its own `fit_<slug>.py` + `predict_<slug>.py` (thin wrappers that define the three variants and call the shared helpers).

- `_helpers.py` — the belief-update Gaussian-mixture losses (`mixture_nll_1d`, `mixture_nll_2d`) + `posterior_mean` / `PRIOR_MEAN` / `EFFORT_PRIOR_MEAN`; per-study data loaders (`load_{desire,joint_de,intimacy,joint_ie}_data`, returning per-trial belief updates); padded table-kwargs builders (`{desire,joint_de,intimacy,joint_ie}_table_kwargs`, which take the variant's `utility_param_names` and derive which optional tables to include); `_build_observer_tables_runs` (runs the observer per run, stacks on a leading K axis); and the joint-fit helpers (`fit_{...}_observer_joint`) that build the K-run observer tables, slice slot 0, compute per-run δ_k, and minimize the mixture NLL (params `[*weights, alpha_observer, sigma]`) with Adam.
- `fit_<slug>.py` — for each ablation, jointly fits the utility weights + `α_observer` + `σ` from this experiment's belief-update data. Writes `outputs/<slug>/fit_results.json` (+ `fit_restarts.jsonl`).
- `predict_<slug>.py` — reads its own `fit_results.json` via `load_fit_results`, runs the observer across runs, writes the per-cell model belief update `delta_<latent>` to `outputs/<slug>/preds_summary.json`.

### Cross-validation (`model/cv/`)

The PRIMARY model-comparison metric is **per-trial held-out log-likelihood** under leave-one-scenario-out (LOSO) CV (`outputs/<slug>/cv_trial_ll.jsonl`, keyed by `subject_id` for the participant bootstrap); the condition-averaged model-vs-human correlation (`cv_preds_summary.json`) is secondary/descriptive.

- `_inverse_dispatcher.py` — LOSO logic for the four inverse studies. Exports `main_{desire,joint_de,intimacy,joint_ie}`. Each loops over 16 scenarios, refits weights + `alpha_observer` + `σ` on the 15-scenario training set via the matching `fit_*_observer_joint` helper, slices slot 0 of the held-out scenario across runs, and scores each held-out trial's belief update under the mixture (`held_out_ll`); also emits the per-cell `delta_<latent>` predictions.
- `cv_<slug>.py` — one per experiment, a thin wrapper around the dispatcher main.

### Outputs (`model/outputs/`)

Per `outputs/<slug>/` (JSON / JSON Lines):
- `fit_results.json` — fitted parameters per ablation (incl. `param_sigma`); `fit_restarts.jsonl` — per-restart diagnostics.
- `preds_summary.json` — per-cell model belief update `delta_<latent>` (+ `delta_effort` for joint studies).
- `cv_trial_ll.jsonl` — per-trial held-out log-likelihood keyed by `subject_id` (**primary** metric); `cv_preds_summary.json` — held-out per-cell `delta_*` (secondary correlation); `cv_folds.jsonl` — per-fold refit diagnostics.

LM-elicited tables live in per-study folders `outputs/lm/<slug>/` (`lm_runs.jsonl`, `lm_given.json`). Preregistration documents are in `preregs/` at the repo root.

### Terminology

`desire_condition` is the observed 2-level desire **condition** for the given-desire studies (2a/2b), indexing `desire_table`; in 1a/1b desire is the inferred continuous latent (`DesireLevels`). The fitted reward-term weight is `w_v` / `param_w_v` (not `w_d`, the risk weight) — keep it named `w_v`. The per-action discomfort feature is **risk** (weight `w_d`).

## Commands

LM tables (require `TOGETHER_API_KEY` in `.env`; Llama-3.3-70B via Together AI; `K_RUNS` elicitation runs per cell, each scored once). Active 3-action pipeline:

```bash
# per-study LM-generated alternatives + per-run scoring (one of the 4 slugs):
uv run python model/lm/generate_alternatives.py --study food_inv_desire
uv run python model/lm/score_merged.py          --study food_inv_desire
# or all four at once (sequential), or in parallel processes:
make lm-alternatives                               # all 4, sequential, K=20
make lm-alternatives K_RUNS=1                      # cheap K=1 smoke test first
make -j4 lm-alternatives SCENARIO_WORKERS=1        # 4 studies in parallel
```

`K_RUNS` (default 20) sets the elicitation runs per cell (the mixture components); `ALT_T` (default 0.7) the generation temperature. `score_merged` scores `--scenario-workers` `(scenario, run)` units concurrently; tune to the Together tier's RPM limit, lowering it when also parallelizing studies with `-j`. After regenerating, the table loaders read `lm_runs.jsonl` automatically (no fit-code change).


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
