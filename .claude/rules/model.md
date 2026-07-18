---
paths:
  - "model/**/*"
---

# Model structure

Models are built using the `memo` DSL with a JAX backend. The roster is four inverse-planning **observer** model families (an actor reasoning inside an observer's `thinks[...]` block) serving six studies — the nonfood studies 3a/3b reuse the joint_de/joint_ie observers on their own stimulus set. They use the risk utility below with a uniform action prior.

## Utility

The canonical (and public-facing) statement of the utility model — the equation, the `w_v · d · g` reward term, the `risk` / `(1 − I)^γ` discomfort term, and the three ablations (`full` / `discomfort_only` / `base`) — is in [README.md](../../README.md#utility-model); the in-code form is the `utility.py` docstring. Implementation details not in README:

- **γ (intimacy exponent)** is a free parameter, initialized at 1.0 (γ = 1 is the linear special case), clipped ≥ 1e-6 by the optimizer. Empirically food prefers γ < 1 (late relaxation) and non-food prefers γ > 1 (early relaxation). `full` and `discomfort_only` fit γ; `base` has no intimacy term, so γ does not apply.
- **`base` uses a relationship-free alternative set.** Because the base utility ignores intimacy, its LM-generated alternatives are elicited **without** the relationship description (so its choice set — and predictions — are relationship-invariant), unlike `full`/`discomfort_only`, which keep the relationship-conditioned set. This applies to the given-relationship studies (1a/1b/3a) only; 2a/2b/3b infer intimacy and never show a relationship paragraph. The base set lives in `lm_runs_base.jsonl` and is broadcast across the relationship axis by the loader; the fit/CV route only the base variant to it. See the `--base` mode in the LM elicitation section below.
- **Fit param vector** is `[*utility weights, alpha_observer, sigma]`: `alpha` (actor softmax temperature) is fixed to 1, `alpha_observer` is the observer softmax temperature, and `sigma` is the response-noise scale (see DV likelihoods). Each ablation fits only the weights its utility requires.
- **Desire `d`** is the inferred continuous latent in 1a/1b/3a (over the 101-bin `DesireLevels` grid) and observer-visible given context in 2a/2b/3b (an LM-rated scalar per (scenario, desire_condition); see `load_lm_scenario_desire`).
- **Intimacy magnitude `I`** for the four relationship levels in the given-relationship studies (1a/1b/3a) is **LM-elicited per run** (mirroring the per-condition desire scalar in 2a/2b) — a `(K, 4)` array (one value per level per elicitation run) loaded by `load_lm_relationship_values` from the per-record `intimacy` field of `lm_runs.jsonl`, sliced per run into the desire/joint_de observer memos as `relationship_values`, falling back to the placeholder `RELATIONSHIP_LEVEL_VALUES` (as K=1) until the elicitation has been run.

## Active roster (six inverse studies, padded LM-alternatives pipeline)

All active studies are on the **3-action** stimulus structure (`scenarios.csv` for the food studies; `scenarios_nonfood.csv` for 3a/3b) and use the LM-generated-alternatives padded-action pipeline: the observer's actor softmaxes over `{observed action} ∪ LM-generated alternatives`, padded to `MAX_ACTIONS = 12` with the participant-observed action in slot 0. Each observer comes in `_full` / `_discomfort_only` / `_base`. The fit and CV slice **slot 0** (the observed action) of the observer table.

| Slug | Study | Observer family | Actor | Infers | Observer-table dims |
|---|---|---|---|---|---|
| `food_inv_desire`   | 1a | `observer_desire_*`    | `actor_discrete_*_padded_desire`     | desire (101-bin)       | (padded_slot, scenario, observed_action, effort, relationship, desire) |
| `food_inv_joint_de` | 1b | `observer_joint_de_*`  | `actor_discrete_*_padded_joint_de`   | desire + effort        | (padded_slot, scenario, observed_action, relationship, desire, effort) |
| `food_inv_intimacy` | 2a | `observer_intimacy_*`  | `actor_continuous_*_padded_intimacy` | intimacy (101-bin)     | (padded_slot, scenario, observed_action, desire_condition, effort, relationship) |
| `food_inv_joint_ie` | 2b | `observer_joint_ie_*`  | `actor_continuous_*_padded_joint_ie` | intimacy + effort      | (padded_slot, scenario, observed_action, desire_condition, relationship, effort) |
| `nonfood_inv_joint_de` | 3a | `observer_joint_de_*` (shared with 1b) | 1b's | desire + effort | as 1b, nonfood scenario axis |
| `nonfood_inv_joint_ie` | 3b | `observer_joint_ie_*` (shared with 2b) | 2b's | intimacy + effort | as 2b, nonfood scenario axis |

The nonfood studies reuse the joint observers, actors, and memo enums unchanged: both stimulus sets have 16 scenarios and the memo `Scenarios` axis is positional, so the only per-domain differences the model code sees are which labels map to which indices (`STUDY_SCENARIO_LABELS` / `scenario_to_idx_for_study` in `tables.py`) and which `outputs/lm/<slug>/` folder the tables come from. The `*_table_kwargs` builders route by `domain="food"|"nonfood"`; the data loaders and the CV dispatcher route by slug (`main_joint_de("nonfood_inv_joint_de")`).

Desire is inferred as a continuous latent in 1a/1b (over the 101-bin `DesireLevels` grid, just like intimacy in 2a/2b); in 2a/2b it is given context, so the actor reads its magnitude from `desire_table[scenario, desire_condition]` and `desire_condition` stays an observed 2-level axis. Studies 1a/1b take intimacy as **observed** (discrete `RelationshipConditions`, 4 levels); Studies 2a/2b **infer** intimacy as a continuous latent (`IntimacyLevels`, 101 bins). The joint observers use memo's `chooses(x in X, y in Y, ...)` multi-choice syntax and return `Pr[a, b]`; downstream code marginalizes for the per-slider predictions.

### Per-study padded table shapes

The alternative set is indexed by the **cell grid** = (scenario, observed_action, + the variables the observer sees). A feature gains an extra axis when the variable it depends on is *inferred* (the alt set is shared across that variable's hypotheses, but the feature value differs): effort gains an `effort_condition` axis when effort is inferred. `g` (goal-satisfaction) is desire-free, so it carries **no** desire axis and has the same shape as `risk` (risk is intimacy- and effort-independent so it's only indexed by the cell grid + slot). Every table carries a **leading run axis** `K` (one elicitation run per simulated-observer mixture component; `K=1` for a single-run elicitation, e.g. a `K_RUNS=1` smoke). With `S = MAX_ACTIONS`:

- **1a desire** — risk (K,16,3,2,4,S), effort (K,16,3,2,4,S), g (K,16,3,2,4,S), prior (K,16,3,2,4,S)
- **1b joint_de** — risk (K,16,3,4,S), effort (K,16,3,4,2,S), g (K,16,3,4,S), prior (K,16,3,4,S)
- **2a intimacy** — risk (K,16,3,2,2,S), effort (K,16,3,2,2,S), g (K,16,3,2,2,S), prior (K,16,3,2,2,S); + `desire_table` (K,16,2)
- **2b joint_ie** — risk (K,16,3,2,S), effort (K,16,3,2,2,S), g (K,16,3,2,S), prior (K,16,3,2,S); + `desire_table` (K,16,2)
- **3a / 3b** — identical shapes to 1b / 2b respectively, with the scenario axis indexed by `NONFOOD_SCENARIO_LABELS`.

The run axis is sliced per-run in the fit loop (`_build_observer_tables_runs`) and the observer memo is run once per run — the run axis is a likelihood-side construct, not a memo axis. The given-magnitude tables (`desire_table`, `relationship_values`) are scored per run too, so they carry the same leading run axis and are sliced per run alongside the features (the observer memo still sees one run's slice: a `(16, 2)` desire table / `(4,)` intimacy vector).

### DV likelihoods (belief-update Gaussian mixture)

The DV is the **belief update** `u = posterior rating − prior rating` (per participant per trial), scored against the model's belief update. Each elicitation run k gives a model update `δ_k = posterior mean − prior mean` for the inferred latent (the posterior mean is `post @ grid`; the prior mean is computed from the uniform prior, = 0.5). A participant's update is scored under the K-component mixture `(1/K) Σ_k N(u | δ_k, σ²)`:

- **desire** (1a) / **intimacy** (2a) → `mixture_nll_1d(u, deltas, sigma)`.
- **joint** (1b desire+effort, 2b intimacy+effort) → `mixture_nll_2d`, a bivariate Gaussian per component with a **single isotropic σ** (covariance σ²·I₂); the cross-dimension correlation comes from the spread of the runs' joint `δ_k`. Effort reduces to `P(effort=HIGH)`, with `δ_effort = P(HIGH) − 0.5`.

`σ` (response-noise scale) is fitted jointly with the weights + `alpha_observer`, per study. Each study fits its own actor utility weights — no transfer between studies. The fit/CV slice **slot 0** (observed action) across all K runs.

## Stimulus sets and LM table families

The food studies use `scenarios.csv` and the nonfood studies `scenarios_nonfood.csv` (both 3-action, 16 scenarios). All LM `risk`/`effort`/`g` ratings are on a 0-6 scale normalized to `[0, 1]` (the absolute scale is absorbed by the freely-fitted weight, so all three share one range). Each study's LM tables live in **its own folder**, `outputs/lm/<slug>/`. The padded LM-alternatives table family is loaded by `load_padded_lm_tables_{desire,joint_de,intimacy,joint_ie}` — the joint loaders take a `slug=` kwarg that routes 3a/3b to their own folder and scenario-label order — which prefer `lm_runs.jsonl` — one record per `(run_id, cell)` holding that run's scored actions (slot 0 = observed action, slots 1..k = alternatives) — and retain a legacy single-run `lm_scenario.csv` + `lm_alternatives.csv` fallback *code path* (as `K=1`) — but those CSVs were removed once the pipeline was validated, so a study now needs its own `lm_runs.jsonl` to load tables (otherwise the loaders return `None` and the fit raises a clear `FileNotFoundError`). The given-desire studies (2a/2b/3b) load the per-run, per-condition desire scalar (K, 16, 2) via `load_lm_scenario_desire(slug)`, and the given-relationship studies (1a/1b/3a) load the per-run, per-level intimacy vector (K, 4) via `load_lm_relationship_values(slug)` — both reading the per-record given field of `lm_runs.jsonl` (the desire CSV fallback was removed; `load_lm_relationship_values` keeps the in-code `RELATIONSHIP_LEVEL_VALUES` placeholder as a K=1 fallback).

All LM table loaders return `None` when their source is missing, so imports stay clean before elicitation has been run. Loaded tables are validated fail-fast: a NaN feature at a valid slot (a failed/null LM rating for an observed action) or a missing given-magnitude scalar raises `ValueError` at load time — NaN would otherwise silently poison every fit gradient, and a missing scalar would silently become desire = 0 / intimacy = 0.

**Base relationship-free alternatives (1a/1b/3a).** The `base` ablation reads a separate alternative set elicited without the relationship paragraph: `load_padded_lm_tables_desire(runs_filename="lm_runs_base.jsonl", broadcast_relationship=True)` reads `lm_runs_base.jsonl` (keyed on effort only, no `intimacy_condition`) and broadcasts the single alt set identically across the 4-level relationship axis, so the base table is relationship-invariant. The `*_table_kwargs` builders take a `base=` flag (`desire_table_kwargs(..., base=(variant=="base"))`, likewise `joint_de_table_kwargs`) that routes only the base variant to this path; `full`/`discomfort_only` keep the default `lm_runs.jsonl` loader. Wired for all three given-relationship studies (1a, 1b, 3a); each study's base elicitation (`make lm-base`) must be run before its fits.

The LM call infrastructure goes through `model/lm/client.py` (`get_ratings_concurrent` + schema helpers, JSON-schema-constrained output, transient-error retries); new LM call sites should reuse it rather than calling Together directly. The K-run pipeline scores each `(scenario, run)` **once** (`num_runs=1`, no inner averaging) — the K runs are the variation axis (both alternatives and feature scores vary run-to-run), which becomes the mixture's spread.

## Layout

`model/` is organized so that every script's name tells you what it does. Per-experiment scripts live in `inverse/`, with CV in `cv/`. LM-elicitation scripts live in `lm/`. Shared math is in the core modules at the top of `model/`.

### Core math (one copy, shared across all experiments)

- `tables.py` — enums (`Scenarios`, `DesireConditions`, `RelationshipConditions`, `EffortConditions`, `IntimacyLevels`, `DesireLevels`, `PaddedActionSlots`, `ObservedActions`), the `actions` array, `SCENARIO_LABELS`, the per-run given-magnitude scalar loaders `load_lm_scenario_desire` (2a/2b) / `load_lm_relationship_values` (1a/1b) reading the per-record given field of `lm_runs.jsonl`, and the per-study padded loaders (`load_padded_lm_tables_{desire,joint_de,intimacy,joint_ie}`, each reading `lm_runs.jsonl` with a leading run axis; a legacy single-run CSV fallback code path remains but those CSVs were removed).
- `utility.py` — jit-compiled utility functions. Active: the padded families `get_utility_{full,discomfort_only,base}_padded_{desire,joint_de,intimacy,joint_ie}` plus their `get_prior_padded_*` and `get_lm_g_padded_*` helpers (the full/base reward term is `w_v · desire · g`; the given-desire studies also take a `desire_table`).
- `actors.py` — actor memos. Active: the padded inverse actors `actor_discrete_*_padded_{desire,joint_de}` (discrete observed intimacy) and `actor_continuous_*_padded_{intimacy,joint_ie}` (continuous inferred intimacy), used inside the observers' `thinks[...]` blocks.
- `observers.py` — observer models, one family per active study (`observer_desire_*`, `observer_joint_de_*`, `observer_intimacy_*`, `observer_joint_ie_*`), each in `_full` / `_discomfort_only` / `_base`. The single-latent observers are memo models. The **joint observers are plain-JAX Bayesian inversions of the actor memos** (`policy^α_obs` normalized over the two latent axes — with the actor's uniform latent prior these are exactly equivalent): the original memo formulation compiled the joint indicator expectation into a 202²-per-cell cross-product (~8 GB of XLA temps, ~14 s/step) versus ~1.5 GB and ~100 ms/step for the inversion. The memo originals are kept as `_*_memo_reference` — the authoritative semantics; **change model semantics in both** — and `test_model_compliance.py` enforces fast ≡ reference (values + gradients, all six variants).
- `test_model_compliance.py` — validation tests: ablation algebra, single and joint observer posterior normalization, the mixture NLLs against a numpy reference, a null-padding probability-mass bound at fitted-scale weights, and the table loaders' fail-fast NaN/missing-scalar validation.

### LM elicitation (`model/lm/`)

- `client.py` — shared LM-call infrastructure (`get_ratings_concurrent`, schema helpers, JSON parsing, `load_api_key`).
- `prompts.py` — prompt templates; `alternatives_user_prompt` composes only the observer-visible condition paragraphs per study.
- `generate_alternatives.py --study <slug>` — LM-generated counterfactual alternatives per cell, repeated for **K runs** (`K_RUNS` env, default 20; nonzero `ALT_T` so runs differ; deterministic per-(cell, run) seed). → `outputs/lm/<slug>/lm_alternatives.jsonl` (stage-1 texts, one JSON record per generated alternative, carrying a `run_id` field). The `_STUDY_CONFIG` registry covers all six active studies, each reading its own scenarios CSV; each iterates scenario × observed_action over only the observer-visible axes (cells: 1a 384, 1b 192, 2a 192, 2b 96, 3a 192, 3b 96 — ×K runs). The `--base` flag (given-relationship studies 1a/1b/3a) overlays a `_BASE_OVERRIDE` that drops intimacy from `show`/`cell_cols` and writes `lm_alternatives_base.jsonl` (1a: 96 cells; 1b/3a: 48 — effort is inferred, so their base generation conditions on scenario × observed action only) — the relationship-free set for the base ablation; `make lm-base` runs the base generate + score.
- `score_merged.py --study <slug>` — for each `(scenario, run)`, scores that run's unified [observed + unique alts] list **once** (no inner averaging) on risk (effort-marginal), effort (per effort_condition), and desire-free goal-satisfaction g, so slot 0 and slots 1..k share one comparative frame. Writes `lm_runs.jsonl` (one record per `(run_id, cell)` with the run's scored actions and that run's given-magnitude scalar for the cell: `desire` for 2a/2b/3b, `intimacy` rated from the **de-anchored** relationship descriptors for 1a/1b/3a — scored per run alongside the features, not once). Relationship intimacy is scenario-independent, so it's scored once per run and reused across that run's scenarios. The `--base` flag scores the relationship-free set into `lm_runs_base.jsonl` with `relationship_given=False` (no per-run intimacy scalar — the base utility has no intimacy term); feature scoring is already relationship-free, so only the file paths and cell grid change. The g instruction names each scenario's `desire_object`; objects phrased as an infinitive outcome (some nonfood scenarios, e.g. "to try the harmonica") render as "actually getting …" instead of "actually getting or consuming …" (see `prompts.user_prompt`).
- `_features_dispatcher.py` — internal helper for `score_merged.py` (prompt formatters, 0-6 → [0,1] normalizers, response parsers incl. desire/intimacy scalar parsers).
- `export_prompts_latex.py` — paper-prep export: renders the `prompts.py` templates into `SIP_journal/si_prompts.tex` (titled monospace boxes for the Supplementary Material). The `.tex` is a generated artifact with an "AUTO-GENERATED — do not edit by hand" header — to read or change a prompt, go to `prompts.py` (not the `.tex`), then re-run this script.

**Three design choices in merged scoring:** (1) observed + alts scored together (shared comparative frame); (2) risk is effort-marginal — risk(a|s) is formally intimacy- and effort-independent (modulated by `(1-I)^γ` in the utility), so it's elicited without the effort paragraph and broadcast; (3) the reward term is `w_v · desire · g`, where g (goal-satisfaction) is LM-elicited desire-free per action and `desire` is the inferred latent (1a/1b) or an LM-rated per-condition scalar (2a/2b); `is_share` is preserved only as diagnostic metadata.

### Inverse planning (`model/inverse/`)

Six active experiments, each with its own `fit_<slug>.py` (a thin wrapper that defines the three variants and calls the shared helpers; the nonfood wrappers pass `domain="nonfood"` to their table-kwargs builder). There is no separate predict step — CV produces the model's predictions out-of-sample.

- `_helpers.py` — the belief-update Gaussian-mixture losses (`mixture_nll_1d`, `mixture_nll_2d`) + `posterior_mean` / `PRIOR_MEAN` / `EFFORT_PRIOR_MEAN`; per-study data loaders (`load_{desire,joint_de,intimacy,joint_ie}_data`, returning per-trial belief updates); padded table-kwargs builders (`{desire,joint_de,intimacy,joint_ie}_table_kwargs`, which take the variant's `utility_param_names` and derive which optional tables to include); `_build_observer_tables_runs` (runs the observer per run, stacks on a leading K axis); and the joint-fit helpers (`fit_{...}_observer_joint`) that build the K-run observer tables, slice slot 0, compute per-run δ_k, and minimize the mixture NLL (params `[*weights, alpha_observer, sigma]`) with Adam.
- `fit_<slug>.py` — for each ablation, jointly fits the utility weights + `α_observer` + `σ` from this experiment's belief-update data. Writes `outputs/<slug>/fit_results.json` (+ `fit_restarts.jsonl`).

### Cross-validation (`model/cv/`)

The PRIMARY model-comparison metric is **per-trial held-out log-likelihood** under leave-one-scenario-out (LOSO) CV (`outputs/<slug>/cv_trial_ll.jsonl`, keyed by `subject_id` for the participant bootstrap); the condition-averaged model-vs-human correlation (`cv_preds_summary.json`) is secondary/descriptive.

- `_inverse_dispatcher.py` — LOSO logic for the inverse studies. Exports `main_{desire,joint_de,intimacy,joint_ie}`; the joint mains take a slug (`main_joint_de("nonfood_inv_joint_de")` runs Study 3a). All four route through the generic `_run_loso(family, slug)` runner: for each of the 16 scenarios it refits weights + `alpha_observer` + `σ` on the 15-scenario training set via the matching `fit_*_observer_joint` helper, slices slot 0 of the held-out scenario across runs, and scores each held-out trial's belief update under the mixture (`held_out_ll`); also emits the per-cell `delta_<latent>` predictions. Each fold refit runs `CV_RESTARTS` restarts (default 2: the full-data warm start plus one cold restart, keeping the better NLL, so no fold depends on an init that saw the held-out scenario). With `CV_WORKERS` > 1 the 48 (variant × fold) jobs run as spawn worker processes with capped XLA/OpenMP threads — each job is deterministic given (variant, fold, warm start, patience), so the outputs are byte-identical to a sequential run. Worker and thread counts default per family from the `_FAMILIES` registry (the single source of truth); since the fast joint observers landed, every family defaults to 8 single-threaded workers (a joint worker is ~1.5 GB). Env `CV_WORKERS` / `CV_WORKER_THREADS` override (execution layout only — results are thread-count-invariant, verified byte-identical). Completed folds are appended to `outputs/<slug>/cv_checkpoint.jsonl` as they finish, so an interrupted run resumes from them (see `_checkpoint.py`). The per-family pieces (data-array loader, table-kwargs builder, fold body) are wired in the `_FAMILIES` registry at the bottom of the module.
- `_checkpoint.py` — the CV fold checkpoint: a fingerprint-guarded JSONL side file holding each completed (variant × fold) refit. The fingerprint hashes everything that determines fold results (data CSV, LM run tables, warm-start fit, patience, restarts, and the model-math source files listed in `_CODE_FILES`), so a resume can never splice folds from different vintages — any mismatch, including a mid-run model-code edit, discards the checkpoint. The final CV outputs are still written only when every fold is present, and the checkpoint is deleted once they land. Unit tests: `test_checkpoint.py` (in `make test`).
- `cv_<slug>.py` — one per experiment, a thin wrapper around the dispatcher main.
- `model_comparison.py` — the paper's numbers, from the CV outputs (`make model-comparison`): full − ablation per-trial held-out LL differences with participant-bootstrap 95% CIs (1,000 resamples), plus the secondary condition-averaged model-vs-human Pearson correlations with subject-cluster bootstrap CIs → `outputs/<slug>/cv_model_comparison.json`.

### Outputs (`model/outputs/`)

Per `outputs/<slug>/` (JSON / JSON Lines):
- `fit_results.json` — fitted parameters per ablation (incl. `param_sigma`); `fit_restarts.jsonl` — per-restart diagnostics.
- `cv_trial_ll.jsonl` — per-trial held-out log-likelihood keyed by `subject_id` (**primary** metric); `cv_preds_summary.json` — held-out per-cell `delta_*` (the model's predictions; secondary correlation); `cv_folds.jsonl` — per-fold refit diagnostics; `cv_model_comparison.json` — the bootstrap model-comparison statistics from `model_comparison.py`. There is no separate in-sample prediction file — CV is the sole prediction source. Fitted-parameter fields are named `param_<name>` (incl. `param_sigma`) with `alpha_observer` bare, consistently across `fit_results.json` and `cv_folds.jsonl`.

LM-elicited tables live in per-study folders `outputs/lm/<slug>/` (`lm_runs.jsonl`, `lm_alternatives.jsonl`). Preregistration documents are in `preregs/` at the repo root.

### Terminology

`desire_condition` is the observed 2-level desire **condition** for the given-desire studies (2a/2b), indexing `desire_table`; in 1a/1b desire is the inferred continuous latent (`DesireLevels`). The fitted reward-term weight is `w_v` / `param_w_v` (not `w_d`, the risk weight) — keep it named `w_v`. The per-action discomfort feature is **risk** (weight `w_d`).

## Commands

LM tables (require `TOGETHER_API_KEY` in `.env`; Llama-3.3-70B via Together AI; `K_RUNS` elicitation runs per cell, each scored once). Active 3-action pipeline:

```bash
# per-study LM-generated alternatives + per-run scoring (one of the 6 slugs):
uv run python model/lm/generate_alternatives.py --study food_inv_desire
uv run python model/lm/score_merged.py          --study food_inv_desire
# or per-domain aggregates (sequential), or in parallel processes:
make lm-alternatives                               # the 4 food studies, K=20
make lm-nonfood                                    # the 2 nonfood studies (3a + 3b)
make lm-alternatives K_RUNS=1                      # cheap K=1 smoke test first
make -j4 lm-alternatives SCENARIO_WORKERS=2 CELL_WORKERS=8   # 4 studies in parallel
```

`K_RUNS` (default 20) sets the elicitation runs per cell (the mixture components); `ALT_T` (default 0.7) the generation temperature. `CELL_WORKERS` (default 32) is generation's concurrent-call count; `score_merged` scores `--scenario-workers` (default 8) `(scenario, run)` units concurrently, each fanning out its 4 feature calls (+2 desire-scalar calls in the given-desire studies), so in-flight requests ≈ 4-6× that value. Together's serverless rate limits are dynamic per-org and the SDK retries 429s with backoff — lower the workers if a run prints repeated rate-limit errors, especially when also parallelizing studies with `-j`. After regenerating, the table loaders read `lm_runs.jsonl` automatically (no fit-code change).


Active inverse fits + CV (CV produces the out-of-sample predictions):

```bash
uv run python model/inverse/fit_food_inv_desire.py      # Study 1a (or any other slug's fit script)
# CV — parallel across the 48 (variant × fold) refits for all six studies,
# with results identical to a sequential run. Worker and thread counts default
# per observer family (currently 8 × 1-thread for all families). Just run the
# script; set CV_WORKERS / CV_WORKER_THREADS only to override:
uv run python model/cv/cv_food_inv_desire.py
uv run python model/cv/cv_food_inv_joint_de.py
uv run python model/cv/model_comparison.py               # bootstrap model comparison, all studies
```

A CV run can be interrupted freely: each completed fold refit is appended to `outputs/<slug>/cv_checkpoint.jsonl`, and rerunning the same study resumes from the completed folds. The checkpoint is fingerprint-guarded — any change to the data CSV, the LM tables, the warm-start fit, the refit config, or the model-math source files discards it automatically. Pass `PYTHONUNBUFFERED=1` for live per-fold progress. Interrupt with Ctrl-C (SIGINT reaches the whole process group); a `kill -9` of the parent instead leaves the spawn workers running as orphans that burn a core each until their current fold finishes — find them with `ps -eo pid,ppid,command | grep spawn_main` (orphans have ppid 1) and kill those too.


Tests:

```bash
uv run python model/test_model_compliance.py
```
