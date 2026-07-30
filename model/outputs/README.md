# Model outputs codebook

Outputs are grouped by experiment slug. Everything the pipeline writes is JSON or JSON
Lines. JSON Lines is used for the large, append/resume-friendly per-record logs (per-run LM
scores, stage-1 alternatives, per-restart and per-fold fit diagnostics, per-trial held-out
likelihoods); plain JSON is used for the smaller structured summaries.

```
outputs/
├── lm/                                   # LM-elicited tables, one folder per study slug
│   └── <slug>/
│       ├── lm_runs.jsonl                     # scored actions + per-run given magnitude, one record per (run, cell)  ← primary
│       ├── lm_alternatives.jsonl             # stage-1 generated alternative texts (one record per alt)
│       ├── lm_runs_base.jsonl                # base ablation's relationship-free analog of lm_runs.jsonl (given-relationship studies)
│       ├── lm_alternatives_base.jsonl        # base ablation's relationship-free analog of lm_alternatives.jsonl
│       ├── lm_alternatives*.rationale.jsonl  # raw response containing the rationale and alternatives array
│       ├── *.manifest.json                   # provenance sidecar per elicited JSONL (model, prompt hash, git SHA, timestamp)
│       ├── lm_embeddings.npz                 # embeddings of the alternatives (semantic diagnostics; where elicited)
│       ├── lm_alternatives_semantic.jsonl    # per-alternative cluster + nearest-observed-action labels
│       ├── lm_clusters.json                  # per-scenario action-type clusters with exemplar texts
│       ├── lm_alternatives_projection.jsonl  # per-scenario 2D projection + mean features, for the R notebook
│       └── figures/                          # quick-look diagnostic PNGs from plot_alternatives.py
└── <slug>/                               # one folder per inverse study (fits + CV)
    ├── fit_results.json                      # fitted params per ablation (incl. param_sigma)
    ├── fit_restarts.jsonl                    # per-restart fit diagnostics
    ├── fit_manifest.json                     # fit provenance: git SHA + sha256 of the fit outputs and input data
    ├── cv_trial_ll.jsonl                     # per-trial held-out log-likelihood, by subject_id  ← primary metric
    ├── cv_preds_summary.json                 # held-out per-cell delta_<latent> (the model's predictions)
    ├── cv_folds.jsonl                        # per-fold refit diagnostics
    ├── cv_manifest.json                      # CV provenance: git SHA + sha256 of the CV outputs and input data
    └── cv_model_comparison.json              # bootstrap model-comparison statistics (the paper's numbers)
```

The slugs are the six inverse studies' directory names; the canonical slug ↔ Study-number
roster is in the [root README](../../README.md#experiments). A study's `outputs/lm/<slug>/`
and `<slug>/` folders appear once its LM elicitation and fits have been run, and the `<slug>/`
folder is populated by running its fit → CV scripts (`make all`, or the per-study
`make fit-<slug>` / `cv-<slug>`). There is no separate in-sample prediction stage: CV is the
sole source of model predictions, because every reported model-vs-human number is
out-of-sample.

The elicited LM tables (`lm_runs.jsonl`, `lm_alternatives.jsonl`, and their `_base` variants
where the study has one), their provenance manifests, and the fit/CV outputs are committed, so
the fit → CV → analysis pipeline is reproducible from a fresh clone without a Together AI key.
They are regenerated when the pipeline changes: the LM tables by `generate_alternatives.py` +
`score_merged.py`, and the fit/CV outputs by the fit and CV scripts. A study whose
`lm_runs.jsonl` is missing has no LM tables — its loaders return `None` and its fit raises a
clear `FileNotFoundError`.

## What the numbers mean

The dependent measure is the **belief update** `u = posterior rating − prior rating` (per
participant per trial). Each LM elicitation run `k` yields a model belief update
`δ_k = posterior mean − prior mean` for the inferred latent, and a trial is scored under the
K-component Gaussian mixture `(1/K) Σ_k N(u | δ_k, σ²)` with a fitted response-noise `σ`. So
the predicted quantities below are all in belief-update space (the `delta_<latent>` fields),
not raw-posterior space, and every fit carries `param_sigma` alongside the utility weights and
`alpha_observer`. The utility model and the naming of the `param_*` fields (the `w_v · desire · g`
reward term, the `risk` feature with weight `w_d`) are defined in
[README.md](../../README.md#utility-model); the model implementation is described in
[`model/README.md`](../README.md).

## LM-elicited tables (`outputs/lm/<slug>/`)

Each study keeps its LM tables in its own folder. The observed actions are re-scored in the
comparative frame of that study's own alternative set, so each study's scores can differ —
keeping them per-folder means no study's elicitation overwrites another's. All `risk`,
`effort`, and `g` values are LM ratings on a 0–6 scale normalized to `[0, 1]`. The pipeline
runs `K` independent elicitation runs per cell (`K_RUNS`, default 20); the runs are the
elicitation-sample mixture's components, so the alternatives, their feature scores, **and** the
given-magnitude scalars all vary run to run.

### `lm_runs.jsonl` — scored actions per run (primary)

One record per `(run_id, cell)`, where a cell is `(scenario, observed_action, +
observer-visible condition levels)`. Each record holds that run's full action list — slot 0 is
the observed action, slots 1.. are that run's alternatives — scored together in one
comparative frame, plus that run's **given-magnitude scalar** for the cell's condition: `desire`
for the given-desire studies (2a/2b), `intimacy` for the given-relationship studies (1a/1b).
Written by `score_merged.py`; consumed by `tables.py` (the `load_padded_lm_tables_*` loaders
for the action features, `load_lm_scenario_desire` / `load_lm_relationship_values` for the
given magnitudes), all stacking the runs on a leading `K` axis.

```json
{
  "run_id": 0,
  "scenario_label": "apples",
  "observed_action": "no_share",
  "intimacy_condition": "somewhat_formal",
  "effort_condition": "low",
  "intimacy": 0.48,
  "actions": [
    {"slot": 0, "is_observed": true,  "action_text": "...", "is_share": 0,
     "risk": 0.0, "effort": 0.0, "g": 0.5},
    {"slot": 1, "alt_idx": 0, "is_observed": false, "action_text": "...", "is_share": 1,
     "risk": 0.17, "effort": 0.33, "g": 1.0}
  ]
}
```

The condition keys between `observed_action` and the given scalar follow the study's cell grid;
`effort_condition` is always present (the loaders key the observed slot on it). The given
scalar is `intimacy` here (1a/1b); for 2a/2b it is `desire` instead. It is denormalized — the
same value repeats across every record sharing a `(run, condition)` — because the given
magnitude is a property of the condition, not of the action list. The `actions` list is ragged:
a run that produced no alternatives for a cell still emits the record with just slot 0.
`risk`/`effort`/`g` are `null` when a rating failed. Resume keys on `(scenario_label, run_id)`.

### `lm_alternatives.jsonl` — stage-1 generated alternatives

The LM-generated counterfactual actions, written by `generate_alternatives.py --study <slug>`
and read back by `score_merged.py`. One record per generated alternative, with fields
`scenario_label`, `observed_action`, the study's generation-cell condition columns, `run_id`,
`alt_idx`, `action_text`, and `is_share`. The feature scores (`risk`/`effort`/`g`) are *not*
here — scoring happens in `score_merged.py` and lands in `lm_runs.jsonl`. This is the stage-1
input to scoring.

### `lm_alternatives_base.jsonl` / `lm_runs_base.jsonl` — the base ablation's tables

The same two-stage pipeline run with `--base` (`make lm-base`), for the given-relationship
studies only (1a/1b/3a). Because the base ablation has no intimacy term, its choice set must
not depend on the relationship either, so these alternatives are elicited **without** the
relationship description and scored into `lm_runs_base.jsonl` with no per-run intimacy scalar.
The record shapes match the main files' (the cell grid just drops the relationship axis); the
base fit and CV load them via `desire_table_kwargs(base=True)`, which broadcasts the
relationship-free set across the relationship conditions.

### `*.manifest.json` — provenance sidecars

Every elicited JSONL gets a small plain-JSON sidecar next to it (`lm_runs.jsonl` →
`lm_runs.manifest.json`), written by `client.write_run_manifest`. The generation and scoring
stages write `status: "in_progress"` before their first paid call and replace it with
`status: "complete"` only after the full intended grid is valid. This makes an interrupted
JSONL checkpoint distinguishable from a completed elicitation. Because the values in these
files are LM-generated, two regenerations must be distinguishable, so each new manifest
records how its file was produced: `stage`
(`generate_alternatives`, `score_merged`, or `priors`), `study`, `model`,
`prompt_sha256` (a short hash of the rendered prompt surfaces that determine that stage's
output, including generation prompts upstream of `score_merged`), `prompts_sha256` (the
legacy whole-file source hash retained for traceability), `rendered_prompt_sha256` (the exact
messages assembled by the production caller for that study and input data), `git_sha`,
`created_utc`, and stage-specific config (`k_runs`, the generation or scoring temperature,
and record counts). A scoring manifest also records hashes of the exact alternatives JSONL,
its generation manifest, and its generation-prompt fingerprint. These fields prevent a
partial scoring resume from using replaced or stale alternatives.

Legacy manifests contain only `prompts_sha256`. The prompt resume guard uses the
stage-specific hash when present and falls back to the legacy whole-file hash, so a generation
run is invalidated by a generation-prompt edit but not by an unrelated rating-prompt or
comment edit. New manifests also guard the exact rendered-message fingerprint. A legacy
generation manifest may resume through the fallback when its whole-source hash matches, but
a partial scoring file whose manifest predates alternatives-input fingerprinting must be
restarted because its upstream input cannot be verified. Set
`LM_RESUME_PROMPT_MISMATCH=allow` only for a deliberate mixed-prompt resume; superseded hashes
are then preserved in the manifest's history fields.

### Semantic diagnostics — `lm_embeddings.npz`, `lm_alternatives_semantic.jsonl`, `lm_clusters.json`, `lm_alternatives_projection.jsonl`, `figures/`

An optional embedding-based view of the generated alternatives (the inverse fit never reads
these), present for the studies where it has been run — currently Study 1a has the full set.
`embed_alternatives.py --study <slug>` embeds each unique alternative text via the Together AI
embeddings API and writes three artifacts: `lm_embeddings.npz` (the mean-centered, normalized
embeddings — `alt_emb`, aligned row-for-row with `lm_alternatives_semantic.jsonl`, plus
`obs_emb` with parallel `obs_scenario`/`obs_action` labels for the observed actions);
`lm_alternatives_semantic.jsonl` (one record per unique `(scenario_label, action_text)` with
its per-scenario `cluster` id, `nearest_observed_action`, and `sim_to_observed_action` cosine,
joined back to `lm_alternatives.jsonl` on those keys); and `lm_clusters.json` (`model`,
`k_per_scenario`, `dup_threshold`, and the per-scenario action-type `clusters` with
nearest-centroid exemplar texts for interpretation).

Downstream of those, `project_alternatives.py` computes a per-scenario 2D UMAP layout and
writes `lm_alternatives_projection.jsonl` — one record per `(scenario_label, action_text)`
with `is_observed`, `observed_action`, the semantic labels, the run-averaged `g`/`risk`/`effort`,
and the `dim1`/`dim2` coordinates — which the R elicitation notebook joins and renders in
ggplot (UMAP is never re-run in R). `plot_alternatives.py` reads the whole artifact family to
render the SI alternatives figures into repo-root `figures/`, and drops two quick-look
diagnostic PNGs (`fig1_semantic_map.png`, a global UMAP colored by scenario and by nearest
observed action; `fig2_decision_space.png`, alternatives vs. the observed action in feature
space with Pareto-dominance flags) into `outputs/lm/<slug>/figures/`.

## Per-study fit and CV outputs (`<slug>/`)

Each study jointly fits its actor utility weights, `alpha_observer`, and the response-noise
`sigma` from its own belief-update data (weights are **not** transferred between studies).

### `fit_results.json`

A list with one object per ablation (`full`, `discomfort_only`, `base`). Each object carries
only the parameters its ablation actually uses, so there are no blank cells:

| Field | Description |
|--------|-------------|
| `model` | `full`, `discomfort_only`, or `base` |
| `experiment` | Slug (e.g. `food_inv_desire`) |
| `nll`, `n_params` | Fit diagnostics (negative log-likelihood of the mixture; parameter count) |
| `param_alpha` | Actor softmax temperature (fixed at 1.0) |
| `alpha_observer` | Fitted observer inverse temperature |
| `param_sigma` | Fitted response-noise scale `σ` |
| `param_w_v`, `param_w_d`, `param_w_e`, `param_gamma` | Fitted utility weights, only those the ablation uses |

`full` carries all four weights; `discomfort_only` carries `w_d` and `gamma` (no reward or
effort term); `base` carries `w_v` and `w_e` (no relational structure, so no `gamma`). Two
caveats when reading the values: a weight sitting at the `1e-6` lower bound has collapsed out
of the model, and some ablation × study combinations leave parameters unidentified — the
`discomfort_only` utility does not depend on desire (or effort), so in the desire studies its
posterior cannot move and its fitted `w_d`/`gamma` are arbitrary leftovers of the
initialization; the same applies to `base` in the intimacy studies. Those values should not be
interpreted.

### `fit_restarts.jsonl`

One record per multi-start restart: `experiment`, `model`, `restart`, the resulting `nll`, and
an `init_<param>` / `param_<param>` pair for each fitted parameter (`w_v`, `w_d`, `w_e`,
`gamma`, `alpha_observer`, `sigma`). Useful for checking that the reported fit is the
best-of-restarts and that restarts converge.

## Cross-validation outputs

All model-vs-human numbers reported in the analysis qmds are **out-of-sample**, from
leave-one-scenario-out (LOSO) CV: for each held-out scenario the weights, `alpha_observer`, and
`sigma` are refit on the other 15 scenarios (a warm start from the full-data fit plus a cold
restart, keeping the better optimum). CV is the only place predictions are generated — there is
no in-sample predict stage.

### `cv_trial_ll.jsonl` — per-trial held-out log-likelihood (primary metric)

The primary model-comparison output. One record per held-out trial: `experiment`, `model`,
`subject_id`, `scenario_label`, and `held_out_ll` (the log mixture-likelihood of that
participant's belief update under the model refit without their scenario). `subject_id` is the
anonymized participant UUID, carried through so the **full − ablation** difference in mean
held-out LL can be bootstrapped over participants.

### `cv_preds_summary.json`

A list with one object per held-out cell, giving the held-out `delta_<latent>` (and
`delta_effort` for the joint studies) tagged with `model`. This is the source the analysis qmds
load for the condition-averaged model-vs-human correlation (secondary/descriptive), and the
model's per-cell predictions generally. The desire study (`food_inv_desire`) additionally stores
`delta_desire_runs` — the K per-run held-out `δ_k` for each cell — which the SI run-spread and
mixture-check figures (`figures/scripts/plot_si_validation.py`) read to show the elicitation-sample
mixture spread against the fitted `σ`, all out-of-sample.

### `cv_folds.jsonl`

Per-fold refit diagnostics (16 folds × 3 ablations). Each record has `experiment`, `variant`,
`fold`, `held_out_scenario`, the refit `alpha_observer` / `param_sigma` / `param_*` weights, and
`train_nll` / `test_nll` with `n_train` / `n_test`.

### `fit_manifest.json`

The fit-side counterpart of `cv_manifest.json`, written by the `fit_*.py` wrappers alongside
`fit_results.json` and `fit_restarts.jsonl`: the study slug, a timestamp, the git SHA the fit
ran at, and SHA-256 hashes of the two fit outputs and of the input data CSV. The CV dispatcher
verifies it before warm-starting folds from the fit, and `model_comparison.py` verifies it
again. The check is deliberately asymmetric: a manifest that is **present but no longer matches**
(the fit outputs were rewritten, or the data CSV changed since the fit ran) is a hard error —
that is genuine staleness; a **missing** manifest only warns and proceeds, so a fit produced
before provenance tracking existed stays usable (CV can still run on it) rather than forcing a
re-fit. Re-run the fit to record provenance before trusting the final published numbers.

### `cv_manifest.json`

A provenance sidecar written by `model/cv/_inverse_dispatcher.py` alongside the three CV
outputs above, recording the study slug, a timestamp, the git SHA the CV ran at, and SHA-256
hashes of the three CV files and of the input data CSV. `model_comparison.py` verifies it with
the same asymmetry as the fit manifest: a **present but mismatched** manifest (the three CV
files were not written together, or the data CSV changed since CV ran) is a hard error — the
mixed-vintage combination the manifest exists to catch — while a **missing** manifest only
warns and proceeds, so CV outputs produced before provenance tracking can still be compared.
Re-run `make cv-<slug>` to record provenance before trusting the final published numbers.

### `cv_model_comparison.json`

The model-comparison statistics reported in the paper, computed from `cv_trial_ll.jsonl` and
`cv_preds_summary.json` by `model/cv/model_comparison.py` (`make model-comparison`):

- `primary` — for each ablation, the mean full − ablation difference in per-trial held-out
  log-likelihood with a 95% CI from bootstrap resampling of participants (`n_boot`, default
  1,000; trials are matched across model variants on subject × scenario).
- `mean_held_out_ll_per_trial` — each model's mean held-out log-likelihood.
- `secondary_correlations` — for each model and dependent variable, the Pearson correlation
  between the condition-averaged human belief updates and the model's held-out per-cell
  predictions, with a subject-cluster bootstrap 95% CI. The percentile interval is
  conservative for r: resampling participants adds noise to the cell means, which attenuates
  the bootstrapped correlations when per-cell trial counts are small.
