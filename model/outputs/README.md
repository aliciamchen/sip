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
│       └── lm_alternatives.jsonl             # stage-1 generated alternative texts (one record per alt)
└── <slug>/                               # one folder per inverse study (fits + CV)
    ├── fit_results.json                      # fitted params per ablation (incl. param_sigma)
    ├── fit_restarts.jsonl                    # per-restart fit diagnostics
    ├── cv_trial_ll.jsonl                     # per-trial held-out log-likelihood, by subject_id  ← primary metric
    ├── cv_preds_summary.json                 # held-out per-cell delta_<latent> (the model's predictions)
    ├── cv_folds.jsonl                        # per-fold refit diagnostics
    └── cv_model_comparison.json              # bootstrap model-comparison statistics (the paper's numbers)
```

Slugs (the four inverse studies, all on the 3-action set): `food_inv_desire` (Study 1a),
`food_inv_joint_de` (1b), `food_inv_intimacy` (2a), `food_inv_joint_ie` (2b). Each slug's
`<slug>/` folder is populated by running its fit → CV scripts (`make all`, or the per-study
`make fit-<slug>` / `cv-<slug>`). There is no separate in-sample prediction stage: CV is the
sole source of model predictions, because every reported model-vs-human number is
out-of-sample.

The elicited LM tables (`lm_runs.jsonl`, `lm_alternatives.jsonl`) and the fit/CV outputs are
committed, so the fit → CV → analysis pipeline is reproducible from a fresh clone without a
Together AI key. They are regenerated when the pipeline changes: the LM tables by
`generate_alternatives.py` + `score_merged.py`, and the fit/CV outputs by the fit and CV
scripts. A study whose `lm_runs.jsonl` is missing has no LM tables — its loaders return
`None` and its fit raises a clear `FileNotFoundError`.

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
simulated-observer mixture's components, so the alternatives, their feature scores, **and** the
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
mixture-check figures (`model/lm/plot_si_validation.py`) read to show the simulated-observer
mixture spread against the fitted `σ`, all out-of-sample.

### `cv_folds.jsonl`

Per-fold refit diagnostics (16 folds × 3 ablations). Each record has `experiment`, `variant`,
`fold`, `held_out_scenario`, the refit `alpha_observer` / `param_sigma` / `param_*` weights, and
`train_nll` / `test_nll` with `n_train` / `n_test`.

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
