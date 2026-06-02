# Model outputs codebook

Outputs are grouped by experiment slug:

```
outputs/
├── lm/                                # LM-elicited tables (lm_*.csv)
└── <slug>/                            # one folder per inverse study
    ├── fit_results.csv
    ├── preds_<variant>.npy            # raw posterior arrays per variant (gitignored)
    ├── preds_summary.csv              # summary scalars (gitignored)
    ├── cv_folds.csv                   # per-fold fit results from LOSO CV
    └── cv_preds_summary.csv           # held-out per-condition summary
```

Slugs (4 inverse studies, all on the 3-action set): `food_inv_desire` (Study 1a),
`food_inv_joint_de` (1b), `food_inv_intimacy` (2a), `food_inv_joint_ie` (2b). None
have collected data yet, so their output folders are created on first fit.

(Earlier forward-planning + pre-3-action inverse outputs were removed in the June 2026
cleanup; only the collected participant data is archived under `data/legacy/`.)

`preds_<variant>.npy` and `preds_summary.csv` are the all-data (non-CV) predictions —
written by the predict scripts but not read downstream (analysis qmds use the CV outputs,
since reported correlations are out-of-sample). They're gitignored; regenerate locally via
`make predict-<slug>`.

## Terminology note

The reward term is `w_v · desire · g`: `w_v` is the fitted weight (kept under that name even
though the concept is desire), `desire` the desire magnitude, `g` the LM-elicited
goal-satisfaction. The per-action discomfort feature is `risk` (weight `w_d`). See
[`model/README.md`](../README.md) for the full terminology.

## LM-elicited tables (`outputs/lm/`)

### `lm_scenario_params.csv` — fixed-action risk + effort

Per (scenario, effort_condition, action) risk and effort ratings for the 3-action set, 96
rows (16 × 2 × 3). Columns: `scenario_label`, `effort_condition`, `action`, plus
`risk`/`risk_raw`/`risk_raw_std`, `effort`/`effort_raw`/`effort_raw_std`, and the
`n_runs_*` / `n_failures_*` run-count columns. Produced by `model/lm/score_features.py`.

### `lm_scenario_params_marginal.csv` — effort-marginal risk

Risk ratings elicited without the effort paragraph (risk is intimacy- and effort-independent
in the utility, modulated by `(1−I)^γ`). Same schema minus the effort columns.

### Per-study padded LM-alternatives tables

`generate_alternatives.py --study <slug>` writes `lm_alternatives_<slug>.csv` (the
LM-generated counterfactual actions per cell), and `score_merged.py --study <slug>` writes
`lm_alternatives_features_<slug>.csv` (risk + effort for the alts), `lm_alternatives_g_<slug>.csv`
(goal-satisfaction g), the shared canonical `lm_scenario_g.csv`, and — for the given-desire
studies (2a/2b) — `lm_scenario_desire.csv` (per-condition desire scalar).

## Per-study outputs (`<slug>/`)

### `<slug>/fit_results.csv`

Summary of fitted observer models, one row per ablation. Each study jointly fits its actor
utility weights **and** `alpha_observer` from its own posterior data (weights are **not**
transferred between studies).

| Column | Description |
|--------|-------------|
| `model` | `full`, `discomfort_only`, or `base` |
| `experiment` | Slug (e.g., `food_inv_desire`) |
| `alpha_observer` | Fitted observer inverse temperature |
| `param_w_v`, `param_w_d`, `param_w_e`, `param_gamma` | Fitted utility weights (NaN where not used by the ablation) |
| `nll`, `n_params`, `aic`, `bic` | Fit diagnostics |

### `<slug>/preds_<variant>.npy`

Raw observer-table array, one file per ablation. Saved as numpy `.npy` because the joint
observers (1b, 2b) return high-dimensional joint posteriors that don't flatten cleanly into a
CSV. The fit/CV slice **slot 0** (the observed action).

### `<slug>/preds_summary.csv`

Lightweight summary of the prediction array per variant (shape + sum sanity check). Full
per-cell tables will be added once data lands and a consumer needs them.

## Cross-validation CSVs

All model-vs-human correlations reported in the analysis qmds are **out-of-sample**, pooled
from leave-one-scenario-out (LOSO) CV (refit utility weights + `alpha_observer` per fold).

### `<slug>/cv_folds.csv`

Per-fold LOSO results (16 folds × 3 ablations). Columns: `fold`, `held_out_scenario`,
`variant`, `train_nll`/`test_nll` (+ per-trial), `n_train`/`n_test`, the per-fold
`param_*` weights, and `test_cell_r`.

### `<slug>/cv_preds_summary.csv`

Held-out per-condition summary pooled across the 16 folds — the source the analysis qmds load
for model-vs-human plots.
