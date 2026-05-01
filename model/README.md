# `model/` — modeling pipeline

This folder contains all the modeling code: LM-based feature elicitation, the
inverse-planning model itself, fit/prediction/CV scripts per experiment, and
all model outputs grouped by experiment slug.

## Pipeline at a glance

```
┌────────────────────────────────────────────────────────────────────────────┐
│  1. LM elicitation         model/lm/                                       │
│     scenario_params.py  →  outputs/lm/lm_scenario_params.csv (access, eff) │
│                         →  outputs/lm/lm_scenario_v.csv      (signed V)    │
│     generate_alternatives.py  →  outputs/lm/lm_alternatives*.csv           │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  2. Forward fit           fit_forward.py --experiment <slug>               │
│     LOSO CV               cv/loso_forward.py --experiment <slug>           │
│       → outputs/<slug>/{fit_results,fits,cv_folds,cv_preds}.csv            │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  3. Inverse fit + predict  fit_inverse_planning_<variant>.py               │
│                            generate_inverse_planning_<variant>_preds.py    │
│     Inverse LOSO CV        cv/loso_inverse_<variant>.py                    │
│       → outputs/<slug>/{fit_results,preds_full,preds_summary,              │
│                          cv_folds,cv_preds_summary}.csv                    │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                           analysis/<slug>-analysis.qmd
```

## Core math (one copy, shared across all experiments)

- `tables.py` — `Scenarios` / `RewardConditions` / `RelationshipConditions` / `EffortConditions` / `PaddedActionSlots` enums; `LLM_TABLES`, `LLM_TABLES_EFFORT`, padded-table loaders; domain-asset loader.
- `utility.py` — jit-compiled utility functions: `get_utility_full / discomfort_only / base` (with `_disc`, `_padded`, `_padded_rel`, and `_effort` siblings). Dimension-agnostic — used by both 4-action canonical and 2-action effort actors.
- `actors.py` — actor memo models: forward, discrete-relationship, continuous-intimacy, padded variants for no-alt observers, plus effort-experiment counterparts.
- `observers.py` — observer memo models: `observer_intimacy_*`, `observer_reward_*` (alt-shown + padded + relationship-keyed), `observer_intimacy_effort_*`, `observer_effort_intimacy_*`.

The three ablations (Full / Discomfort-only / Base) are defined in `utility.py` and used everywhere via `--variant` flags or registry entries inside fit scripts.

## Per-experiment scripts

The experiment slug is the stable identifier — it matches the folder under `data/`, `experiments/`, and `outputs/`.

| Slug | Fit script | Predictions script | CV script |
|---|---|---|---|
| `food_forw_intimacy_desire` | `fit_forward.py --experiment food_forw_intimacy_desire` | (forward; predictions live in `fits.csv` from the fit) | `cv/loso_forward.py --experiment food_forw_intimacy_desire` |
| `nonfood_forw_intimacy_desire` | `fit_forward.py --experiment nonfood_forw_intimacy_desire` | — | `cv/loso_forward.py --experiment nonfood_forw_intimacy_desire` |
| `food_forw_intimacy_effort` | `fit_forward.py --experiment food_forw_intimacy_effort` | — | `cv/loso_forward.py --experiment food_forw_intimacy_effort` |
| `food_inv-intimacy_desire_alt` | `fit_inverse_planning_alt.py` (joint) | `generate_inverse_planning_alt_preds.py` (joint) | `cv/loso_inverse_alt.py` (joint) |
| `food_inv-desire_intimacy_alt` | `fit_inverse_planning_alt.py` (joint) | `generate_inverse_planning_alt_preds.py` (joint) | `cv/loso_inverse_alt.py` (joint) |
| `food_inv-intimacy_desire_noalt` | `fit_inverse_planning_intimacy_noalt.py` | `generate_inverse_planning_intimacy_noalt_preds.py` | `cv/loso_inverse_intimacy_noalt.py` |
| `food_inv-desire_intimacy_noalt` | `fit_inverse_planning_desire_noalt.py` | `generate_inverse_planning_desire_noalt_preds.py` | `cv/loso_inverse_desire_noalt.py` |
| `food_inv-intimacy_effort_alt` | `fit_inverse_planning_intimacy_effort.py` | `generate_inverse_planning_intimacy_effort_preds.py` | `cv/loso_inverse_intimacy_effort.py` |
| `food_inv-effort_intimacy_alt` | `fit_inverse_planning_effort_intimacy.py` | `generate_inverse_planning_effort_intimacy_preds.py` | `cv/loso_inverse_effort_intimacy.py` |

The two alt-shown experiments share fit/predict/CV scripts because the joint fit estimates a single `α_observer` over both targets simultaneously. Each script writes per-experiment CSVs into the appropriate `outputs/<slug>/` dir.

## Output layout

```
outputs/
├── lm/                                # LM-elicited tables (access, effort, V, alternatives)
│   ├── lm_scenario_params.csv
│   ├── lm_scenario_v.csv
│   ├── lm_alternatives*.csv
│   └── ... (food + nonfood + effort)
└── <slug>/                            # one folder per experiment
    ├── fit_results.csv                # fitted parameters + AIC/BIC/r per variant
    ├── fits.csv                       # forward only — per-trial predictions
    ├── preds_full.csv                 # inverse only — per-(scenario, action, condition) predictions
    ├── preds_summary.csv              # inverse only — summary scalar per condition
    ├── cv_folds.csv                   # per-fold fit results from LOSO CV
    ├── cv_preds.csv                   # forward only — per-trial held-out predictions
    └── cv_preds_summary.csv           # inverse only — held-out per-condition summary
```

The 9 experiment slugs (8 food + 1 nonfood) plus `lm/` are the only top-level folders under `outputs/`.

## Tests

```bash
uv run python model/test_model_compliance.py
```

Validates that the utility / actor / observer math behaves as specified — V-independence in `discomfort_only`, intimacy-independence in `base`, padded observers' posteriors normalize, etc.

## Sandbox / preregs

- `model/sandbox/` — exploratory/comparison code that's not part of the main pipeline (frozen-param verification, prompt-comparison sweeps, etc.).
- `model/preregs/` — preregistration documents per experiment.
