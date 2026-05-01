# `model/` — modeling pipeline

Every script in this folder is named after either the experiment it serves
(fit/predict/CV) or the LM output it produces. No multi-experiment dispatchers
in script names; no `--feature` flags hiding what a script does.

## Pipeline at a glance

```
LM elicitation  (model/lm/)
    score_canonical_features.py   →  outputs/lm/lm_scenario_params{,_nonfood}.csv
    score_canonical_v.py          →  outputs/lm/lm_scenario_v{,_nonfood}.csv
    score_alternative_features.py →  outputs/lm/lm_alternatives_features.csv (motivation)
                                  →  outputs/lm/lm_alternatives_relationship_features.csv (relationship)
    score_alternative_v.py        →  outputs/lm/lm_alternatives_v.csv / lm_alternatives_relationship_v.csv
    score_effort_features.py      →  outputs/lm/lm_scenario_params_effort{,_marginal}.csv
    generate_alternatives_motivation.py    →  outputs/lm/lm_alternatives.csv
    generate_alternatives_relationship.py  →  outputs/lm/lm_alternatives_relationship.csv
        ↓
Forward planning  (model/forward/)
    fit_<slug>.py     → outputs/<slug>/fit_results.csv
    predict_<slug>.py → outputs/<slug>/fits.csv
        ↓
Inverse planning  (model/inverse/)
    fit_<slug>.py     → outputs/<slug>/fit_results.csv
    predict_<slug>.py → outputs/<slug>/preds_full.csv + preds_summary.csv
        ↓
Cross-validation  (model/cv/)
    cv_<slug>.py → outputs/<slug>/cv_folds.csv + cv_preds[_summary].csv
        ↓
Analysis qmds   (analysis/<slug>-analysis.qmd)
```

## Per-experiment files

For every experiment slug, exactly two scripts in fit/predict + one CV script:

| Slug | Fit | Predict | CV |
|---|---|---|---|
| `food_forw_intimacy_desire` | `forward/fit_food_forw_intimacy_desire.py` | `forward/predict_food_forw_intimacy_desire.py` | `cv/cv_food_forw_intimacy_desire.py` |
| `food_forw_intimacy_effort` | `forward/fit_food_forw_intimacy_effort.py` | `forward/predict_food_forw_intimacy_effort.py` | `cv/cv_food_forw_intimacy_effort.py` |
| `nonfood_forw_intimacy_desire` | `forward/fit_nonfood_forw_intimacy_desire.py` | `forward/predict_nonfood_forw_intimacy_desire.py` | `cv/cv_nonfood_forw_intimacy_desire.py` |
| `food_inv-intimacy_desire_alt` | `inverse/fit_food_inv-intimacy_desire_alt.py` | `inverse/predict_food_inv-intimacy_desire_alt.py` | `cv/cv_food_inv-intimacy_desire_alt.py` |
| `food_inv-desire_intimacy_alt` | `inverse/fit_food_inv-desire_intimacy_alt.py` | `inverse/predict_food_inv-desire_intimacy_alt.py` | `cv/cv_food_inv-desire_intimacy_alt.py` |
| `food_inv-intimacy_desire_noalt` | `inverse/fit_food_inv-intimacy_desire_noalt.py` | `inverse/predict_food_inv-intimacy_desire_noalt.py` | `cv/cv_food_inv-intimacy_desire_noalt.py` |
| `food_inv-desire_intimacy_noalt` | `inverse/fit_food_inv-desire_intimacy_noalt.py` | `inverse/predict_food_inv-desire_intimacy_noalt.py` | `cv/cv_food_inv-desire_intimacy_noalt.py` |
| `food_inv-intimacy_effort_alt` | `inverse/fit_food_inv-intimacy_effort_alt.py` | `inverse/predict_food_inv-intimacy_effort_alt.py` | `cv/cv_food_inv-intimacy_effort_alt.py` |
| `food_inv-effort_intimacy_alt` | `inverse/fit_food_inv-effort_intimacy_alt.py` | `inverse/predict_food_inv-effort_intimacy_alt.py` | `cv/cv_food_inv-effort_intimacy_alt.py` |

Run any script directly as `uv run python <path>`. No flags needed for the per-experiment ones.

## Core math (one copy, shared across all experiments)

- `tables.py` — enums, scenario maps, `LLM_TABLES`, `LLM_TABLES_EFFORT`, padded-table loaders, domain-asset loader.
- `utility.py` — jit-compiled utility functions (Full / Discomfort-only / Base) with all variants.
- `actors.py` — actor memo models (forward + inverse + padded + effort counterparts).
- `observers.py` — observer memo models (intimacy / reward / effort, alt-shown + padded + relationship-keyed).

## Shared infrastructure (under each pipeline subfolder)

- `forward/_shared.py` — NLL/AIC/BIC, `_fit_with_adam`, predict/fit functions, data loaders.
- `inverse/_helpers.py` — observer fit loops, NLL functions, data loaders, frozen-param loaders, variant registries.
- `cv/_forward_dispatcher.py`, `cv/_alt_dispatcher.py` — joint LOSO logic shared by per-experiment CV scripts.
- `lm/_features_dispatcher.py`, `lm/_alternatives_dispatcher.py` — multi-mode internals shared by per-output LM scripts.
- `lm/client.py`, `lm/prompts.py` — LM-call infrastructure and prompt templates.

## Tests

```bash
uv run python model/test_model_compliance.py
```

## Sandbox / preregs

- `model/sandbox/` — exploratory/comparison code (not part of the main pipeline).
- `model/preregs/` — preregistration documents per experiment.
