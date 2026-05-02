# `model/` — modeling pipeline

Every script in this folder is named after either the experiment it serves
(fit/predict/CV) or the LM output it produces. No multi-experiment dispatchers
in script names; no `--feature` flags hiding what a script does.

## Pipeline at a glance

```
LM elicitation  (model/lm/)
    score_canonical_features.py   →  outputs/lm/lm_scenario_params{,_nonfood}.csv
    score_canonical_v.py          →  outputs/lm/lm_scenario_v{,_nonfood}.csv
    score_alternative_features.py →  outputs/lm/lm_alternatives_features_food_inv_intimacy_desire_noalt.csv (motivation)
                                  →  outputs/lm/lm_alternatives_features_food_inv_desire_intimacy_noalt.csv (relationship)
    score_alternative_v.py        →  outputs/lm/lm_alternatives_v_food_inv_intimacy_desire_noalt.csv / lm_alternatives_v_food_inv_desire_intimacy_noalt.csv
    score_effort_features.py      →  outputs/lm/lm_scenario_params_effort{,_marginal}.csv
    generate_alternatives_motivation.py    →  outputs/lm/lm_alternatives_food_inv_intimacy_desire_noalt.csv
    generate_alternatives_relationship.py  →  outputs/lm/lm_alternatives_food_inv_desire_intimacy_noalt.csv
        ↓
Forward planning  (model/forward/)
    fit_<slug>.py     → outputs/<slug>/fit_results.csv
    predict_<slug>.py → outputs/<slug>/preds.csv
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
| `food_inv_intimacy_desire_alt` | `inverse/fit_food_inv_intimacy_desire_alt.py` | `inverse/predict_food_inv_intimacy_desire_alt.py` | `cv/cv_food_inv_intimacy_desire_alt.py` |
| `food_inv_desire_intimacy_alt` | `inverse/fit_food_inv_desire_intimacy_alt.py` | `inverse/predict_food_inv_desire_intimacy_alt.py` | `cv/cv_food_inv_desire_intimacy_alt.py` |
| `food_inv_intimacy_desire_noalt` | `inverse/fit_food_inv_intimacy_desire_noalt.py` | `inverse/predict_food_inv_intimacy_desire_noalt.py` | `cv/cv_food_inv_intimacy_desire_noalt.py` |
| `food_inv_desire_intimacy_noalt` | `inverse/fit_food_inv_desire_intimacy_noalt.py` | `inverse/predict_food_inv_desire_intimacy_noalt.py` | `cv/cv_food_inv_desire_intimacy_noalt.py` |
| `food_inv_intimacy_effort_alt` | `inverse/fit_food_inv_intimacy_effort_alt.py` | `inverse/predict_food_inv_intimacy_effort_alt.py` | `cv/cv_food_inv_intimacy_effort_alt.py` |
| `food_inv_effort_intimacy_alt` | `inverse/fit_food_inv_effort_intimacy_alt.py` | `inverse/predict_food_inv_effort_intimacy_alt.py` | `cv/cv_food_inv_effort_intimacy_alt.py` |

Run any script directly as `uv run python <path>`. No flags needed for the per-experiment ones.

### Why dispatchers?

The few places where logic is naturally shared across experiments (the joint LOSO loops in `cv/`, the multi-mode LM dispatchers in `lm/`) live in `_dispatcher.py` files. Each per-experiment script is a thin wrapper: it imports from the dispatcher and calls its main with the experiment slug hardcoded. To trace what `cv/cv_food_forw_intimacy_desire.py` does, follow the import to `cv/_forward_dispatcher.py`. Same pattern for `cv/cv_food_inv_intimacy_desire_alt.py` → `cv/_alt_dispatcher.py:main_intimacy_alt`, and `lm/score_canonical_features.py` → `lm/_features_dispatcher.py:main`.

## Core math (one copy, shared across all experiments)

- `tables.py` — enums, scenario maps, `LLM_TABLES`, `LLM_TABLES_EFFORT`, padded-table loaders, domain-asset loader.
- `utility.py` — jit-compiled utility functions (Full / Discomfort-only / Base) with all variants.
- `actors.py` — actor memo models (forward + inverse + padded + effort counterparts).
- `observers.py` — observer memo models (intimacy / reward / effort, alt-shown + padded + relationship-keyed).

### Terminology: V, reward, desire, motivation

These four words all relate to the actor's motivational state, but they're not interchangeable in code:

- **V** is the *signed valence* of an action with respect to the actor's motivational state, in `[-1, +1]`. Positive = action serves the state; negative = action is counterproductive; 0 = neutral. V is what enters the utility as `w_v · V`.
- The **canonical 4-action pipeline** elicits V from the LM per `(scenario, action, motivation)` and stores it in `outputs/lm/lm_scenario_v.csv`, loaded via `tables.load_lm_v(domain)`.
- The **effort 2-action pipeline** stipulates V=1 for both actions in `utility.get_stipulated_reward_effort` because reward is held fixed at HIGH and both actions involve eating, so V is uniform by construction. With V uniform across actions, `w_v` is non-identified under the softmax — it's kept in the utility for parallelism with the canonical pipeline but stays near initialization during fitting.
- "Reward" appears in code (`reward_condition`, `param_w_v`, internal `experiment="reward"` column values) and "motivation" appears in the data CSVs (`motivation` column with values `low`/`high`); both refer to the same underlying motivational state. The paper-facing word is **desire**. We use whichever fits the local context — paper text says "desire", data files say "motivation", code uses "reward" for V/condition variables. None of these are different quantities; the multiplicity is purely terminological drift.

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
