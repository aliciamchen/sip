# `model/` — modeling pipeline

Every script in this folder is named after either the experiment it serves (fit/predict/CV) or the LM output it produces. No multi-experiment dispatchers in script names; no `--feature` flags hiding what a script does.

The active roster is four inverse-planning studies, all on the 3-action stimulus set and all using the LM-generated-alternatives padded-action pipeline: `food_inv_desire` (Study 1a), `food_inv_joint_de` (1b), `food_inv_intimacy` (2a), `food_inv_joint_ie` (2b). The forward-planning experiments and the pre-3-action inverse experiments are legacy (see below).

## Pipeline at a glance

```
LM elicitation  (model/lm/)
    score_3act_features.py   →  outputs/lm/lm_scenario_params_3act{,_marginal}.csv  (fixed-action access + effort)
    score_3act_v.py          →  outputs/lm/lm_scenario_v_3act.csv                   (fixed-action V)
    generate_alternatives_3act.py --study <slug>  →  outputs/lm/lm_alternatives_<slug>.csv          (per-study counterfactuals)
    score_3act_merged.py          --study <slug>  →  outputs/lm/lm_alternatives_{features,v}_<slug>.csv + canonical CSVs
        ↓
Inverse planning  (model/inverse/)       active Studies 1a, 1b, 2a, 2b
    fit_<slug>.py     → outputs/<slug>/fit_results.csv
    predict_<slug>.py → outputs/<slug>/preds_<variant>.npy + preds_summary.csv
        ↓
Cross-validation  (model/cv/)
    cv_<slug>.py → outputs/<slug>/cv_folds.csv + cv_preds_summary.csv
        ↓
Analysis qmds   (analysis/<slug>-analysis.qmd)
```

Legacy experiments keep parallel `model/forward/` and `model/inverse/*_noalt` scripts that write to `outputs/legacy/<slug>/` (see "Legacy" below).

## Per-experiment files (active)

All four active studies infer one or two latent variables from a single observed action; the observer reasons over a padded action space (`{observed action} ∪ LM-generated alternatives`) and the fit/CV slice slot 0.

| Slug | Study | Infers | Fit | Predict | CV |
|---|---|---|---|---|---|
| `food_inv_desire`   | 1a | desire           | `inverse/fit_food_inv_desire.py`   | `inverse/predict_food_inv_desire.py`   | `cv/cv_food_inv_desire.py` |
| `food_inv_joint_de` | 1b | desire + effort  | `inverse/fit_food_inv_joint_de.py` | `inverse/predict_food_inv_joint_de.py` | `cv/cv_food_inv_joint_de.py` |
| `food_inv_intimacy` | 2a | intimacy         | `inverse/fit_food_inv_intimacy.py` | `inverse/predict_food_inv_intimacy.py` | `cv/cv_food_inv_intimacy.py` |
| `food_inv_joint_ie` | 2b | intimacy + effort| `inverse/fit_food_inv_joint_ie.py` | `inverse/predict_food_inv_joint_ie.py` | `cv/cv_food_inv_joint_ie.py` |

Run any script directly as `uv run python <path>`, or via `make fit-<slug>` / `make predict-<slug>` / `make cv-<slug>`. The CV scripts are thin wrappers around the LOSO mains in `cv/_inverse_dispatcher.py`. No active study has collected data yet, so the fit scripts run only once data lands in `data/<slug>/` (and the per-study LM-alternatives CSVs have been elicited).

### Legacy

Model scripts for legacy experiments remain runnable (per-slug `make` targets only; not part of `make all`) and write to `outputs/legacy/<slug>/`:

- **Forward planning** — `model/forward/{fit,predict}_<slug>.py` + `model/cv/cv_<slug>.py` for `food_forw_intimacy_desire`, `food_forw_intimacy_effort`, `nonfood_forw_intimacy_desire` (data in `data/legacy/`, registered under the Makefile's `LEGACY_FORWARD`).
- **Pre-3-action inverse** — the two `food_inv_*_noalt` slugs keep `model/inverse/{fit,predict}_*_noalt.py` + `model/cv/cv_*_noalt.py` (`LEGACY_INVERSE`); the four `_alt` siblings are data-only.
- The Study 1a pilot fit lives at `outputs/legacy/food_inv_desire_pilot/`.

### Why dispatchers?

Logic shared across experiments (the LOSO loops in `cv/`, the multi-mode features dispatcher in `lm/`) lives in `_dispatcher.py` / `_helpers.py` files. Each per-experiment script is a thin wrapper: it imports the shared main and calls it with the experiment slug hardcoded.

## Core math (one copy, shared across all experiments)

- `tables.py` — enums (`Scenarios`, `RewardConditions`, `RelationshipConditions`, `EffortConditions`, `PaddedActionSlots`, `PaddedActionSlots3Act`), action arrays (`actions`, `actions_effort`, `actions_3act`), `SCENARIO_LABELS`, and LM table loaders: the fixed-action 3-action tables (`LLM_TABLES_3ACT`, `load_lm_v_3act`) and the per-study padded LM-alternatives loaders (`load_padded_lm_tables_3act_{desire,joint_de,intimacy,joint_ie}`). The legacy 4-action / 2-action loaders (`LLM_TABLES`, `LLM_TABLES_EFFORT`, `load_lm_v`) remain for the forward experiments.
- `utility.py` — jit-compiled utility functions. The active studies use the padded families `get_utility_3act_*_padded_{desire,joint_de,intimacy,joint_ie}` (+ prior/V helpers); the fixed-action `get_utility_3act_*` and the legacy canonical/effort families remain.
- `actors.py` — actor memo models. The active studies use the padded inverse actors `actor_{discrete,continuous}_3act_*_padded_{desire,joint_de,intimacy,joint_ie}` inside the observers' `thinks[...]` blocks. Forward and fixed-action 3-action actors remain for legacy + reference.
- `observers.py` — observer memos, one family per active study, each in `_full` / `_discomfort_only` / `_base`:
  - `observer_reward_*` — Study 1a (infers desire)
  - `observer_joint_de_*` — Study 1b (joint posterior over reward × effort, given intimacy)
  - `observer_intimacy_*` — Study 2a (infers intimacy, 101-bin continuous posterior)
  - `observer_joint_ie_*` — Study 2b (joint posterior over intimacy × effort, given desire)

  Joint observers use memo's `chooses(x in X, y in Y, ...)` multi-choice syntax and return `Pr[a, b]`; downstream code marginalizes for the per-slider predictions.

### Terminology: V, reward, desire, motivation

These four words all relate to the actor's motivational state, but they're not interchangeable in code:

- **V** is the *signed valence* of an action with respect to the actor's motivational state, in `[-1, +1]`. Positive = action serves the state; negative = action is counterproductive; 0 = neutral. V enters the utility as `w_v · V`.
- V is elicited from the LM per `(scenario, action, motivation)` and stored in `outputs/lm/lm_scenario_v_3act.csv` (canonical) plus `outputs/lm/lm_alternatives_v_<slug>.csv` (alternatives), loaded by the padded-table loaders.
- "Reward" appears in code (`reward_condition`, `param_w_v`); "motivation" appears in the data CSVs (`motivation` column, `low`/`high`); both refer to the same motivational state. The paper-facing word is **desire**, and its DV is a 1–7 Likert in Studies 1a/1b.

## Shared infrastructure

- `inverse/_helpers.py` — observer fit loops (`fit_{desire,joint_de,intimacy,joint_ie}_observer_joint`), NLL functions (`compute_intimacy_nll`, `compute_reward_nll`, `compute_desire_likert_se`), per-study data loaders, frozen-param loaders, and the padded table-kwargs helpers (`desire_table_kwargs`, `joint_de_table_kwargs`, `intimacy_table_kwargs`, `joint_ie_table_kwargs`). Legacy `_noalt` loaders/fitters remain.
- `cv/_inverse_dispatcher.py` — LOSO logic for the four active inverse studies (`main_{desire,joint_de,intimacy,joint_ie}`).
- `cv/_forward_dispatcher.py` — joint LOSO logic for the legacy forward CV scripts.
- `forward/_shared.py` — NLL/AIC/BIC, `_fit_with_adam`, predict/fit functions, data loaders (legacy forward experiments).
- `lm/_features_dispatcher.py` — multi-mode internals shared by the canonical scorers; `lm/client.py`, `lm/prompts.py` — LM-call infrastructure and prompt templates.

## Tests

```bash
uv run python model/test_model_compliance.py
```

## Sandbox

- `model/sandbox/` — exploratory/comparison code (not part of the main pipeline).

Preregistration documents are at the repo root under [`preregs/`](../preregs/), one file per experiment slug.
