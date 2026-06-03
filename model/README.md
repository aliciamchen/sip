# `model/` — modeling pipeline

Every script in this folder is named after either the experiment it serves (fit/predict/CV) or the LM output it produces. No multi-experiment dispatchers in script names; no `--feature` flags hiding what a script does.

The roster is four inverse-planning studies, all on the 3-action stimulus set and all using the LM-generated-alternatives padded-action pipeline: `food_inv_desire` (Study 1a), `food_inv_joint_de` (1b), `food_inv_intimacy` (2a), `food_inv_joint_ie` (2b). (Earlier forward-planning and pre-3-action inverse code was removed in the June 2026 cleanup; only the collected data is archived under `data/legacy/`.)

## Pipeline at a glance

```
LM elicitation  (model/lm/)
    generate_alternatives.py --study <slug>  →  outputs/lm/<slug>/lm_alternatives.csv          (per-study counterfactuals)
    score_merged.py          --study <slug>  →  outputs/lm/<slug>/lm_scenario.csv      (canonical risk+effort+g, this study's frame)
                                                outputs/lm/<slug>/lm_alternatives.csv  (the alt list + its risk/effort/g; + lm_scenario_desire.csv for 2a/2b)
        ↓
Inverse planning  (model/inverse/)       Studies 1a, 1b, 2a, 2b
    fit_<slug>.py     → outputs/<slug>/fit_results.csv
    predict_<slug>.py → outputs/<slug>/preds_<variant>.npy + preds_summary.csv
        ↓
Cross-validation  (model/cv/)
    cv_<slug>.py → outputs/<slug>/cv_folds.csv + cv_preds_summary.csv
        ↓
Analysis qmds   (analysis/<slug>-analysis.qmd)
```

## Per-experiment files

All four studies infer one or two latent variables from a single observed action; the observer reasons over a padded action space (`{observed action} ∪ LM-generated alternatives`) and the fit/CV slice slot 0.

| Slug | Study | Infers | Fit | Predict | CV |
|---|---|---|---|---|---|
| `food_inv_desire`   | 1a | desire           | `inverse/fit_food_inv_desire.py`   | `inverse/predict_food_inv_desire.py`   | `cv/cv_food_inv_desire.py` |
| `food_inv_joint_de` | 1b | desire + effort  | `inverse/fit_food_inv_joint_de.py` | `inverse/predict_food_inv_joint_de.py` | `cv/cv_food_inv_joint_de.py` |
| `food_inv_intimacy` | 2a | intimacy         | `inverse/fit_food_inv_intimacy.py` | `inverse/predict_food_inv_intimacy.py` | `cv/cv_food_inv_intimacy.py` |
| `food_inv_joint_ie` | 2b | intimacy + effort| `inverse/fit_food_inv_joint_ie.py` | `inverse/predict_food_inv_joint_ie.py` | `cv/cv_food_inv_joint_ie.py` |

Run any script directly as `uv run python <path>`, or via `make fit-<slug>` / `make predict-<slug>` / `make cv-<slug>`. The CV scripts are thin wrappers around the LOSO mains in `cv/_inverse_dispatcher.py`. The fit scripts run only once data lands in `data/<slug>/` and the per-study LM-alternatives CSVs have been elicited (otherwise the table-kwargs helpers raise a clear `FileNotFoundError`).

### Why dispatchers?

Logic shared across experiments (the LOSO loops in `cv/`, the multi-mode helpers in `lm/`) lives in `_dispatcher.py` / `_helpers.py` files. Each per-experiment script is a thin wrapper: it imports the shared main and calls it with the experiment slug hardcoded.

## Core math (one copy, shared across all experiments)

- `tables.py` — enums (`Scenarios`, `DesireConditions`, `RelationshipConditions`, `EffortConditions`, `IntimacyLevels`, `DesireLevels`, `PaddedActionSlots`, `ObservedActions`), the `actions` array, `SCENARIO_LABELS`, the per-study padded LM-alternatives loaders (`load_padded_lm_tables_{desire,joint_de,intimacy,joint_ie}`, each reading `lm_scenario.csv` + `lm_alternatives.csv`), and `load_lm_scenario_desire` (per-condition desire scalar for 2a/2b).
- `utility.py` — jit-compiled utility functions: the padded families `get_utility_{full,discomfort_only,base}_padded_{desire,joint_de,intimacy,joint_ie}` plus their `get_prior_padded_*` and `get_lm_g_padded_*` helpers. The reward term is `w_v · desire · g`.
- `actors.py` — actor memo models: the padded inverse actors `actor_discrete_*_padded_{desire,joint_de}` (discrete observed intimacy) and `actor_continuous_*_padded_{intimacy,joint_ie}` (continuous inferred intimacy), used inside the observers' `thinks[...]` blocks.
- `observers.py` — observer memos, one family per study, each in `_full` / `_discomfort_only` / `_base`:
  - `observer_desire_*` — Study 1a (infers desire, 101-bin continuous posterior)
  - `observer_joint_de_*` — Study 1b (joint posterior over desire × effort, given intimacy)
  - `observer_intimacy_*` — Study 2a (infers intimacy, 101-bin continuous posterior)
  - `observer_joint_ie_*` — Study 2b (joint posterior over intimacy × effort, given desire)

  Joint observers use memo's `chooses(x in X, y in Y, ...)` multi-choice syntax and return `Pr[a, b]`; downstream code marginalizes for the per-slider predictions.

### Terminology: g, desire, reward, risk

- **g** (goal-satisfaction) is how fully an action delivers the outcome, in `[0, 1]`; desire-free. **desire** (`d`, in `[0, 1]`) is how much the dyad wants the outcome. They enter the utility together as the reward term `w_v · desire · g`. `g` replaced the old signed-valence `V`, which was legacy and is gone.
- The code, data, and paper all use **desire** (the June 2026 cleanup renamed the model-side `reward_condition` → `desire_condition`, the `RewardConditions` enum → `DesireConditions`, and the processed-CSV `motivation` column → `desire`). The one name kept is the fitted weight `w_v` (and `param_w_v` in `fit_results.csv`) — the weight on the `w_v · desire · g` term, left as `w_v` to avoid colliding with `w_d`.
- The per-action discomfort feature is **risk** (`w_d` is its weight; renamed from `access` in the same cleanup).

## Shared infrastructure

- `inverse/_helpers.py` — `_fit_with_adam`, NLL functions (`compute_intimacy_nll`, `compute_effort_nll`, `compute_desire_nll`), per-study data loaders, the joint observer fit loops (`fit_{desire,joint_de,intimacy,joint_ie}_observer_joint`), and the padded table-kwargs helpers (`desire_table_kwargs`, `joint_de_table_kwargs`, `intimacy_table_kwargs`, `joint_ie_table_kwargs`).
- `cv/_inverse_dispatcher.py` — LOSO logic for the four inverse studies (`main_{desire,joint_de,intimacy,joint_ie}`).
- `lm/_features_dispatcher.py`, `lm/_alternatives_dispatcher.py` — multi-mode internals shared by the scorers / alternative generation; `lm/client.py`, `lm/prompts.py` — LM-call infrastructure and prompt templates.

## Tests

```bash
uv run python model/test_model_compliance.py
```

Preregistration documents are at the repo root under [`preregs/`](../preregs/), one file per experiment slug.
