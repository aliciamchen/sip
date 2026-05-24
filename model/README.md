# `model/` — modeling pipeline

Every script in this folder is named after either the experiment it serves (fit/predict/CV) or the LM output it produces. No multi-experiment dispatchers in script names; no `--feature` flags hiding what a script does.

## Pipeline at a glance

```
LM elicitation  (model/lm/)
    score_canonical_features.py   →  outputs/lm/lm_scenario_params{,_nonfood}.csv      (Study 1a, archived 4-action inverses)
    score_canonical_v.py          →  outputs/lm/lm_scenario_v{,_nonfood}.csv           (Study 1a)
    score_effort_features.py      →  outputs/lm/lm_scenario_params_effort{,_marginal}.csv  (Study 1b)
    score_3act_features.py        →  outputs/lm/lm_scenario_params_3act{,_marginal}.csv    (Studies 2/3/4)
    score_3act_v.py               →  outputs/lm/lm_scenario_v_3act{,_nonfood}.csv     (Studies 2/3/4)
        ↓
Forward planning  (model/forward/)       Studies 1a, 1b, non-food forward
    fit_<slug>.py     → outputs/<slug>/fit_results.csv
    predict_<slug>.py → outputs/<slug>/preds.csv
        ↓
Inverse planning  (model/inverse/)       Studies 2, 3a, 3b, 4a, 4b
    fit_<slug>.py     → outputs/<slug>/fit_results.csv
    predict_<slug>.py → outputs/<slug>/preds_<variant>.npy + preds_summary.csv
        ↓
Cross-validation  (model/cv/)
    cv_<slug>.py → outputs/<slug>/cv_folds.csv + cv_preds[_summary].csv
        ↓
Analysis qmds   (analysis/<slug>-analysis.qmd)
```

## Per-experiment files (active)

| Slug | Study | Fit | Predict | CV |
|---|---|---|---|---|
| `food_forw_intimacy_desire` | 1a | `forward/fit_food_forw_intimacy_desire.py` | `forward/predict_food_forw_intimacy_desire.py` | `cv/cv_food_forw_intimacy_desire.py` |
| `food_forw_intimacy_effort` | 1b | `forward/fit_food_forw_intimacy_effort.py` | `forward/predict_food_forw_intimacy_effort.py` | `cv/cv_food_forw_intimacy_effort.py` |
| `nonfood_forw_intimacy_desire` | — | `forward/fit_nonfood_forw_intimacy_desire.py` | `forward/predict_nonfood_forw_intimacy_desire.py` | `cv/cv_nonfood_forw_intimacy_desire.py` |
| `food_inv_intimacy_3act` | 2  | `inverse/fit_food_inv_intimacy_3act.py` | `inverse/predict_food_inv_intimacy_3act.py` | `cv/cv_food_inv_intimacy_3act.py` |
| `food_inv_effort_3act`   | 3a | `inverse/fit_food_inv_effort_3act.py`   | `inverse/predict_food_inv_effort_3act.py`   | `cv/cv_food_inv_effort_3act.py` |
| `food_inv_desire_3act`   | 3b | `inverse/fit_food_inv_desire_3act.py`   | `inverse/predict_food_inv_desire_3act.py`   | `cv/cv_food_inv_desire_3act.py` |
| `food_inv_joint_de_3act` | 4a | `inverse/fit_food_inv_joint_de_3act.py` | `inverse/predict_food_inv_joint_de_3act.py` | `cv/cv_food_inv_joint_de_3act.py` |
| `food_inv_joint_di_3act` | 4b | `inverse/fit_food_inv_joint_di_3act.py` | `inverse/predict_food_inv_joint_di_3act.py` | `cv/cv_food_inv_joint_di_3act.py` |

Run any script directly as `uv run python <path>`. No flags needed for the per-experiment ones. The five inverse CV scripts are currently stubs pending a full LOSO loop; for now use `make fit-<slug>` + `make predict-<slug>` directly.

The six pre-3-action inverse-food slugs (`food_inv_*_alt`, `food_inv_*_noalt`) remain on disk as legacy and will be removed once the new design supersedes them; their Makefile targets are registered under `LEGACY_INVERSE`.

### Why dispatchers?

The few places where logic is naturally shared across experiments (the joint LOSO loop in `cv/`, the multi-mode features dispatcher in `lm/`) live in `_dispatcher.py` files. Each per-experiment script is a thin wrapper: it imports from the dispatcher and calls its main with the experiment slug hardcoded.

## Core math (one copy, shared across all experiments)

- `tables.py` — enums (`Scenarios`, `RewardConditions`, `RelationshipConditions`, `EffortConditions`, `PaddedActionSlots`), action arrays (`actions`, `actions_effort`, `actions_3act`), `SCENARIO_LABELS`, and LM table loaders for all three stimulus structures (`LLM_TABLES`, `LLM_TABLES_EFFORT`, `LLM_TABLES_3ACT`, `load_lm_v`, `load_lm_v_3act`). The legacy padded-table loaders also live here.
- `utility.py` — jit-compiled utility functions. Three families: canonical 4-action (`get_utility_full/discomfort_only/base`), effort 2-action (`get_utility_effort_*`), and 3-action (`get_utility_3act_*` with `_disc` siblings for discrete relationship).
- `actors.py` — actor memo models. Forward families for all three stimulus structures (`actor_forw_*`, `actor_forw_effort_*`, `actor_forw_3act_*`) plus inverse actors used inside observer `thinks[...]` blocks (`actor_continuous_3act_*`, `actor_discrete_3act_*` for the new design; the legacy `actor_discrete_*`, `actor_continuous_*`, `_padded`, `_padded_rel` families remain for archived experiments).
- `observers.py` — observer memos. New roster (single-target: `observer_intimacy_3act_*` for Study 2, `observer_effort_3act_*` for Study 3a, `observer_reward_3act_*` for Study 3b; joint: `observer_joint_de_3act_*` for Study 4a, `observer_joint_di_3act_*` for Study 4b). Joint observers use memo's `chooses(x in X, y in Y, ...)` multi-choice syntax and return `Pr[a, b]`.

### Terminology: V, reward, desire, motivation

These four words all relate to the actor's motivational state, but they're not interchangeable in code:

- **V** is the *signed valence* of an action with respect to the actor's motivational state, in `[-1, +1]`. Positive = action serves the state; negative = action is counterproductive; 0 = neutral. V is what enters the utility as `w_v · V`.
- The canonical 4-action and 3-action pipelines elicit V from the LM per `(scenario, action, motivation)` and store it in `outputs/lm/lm_scenario_v{,_3act}.csv`, loaded via `tables.load_lm_v(domain)` / `tables.load_lm_v_3act(domain)`.
- The effort 2-action pipeline stipulates V=1 for both actions in `utility.get_stipulated_reward_effort` because reward is held fixed at HIGH and both actions involve eating, so V is uniform by construction. With V uniform across actions, `w_v` is non-identified under the softmax — it's kept in the utility for parallelism but stays near initialization during fitting.
- "Reward" appears in code (`reward_condition`, `param_w_v`); "motivation" appears in the data CSVs (`motivation` column with values `low`/`high`); both refer to the same underlying motivational state. The paper-facing word is **desire**.

## Shared infrastructure

- `forward/_shared.py` — NLL/AIC/BIC, `_fit_with_adam`, predict/fit functions, data loaders, `build_canonical_cells(scenario_labels, n_actions)`.
- `inverse/_helpers.py` — observer fit loops, NLL functions, data loaders for the 5 active 3-action inverse experiments plus the 2 legacy `_noalt` experiments, frozen-param loaders, table-kwargs helpers (`intimacy_3act_table_kwargs`, `effort_3act_table_kwargs`, `desire_3act_table_kwargs`, `joint_3act_table_kwargs`).
- `cv/_forward_dispatcher.py` — joint LOSO logic shared by per-experiment forward CV scripts.
- `lm/_features_dispatcher.py` — multi-mode internals shared by `score_canonical_features.py`.
- `lm/client.py`, `lm/prompts.py` — LM-call infrastructure and prompt templates.

## Tests

```bash
uv run python model/test_model_compliance.py
```

## Sandbox

- `model/sandbox/` — exploratory/comparison code (not part of the main pipeline).

Preregistration documents are at the repo root under [`preregs/`](../preregs/), one file per active experiment slug.
