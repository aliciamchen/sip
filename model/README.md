# `model/` — modeling pipeline

Every script in this folder is named after either the experiment it serves (fit/CV) or the LM output it produces. No multi-experiment dispatchers in script names; no `--feature` flags hiding what a script does.

The roster is four inverse-planning studies, all on the 3-action stimulus set and all using the LM-generated-alternatives padded-action pipeline: `food_inv_desire` (Study 1a), `food_inv_joint_de` (1b), `food_inv_intimacy` (2a), `food_inv_joint_ie` (2b).

## Pipeline at a glance

```
LM elicitation  (model/lm/)   — K independent runs per cell (the simulated-observer mixture)
    generate_alternatives.py --study <slug>  →  outputs/lm/<slug>/lm_alternatives.jsonl  (per-run counterfactual texts; run_id field)
    score_merged.py          --study <slug>  →  outputs/lm/<slug>/lm_runs.jsonl  (one record per run × cell: every action's risk/effort/g
                                                  + that run's given magnitude — desire for 2a/2b, intimacy for 1a/1b — all scored once per run)
        ↓
Inverse planning  (model/inverse/)       Studies 1a, 1b, 2a, 2b
    fit_<slug>.py  → outputs/<slug>/fit_results.json  (fitted weights + α_observer + σ; + fit_restarts.jsonl)
        ↓
Cross-validation  (model/cv/)   — the model's predictions, all out-of-sample
    cv_<slug>.py → outputs/<slug>/cv_trial_ll.jsonl  (primary metric: per-trial held-out log-likelihood, keyed by subject_id)
                   + cv_preds_summary.json (per-cell delta_<latent>) + cv_folds.jsonl
        ↓
Analysis qmds   (analysis/<slug>-analysis.qmd)
```

## Per-experiment files

All four studies infer one or two latent variables from a single observed action; the observer reasons over a padded action space (`{observed action} ∪ LM-generated alternatives`) and the fit/CV slice slot 0. The dependent measure is the **belief update** (posterior − prior rating). Each elicitation run k yields a model update `δ_k = posterior mean − prior mean`, and a participant's update is scored under the K-component Gaussian mixture `(1/K) Σ_k N(u | δ_k, σ²)` with a fitted response-noise `σ` (a single isotropic σ for the joint studies). The fitted parameters are the utility weights, `α_observer`, and `σ`; the primary model-comparison metric is per-trial held-out log-likelihood under leave-one-scenario-out CV.

| Slug | Study | Infers | Fit | CV |
|---|---|---|---|---|
| `food_inv_desire`   | 1a | desire           | `inverse/fit_food_inv_desire.py`   | `cv/cv_food_inv_desire.py` |
| `food_inv_joint_de` | 1b | desire + effort  | `inverse/fit_food_inv_joint_de.py` | `cv/cv_food_inv_joint_de.py` |
| `food_inv_intimacy` | 2a | intimacy         | `inverse/fit_food_inv_intimacy.py` | `cv/cv_food_inv_intimacy.py` |
| `food_inv_joint_ie` | 2b | intimacy + effort| `inverse/fit_food_inv_joint_ie.py` | `cv/cv_food_inv_joint_ie.py` |

The model's per-cell predictions come from CV (out-of-sample); there is no separate
in-sample predict step.

Run any script directly as `uv run python <path>`, or via `make fit-<slug>` / `make cv-<slug>`. The CV scripts are thin wrappers around the LOSO mains in `cv/_inverse_dispatcher.py`. The fit scripts run only once data lands in `data/<slug>/` and the per-study LM alternatives have been elicited (otherwise the table-kwargs helpers raise a clear `FileNotFoundError`).

### Why dispatchers?

Logic shared across experiments (the LOSO loops in `cv/`, the multi-mode helpers in `lm/`) lives in `_dispatcher.py` / `_helpers.py` files. Each per-experiment script is a thin wrapper: it imports the shared main and calls it with the experiment slug hardcoded.

## Core math (one copy, shared across all experiments)

- `tables.py` — enums (`Scenarios`, `DesireConditions`, `RelationshipConditions`, `EffortConditions`, `IntimacyLevels`, `DesireLevels`, `PaddedActionSlots`, `ObservedActions`), the `actions` array, `SCENARIO_LABELS`, the per-study padded LM-alternatives loaders (`load_padded_lm_tables_{desire,joint_de,intimacy,joint_ie}`, each reading `lm_runs.jsonl` into tables that carry a leading elicitation-run axis), and the given-magnitude scalar loaders `load_lm_scenario_desire` (per-condition desire, 2a/2b) / `load_lm_relationship_values` (per-level intimacy, 1a/1b), both reading the per-record given field of `lm_runs.jsonl` with the same leading run axis, so the given magnitudes vary run-to-run alongside the features.
- `utility.py` — jit-compiled utility functions: the padded families `get_utility_{full,discomfort_only,base}_padded_{desire,joint_de,intimacy,joint_ie}` plus their `get_prior_padded_*` and `get_lm_g_padded_*` helpers. The reward term is `w_v · desire · g`.
- `actors.py` — actor memo models: the padded inverse actors `actor_discrete_*_padded_{desire,joint_de}` (discrete observed intimacy) and `actor_continuous_*_padded_{intimacy,joint_ie}` (continuous inferred intimacy), used inside the observers' `thinks[...]` blocks.
- `observers.py` — observer memos, one family per study, each in `_full` / `_discomfort_only` / `_base`:
  - `observer_desire_*` — Study 1a (infers desire, 101-bin continuous posterior)
  - `observer_joint_de_*` — Study 1b (joint posterior over desire × effort, given intimacy)
  - `observer_intimacy_*` — Study 2a (infers intimacy, 101-bin continuous posterior)
  - `observer_joint_ie_*` — Study 2b (joint posterior over intimacy × effort, given desire)

  Joint observers use memo's `chooses(x in X, y in Y, ...)` multi-choice syntax and return `Pr[a, b]`; downstream code marginalizes for the per-slider predictions.

### Terminology: g, desire, reward, risk

The utility model and the meaning of `g`, `desire`, `risk`, and the `w_v · desire · g` reward term are defined in [README.md](../README.md#utility-model); the `w_v`-vs-`w_d` naming convention (`w_v` kept even though the concept is desire) is in [`.claude/CLAUDE.md`](../.claude/CLAUDE.md). One model-implementation detail not in those:

- **Intimacy magnitude** `I ∈ [0, 1]` for the four relationship levels is LM-elicited per run (`load_lm_relationship_values` ← the per-record `intimacy` field of `lm_runs.jsonl`), mirroring the per-condition desire scalar in 2a/2b; it's passed into the desire/joint_de observer memos as `relationship_values` (sliced per run), falling back to the placeholder `RELATIONSHIP_LEVEL_VALUES` until the elicitation has been run.

## Shared infrastructure

- `inverse/_helpers.py` — `_fit_with_adam` / `_fit_multistart` (with optional warm-start `init_params`), the belief-update Gaussian-mixture losses (`mixture_nll_1d`, `mixture_nll_2d`) plus `posterior_mean` / `PRIOR_MEAN` / `EFFORT_PRIOR_MEAN`, per-study data loaders (returning per-trial belief updates), `_build_observer_tables_runs` (runs the observer once per elicitation run and stacks the posteriors on a leading K axis), the joint observer fit loops (`fit_{desire,joint_de,intimacy,joint_ie}_observer_joint`, fitting weights + `α_observer` + `σ`), and the padded table-kwargs helpers (`desire_table_kwargs`, `joint_de_table_kwargs`, `intimacy_table_kwargs`, `joint_ie_table_kwargs`).
- `cv/_inverse_dispatcher.py` — LOSO logic for the four inverse studies (`main_{desire,joint_de,intimacy,joint_ie}`).
- `lm/_features_dispatcher.py`, `lm/_alternatives_dispatcher.py` — multi-mode internals shared by the scorers / alternative generation; `lm/client.py`, `lm/prompts.py` — LM-call infrastructure and prompt templates.

## Tests

```bash
uv run python model/test_model_compliance.py
```

Preregistration documents are at the repo root under [`preregs/`](../preregs/); Study 1a's is present, with 1b/2a/2b pending.
