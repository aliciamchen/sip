# `model/` — modeling pipeline

Every script in this folder is named after either the experiment it serves (fit/CV) or the LM output it produces. There are six inverse-planning studies, all on the same 3-action stimulus structure — four on the food scenario set, plus two non-food studies that repeat the joint designs on the non-food set. The [per-experiment table below](#per-experiment-files) maps each slug to its fit/CV scripts; the slug ↔ study mapping is in the [root README](../README.md#experiments).

## Pipeline at a glance

```
LM elicitation  (lm/)   — K = 20 independent samples per cell (the elicitation-sample mixture)
    generate_alternatives.py --study <slug>  →  outputs/lm/<slug>/lm_alternatives.jsonl
    score_merged.py          --study <slug>  →  outputs/lm/<slug>/lm_runs.jsonl
        ↓
Fits  (inverse/)
    fit_<slug>.py  →  outputs/<slug>/fit_results.json  (+ fit_restarts.jsonl)
        ↓
Leave-one-scenario-out CV  (cv/)   — the model's predictions, all out-of-sample
    cv_<slug>.py  →  outputs/<slug>/cv_trial_ll.jsonl + cv_preds_summary.json + cv_folds.jsonl
        ↓
Model comparison  (cv/model_comparison.py)
    →  outputs/<slug>/cv_model_comparison.json  (the paper's statistics)
```

## Per-experiment files

All six studies infer one or two latent variables from a single observed action; the observer reasons over `{observed action} ∪ LM-generated alternatives`. The dependent measure is the **belief update** (posterior − prior rating). Each elicitation run k yields a model update `δ_k`, and a participant's update is scored under the K-component Gaussian mixture `(1/K) Σ_k N(u | δ_k, σ²)` with a fitted response-noise `σ` (bivariate with isotropic σ for the joint studies). The fitted parameters are the ablation's utility weights, the observer temperature `α_observer`, and `σ`; the primary model-comparison metric is per-trial held-out log-likelihood under leave-one-scenario-out CV.

| Slug | Study | Infers | Fit | CV |
|---|---|---|---|---|
| `food_inv_desire`   | 1a | desire            | `inverse/fit_food_inv_desire.py`   | `cv/cv_food_inv_desire.py` |
| `food_inv_joint_de` | 1b | desire + effort   | `inverse/fit_food_inv_joint_de.py` | `cv/cv_food_inv_joint_de.py` |
| `food_inv_intimacy` | 2a | intimacy          | `inverse/fit_food_inv_intimacy.py` | `cv/cv_food_inv_intimacy.py` |
| `food_inv_joint_ie` | 2b | intimacy + effort | `inverse/fit_food_inv_joint_ie.py` | `cv/cv_food_inv_joint_ie.py` |
| `nonfood_inv_joint_de` | 3a | desire + effort   | `inverse/fit_nonfood_inv_joint_de.py` | `cv/cv_nonfood_inv_joint_de.py` |
| `nonfood_inv_joint_ie` | 3b | intimacy + effort | `inverse/fit_nonfood_inv_joint_ie.py` | `cv/cv_nonfood_inv_joint_ie.py` |

Run any script directly as `uv run python <path>`, or via `make fit-<slug>` / `make cv-<slug>`. Per-experiment scripts are thin wrappers: shared logic lives in `inverse/_helpers.py` and `cv/_inverse_dispatcher.py`, and each wrapper calls the shared main with its slug hardcoded. The non-food studies reuse the joint studies' observers and helpers wholesale — both stimulus sets have 16 scenarios, so only the scenario labels and the LM-table folder differ. A fit runs only once its study's data is in `data/<slug>/` and its LM tables have been elicited (otherwise it raises a `FileNotFoundError` naming the elicitation commands to run).

## Layout

- `tables.py` — enums and axes, plus the loaders that assemble `outputs/lm/<slug>/lm_runs.jsonl` into the padded per-study feature tables. A missing or failed rating raises an error at load time rather than flowing silently into a fit.
- `utility.py` — jit-compiled utility functions implementing `w_v · d · g − w_d · risk · (1 − I)^γ − w_e · effort` and its ablations (see the [utility model](../README.md#utility-model)).
- `actors.py` / `observers.py` — the actor policies and their Bayesian observers, one family per study, each in `full` / `discomfort_only` / `base` variants. Both are plain JAX: the actor is a softmax policy over the padded action space, and every observer conditions on the observed action and returns the posterior over the study's latent(s) by direct Bayesian inversion of the actor policy, computed in log space for numerical stability. The joint studies return a joint posterior that downstream code marginalizes to the two sliders.
- `memo_spec.py` — the same actors and observers written as [memo](https://github.com/kach/memo) probabilistic programs. This is the executable specification of the model, and the test suite verifies the production code against it on every variant. The fits and cross-validation do not run it: the memo observers needed several gigabytes of intermediate memory per gradient step and lose float32 precision at large observer sharpening, which is why the plain-JAX forms in `actors.py` and `observers.py` exist.
- `inverse/_helpers.py` — mixture likelihoods, data loaders, the Adam multi-start fit loop, and the per-study fit entry points.
- `cv/_inverse_dispatcher.py` — the leave-one-scenario-out loops (each fold refits on 15 scenarios with a warm start from the full-data fit plus a cold restart, keeping the better optimum).
- `cv/model_comparison.py` — the paper's statistics from the CV outputs: full − ablation per-trial held-out log-likelihood with participant-bootstrap 95% CIs, plus the secondary model-vs-human correlations.
- `lm/` — elicitation scripts, prompt templates (`prompts.py`), and the shared Together AI client (`client.py`).

The output files are documented field-by-field in the [outputs codebook](outputs/README.md). Finer-grained implementation notes (cell grids, table shapes, run-axis handling) are in [`.claude/rules/model.md`](../.claude/rules/model.md) and the module docstrings.

## Tests

```bash
make test                                        # the full suite
uv run python model/test_model_compliance.py     # just the model checks
```

The model checks cover the utility ablation algebra, observer posterior normalization (single and joint), the mixture likelihoods against a plain-numpy reference, a bound on the probability mass reaching null padding slots, and the table loaders' error checks on missing or failed ratings. `make test` adds the fit/CV protocol, checkpoint, statistics-module, elicitation-guard, data-conversion, and experiment-list tests.
