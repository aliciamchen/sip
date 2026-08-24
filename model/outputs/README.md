# Model outputs codebook

The `model/outputs/` directory contains the language-model ratings used as
model inputs and the results produced by model fitting and cross-validation.
Files are grouped by study slug. Large collections of records use JSON Lines
(`.jsonl`), with one JSON object per line; smaller summaries use regular JSON.

## Directory structure

```text
outputs/
|-- lm/
|   `-- <slug>/
|       |-- lm_alternatives.jsonl
|       |-- lm_runs.jsonl
|       |-- lm_alternatives_base.jsonl   # some studies only
|       |-- lm_runs_base.jsonl           # some studies only
|       `-- *.manifest.json
`-- <slug>/
    |-- fit_results.json
    |-- fit_restarts.jsonl
    |-- fit_manifest.json
    |-- cv_trial_ll.jsonl
    |-- cv_preds_summary.json
    |-- cv_folds.jsonl
    |-- cv_manifest.json
    `-- cv_model_comparison.json
```

The language-model ratings and the main fit and cross-validation results are
included in the repository. This allows readers to inspect the reported
results and regenerate figures without repeating the paid language-model
elicitation.

## Language-model ratings

The model uses a language model to suggest other actions the characters could
have taken and to rate the observed and alternative actions. Ratings of risk,
effort, and goal satisfaction are collected on a scale from 0 to 6 and
converted to values between 0 and 1 before model fitting. Each scenario and
condition is elicited 20 times so that the analysis does not depend on a
single language-model response.

### `lm_alternatives.jsonl`

This file contains the alternative actions generated for each scenario,
condition, and elicitation run. Important fields include:

| Field | Description |
|---|---|
| `scenario_label` | The scenario name. |
| `observed_action` | The action shown in the experiment. |
| `run_id` | The elicitation run, from 0 to 19. |
| `alt_idx` | The alternative's position within that run. |
| `action_text` | The generated alternative action. |
| `is_share` | Whether the action involves sharing. |

The condition columns differ by study because participants are given different
information in different studies.

### `lm_runs.jsonl`

This file contains the rated actions used by the fitted models. Each record
represents one elicitation run for one scenario and condition. Its `actions`
list contains the observed action followed by the generated alternatives.

Each action has an `action_text`, an `is_observed` indicator, and ratings for
`risk`, `effort`, and `g`, where `g` is goal satisfaction. Studies in which
desire or intimacy is given to participants also include the language model's
numeric rating of that condition.

### Base-model files

Studies 1a, 1b, and 3a also have `lm_alternatives_base.jsonl` and
`lm_runs_base.jsonl`. These files contain actions elicited without a
relationship description for the preregistered base model.

### Manifest files

Each elicited file has a neighboring `*.manifest.json` file. The manifest
records whether the elicitation finished, which language model and settings
were used, when it ran, and identifiers for the prompts and input files. These
records make it possible to tell whether two output files came from the same
elicitation.

The `*.rationale.jsonl` files preserve the language model's complete response
from the alternative-generation step. They are included for inspection but
are not read by the fitted models.

## Fitted models

Each study's main results are in `outputs/<slug>/`. The parameters are fit to
the change between each participant's prior and posterior rating.

### `fit_results.json`

This file contains one record for each fitted model variant. All studies have
`full`, `discomfort_only`, and `base` variants. Studies in which intimacy is
given also have `base_shared`, which uses the base model with the full model's
relationship-specific set of possible actions.

| Field | Description |
|---|---|
| `model` | The model variant. |
| `experiment` | The study slug. |
| `nll` | The negative log-likelihood for the fit. |
| `n_params` | The number of fitted parameters. |
| `param_alpha` | The actor's fixed choice parameter. |
| `alpha_observer` | The fitted strength of the observer's belief update. |
| `alpha_observer_at_bound` | Whether `alpha_observer` reached its allowed upper limit. |
| `param_sigma` | The fitted amount of response noise. |
| `param_w_v` | The weight on goal satisfaction. |
| `param_w_d` | The weight on risk or discomfort. |
| `param_w_e` | The weight on effort. |
| `param_gamma` | How strongly intimacy changes the cost of risk. |
| `param_eta` | The fitted comparison-set reweighting strength, when reweighting applies. |
| `reweighting_targets` | The action categories reweighted by `param_eta`, when reweighting applies. |

A record contains only the parameters and reweighting fields used by that
model variant.

### `fit_restarts.jsonl`

The fitting procedure starts from several initial parameter values. This file
records the initial and final parameters and the negative log-likelihood for
each attempt. `fit_results.json` contains the best result.

### `fit_manifest.json`

This file records when the fit ran, the Git commit used, and checksums for the
fit results, participant data, and language-model ratings. Cross-validation
checks this file before using a fit.

## Cross-validation

The reported predictions come from leave-one-scenario-out cross-validation.
For each of the 16 scenarios, the model is fit to the other 15 and then used to
predict responses to the held-out scenario.

### `cv_trial_ll.jsonl`

This is the main input to the model comparisons. Each record contains one
participant trial's held-out log-likelihood under one model variant.

| Field | Description |
|---|---|
| `experiment` | The study slug. |
| `model` | The model variant. |
| `subject_id` | The anonymized participant ID. |
| `scenario_label` | The scenario left out during fitting. |
| `held_out_ll` | The log-likelihood of that participant's response. |

### `cv_preds_summary.json`

This file contains the model's predicted belief change for every held-out
scenario and condition. The predicted means are stored as `delta_desire`,
`delta_intimacy`, or `delta_effort`, depending on the study. The corresponding
`delta_*_runs` fields contain the 20 individual predictions whose mean is
reported. Figure scripts use this file for model predictions.

### `cv_folds.jsonl`

This file records the fitted parameters and training and test loss for each
held-out scenario and model variant. It is useful for checking individual
cross-validation fits.

### `cv_manifest.json`

This file records when cross-validation ran, the Git commit used, and checksums
for the cross-validation outputs and their input data. The model-comparison
code uses it to avoid combining files from different runs.

### `cv_model_comparison.json`

This file contains the statistics reported from the cross-validated results,
including:

- differences in held-out log-likelihood between the full model and its
  comparison models;
- confidence intervals based on resampling participants;
- correlations between model predictions and participants' average belief
  changes; and
- the additional contrasts and reliability estimates reported in the
  manuscript.

Run `make model-comparison` to regenerate these files from the cross-validation
outputs.

## Additional analyses

Alternative model settings are written under each study's `alt/` directory so
that they do not replace the main results. The exploratory transfer, pooled,
and generalization analyses also write summaries under `model/outputs/`. Run
`make help` for the commands that generate them.
