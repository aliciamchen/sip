# Model pipeline

The `model/` directory contains the code that builds the model inputs, fits the
models, evaluates their predictions, and compares the model variants. Each of
the six studies is fit separately.

## Workflow

```text
language-model elicitation (`lm/`)
    -> ratings of possible actions (`outputs/lm/<slug>/`)
    -> model fitting (`inverse/`)
    -> leave-one-scenario-out cross-validation (`cv/`)
    -> model comparisons and figure inputs (`outputs/<slug>/`)
```

The language-model ratings are included in the repository. They do not need to
be regenerated in order to fit or evaluate the models.

## Running the models

The `Makefile` is the recommended way to run the pipeline:

```bash
make fit                 # fit all six studies
make cv                  # run cross-validation for all six studies
make model-comparison    # compare models using held-out predictions
```

The same stages can be run for one study by adding its slug:

```bash
make fit-food_inv_desire
make cv-food_inv_desire
```

Cross-validation leaves out one scenario at a time, refits the model on the
other 15 scenarios, and predicts responses for the held-out scenario. The
figures and model comparisons use these held-out predictions.

Regenerating the language-model ratings is a separate, optional step. It
requires `TOGETHER_API_KEY` in a `.env` file and incurs API costs:

```bash
make lm
```

## Directory contents

- `lm/` contains the prompts and scripts that generate possible actions and
  rate their risk, effort, and goal satisfaction.
- `inverse/` contains the shared fitting code and one script for each study.
- `cv/` contains the shared cross-validation code, one script for each study,
  and the scripts that compare models across studies.
- `actors.py` defines the model's action choices, and `observers.py` defines
  how an observer updates their beliefs after seeing an action.
- `memo_spec.py` expresses the same models in the memo probabilistic
  programming language. The tests compare this specification with the JAX
  code used for fitting.
- `tables.py` loads the language-model ratings into the arrays used by the
  models.
- `outputs/` contains the language-model ratings, fits, cross-validation
  results, and model comparisons. See the [model outputs
  codebook](outputs/README.md).

## Tests

Run the complete test suite from the repository root:

```bash
make test
```

To run only the tests that compare the model implementations:

```bash
uv run python model/test_model_compliance.py
```
