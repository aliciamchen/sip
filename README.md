# Inverse planning of social interactions in relationships

This repository contains the experiments, de-identified data, model code, and
figure code for the manuscript "Inverse planning of social interactions in relationships."

Preprint: [TODO]

## Reproducing the results

The project uses Python 3.12 or later, with dependencies managed by
[uv](https://docs.astral.sh/uv/). To fit the models, run cross-validation,
compute the model comparisons, and generate the main figure panels:

```bash
uv sync
make all
```

Model fitting and cross-validation can take a long time, but their outputs are
included in the repository. Because Git does not preserve file timestamps, a
fresh clone may rerun calculations whose outputs are already included. To use
the included outputs without recomputing them, run:

```bash
uv sync
make freshen-outputs
make all
```

`make freshen-outputs` updates only the timestamps used by `make`; it does not
change any results. Run `make help` to see commands for individual studies and
pipeline stages. Run the test suite with:

```bash
make test
```

## Studies

| Slug | Study | What participants infer | Scenarios |
|---|---|---|---|
| `food_inv_desire` | 1a | desire | food sharing |
| `food_inv_joint_de` | 1b | desire and effort | food sharing |
| `food_inv_intimacy` | 2a | intimacy | food sharing |
| `food_inv_joint_ie` | 2b | intimacy and effort | food sharing |
| `nonfood_inv_joint_de` | 3a | desire and effort | non-food sharing |
| `nonfood_inv_joint_ie` | 3b | intimacy and effort | non-food sharing |

## Repository contents

- `data/` contains the de-identified participant data. The
  [data codebook](data/README.md) describes the files and columns.
- `data_prep/` contains the script that converts raw data to the
  processed CSV files.
- `experiments/` contains the jsPsych experiments and the source files for the
  food and non-food scenarios. See the [experiments
  README](experiments/README.md).
- `model/` contains the language-model elicitation, model fitting,
  cross-validation, and model-comparison code. See the [model
  README](model/README.md).
- `model/outputs/` contains the language-model ratings, fitted parameters,
  cross-validated predictions, and model comparisons. The [outputs
  codebook](model/outputs/README.md) explains these files.
- `figures/` contains the plotting code and generated figure files. See the
  [figures README](figures/README.md).
- `preregs/` contains readable copies of the six preregistrations.

## Analysis workflow

```text
participant data       +
                       +-> model fits -> cross-validation -> model comparisons -> figures
language-model ratings +
```

The language-model ratings in `model/outputs/lm/` are included in the
repository. Regenerating them is optional, requires a Together AI key, and
incurs API costs. The `make lm` command runs that separate step.

The Python files in `experiments/` define the scenario text and generate the
CSV and JSON files used by the experiments. After editing a scenario, run
`make experiments` to regenerate the experiment files. The `make
check-experiments` command checks that the generated files match their Python
sources.
