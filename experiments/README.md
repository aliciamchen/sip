# Experiments

The `experiments/` directory contains the six jsPsych studies and the scenario
text they use.

## Study designs

| Directory | Study | Conditions shown to participants | Ratings collected |
|---|---|---|---|
| `food_inv_desire/` | 1a | effort, intimacy, and observed action | desire |
| `food_inv_joint_de/` | 1b | intimacy and observed action | desire and effort |
| `food_inv_intimacy/` | 2a | desire, effort, and observed action | intimacy |
| `food_inv_joint_ie/` | 2b | desire and observed action | intimacy and effort |
| `nonfood_inv_joint_de/` | 3a | intimacy and observed action | desire and effort |
| `nonfood_inv_joint_ie/` | 3b | desire and observed action | intimacy and effort |

Studies 1a through 2b use scenarios about food sharing. Studies 3a and 3b use
the same joint-inference designs with non-food scenarios involving shared
objects, spaces, or information.

## Scenario files

The scenario text is defined in two Python files:

- `scenarios.py` defines the food scenarios used in Studies 1a through 2b.
- `scenarios_nonfood.py` defines the non-food scenarios used in Studies 3a and
  3b.

These scripts generate `scenarios.csv` and `scenarios_nonfood.csv`. The build
scripts then turn the CSV files into the `json/stimuli.json` file in each study
directory. Edit the Python files rather than the generated CSV or JSON files.

Run the complete experiment build with:

```bash
make experiments
```

This command regenerates the scenario CSV files, each study's stimuli and
counterbalancing files, and the shared entry files. To check whether the
generated files are up to date, run:

```bash
make check-experiments
```

The check regenerates the files and reports anything that changed, leaving the
updated files available for review. `make help` lists commands for rebuilding
only one type of file or one study.

## Shared experiment code

The study directories contain their study-specific trial code. Code used by
all six experiments is in `_lib/`, including the consent flow, instructions,
comprehension and attention checks, memory checks, survey pages, and shared
styles. The generated `index.html` and `experiment.js` files connect each
study's trials to this shared code.

The `build/` directory contains the scripts that generate experiment files:

- `csv_to_json.py` creates each study's `json/stimuli.json`.
- `counterbalancing.py` creates each study's
  `json/full_counterbalancing.json`.
- `sync_entry_files.py` creates each study's `index.html` and `experiment.js`.

The experiment folders therefore need `_lib/` in order to run; they are not
independent copies of the shared code.

## Previewing the studies

The preview page displays any study, scenario, and condition without recording
data. Start it with:

```bash
make preview
```

Then open `http://localhost:8000/preview/` in a browser.
