---
paths:
  - "experiments/**/*"
---

# Experiment structure

Each study directory contains generated `index.html` and `experiment.js` entry files plus a thin, study-specific `trials.js`. Shared behavior belongs in `experiments/_lib/`; do not copy it into study directories.

Key shared sources are:

- `_lib/bootstrap.js` for the full timeline.
- `_lib/config.js` for `DATAPIPE_IDS`, shared timing, and completion settings.
- `_lib/instructions.js` and `_lib/comprehension-check.js` for study-specific text and questions.
- `_lib/two-slider.js` for Studies 1b, 2b, 3a, and 3b.
- `build/sync_entry_files.py` for the generated entry files.

The comprehension gate allows three attempts and aborts before the DataPipe save when a participant never passes. Each active slug needs its own DataPipe ID. Study directories reference `../_lib/`, so `_lib/` must be deployed with them.

## Generated scenarios and assets

Edit `experiments/scenarios.py` or `scenarios_nonfood.py`, never their generated CSVs. The build flows from those Python sources to `scenarios*.csv`, then through `build/csv_to_json.py` into per-study `json/stimuli.json`. Counterbalancing files and entry files are generated too.

Run `make experiments` to regenerate assets and `make check-experiments` to fail if tracked artifacts were stale. A deploy runs the same drift guard automatically. Regenerate the manuscript's SI scenario tables separately with `uv run python experiments/export_scenarios_latex.py`.

## Preview and deploy

Use the Makefile or `bin/deploy-experiment`; do not assemble an ad hoc deployment.

```bash
make preview
bin/deploy-experiment <slug> --dry-run
bin/deploy-experiment <slug>
make deploy-all
```

`bin/deploy-experiment` accepts only active slugs and special preview modes, deploys the required shared library, and deletes stale remote files within the experiment target. Verify the resolved target with `--dry-run` before a consequential deploy.
