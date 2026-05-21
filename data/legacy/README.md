# Legacy data

Older experiment data that is no longer part of the active pipeline. The processed CSVs and `raw_data/` directories are kept here so that legacy fits, predictions, CV runs, and analysis qmds remain reproducible, but `make all` no longer touches them.

## Archived inverse-planning experiments (2025–2026)

Six food-domain inverse-planning experiments collected against the original 4-action canonical set (`scenarios.csv`) and 2-action effort set (`scenarios_effort.csv`). These were superseded when the manuscript's experimental structure was reorganized around a single 3-action canonical set (`scenarios_3act.csv`) and multi-factor crossings for Studies 2, 3a, 3b, 4a, and 4b.

| Slug | Inferred | Conditioning | Action set |
|---|---|---|---|
| `food_inv_intimacy_desire_alt`    | intimacy | desire        | 4 actions, alternatives shown |
| `food_inv_desire_intimacy_alt`    | desire   | intimacy      | 4 actions, alternatives shown |
| `food_inv_intimacy_desire_noalt`  | intimacy | desire        | 4 actions, single action shown (LM counterfactuals) |
| `food_inv_desire_intimacy_noalt`  | desire   | intimacy      | 4 actions, single action shown (LM counterfactuals) |
| `food_inv_intimacy_effort_alt`    | intimacy | effort        | 2 actions, alternatives shown |
| `food_inv_effort_intimacy_alt`    | effort   | intimacy      | 2 actions, alternatives shown |

The model scripts (`model/inverse/{fit,predict}_<slug>.py`, `model/cv/cv_<slug>.py`) and analysis qmds (`analysis/food-inv-*.qmd`, `analysis/inv-plan-combined-correlation*.qmd`) for these experiments are still in the repo and remain runnable: their hardcoded data paths point to `data/legacy/<slug>/`, and the `Makefile` registers per-slug targets (`fit-<slug>`, `predict-<slug>`, `cv-<slug>`, `analysis-<name>`) outside of `EXPERIMENTS_ALL`. Per-CSV column documentation is in [data/README.md](../README.md).

## Earlier pilots and side projects

- `pilots/` — early pilots collected while iterating on the experimental design (discomfort, effort, forw_plan, planning_priors, planning-1, risk).
- `planning_comm/` — a parallel side project on planning and communication.

The whole `data/legacy/` directory is covered by the `legacy` gitignore rule at the repo root.
