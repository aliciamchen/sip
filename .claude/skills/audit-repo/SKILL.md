---
name: audit-repo
description: Run a state-of-the-repo audit across the active inverse-planning experiments, the model pipeline, and the analysis qmds. Reports broken scripts, stale references, drift across per-experiment scripts, output CSV consistency, test status, orphaned files, and gitignore correctness — grouped by severity. Use after large refactors or before submission.
allowed-tools: Bash, Read, Grep, Glob
---

# Repo state audit

Run a thorough audit of this repository. The active roster is six inverse-planning studies on the 3-action structure — `food_inv_desire` (1a), `food_inv_joint_de` (1b), `food_inv_intimacy` (2a), `food_inv_joint_ie` (2b) on the food scenario set, plus `nonfood_inv_joint_de` (3a) and `nonfood_inv_joint_ie` (3b) on the nonfood set (`scenarios_nonfood.csv`). All six are live and complete — Study 3 collected in July 2026 and was folded into `EXPERIMENTS_INVERSE`, so all six have participant data, LM tables, fits and CV, and a *missing* `data/<slug>/` or `model/outputs/<slug>/` is now a finding rather than an expected gap. The roster is fed by a multi-stage pipeline (LM elicitation → fit → CV → analysis; CV is the sole source of model predictions, all out-of-sample). The audit catches things that broke during recent refactors and surfaces drift before it compounds.

For broad checks, delegate to the Explore subagent in parallel chunks. For targeted file reads or path checks, do them directly.

If this skill's own assumptions look stale against the repo (slugs, file names, expected columns), flag that as a finding too — it has drifted before.

## Checks to run

1. **Will every script run cleanly?** Spot-check imports, signatures, and paths in `model/{inverse,cv,lm}/*.py`. The per-experiment scripts are ~25-line wrappers carrying only a slug: `model/inverse/fit_<slug>.py` calls `_fit_dispatcher.main(slug)`, and `model/cv/cv_<slug>.py` calls `_inverse_dispatcher.main_<family>(slug)` (both dispatchers key off a `_FAMILIES` registry; `model/test_fit_protocol.py` enforces that no wrapper reimplements a protocol step) (plus the non-wrapper modules under `model/cv/`: `model_comparison.py` for the bootstrap comparison, `contrast_tests.py` for the variance decomposition and condition gradients it reports, `_checkpoint.py`, and the exploratory `transfer.py` / `pooled.py`); the LM scripts keep per-study registries (`_STUDY_CONFIG`) in `model/lm/generate_alternatives.py` and `score_merged.py`. Look for: calls with wrong kwargs, references to LM table paths that don't exist under `model/outputs/lm/<slug>/` (expected: `lm_runs.jsonl` — scored actions plus each run's given magnitude (`desire`/`intimacy`) — and `lm_alternatives.jsonl` (stage-1 texts); the given-relationship studies 1a/1b additionally get `lm_runs_base.jsonl` + `lm_alternatives_base.jsonl`, the relationship-free base-ablation set; the legacy `lm_scenario.csv`/`lm_alternatives.csv`/`lm_scenario_desire.csv` are kept only as the K=1 fallback in the table loaders), and references to removed code (forward pipeline, `_3act` slugs, `_alt`/`_noalt` actives, `score_features.py`, `lm_scenario_params_marginal`, the categorical `compute_{desire,intimacy,effort}_nll` losses, `NUM_RUNS` rating-averaging in `score_merged.py`, the dropped `predict_<slug>.py` scripts / `preds_summary.json`, and `lm_given.json` / `load_fit_results` reading a separate given file).

2. **Drift across the per-experiment scripts.** Each of the six experiments has fit and CV wrappers (there is no predict wrapper — CV is the sole prediction source). Flag any wrapper that does more than pass its slug through to the shared logic (extra steps, missing params, hardcoded values), and any study missing from shared-module docstrings or registries.

   **`study_registry.py` (repo root) is the single source of per-study metadata** — given conditions, inferred latents with their `<rating>_update` / `delta_<latent>` column pairs, paper label, and stimulus domain. `model/cv/model_comparison.py`, `model/export_results_latex.py`, and the figure scripts all import it (it is also in the Makefile's `FIG_SHARED`). So a per-study fact hardcoded in a *consumer* rather than read from the registry is drift, even when its value is currently right.

3. **Output consistency (JSON/JSONL).** Sample `model/outputs/<slug>/fit_results.json` and `cv_folds.jsonl` across experiments; the PRIMARY model-comparison metric is `cv_trial_ll.jsonl` (per-trial held-out log-likelihood, keyed by `subject_id` for the participant bootstrap), with `cv_preds_summary.json` carrying the per-cell `delta_<latent>` predictions (out-of-sample; there is no in-sample `preds_summary.json` — predict was dropped) and `cv_model_comparison.json` the bootstrap statistics from `model/cv/model_comparison.py`. The `experiment` field should hold the slug in every file (including `cv_preds_summary.json`). Every fit carries all utility weights plus `alpha_observer` and `param_sigma` — the `param_<name>` spelling (incl. `param_sigma`) is shared by `fit_results.json` and `cv_folds.jsonl` (`param_w_v`, `param_w_d`, `param_w_e`, `param_gamma` — the reward-term weight is intentionally named `w_v`, not a typo for `w_d`). Also scan fitted values for parameters pinned at the 1e-6 lower bound and report them: a boundary fit means that term collapsed out. Caveat: `discomfort_only` on a desire-inference study (1a/1b) and `base` on an intimacy-inference study (2a/2b) have likelihoods that are FLAT in some or all of their params (their posteriors cannot move), so those fitted values are unidentifiable leftovers of the init — they may sit at the floor, near 1, or anywhere; only flag boundary values in variants whose params are actually identified.

   The variant set is no longer just base / discomfort_only / full: `base_shared` (same utility and observer as `base`, but on full's comparison set, so it isolates the utility term from the comparison set) and `base_prereg` also exist. Which variant the paper's "Base" column means is per study and comes from `study_registry.reported_base(slug)` — never assume it is `base`.

4. **Tests pass.** `make test` — the whole suite must exit 0 (11 files: `model/test_model_compliance.py`, `test_run_config.py`, `test_fit_protocol.py`, `model/cv/test_{checkpoint,model_comparison,contrast_tests,transfer,pooled}.py`, `model/lm/test_elicitation_guards.py`, `analysis/test_json_to_csv.py`, `test_roster_sync.py`). Also confirm each test file is tracked (`git ls-files` non-empty); a broad gitignore pattern once silently excluded `test_model_compliance.py`.

5. **Orphaned files.** Survey `preregs/`, `experiments/pilots/`, `model/model_viz_nonfit/`. For each, grep for references in tracked code/docs and flag content not referenced anywhere (`data/legacy/` is entirely local-only and gitignored — flag it if anything tracked starts reading it again). For `preregs/`, flag prereg files whose slug is no longer in the active roster, and active slugs that have no prereg.

6. **Doc-vs-code drift.** Check the last few commits for refactors (file renames, schema changes, slug changes). Then grep all .md/.qmd files — including `.claude/CLAUDE.md`, `.claude/rules/*.md`, every `.claude/skills/*/SKILL.md`, and this skill — for the old patterns. Exclude `.claude/worktrees/`: those are checkouts of old branches, stale by design, and they swamp the results. Patterns worth checking after past refactors: forward-pipeline script names, `_3act` slugs, "reward"/"motivation"/"access" terminology in active docs (the data columns and `param_w_v` are intentional exceptions; `data/legacy/` keeps old names by design), old LM CSV names, and the June-2026 CSV→JSON/JSONL output migration (`fit_results.csv`→`.json`; `cv_folds.csv`→`.jsonl`; `cv_preds_summary.csv`→`.json`; `lm_scenario.csv`+`lm_alternatives.csv`→`lm_runs.jsonl`; `lm_alternatives.csv`→`lm_alternatives.jsonl` for the stage-1 texts; the per-run given-magnitude scalars are folded into `lm_runs.jsonl`, so `lm_scenario_desire.csv` and the former `lm_given.json` are gone; the non-CV predict stage — `predict_<slug>.py`, `preds_*.npy`, `preds_summary.csv`/`.json` — was dropped entirely, CV being the sole prediction source) plus the belief-update mixture / fitted-σ rewrite (categorical NLLs and per-rating `NUM_RUNS` averaging are gone).

7. **Git state.** `git status`, large untracked files, branches diverged from origin/main, stale worktrees under `.claude/worktrees/`. Note anything weird without acting on it.

8. **Gitignore correctness.** Anything tracked that shouldn't be (raw_data, large binaries, secrets, build artifacts) — or vice versa: files that docs claim are tracked but `git ls-files` doesn't know about. Check `git ls-files | grep -E '\.json$' | grep raw_data` returns zero, sample subject_id columns to confirm anonymization (UUIDs, not Prolific IDs), and `git check-ignore` anything suspicious. Be suspicious of broad patterns (a bare `*test*`-style glob has bitten this repo before).

9. **Anything else surprising.** Half-finished implementations, scattered TODOs, dead branches, files that look like debugging leftovers.

## Output format

Group findings by severity:

- 🔴 **BROKEN** — will fail when run, or data/code invisible to git
- 🟡 **STALE / DRIFT** — incorrect docs, dead code, naming inconsistency
- 🟢 **NICE-TO-HAVE** — improvements but not problems

For each finding, give: short description, file path, line number (when applicable), evidence. Skip categories where nothing notable was found. Don't pad with reassurances. Aim for ~600–800 words.

After reporting, ask the user which findings to act on rather than auto-fixing.
