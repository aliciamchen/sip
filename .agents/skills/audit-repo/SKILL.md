---
name: audit-repo
description: Run a state-of-the-repo audit across the active inverse-planning experiments, the model pipeline, and the analysis qmds. Reports broken scripts, stale references, drift across per-experiment scripts, output CSV consistency, test status, orphaned files, and gitignore correctness — grouped by severity. Use after large refactors or before submission.
allowed-tools: Bash, Read, Grep, Glob
---

# Repo state audit

Run a thorough audit of this repository. The active roster is six inverse-planning studies on the 3-action structure — `food_inv_desire` (1a), `food_inv_joint_de` (1b), `food_inv_intimacy` (2a), `food_inv_joint_ie` (2b) on the food scenario set, plus `nonfood_inv_joint_de` (3a) and `nonfood_inv_joint_ie` (3b) on the nonfood set (`scenarios_nonfood.csv`; no participant data or LM tables until Study 3 collection starts, so their missing `data/` and `outputs/` folders are expected, not findings) — fed by a multi-stage pipeline (LM elicitation → fit → CV → analysis; CV is the sole source of model predictions, all out-of-sample). The audit catches things that broke during recent refactors and surfaces drift before it compounds.

For broad checks, delegate to the Explore subagent in parallel chunks. For targeted file reads or path checks, do them directly.

If this skill's own assumptions look stale against the repo (slugs, file names, expected columns), flag that as a finding too — it has drifted before.

## Checks to run

1. **Will every script run cleanly?** Spot-check imports, signatures, and paths in `model/{inverse,cv,lm}/*.py`. The per-experiment scripts are thin wrappers: `model/inverse/fit_<slug>.py` imports from `model/inverse/_helpers.py`, and `model/cv/cv_<slug>.py` imports from `model/cv/_inverse_dispatcher.py` (plus `model/cv/model_comparison.py`, the bootstrap model-comparison script); the LM scripts keep per-study registries (`_STUDY_CONFIG`) in `model/lm/generate_alternatives.py` and `score_merged.py`. Look for: calls with wrong kwargs, references to LM table paths that don't exist under `model/outputs/lm/<slug>/` (expected: `lm_runs.jsonl` — scored actions plus each run's given magnitude (`desire`/`intimacy`) — and `lm_alternatives.jsonl` (stage-1 texts); the given-relationship studies 1a/1b additionally get `lm_runs_base.jsonl` + `lm_alternatives_base.jsonl`, the relationship-free base-ablation set; the legacy `lm_scenario.csv`/`lm_alternatives.csv`/`lm_scenario_desire.csv` are kept only as the K=1 fallback in the table loaders), and references to removed code (forward pipeline, `_3act` slugs, `_alt`/`_noalt` actives, `score_features.py`, `lm_scenario_params_marginal`, the categorical `compute_{desire,intimacy,effort}_nll` losses, `NUM_RUNS` rating-averaging in `score_merged.py`, the dropped `predict_<slug>.py` scripts / `preds_summary.json`, and `lm_given.json` / `load_fit_results` reading a separate given file).

2. **Drift across the per-experiment scripts.** Each of the four experiments has fit and CV wrappers (there is no predict wrapper — CV is the sole prediction source). Flag any wrapper that does more than pass its slug through to the shared logic (extra steps, missing params, hardcoded values), and any study missing from shared-module docstrings or registries.

3. **Output consistency (JSON/JSONL).** Sample `model/outputs/<slug>/fit_results.json` and `cv_folds.jsonl` across experiments; the PRIMARY model-comparison metric is `cv_trial_ll.jsonl` (per-trial held-out log-likelihood, keyed by `subject_id` for the participant bootstrap), with `cv_preds_summary.json` carrying the per-cell `delta_<latent>` predictions (out-of-sample; there is no in-sample `preds_summary.json` — predict was dropped) and `cv_model_comparison.json` the bootstrap statistics from `model/cv/model_comparison.py`. The `experiment` field should hold the slug in every file (including `cv_preds_summary.json`). Every fit carries all utility weights plus `alpha_observer` and `param_sigma` — the `param_<name>` spelling (incl. `param_sigma`) is shared by `fit_results.json` and `cv_folds.jsonl` (`param_w_v`, `param_w_d`, `param_w_e`, `param_gamma` — the reward-term weight is intentionally named `w_v`, not a typo for `w_d`). Also scan fitted values for parameters pinned at the 1e-6 lower bound and report them: a boundary fit means that term collapsed out. Caveat: `discomfort_only` on a desire-inference study (1a/1b) and `base` on an intimacy-inference study (2a/2b) have likelihoods that are FLAT in some or all of their params (their posteriors cannot move), so those fitted values are unidentifiable leftovers of the init — they may sit at the floor, near 1, or anywhere; only flag boundary values in variants whose params are actually identified.

4. **Tests pass.** `uv run python model/test_model_compliance.py` — must exit 0. Also confirm the test file is tracked (`git ls-files model/test_model_compliance.py` non-empty); a broad gitignore pattern once silently excluded it.

5. **Orphaned files.** Survey `preregs/`, `data/legacy/`, `experiments/pilots/`, `model/model_viz_nonfit/`. For each, grep for references in tracked code/docs and flag content not referenced anywhere. For `data/legacy/`, verify the tracked-vs-local-only split still matches `data/legacy/README.md` (forward experiments and the 1a pilot tracked; `*_alt`/`*_noalt`, `pilots/`, `planning_comm/` local-only). For `preregs/`, flag prereg files whose slug is no longer in the active roster, and active slugs that have no prereg.

6. **Doc-vs-code drift.** Check the last few commits for refactors (file renames, schema changes, slug changes). Then grep all .md/.qmd files — including `.Codex/AGENTS.md`, `.Codex/rules/*.md`, and this skill — for the old patterns. Patterns worth checking after past refactors: forward-pipeline script names, `_3act` slugs, "reward"/"motivation"/"access" terminology in active docs (the data columns and `param_w_v` are intentional exceptions; `data/legacy/` keeps old names by design), old LM CSV names, and the June-2026 CSV→JSON/JSONL output migration (`fit_results.csv`→`.json`; `cv_folds.csv`→`.jsonl`; `cv_preds_summary.csv`→`.json`; `lm_scenario.csv`+`lm_alternatives.csv`→`lm_runs.jsonl`; `lm_alternatives.csv`→`lm_alternatives.jsonl` for the stage-1 texts; the per-run given-magnitude scalars are folded into `lm_runs.jsonl`, so `lm_scenario_desire.csv` and the former `lm_given.json` are gone; the non-CV predict stage — `predict_<slug>.py`, `preds_*.npy`, `preds_summary.csv`/`.json` — was dropped entirely, CV being the sole prediction source) plus the belief-update mixture / fitted-σ rewrite (categorical NLLs and per-rating `NUM_RUNS` averaging are gone).

7. **Git state.** `git status`, large untracked files, branches diverged from origin/main, stale worktrees under `.Codex/worktrees/`. Note anything weird without acting on it.

8. **Gitignore correctness.** Anything tracked that shouldn't be (raw_data, large binaries, secrets, build artifacts) — or vice versa: files that docs claim are tracked but `git ls-files` doesn't know about. Check `git ls-files | grep -E '\.json$' | grep raw_data` returns zero, sample subject_id columns to confirm anonymization (UUIDs, not Prolific IDs), and `git check-ignore` anything suspicious. Be suspicious of broad patterns (a bare `*test*`-style glob has bitten this repo before).

9. **Anything else surprising.** Half-finished implementations, scattered TODOs, dead branches, files that look like debugging leftovers.

## Output format

Group findings by severity:

- 🔴 **BROKEN** — will fail when run, or data/code invisible to git
- 🟡 **STALE / DRIFT** — incorrect docs, dead code, naming inconsistency
- 🟢 **NICE-TO-HAVE** — improvements but not problems

For each finding, give: short description, file path, line number (when applicable), evidence. Skip categories where nothing notable was found. Don't pad with reassurances. Aim for ~600–800 words.

After reporting, ask the user which findings to act on rather than auto-fixing.
