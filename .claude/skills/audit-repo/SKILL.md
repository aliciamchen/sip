---
name: audit-repo
description: Run a state-of-the-repo audit across all 9 experiments, the model pipeline, and the analysis qmds. Reports broken scripts, stale references, drift across per-experiment scripts, output CSV consistency, test status, orphaned files, and gitignore correctness — grouped by severity. Use after large refactors or before submission.
allowed-tools: Bash, Read, Grep, Glob
---

# Repo state audit

Run a thorough audit of this repository. The codebase has 9 experiments (3 forward, 6 inverse) and a multi-stage pipeline (LM elicitation → fit → predict → CV → analysis). The audit catches things that broke during recent refactors and surfaces drift before it compounds.

For broad checks, delegate to the Explore subagent in parallel chunks. For targeted file reads or path checks, do them directly.

## Checks to run

1. **Will every script run cleanly?** Spot-check imports, signatures, and paths in `model/{forward,inverse,cv}/*.py`. Look for: missing `EXPERIMENT_SLUG` constants, calls to functions with wrong kwargs (e.g. `relationship_idx` vs. `relationship_condition`), tuple unpacks that don't match the variant registry's tuple width, references to CSV paths that don't exist (`outputs/<old-name>.csv` patterns).

2. **Drift across the 9 per-experiment scripts.** Each experiment has fit, predict, and CV scripts. Forward scripts share `_shared.py`; alt-shown inverse uses `_helpers.py`; no-alt inverse uses padded helpers; the alt CV uses `_alt_dispatcher.py`. Flag scripts that diverge weirdly from their siblings (extra steps, missing params, hardcoded values).

3. **Output CSV consistency.** Sample headers from `model/outputs/<slug>/fit_results.csv` and `cv_folds.csv` across experiments. The `experiment` column should always hold the slug. Forward fits include AIC/BIC/r columns; inverse alt fits have only `alpha_observer`; inverse no-alt fits have all utility weights. Verify these expectations hold.

4. **Tests pass.** `uv run python model/test_model_compliance.py` — must exit 0.

5. **Orphaned files.** Survey `model/sandbox/`, `model/preregs/`, `analysis/legacy/`, `data/legacy/`. For each, grep for references in tracked code/docs. Flag content not referenced anywhere.

6. **Doc-vs-code drift.** Check the last few commits for refactors (file renames, schema changes, slug changes). Then grep all .md/.qmd files for the old patterns. Common patterns to check after refactors: legacy script names (`fit_forward_planning.py` etc.), legacy CSV names (`forward_planning_*.csv`), pre-slug-reorg paths.

7. **Git state.** `git status`, large untracked files, branches diverged from origin/main. Note anything weird without acting on it.

8. **Gitignore correctness.** Anything tracked that shouldn't be (raw_data, large binaries, secrets, build artifacts) — or vice versa. Check `git ls-files | grep -E '\.json$' | grep raw_data` returns zero, sample subject_id columns to confirm anonymization, etc.

9. **Anything else surprising.** Half-finished implementations, scattered TODOs, dead branches, files that look like debugging leftovers.

## Output format

Group findings by severity:

- 🔴 **BROKEN** — will fail when run
- 🟡 **STALE / DRIFT** — incorrect docs, dead code, naming inconsistency
- 🟢 **NICE-TO-HAVE** — improvements but not problems

For each finding, give: short description, file path, line number (when applicable), evidence. Skip categories where nothing notable was found. Don't pad with reassurances. Aim for ~600–800 words.

After reporting, ask the user which findings to act on rather than auto-fixing.
