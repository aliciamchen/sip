---
name: audit-repo
description: Use when auditing repository health after a large refactor or before submission. Check the active study roster, pipeline scripts, generated artifacts, output consistency, tests, documentation drift, orphaned files, Git state, and ignore rules; report findings by severity.
allowed-tools: Bash, Read, Grep, Glob
---

# Audit repository state

Read the active roster from the `Makefile` and compare it with `study_registry.py`; do not copy a roster from this skill. CV outputs are the sole prediction source.

## Checks

1. **Entrypoints and registries.** Import or inspect scripts under `model/inverse`, `model/cv`, and `model/lm`. Confirm every active slug is registered. Fit and CV study files must remain thin dispatcher wrappers.
2. **Study metadata.** Flag per-study labels, domains, given conditions, inferred variables, or paper-facing base variants duplicated in consumers instead of read from `study_registry.py`.
3. **Experiment artifacts.** Run `make check-experiments`. Confirm authored scenario Python files, generated CSV/JSON files, shared experiment modules, and active slugs agree.
4. **Output artifacts.** For every active study, verify required LM, fit, CV, manifest, and comparison files and sample their schemas. Check slugs, variant sets, finite values, anonymized participant IDs, and matching provenance. Use `model/outputs/README.md` as the codebook.
5. **Variant semantics.** Given-relationship studies use `base_shared` as the paper-facing Base through `study_registry.reported_base`; `base_prereg` is a reporting label, not a stored fit variant. Treat flat or unidentified ablation parameters according to the model code rather than flagging every boundary value.
6. **Tests and reported outputs.** Run `make test`; before submission also run `make check-reported`. Do not rely on a hardcoded test-file count.
7. **Orphans and dead code.** Search tracked code and docs for references to unusual files, public shared-module exports, TODOs, debugging remnants, retired paths, and generated artifacts. Report candidates with evidence; do not delete them during an audit.
8. **Documentation and skills.** Check the root guide, scoped rules, canonical skill tree, hooks, and public docs against the current code. Exclude `.claude/worktrees/` and frozen preregistrations from modernization sweeps.
9. **Git and ignore state.** Report worktree changes, branch divergence, unexpected worktrees, large untracked files, secrets, raw participant data, broad ignore patterns, and files that docs claim are tracked but are not.
10. **Reproducibility.** Check that outputs from one analysis vintage are not mixed across data, LM inputs, fits, CV, statistics, and figures. Use manifests and repository validation scripts where available.

## Report

Group evidence-backed findings as:

- **Broken:** execution failures, invalid or missing required artifacts, exposed sensitive data, or invisible required code.
- **Stale or drifted:** incorrect guidance, schema disagreement, duplicated metadata, mixed vintages, or dead references.
- **Nice to have:** maintainability improvements without a current failure.

For each finding, give a concise description, path and line when applicable, and the evidence. Omit empty categories and avoid reassuring filler. Ask which fixes to apply; do not mutate the repository during an audit-only request.
