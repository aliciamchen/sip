---
name: run-overnight
description: Use when launching, supervising, checking, or reporting on a long unattended fit, CV, elicitation, sweep, or other compute run.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
---

# Supervise an unattended run

Before launch, state the expected duration, resource use, and monetary cost. Check for existing jobs and choose worker and thread counts within the machine and provider limits.

## Launch and monitor

- Stream unbuffered output to a temporary log and use an appropriate mechanism to prevent laptop sleep.
- Record the exact command, working tree or worktree, start time, log path, output paths, and process IDs.
- Do not edit source files that a running process imported. If code must change, stop safely or fix and relaunch from an isolated worktree so one run cannot mix code vintages.
- CV can resume from fingerprinted checkpoints. Check progress from the current log, checkpoint row counts, output manifests, and live processes rather than from memory.
- Inspect child processes after abnormal termination so orphaned workers do not continue consuming resources.

Do not silently change restart counts, bounds, priors, patience, K, or other protocol-relevant settings. Code fixes stay within the user's authorization; protocol changes require separate approval. Do not commit unless asked.

## Final report

Report:

1. what completed, failed, or remains running;
2. exact fixes or restarts and whether any affected the protocol;
3. output and manifest validation;
4. model-comparison statistics when all required CV outputs exist;
5. concise qualitative diagnostics from scratch plots when useful; and
6. concrete next steps that require a user choice.
