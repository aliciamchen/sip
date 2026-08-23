---
name: run-overnight
description: Use when the user starts a long unattended compute run (overnight fits, CV, LM elicitation, sweeps) and leaves Claude to supervise — "start the overnight run", "run this while i sleep", "how is it going", "did runs finish".
allowed-tools: Bash, Read, Edit, Write, Grep, Glob, Agent
---

# Supervise an unattended compute run

She launches multi-hour fit/CV/elicitation runs before stepping away and expects
results plus a report when she returns. Four standing rules, each from a real
correction:

1. **Never silently change fit-protocol hyperparameters.** Restart counts,
   bounds, priors, patience, and K are preregistration-relevant. If a run fails
   and a hyperparameter change would fix it, flag it in the report as a possible
   deviation instead of quietly applying it (a silently increased restart count
   drew "why did you increase the number of restarts?"). Pre-authorized fixes
   ("if issues come up then fix them") cover code bugs, not protocol changes.
2. **Fix breakage in a worktree, not the running tree.** The jobs read the tree
   they were launched from; editing under them mixes vintages. Fix and verify in
   a worktree, relaunch from there if needed.
3. **Never commit unasked.** The end state is a report and intact outputs; she
   decides what lands ("i dont want you to commit anything just now").
4. **Answer "how is it going?" from evidence, not memory of launching it** —
   the run's log, checkpoint progress (`wc -l model/outputs/<slug>/cv_checkpoint.jsonl`;
   a study's CV is 48–64 fold jobs), and `ps`. "Is it running" recurs several
   times per run; keep the answer to a line or two of current state.

## Launching

- Run in the background with output to a scratchpad log (`PYTHONUNBUFFERED=1`
  for live per-fold progress); `caffeinate -i` so a closing laptop doesn't
  pause it.
- CV resumes from its fingerprint-guarded checkpoint, so an interrupted study
  is safely re-runnable. A killed parent can orphan spawn workers — find them
  with `ps -eo pid,ppid,command | grep spawn_main` (orphans have ppid 1).
- Respect the machine budget (workers × threads ≲ cores; the Makefile comments
  give per-stage knobs), and check for her own running jobs before adding load.

## The morning report

A fixed shape, in her order of interest:

1. What finished, what failed, what was fixed — and exactly what any fix
   touched (file + nature of change), with protocol-relevant changes flagged as
   possible deviations.
2. A qualitative model-vs-human look (scratch plots reusing
   `model/cv/model_comparison.py`'s cell specs — see iterate-figures; scratch
   PNGs, not committed figures).
3. The comparison statistics (`make model-comparison`) if every CV landed.
4. Concrete next-step suggestions as a lettered menu she can pick from.
