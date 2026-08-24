---
name: verify-unchanged
description: Use after any refactor, dependency change, rename, or cleanup that must not move reported numbers — import/layout changes, dead-code removal, deduplication — to verify the pipeline's outputs are unchanged before committing.
allowed-tools: Bash, Read, Grep, Glob
---

# Verify a refactor changed nothing

The 2026-08 release cleanup converged on this recipe; each step caught (or
ruled out) a real difference at least once. Run the cheap steps always, the
expensive ones when the change touches that layer. State in the commit message
which steps ran and what they showed.

1. **Full suite** — `make test > "$SCRATCH/test.log" 2>&1; echo $?`, then read
   the log. Never `make test 2>&1 | tail`: the pipeline's exit status is
   tail's, so a failure reads as green.
2. **Statistics byte-identity** (always): rerun
   `uv run python model/cv/model_comparison.py` and
   `model/cv/generalization_primary.py`; `git status model/outputs/` must stay
   clean. These scripts rewrite their JSONs in full, so any diff is a real
   numeric change, not formatting.
3. **Table bit-identity** (loader/table changes): load all five padded table
   families plus the given-magnitude scalars under the old and new code (import
   the old module from a `git show HEAD:model/tables.py` copy) and
   `np.array_equal` every array.
4. **Figure content-identity** (figure or shared-module changes): regenerate,
   then compare each PDF against `git show HEAD:<path>` with `/CreationDate`
   and `/ModDate` stripped from both byte strings. Identical → `git checkout --`
   the file (metadata churn is not a change). Differences at float
   reassociation scale (~1e-16 in the drawn coordinates) are fine for
   illustrative panels — note them, restore the committed PDFs.
5. **Fit/CV bit-identity** (anything near the likelihood, tables, or fit loop):
   rerun one study's CV and compare the fold refits' params and NLLs
   field-by-field against the committed `cv_folds.jsonl`. Even a partial run
   verifies the folds it completed — the checkpoint
   (`cv_checkpoint.jsonl` `fold_row`s) is comparable directly, no need to let
   the run finish.
6. **Restore the no-op state**: `make freshen-outputs`, then `make all` should
   print nothing but sub-make lines. (Code edits make `FIG_SHARED` prereqs
   newer than the committed figure PDFs, so skipping this leaves make wanting
   to rebuild figures.)

Never create a file with `touch` on an unmatched glob in a shell recipe — use
make's `$(wildcard ...)`, which expands to existing files only (a bare `touch
.../cv_run_deltas.json` once created a spurious empty sidecar; 1a legitimately
has none).
