---
name: rerun-lm-elicitation
description: Use when Together AI elicitation tables need smoke-testing or regeneration after prompt, scenario, generation, or scoring changes, or when the user asks to run an lm or lm-base target. Protect canonical paid outputs and require approval for full runs.
allowed-tools: Bash, Read, Grep, Glob
---

# Rerun an LM elicitation safely

Elicitations spend API money and feed every downstream fit. Never start a full run merely because upstream files changed. A direct request to rerun authorizes the workflow, but still state the estimated cost and elapsed time before the paid full run.

## Output vintages

Canonical `lm_alternatives*.jsonl` and `lm_runs*.jsonl` files are committed model inputs. Diagnostic `*_diag*` files are gitignored and cannot overwrite them. Smoke-test only through diagnostic targets; promote an accepted change by re-eliciting canonical files, not by copying diagnostic files.

The resume guard compares prompt hashes in manifests and rejects incompatible appends. Override it with `LM_RESUME_PROMPT_MISMATCH=allow` only for an intentional extension whose mixed prompt history has been reviewed and will be recorded.

## Preconditions

1. Confirm `.env` contains `TOGETHER_API_KEY` without printing its value.
2. Check for active elicitation, fit, or CV processes before adding load.
3. Review the diff and identify affected studies and stages. A scoring prompt or JSON schema change must keep prompt fields and schema fields aligned.
4. Confirm the prompt, input, and output filenames in the generated manifest will identify the intended vintage.

## Diagnostic smoke

Run one elicitation component per cell into the diagnostic vintage:

```bash
make lm-diag-<slug> K_RUNS=1
make lm-base-diag-<slug> K_RUNS=1  # Studies 1a, 1b, and 3a only
```

Validate expected cell and run coverage against the study registry, empty or singleton alternative sets, parse failures, rationale leakage into action text, null observed-action features, and successful loading through `model/tables.py`. Compare distributions with a previously validated vintage; do not infer stability from a few K=1 events.

## Full run

Proceed only after the cost and time estimate has been acknowledged. Preserve the previous canonical vintage through Git or a scratch backup, make the intended output scope explicit, then run the applicable per-study targets. Use Makefile worker controls, reduce aggregate concurrency when running studies in parallel, stream logs to a temporary file, and prevent laptop sleep for unattended runs.

After completion:

1. Verify record counts, run IDs per cell, manifests, null features, loader behavior, and rating distributions against the previous vintage.
2. Run `make test`.
3. Remove a `VINTAGE MARKER` from `model/lm/prompts.py` only when the canonical tables now match it, then regenerate `SIP_journal/si_prompts.tex`.
4. Regenerate fits, CV, model comparison, and affected figures as one coherent vintage. Do not mix old and new artifacts.
5. Commit only when asked and stage explicit files.
