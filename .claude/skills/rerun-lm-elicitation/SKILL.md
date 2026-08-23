---
name: rerun-lm-elicitation
description: Use when LM elicitation tables need regenerating or smoke-testing — after edits to model/lm/prompts.py, the scenarios, or generate_alternatives.py / score_merged.py, when the user asks to rerun an elicitation, or before running make lm-<slug> / lm-base.
allowed-tools: Bash, Read, Grep, Glob
---

# Rerun an LM elicitation safely

Elicitations are **paid Together AI runs** (K=20 over hundreds of cells) and
their outputs are expensive to reproduce. Two standing rules, both from real
incidents:

1. **Never launch a full K=20 run as a side effect of another task.** A direct
   user request to rerun an elicitation is the go-ahead: smoke-test first, state
   the cost/time estimate, then proceed. Without a direct request, give the
   estimate and stop — she often defers ("later i will rerun all elicitations");
   build and plot against the current tables and flag the staleness.
2. **A smoke must never write to the canonical filenames.** Use the diagnostic
   vintage (below), which is gitignored and cannot displace the tables the
   reported fits read.

## Two output vintages

The canonical tables — `lm_alternatives.jsonl`, `lm_runs.jsonl`, plus
`*_base.jsonl` for the given-relationship studies (1a/1b/3a) — are committed and
feed every reported fit.

`--arm-output-only` (generate) and `--arm` (score) write a parallel
**diagnostic** vintage instead: `lm_alternatives_diag.jsonl`,
`lm_runs_diag.jsonl`, plus `.empty_units.jsonl`, `.rationale.jsonl` and
`.manifest.json` siblings. It is available for **all six studies**, and
`.gitignore` excludes `*_diag*` under `model/outputs/lm/`, so a smoke there is
free: no backup dance, no risk of degrading a downstream fit. This supersedes
the old "back up the study folder, then restore" runbook.

An adopted arm is promoted by **re-eliciting into the canonical filenames**,
never by copying the diag files over them.

## Preconditions

- Confirm the key without printing it: `grep -c TOGETHER_API_KEY .env`.
- Check for her own running jobs before adding load:
  `ps aux | grep -E "generate_alternatives|score_merged|fit_|cv_" | grep -v grep`.
- **Prompt/schema pairing check** — feature scoring only. `score_merged.py` /
  `_features_dispatcher.py` still use grammar-constrained `json_schema`
  decoding, and a prompt whose fields no longer match the schema does NOT
  error: Together jams the model's intended JSON into free-text fields,
  silently corrupting output (the `is_share` incident). Alternatives generation
  no longer uses a schema (free decoding, adopted after grammar constraint
  collapsed ~27% of Study 3 cells to empty arrays), so it is not exposed to this
  — but it IS exposed to parse failures, which the tolerant parser retries.
- **There is exactly one alternatives prompt and it reasons before answering.**
  `ALTERNATIVES_SYSTEM_PROMPT` carries the think-step-by-step preamble and the
  explain-then-JSON close; generation always requests the raw text
  (`ALT_MAX_TOKENS = 1400`) and always writes a `.rationale.jsonl` sidecar. The
  sidecar stores a generated rationale for auditing the comparison set, not
  evidence about the model's hidden reasoning process. Do not add a selectable
  second variant unless the selected variant is recorded explicitly in the
  manifest.

## The resume guard

`guard_resume_prompt_mismatch` **hard-errors** when the existing manifest's
stage-specific `prompt_sha256` differs from the current rendered prompt
surfaces, so a rerun after a relevant prompt edit cannot silently append onto
the old vintage. Legacy manifests have only the whole-file `prompts_sha256`;
the guard falls back to that field for old artifacts. Consequences:

- Deleting stale JSONL before regenerating is still the right move, but the
  guard is what protects you if you forget.
- For new manifests, an edit blocks only the stage whose rendered prompt
  surfaces changed or feed it upstream: a scoring table is also invalidated by
  a generation-prompt edit because it embeds those generated alternatives.
  Comments and prompts used only by an unrelated stage do not block resume.
- To extend K across an intentional prompt mismatch, set
  `LM_RESUME_PROMPT_MISMATCH=allow`; the superseded hash is then recorded in
  the manifest's history rather than hidden.

## Runbook

1. **K=1 smoke into the diag vintage**, generation only:
   `K_RUNS=1 uv run python model/lm/generate_alternatives.py --study <slug> --arm-output-only`
   Expected cells: 1a 384, 1b/2a/3a 192, 2b/3b 96 (records = cells × K).
   Validate: full cell coverage, zero empty units, no singleton sets, mean set
   size in the 2.5–3.5 band, no conditional "if available" hedging about the
   unknown world state (it neutralises the effort swing at scoring), and no
   generated-rationale prose leaking into `action_text`.
   **Calibrate against a study whose vintage is already validated** rather than
   absolute thresholds, and check counts before believing a rate: at K=1 a
   handful of events is noise, not a signal.
2. **Score the smoke** if the feature stage is in scope:
   `uv run python model/lm/score_merged.py --study <slug> --arm`, then load
   through `model/tables.py` and scan for null slot-0 features.
3. **Full run — only on her explicit go-ahead.** State cost + ETA first. Delete
   the stale canonical JSONL for the studies in scope, then
   `make lm-<slug>` (and `lm-base-<slug>` for 1a/1b/3a). Run in the background
   with `tee` to a scratchpad log; tune `SCENARIO_WORKERS` / `CELL_WORKERS` per
   the Makefile comments (lower under `make -j`). `caffeinate -i` stops a closed
   laptop pausing it. "Failed to parse JSON" retry lines are noise.
4. **Post-run validation**: record counts and run_id coverage per cell,
   null-feature scan, per-scenario g/risk/effort means against the previous
   vintage, a loader check, then `make test`.
5. **Post-run bookkeeping that is easy to forget:**
   - Delete the `VINTAGE MARKER` block in `prompts.py` **if one is present** (it
     exists precisely to say the prompt is ahead of the tables — once it isn't,
     the marker lies; there is none while the two are in sync).
   - Re-run `model/lm/export_prompts_latex.py` to refresh
     `SIP_journal/si_prompts.tex`, which a marker would have been blocking.
   - Regenerate fits → CV → model-comparison → figures. Fit and CV outputs
     must never mix vintages; the CV checkpoint fingerprint hashes the LM tables, so
     a stale checkpoint is discarded rather than spliced.
6. Commit what she asks — pipeline code and outputs separately; never
   `git add -A` (other tabs may have in-flight work).
