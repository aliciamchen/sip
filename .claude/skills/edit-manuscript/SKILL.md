---
name: edit-manuscript
description: Use when writing, editing, restructuring, or auditing manuscript text (SIP_journal main.tex, SI, preregistrations, cover letters) — including "add this to the paper", "make this section simpler", and "audit the methods section".
allowed-tools: Bash, Read, Edit, Grep, Glob, WebSearch, WebFetch
---

# Write, edit, and audit manuscript text

The manuscript is her design document: she writes the plan into the paper first and brings code up to it later. During a design conversation, **don't implement code** — "leave the code for later" is her stated workflow. And always work from the actual current `.tex` (she edits it herself between and during sessions); never from a summary or recall of it.

## Writing style (every one of these came from repeated corrections)

- **Short and compact.** Her most common edit to Claude-written text is "too long, too much detail." Default to the minimal addition; she'd rather expand than cut.
- **Intuitive over formal.** Prefer "we fit a Gaussian" + a package citation over derivations; equations only where they earn their place.
- **Understate.** No "heavy handed" novelty claims ("captures two advances… not previously brought together" got rejected). Every empirical claim needs computed evidence or a hedge — she interrogates "what is the evidence that…" line by line.
- **Synthesize, don't parallel.** Later studies' sections must not mirror earlier sections' structure; integrate points smoothly rather than bolting on paragraphs.
- **Two-sided related work**: what this paper adds over X *and* what X does that this paper doesn't.
- **Never silently drop rationale.** When restructuring, existing justification paragraphs must survive or be flagged ("where did the stuff about the point estimate go?" is a failure).
- Ground construct definitions in the actual literature (live search + read), not plausible glosses; verify every citation (verify-citations skill).
- Mechanics: sentence case headings, ` -- ` em dashes, no digits in macro names.

## Audit mode ("audit/evaluate this section")

1. Read the **full** main text first for the claims and framing; audit against the main text — never against the generated `si_*.tex` (they mirror `prompts.py`/the scenario CSVs by construction).
2. Look for unknown-unknowns: claims-vs-design mismatches (e.g., an ablation structurally unable to produce the DV), internal contradictions, statistical incoherence, unsupported citations. Don't re-run checklists from previous audits — check `notes/` for prior findings and skip what's known.
3. Output a **numbered/lettered findings menu**, ordered by severity, internal-coherence problems separated from suggestions. She replies "do B, C, D". When referring back to a finding, restate its content — never bare codes ("what is c3 again").
4. Report only; no edits until she picks.
5. Code-vs-manuscript discrepancies: surface them and ask which side is newer — the direction genuinely alternates.

## Explaining methods to her

If she repeats a question near-verbatim ("i still dont get why…"), the explanation is at the wrong altitude — stop re-describing and locate the specific quantity or mechanism in doubt (e.g., "is σ identifiable given the run spread?"). Expect design churn on statistical-machinery items; converge in prose before touching code.
