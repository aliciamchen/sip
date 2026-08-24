---
name: build-manuscript
description: Use for any routine compile, page-preview, or figure-verification of SIP_journal (or cogsci-cr) — checking that an edit compiled, that a figure landed correctly, or that references resolve. For corruption signatures (empty bbl, NUL bytes, truncated PDF), use fix-latex-build instead.
allowed-tools: Bash, Read, Grep, Glob
---

# Build and verify the manuscript

`SIP_journal/` is inside Dropbox, and local `latexmk` races Dropbox sync (the cause of the recurring corruption). So **routine builds also go through a scratchpad copy outside Dropbox** — don't run `latexmk` in the Dropbox tree.

## Procedure

1. If figures changed, sync them first: `make sync-journal-figures` (it fails loudly on MISSING sources — regenerate those before building).
2. **Build with the bundled script — don't hand-roll a scratchpad build.** It does the copy-out-of-Dropbox, the build, the verification, and the copy-back in one command:

   ```bash
   bash .agents/skills/fix-latex-build/rebuild-outside-dropbox.sh          # SIP_journal/main.tex
   bash .agents/skills/fix-latex-build/rebuild-outside-dropbox.sh cogsci-cr/cogsci-2026 main.tex
   ```

   It prints the undefined-citation, undefined-reference and overfull counts plus the page count, copies `main.pdf` and `main.log` back into the project, and on failure keeps the scratch dir and prints the real errors from the log. (It is a fresh full build every time, so it costs more than an incremental one — that is the price of not racing Dropbox. Reserve hand-rolled builds for the case where you genuinely need repeated fast passes on one file, and even then build outside Dropbox.)
3. Check the residual problems the script's counts don't cover:
   - Missing-figure warnings, and `grep "Float too large" main.log` — a figure taller than `\textheight` can't be placed as any float.
   - Duplicate bib keys (co-occurred with a past cache corruption): `grep -o '^@[a-z]*{[^,]*' references.bib | sort -f | uniq -di`.
   - A 0-byte `main.bbl` or NUL-filled aux → that's the corruption path; the same script diagnoses it with `--check` (see fix-latex-build).
4. **Visual check of a specific figure/page**: find the page with `pdftotext main.pdf - | grep -n ...` (or the figure's caption text), then render just that page — `pdftoppm -png -r 75 -f <page> -l <page> main.pdf "$SCRATCH/page"` — and Read the PNG. Never claim a figure "looks right" without rendering it.
5. Don't commit or push the SIP_journal repo unless asked — it's a separate git repo synced to Overleaf.

## Notes

- Two LaTeX-project-level house rules: no digits in `\newcommand` macro names, and multi-`\documentclass` projects pushed to Overleaf need a `latexmkrc` with `@default_files`.
- The SI `\input`s generated files (`si_prompts.tex`, `si_scenarios_*.tex`) — if a prompt or scenario looks wrong in the PDF, fix the source and re-export (regen-experiments skill), never the `.tex`.
