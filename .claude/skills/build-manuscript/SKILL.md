---
name: build-manuscript
description: Use when routinely compiling, previewing pages, checking figures, or verifying citations and references in SIP_journal or cogsci-cr. Use fix-latex-build for corruption signatures such as an empty bbl, NUL bytes, or a truncated PDF.
allowed-tools: Bash, Read, Grep, Glob
---

# Build and verify the manuscript

Build outside Dropbox to avoid sync races. Do not run `latexmk` directly in either manuscript tree.

1. If repository figures changed, run `make sync-journal-figures` and resolve missing sources.
2. Build with the bundled outside-Dropbox script:

   ```bash
   bash .claude/skills/fix-latex-build/rebuild-outside-dropbox.sh
   bash .claude/skills/fix-latex-build/rebuild-outside-dropbox.sh cogsci-cr/cogsci-2026 main.tex
   ```

3. Review its citation, reference, overfull-box, and page-count results. Also search the copied-back log for missing figures and `Float too large`.
4. To verify a specific page, locate it with `pdftotext`, create a temporary preview directory with `mktemp -d`, render only that page with `pdftoppm -png`, and inspect the PNG. Do not claim visual correctness without rendering.
5. If corruption signatures remain, switch to the fix-latex-build procedure. If the outside-Dropbox build reports a source error, inspect the retained scratch log.

Generated `si_prompts.tex` and `si_scenarios_*.tex` files are downstream artifacts. Correct their Python sources and re-export them rather than editing the LaTeX. Do not commit or push either manuscript repository unless asked.
