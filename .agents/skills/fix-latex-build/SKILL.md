---
name: fix-latex-build
description: Repair a corrupted local LaTeX build in a Dropbox-synced manuscript (SIP_journal, cogsci-cr). Use when the compiled PDF looks broken — every citation undefined, an empty main.bbl, a truncated/short PDF, sudden page-count drops, aux files with NUL bytes, or a stray main.synctex(busy) file. Diagnoses the cause, clears the biber PAR cache, and rebuilds outside Dropbox.
allowed-tools: Bash, Read, Grep, Glob
---

# Fix a corrupted LaTeX build

The manuscript directories (`SIP_journal/`, and the CogSci fork under `cogsci-cr/`) live inside the user's Dropbox folder. Two recurring failure modes corrupt the *local* build (Overleaf, which builds remotely from source, is unaffected):

1. **Stale biber PAR cache** → **empty `main.bbl` (0 bytes)**, so every citation is undefined and the References section disappears. biber is a PAR-packed binary that unpacks Perl libs into `${TMPDIR}/par-<userhex>/`; macOS prunes part of that cache, and biber then dies mid-parse — often writing a 0-byte `.bbl` while still **exiting 0** (exit code is not a reliable tell; the 0-byte `.bbl` is). Not a `references.bib` problem.
2. **Dropbox sync race** → **truncated `main.pdf`** (no `%%EOF`), empty or NUL-filled `main.aux`/`.bcf`/`.out`/`.toc`, a `main.synctex(busy)` file, or a sudden page-count drop. Local `latexmk` and Dropbox fight over the same rapidly-rewritten build files. Often transient: files may be changing under you as Dropbox syncs, and can settle on their own — but the settled PDF may be stale relative to the current source.

Both are fixed the same way: clear the PAR cache, then rebuild in a scratch dir **outside** Dropbox and copy the finished artifacts back. The build artifacts are gitignored (except tracked `main.pdf`/`main.tex`), so only `main.pdf` matters for git/Overleaf.

## How to use

Run the bundled script from the repo root. It diagnoses, clears the cache, rebuilds outside Dropbox, verifies, and copies the good artifacts back:

```bash
bash .Codex/skills/fix-latex-build/rebuild-outside-dropbox.sh
```

- Defaults to `SIP_journal/main.tex`. For the CogSci fork or another project/file, pass them:
  ```bash
  bash .Codex/skills/fix-latex-build/rebuild-outside-dropbox.sh cogsci-cr/cogsci-2026 main.tex
  ```
- `--check` diagnoses only (no cache clear, no rebuild, nothing modified) — use it to see *which* failure mode is present before acting.
- `--force` rebuilds even if a local latex/biber process is detected (by default the script refuses, to avoid racing a build that is actually running).

After it finishes, report to the user: which corruption signature was found, the final page count, and that `main.pdf` now matches the current source. Do **not** commit or push the Dropbox/Overleaf repo unless the user asks.

## What the script does (so you can do it by hand if needed)

1. **Diagnose** the in-place artifacts for the signatures above and name the likely cause.
2. **Guard**: abort if a local `latexmk`/`pdflatex`/`biber` is running (would race it).
3. **Clear the biber PAR cache**: `find "${TMPDIR:-/tmp}" -maxdepth 1 -type d -name 'par-*' -exec rm -rf {} +` (safe — biber re-extracts on next run).
4. **Rebuild outside Dropbox**: copy the project (minus `.git`) to a `mktemp -d` dir, delete the root-level build artifacts so `latexmk` builds fresh (figure PDFs under `figures/` are kept), then `latexmk -pdf -interaction=nonstopmode main.tex`.
5. **Verify**: PDF has a `%PDF-` header and `%%EOF`; 0 undefined citations, 0 undefined references, 0 overfull boxes; sensible page count.
6. **Copy back** `main.pdf` + `.bbl`/`.aux`/`.bcf`/etc. into the project, and delete the stale `main.fdb_latexmk`/`main.fls`/`main.synctex(busy)` so the next in-place build re-initializes cleanly.

## Notes and prevention

- If the script reports **no corruption signatures**, it was likely a transient mid-sync state that already settled. Rebuilding anyway is still worthwhile: it produces a PDF guaranteed to match the current `main.tex`.
- If the build **fails** (nonzero exit / undefined refs / no PDF), it is probably a real source error, not the sync/cache — the script keeps the scratch dir and prints the first errors from the log. Read the log there.
- To stop this recurring: build outside Dropbox (or pause Dropbox during builds) and treat Overleaf as the source of truth for the compiled PDF; and add `export PAR_GLOBAL_TEMP=$HOME/.parcache` to the shell profile so the biber cache lives outside the auto-pruned system temp dir.
