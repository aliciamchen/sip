---
name: fix-latex-build
description: Use when a Dropbox-synced LaTeX build has corruption signatures such as undefined citations, an empty bbl, NUL-filled auxiliaries, a truncated PDF, a sudden page-count drop, or a synctex busy file. Diagnose and rebuild safely outside Dropbox.
allowed-tools: Bash, Read, Grep, Glob
---

# Repair a corrupted LaTeX build

Dropbox can race rapidly rewritten LaTeX artifacts, and biber's unpacked PAR cache can become incomplete. Both can leave misleading local outputs even when the source is valid. Do not debug the bibliography or source until these failure modes are excluded.

Use the bundled script from the repository root:

```bash
bash .claude/skills/fix-latex-build/rebuild-outside-dropbox.sh --check
bash .claude/skills/fix-latex-build/rebuild-outside-dropbox.sh
```

For the CogSci fork:

```bash
bash .claude/skills/fix-latex-build/rebuild-outside-dropbox.sh cogsci-cr/cogsci-2026 main.tex
```

The script diagnoses artifacts, refuses to race an active LaTeX process, clears the stale biber cache when rebuilding, copies the project to a temporary directory outside Dropbox, builds and verifies it there, and copies good artifacts back. Use `--force` only when the user wants to override the active-process guard.

If the outside-Dropbox build fails, inspect the retained scratch log for a real source error. Report the detected signature, whether rebuilding succeeded, the final page count, and any remaining undefined citations, references, or overfull boxes. Do not commit or push the manuscript repository unless asked.
