---
name: rename-concept
description: Use when renaming a concept or term across the repo — code identifiers, data keys, docs, prose, figure slugs (like the desire/risk and canonical→observed renames), or when a term collides with another concept and needs replacing.
allowed-tools: Bash, Read, Edit, Grep, Glob, Agent
---

# Rename a concept across the repo

She renames aggressively when a term collides with another concept, and three renames (reward/motivation→desire, canonical→observed, and canonical→preregistered for the run config) converged on the same recipe. Renames here touch **persisted data keys**, not just code, so a grep-and-replace alone breaks the loaders.

## Recipe

1. **Inventory and classify** every occurrence (parallel Explore agents for big terms): (a) prose/docs, (b) code identifiers, (c) reader-visible labels (figures, experiment text, manuscript), (d) persisted data keys (JSONL fields, npz keys, CSV columns), (e) file/figure slugs. Search with the standard exclusions: `.venv/`, `cogsci-cr/`, `SIP_journal/`, `.claude/worktrees/`, `data/legacy/`, `notes/`. Include both skill trees, `.claude/skills/*/SKILL.md` and `.agents/skills/*/SKILL.md` — a skill naming the old term will keep reintroducing it.

Two lessons from the run-config rename specifically: a term can be doing real work in a **string comparison** (`model_comparison.py` routed on `tag == "canonical"`), which a prose-level grep misses; and where a retired spelling could still be passed in, reject it with a pointer to the new name rather than aliasing it, so there is never a window with two live spellings.
2. **Present the scope as a menu by tier** and get approval before touching anything — especially data keys (she has approved partial scopes: "can you do code identifiers and data keys"). Reader-visible labels and manuscript prose may deliberately keep a different word than the code.
3. Batch-replace with word boundaries: `perl -i -pe 's/\bOLD\b/NEW/g'` over the approved file list (watch camelCase/snake_case variants separately).
4. **Migrate persisted data** with a small script (rewrite JSONL/npz keys), never by hand; keep old-key reads out — fail fast instead.
5. `git mv` renamed files/figure slugs; delete orphaned outputs from the old naming.
6. **Verify**: `python -m py_compile` sweep over touched .py, the full `make test` suite, and an exhaustive `rg '\bOLD\b'` with the exclusions to confirm zero survivors (decide explicitly about intentional survivals, e.g. a legacy figure slug).
7. Work on a branch, `git merge --ff-only` to main; add a memory/AGENTS.md note recording the rename and any intentional exceptions.
