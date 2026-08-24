---
name: audit-docs
description: Use when auditing or cleaning repository documentation, agent guides, rules, skills, hooks, or agent configuration for staleness, duplication, broken references, and excess length. Report concrete fixes separately from optional structural changes.
allowed-tools: Bash, Read, Grep, Glob
---

# Audit documentation

Audit repository guidance against the current code and configuration. Include the skill itself.

## Scope

Derive tracked files instead of relying on a fixed inventory:

```bash
git ls-files '*.md' '.claude/**' '.codex/**' AGENTS.md .agents/skills
```

Exclude `.claude/worktrees/`, nested manuscript repositories, local notes, and generated files unless the user places them in scope. Treat `preregs/` as frozen records: report broken references or internal inconsistencies, but do not modernize them to match later implementation choices.

Check these canonical relationships:

- `AGENTS.md` is a symlink to `.claude/CLAUDE.md`.
- `.agents/skills` is a symlink to `.claude/skills`.
- Mirrored hooks should be symlinks or otherwise have one canonical implementation.

Shared guidance must remain agent-neutral. Harness-specific configuration can use its native vocabulary.

## Checks

1. Verify file paths, commands, Make targets, output schemas, study rosters, and module names mechanically against the repository.
2. Test relative Markdown links. Split anchors from paths, then verify both the file and normalized heading.
3. Compare repeated facts across the README, guide, scoped rules, skills, hook comments, and configuration. Recommend one source of truth for facts likely to change.
4. Check audience and loading cost:
   - `README.md` serves reviewers and new users.
   - The root guide holds durable cross-cutting constraints.
   - Scoped rules contain directory-specific invariants.
   - Skills contain procedures that should load only when triggered.
   - Subdirectory READMEs describe their public interfaces or artifacts.
5. Flag volatile history, dated numerical results, exhaustive inventories, and implementation narration when code, tests, manifests, or scripts already encode them.
6. Check sentence-case headings, complete public prose, US spelling, conventional commit examples, and ` -- ` for prose em dashes. Flag Unicode em dashes and three consecutive prose hyphens; ignore YAML frontmatter delimiters.
7. Inspect tracked hooks and agent settings for stale paths, duplicate implementations, mismatched permissions, or behavior not documented where users need it.
8. Confirm skill folder names match frontmatter names, descriptions begin with `Use when`, triggers are specific, and bodies do one job without duplicating the root guide.

## Report

Return two sections:

1. Concrete fixes with path, line number, evidence, and exact correction.
2. Optional structural suggestions with their tradeoffs.

Lead with the highest-impact findings, omit clean-file filler, and keep the report concise. Ask which changes to apply; do not edit during an audit-only request.
