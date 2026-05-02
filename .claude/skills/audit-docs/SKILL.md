---
name: audit-docs
description: Audit READMEs, CLAUDE.md, and rules files for staleness, redundancy, broken links, and structural issues. Reports concrete fixes (with line numbers) separately from bigger structural suggestions the user can decide on.
allowed-tools: Bash, Read, Grep, Glob
---

# Documentation audit

Scan documentation files for staleness against the current code, redundancy across files, broken links, and structural issues. Two outputs: concrete fixes, and bigger structural ideas.

## Files in scope

- `README.md` (project root, public-facing — reviewers/cloners read this first)
- `data/README.md` (data codebook)
- `experiments/README.md` (experiment list/structure)
- `model/README.md` (model directory overview)
- `model/outputs/README.md` (output CSV codebook)
- `.claude/CLAUDE.md` (Claude-only project context)
- `.claude/rules/{analysis,data,experiments,model}.md` (path-conditional rules)

## Audit dimensions

For each file:

1. **Staleness vs. current code.** Pre-refactor paths/scripts/CSV names that no longer exist. Common culprits after recent refactors: hyphenated slugs that became underscored, legacy flat CSV names like `forward_planning_*.csv` that became `<slug>/fit_results.csv`, pre-`lm/` LM table paths. Cross-check claims against the actual filesystem.

2. **Redundancy.** What appears in multiple files? The 9-experiment list, the utility-model math, the repository layout, terminology notes — flag specific duplicate sections so the user can pick a single source of truth.

3. **Audience fit.** Each doc should serve one audience clearly:
   - `README.md` → reviewers / cloners (lead with what the project is + a quick start)
   - `CLAUDE.md` → Claude (terminology drift, naming conventions, gitignored Overleaf folders, anything not in the code or public docs)
   - rules files → Claude when working in a specific dir (directory-specific concerns, not general project context)
   - subfolder READMEs → developers in that subfolder
   Flag files that are doing two jobs at once.

4. **Onboarding path.** Could a fresh reviewer reach a `make all` invocation from `README.md` within the first 50 lines? Front-loaded context that delays the practical instructions is a smell.

5. **Style.** Sentence-case headings (only first word capitalized). Complete sentences over telegraphic shorthand. Native R pipe `|>` not `%>%`. Lowercase casual commit-message style. LaTeX em-dashes use ` -- ` not `---`. Flag style violations with line numbers.

6. **Broken links.** Verify all relative markdown links (`[text](path)`) resolve to real files. Use `grep -oE '\[[^]]+\]\([^)]+\)'` and check each path.

7. **Factual accuracy.** Cross-check claims like "n forward experiments", "data collected for X", "scripts in directory Y" against what's actually there. Counts/dates/states drift fastest.

## Output format

**Two sections:**

### 1. Concrete staleness fixes
File + line number + the specific change. One bullet per fix. Skip files that are clean.

### 2. Structural suggestions
Bigger ideas the user can decide whether to act on. Frame each as a recommendation + tradeoff. Examples: "extract experiment list to one source of truth (annoying when you add an experiment, fine for now)", "split column codebook from architecture overview".

Lead with the strongest items. Don't pad with reassurances. Aim for ~500–700 words total.

After reporting, ask the user which fixes to apply (don't auto-fix).
