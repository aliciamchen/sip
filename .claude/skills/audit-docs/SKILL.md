---
name: audit-docs
description: Audit READMEs, CLAUDE.md, and rules files for staleness, redundancy, broken links, and structural issues. Reports concrete fixes (with line numbers) separately from bigger structural suggestions the user can decide on.
allowed-tools: Bash, Read, Grep, Glob
---

# Documentation audit

Scan documentation files for staleness against the current code, redundancy across files, broken links, and structural issues. Two outputs: concrete fixes, and bigger structural ideas.

## Files in scope

Derive the list rather than trusting one written here — a hardcoded list is the
first thing to go stale (this skill's own list was three files short):

```
git ls-files '*.md' | grep -vE '^(SIP_journal|cogsci-cr|notes)/'
```

plus **both** skill trees, `.claude/skills/*/SKILL.md` and
`.agents/skills/*/SKILL.md`. Skills drift exactly like docs do, and they are
worse when stale because they instruct rather than inform — **this file
included**.

### Check the mirrored pairs, not each file alone

Three pairs are maintained by hand as near-copies, and each one has drifted.
**Don't assume a direction** — it goes both ways, so establish which side is
right by checking the *code*, not by preferring one tree. (The Claude side has
been the newer one more often, but `rerun-lm-elicitation` drifted the other way:
the `.agents` copy correctly described the `.rationale.jsonl` sidecar and the
stage-specific `prompt_sha256` while the `.claude` copy still described the old
`.reasoning.jsonl` and whole-file hash.) Drift is worse for instructions than
for docs, because they direct behavior rather than inform it. Diff each pair
modulo the agent name and report any difference that is not just "Claude" vs
"Codex":

```
diff <(sed 's/Claude/AGENT/g' .claude/CLAUDE.md)     <(sed 's/Codex/AGENT/g' AGENTS.md)
diff <(sed 's/Claude/AGENT/g' ~/.claude/CLAUDE.md)   <(sed 's/Codex/AGENT/g' ~/.codex/AGENTS.md)
for n in .claude/skills/*/; do diff "$n/SKILL.md" ".agents/skills/$(basename $n)/SKILL.md"; done
```

1. **`.claude/CLAUDE.md` vs the root `AGENTS.md`** (project; the root copy is
   what Codex discovers — a third copy in `.agents/` had no reader and was
   dropped 2026-08-24). AGENTS.md once claimed `EXPERIMENTS_NONFOOD` holds the
   nonfood studies with "no participant data yet", long after Study 3 went live
   and that list was emptied.
2. **`~/.claude/CLAUDE.md` vs `~/.codex/AGENTS.md`** (global preferences —
   outside the repo, but in scope precisely because nothing else audits them).
   The commit-message convention was updated in one and not the other.
3. **The two skill trees.** Copies diverge in batches, and the `.agents` copies
   once referenced a `.Codex/` directory that has never existed. Report the
   drifted pairs by name rather than counting them.

Expected, legitimate differences: the agent's name, the README link path
(`README.md` from the root copy vs `../README.md` from `.claude/`), and
tool-specific instructions (Claude slash commands, Codex hooks). Everything
else is drift.

Never search inside `.claude/worktrees/`: those are checkouts of old branches,
stale by design, and they swamp every grep.

Audience each file should serve:

- `README.md` → reviewers / cloners (what the project is + a quick start)
- `data/`, `experiments/`, `model/`, `model/outputs/` READMEs → developers in
  that subfolder; the outputs one is the artifact codebook
- `.claude/CLAUDE.md` / root `AGENTS.md` → the agent: terminology drift,
  naming conventions, gitignored Overleaf folders, anything not in the code or
  public docs. Keep the pair in sync.
- `~/.claude/CLAUDE.md` / `~/.codex/AGENTS.md` → cross-project preferences
  (style, commit convention, citation rules). Keep the pair in sync.
- `.claude/rules/*.md` → Claude working in a specific directory
- `.claude/skills/*/SKILL.md`, `.agents/skills/*/SKILL.md` → a procedure, so
  correctness matters most, and a broken path is worse than a stale sentence
- `.codex/hooks/*.sh` → runs automatically, so a stale assumption fires silently

## Audit dimensions

For each file:

1. **Staleness vs. current code.** Paths, scripts, Makefile targets and file
   names that no longer exist. Verify mechanically, not by eye: extract every
   `path/to/file.py`-looking token and every `make <target>` and test each one
   (`grep -qE '^<target>:' Makefile`). Nonexistent make targets are a recurring
   find and are invisible on a read-through.

   Claims that drift fastest here: which module a wrapper imports from (the fit
   and CV wrappers now call `_fit_dispatcher` / `_inverse_dispatcher`, not the
   helpers directly), where the figure scripts live (all under
   `figures/scripts/`, writing to `figures/panels/` and `figures/si/` — they are NOT under
   `model/lm/`), and run-config vocabulary (the uniform-prior config is
   **preregistered**; "canonical" was retired because it read as "the reported
   model", which it is not — the reported fits add the comparison-set
   reweighting on top).

2. **Redundancy.** What appears in multiple files? The six-study roster, the utility-model math, the repository layout, terminology notes — flag specific duplicate sections so the user can pick a single source of truth.

3. **Audience fit.** Each doc should serve one audience clearly:
   - `README.md` → reviewers / cloners (lead with what the project is + a quick start)
   - `CLAUDE.md` → Claude (terminology drift, naming conventions, gitignored Overleaf folders, anything not in the code or public docs)
   - rules files → Claude when working in a specific dir (directory-specific concerns, not general project context)
   - subfolder READMEs → developers in that subfolder
   Flag files that are doing two jobs at once.

4. **Onboarding path.** Could a fresh reviewer reach a `make all` invocation from `README.md` within the first 50 lines? Front-loaded context that delays the practical instructions is a smell.

5. **Style.** Sentence-case headings (only first word capitalized). Complete sentences over telegraphic shorthand. Lowercase casual commit-message style. LaTeX em-dashes use ` -- ` not `---`. Flag style violations with line numbers.

6. **Broken links.** Verify relative markdown links resolve — and split the
   target on `#` first: the file part must exist, and the anchor must match a
   real heading (lowercased, non-alphanumerics stripped, spaces to hyphens).
   Testing the whole `file.md#anchor` string as a path reports every anchored
   link as broken, which is a false alarm, not a finding.

7. **History-dependent framing** (public docs only — READMEs). A fresh reader has no "before": flag wording that assumes project history, like "the *current* desire naming", "not standalone *anymore*", pointers to local archives or git history, and descriptions of removed mechanisms. Grep for telltales: `current|renamed|previously|no longer|anymore|legacy|earlier|git history|used to`. Rewrite as timeless statements or delete; self-contained history a reviewer needs (why an exclusion rule differs between studies) stays.

8. **Factual accuracy.** Cross-check counts and states against the filesystem: the roster is six inverse-planning studies (there is no forward pipeline — it was removed, so a doc still describing one is stale, not merely out of date). Read `EXPERIMENTS_INVERSE` in the Makefile as the roster's source of truth.

## Output format

**Two sections:**

### 1. Concrete staleness fixes
File + line number + the specific change. One bullet per fix. Skip files that are clean.

### 2. Structural suggestions
Bigger ideas the user can decide whether to act on. Frame each as a recommendation + tradeoff. Examples: "extract experiment list to one source of truth (annoying when you add an experiment, fine for now)", "split column codebook from architecture overview".

Lead with the strongest items. Don't pad with reassurances. Aim for ~500–700 words total.

After reporting, ask the user which fixes to apply (don't auto-fix).
