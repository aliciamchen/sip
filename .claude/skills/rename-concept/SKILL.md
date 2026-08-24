---
name: rename-concept
description: Use when renaming a concept across code identifiers, persisted data keys, documentation, manuscript prose, reader-visible labels, or file and figure slugs.
allowed-tools: Bash, Read, Edit, Grep, Glob
---

# Rename a concept

1. Search every relevant tracked file and classify matches as prose, identifiers, reader-visible labels, persisted keys, filenames, generated artifacts, or intentional historical records. Include the canonical skill tree.
2. Derive exclusions from the requested scope. Do not automatically exclude nested manuscript repositories or preregistrations when the user included them; treat frozen records and local archives separately.
3. If the request leaves persisted keys, public labels, nested repositories, or frozen records ambiguous, show the classified scope and obtain approval for those categories before writing.
4. Update approved occurrences with syntax-aware or boundary-aware edits. Check string comparisons, configuration tags, serialization keys, and camelCase or snake_case variants separately.
5. Migrate persisted formats with a reproducible script or existing converter. Prefer a clear failure for retired live keys over maintaining two spellings indefinitely.
6. Rename files with `git mv`, update references, regenerate downstream artifacts, and remove superseded outputs only when their replacement is verified.
7. Verify with `make test`, relevant build checks, and an exhaustive old-term search. List intentional survivors and why they remain.

Do not commit, branch, or merge unless requested. If a commit is requested, stage explicit files and use the repository's conventional commit format.
