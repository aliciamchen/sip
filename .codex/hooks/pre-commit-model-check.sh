#!/bin/bash
# PreToolUse hook for the Bash tool.
#
# When the agent is about to run `git commit`, check whether any staged file
# is under model/. If so, run the whole test suite first; if it fails, block the
# commit and report what failed.
#
# It runs `make test` rather than naming individual test files, deliberately: the
# suite has grown from one file to five, and a hook that hardcodes a list quietly
# stops covering the newest tests — which are the ones most likely to be broken.
# `make test` is the single source of truth for what must pass, and it costs 22s
# against 18s for the compliance test alone.
#
# Other Bash commands pass through untouched.

set -o pipefail

# Read the PreToolUse JSON payload from stdin
input=$(cat)
command=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')

# Only fire on git commit invocations
case "$command" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac

# Resolve project dir; CLAUDE_PROJECT_DIR is set when Claude invokes the hook
project_dir="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}"
cd "$project_dir" 2>/dev/null || exit 0

# Skip if no model/ files are staged
staged_model=$(git diff --cached --name-only | grep -E '^model/' || true)
if [ -z "$staged_model" ]; then
  exit 0
fi

# Every test file is tracked (there is no `*test*` gitignore rule -- an earlier
# version of this hook assumed one and skipped itself on that basis), so a clone
# always has them. Still bail out rather than block if `make` is unavailable.
if ! command -v make >/dev/null 2>&1; then
  exit 0
fi

# Run the suite, capturing output
log=$(mktemp)
if make test >"$log" 2>&1; then
  rm -f "$log"
  exit 0
fi

# Test failed — block the commit
output=$(cat "$log")
rm -f "$log"

jq -n --arg out "$output" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: ("make test failed; commit blocked.\n\nTest output:\n" + $out + "\n\nFix the failures and retry. To bypass this check for a single commit (rare; only when committing docs alongside an unrelated test failure), use `git commit --no-verify` — but the hook still fires, so prefer fixing the test.")
  }
}'
exit 0
