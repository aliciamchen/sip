#!/bin/bash
# PreToolUse hook for the Bash tool.
#
# When Claude is about to run `git commit`, check whether any staged file
# is under model/. If so, run model/test_model_compliance.py first; if the
# test fails, block the commit and tell Claude what failed.
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

# Defensive: skip if the test file isn't present. It's tracked in git, so a
# normal checkout has it — this guard just avoids blocking commits spuriously
# if it's ever missing.
if [ ! -f model/test_model_compliance.py ]; then
  exit 0
fi

# Run the compliance test, capturing output
log=$(mktemp)
if uv run python model/test_model_compliance.py >"$log" 2>&1; then
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
    permissionDecisionReason: ("model/test_model_compliance.py failed; commit blocked.\n\nTest output:\n" + $out + "\n\nFix the failures and retry. To bypass this check for a single commit (rare; only when committing docs alongside an unrelated test failure), use `git commit --no-verify` — but the hook still fires, so prefer fixing the test.")
  }
}'
exit 0
