"""Consistency checks for repository agent guidance and skills.

Run standalone with ``uv run python test_agent_docs.py``.
"""

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_SYMLINKS = {
    "AGENTS.md": ".claude/CLAUDE.md",
    ".agents/skills": ".claude/skills",
    ".codex/hooks/pre-commit-model-check.sh": (
        ".claude/hooks/pre-commit-model-check.sh"
    ),
}


def _frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text()
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if match is None:
        raise AssertionError(f"{path.relative_to(ROOT)} has no YAML frontmatter")
    fields = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields, text[match.end() :]


def run_all_checks() -> None:
    failures = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        if not ok:
            failures.append(f"{label}: {detail}")

    for link, target in EXPECTED_SYMLINKS.items():
        path = ROOT / link
        expected = ROOT / target
        check(path.is_symlink(), f"{link} is a symlink", f"expected {target}")
        if path.is_symlink():
            check(
                path.resolve() == expected.resolve(),
                f"{link} resolves to its canonical file",
                f"found {path.readlink()}, expected {target}",
            )

    skill_paths = sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md"))
    descriptions = []
    guidance_paths = [
        ROOT / ".claude" / "CLAUDE.md",
        *(ROOT / ".claude" / "rules").glob("*.md"),
    ]

    for path in skill_paths:
        fields, body = _frontmatter(path)
        rel = path.relative_to(ROOT)
        name = fields.get("name", "")
        description = fields.get("description", "")
        descriptions.append(description)
        check(name == path.parent.name, f"{rel} name matches its directory")
        check(
            description.startswith("Use when "),
            f"{rel} description starts with 'Use when'",
            description,
        )
        check(
            len(description) <= 500,
            f"{rel} description is at most 500 characters",
            str(len(description)),
        )
        check(
            len(path.read_text().splitlines()) <= 500,
            f"{rel} stays within the 500-line skill limit",
        )
        guidance_paths.append(path)
        check("---" not in body, f"{rel} body contains no prose triple dash")

    check(
        sum(map(len, descriptions)) <= 16_000,
        "skill descriptions fit a compact discovery budget",
        str(sum(map(len, descriptions))),
    )

    stale_commit_phrases = (
        "lowercase casual commit-message style",
        "lowercase verb-first message",
    )
    for path in guidance_paths:
        text = path.read_text()
        rel = path.relative_to(ROOT)
        check("—" not in text, f"{rel} contains no Unicode em dash")
        for phrase in stale_commit_phrases:
            check(phrase not in text, f"{rel} omits stale commit guidance")

    if failures:
        for failure in failures:
            print(f"FAILED -- {failure}")
        sys.exit(1)
    print("All agent-documentation checks passed.")


if __name__ == "__main__":
    run_all_checks()
