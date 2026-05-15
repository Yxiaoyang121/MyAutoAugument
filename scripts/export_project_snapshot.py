from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "project_snapshot_latest.md"


def main() -> None:
    snapshot = build_snapshot()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(snapshot, encoding="utf-8")
    print(str(OUTPUT_PATH))


def build_snapshot() -> str:
    git_status = run_git(["status", "--short"])
    branch = run_git(["branch", "--show-current"]).strip() or "<detached>"
    commit = run_git(["rev-parse", "HEAD"]).strip()
    remote = run_git(["remote", "get-url", "origin"]).strip()
    tracked = run_git(["ls-files"]).splitlines()
    important_files = [
        "README.md",
        "PROJECT_STATE.md",
        "CODEX_HANDOFF.md",
        "EXPERIMENT_LOG.md",
        "AGENTS.md",
        "docs/ARCHITECTURE_CURRENT.md",
        "docs/diagnostic_augmentation_framework.md",
        "docs/experiment_protocol.md",
        "scripts/run_diagnostic_augmentation_pipeline.py",
        "AutoAugment/diagnostic_pipeline/__init__.py",
    ]
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Project Snapshot",
        "",
        f"- Generated: {now}",
        f"- Branch: {branch}",
        f"- Commit: {commit}",
        f"- Remote: {remote}",
        "",
        "## Working Tree",
        "",
    ]
    if git_status.strip():
        lines.append("```text")
        lines.append(git_status.rstrip())
        lines.append("```")
    else:
        lines.append("Clean working tree.")
    lines.extend(
        [
            "",
            "## Key Files",
            "",
        ]
    )
    for rel in important_files:
        path = PROJECT_ROOT / rel
        if path.exists():
            lines.append(f"- {rel}")
    lines.extend(
        [
            "",
            "## Tracked File Count",
            "",
            f"- {len(tracked)} tracked files",
            "",
            "## Notes",
            "",
            "- This snapshot reflects the current local repository state.",
            "- It does not invent benchmark results.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return completed.stdout


if __name__ == "__main__":
    main()
