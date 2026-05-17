from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = PROJECT_ROOT / "outputs" / "snapshots" / "project_snapshot_latest.md"
COMPAT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "project_snapshot_latest.md"


def main() -> None:
    snapshot = build_snapshot()
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(snapshot, encoding="utf-8")
    COMPAT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPAT_OUTPUT_PATH.write_text(snapshot, encoding="utf-8")
    print(str(SNAPSHOT_PATH))
    print(str(COMPAT_OUTPUT_PATH))


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
        "docs/output_convention.md",
        "scripts/audit_dataset_mapping.py",
        "scripts/audit_artifacts.py",
        "scripts/build_yolo_tiled_dataset.py",
        "scripts/run_gpu_preflight.py",
        "scripts/run_diagnostic_augmentation_pipeline.py",
        "outputs/audits/dataset_mapping/dataset_mapping_audit.md",
        "outputs/audits/dataset_mapping/dataset_mapping_audit.json",
        "outputs/audits/gpu_preflight/gpu_preflight_report.md",
        "outputs/audits/gpu_preflight/gpu_preflight_report.json",
        "outputs/audits/artifact_inventory/artifact_inventory.md",
        "outputs/experiments/20260517_tiled_baseline_20epoch/reports/summary.md",
        "outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_report.md",
        "outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_metrics.json",
        "outputs/datasets/tiled/tiled_1024_ov20_smoke/dataset_summary.md",
        "outputs/snapshots/project_snapshot_latest.md",
        "outputs/project_snapshot_latest.md",
        "AutoAugment/diagnostic_pipeline/__init__.py",
        "AutoAugment/diagnostic_pipeline/strategy_memory.py",
        "AutoAugment/diagnostic_pipeline/metric_audit.py",
        "AutoAugment/diagnostic_pipeline/proxy_evaluation.py",
        "AutoAugment/diagnostic_pipeline/policy_mapping.py",
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
