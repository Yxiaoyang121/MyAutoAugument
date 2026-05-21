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
        "scripts/audit_tiling_quality.py",
        "scripts/filter_tiled_dataset.py",
        "scripts/audit_artifacts.py",
        "scripts/build_yolo_tiled_dataset.py",
        "scripts/run_gpu_preflight.py",
        "scripts/run_diagnostic_augmentation_pipeline.py",
        "scripts/generate_top3_policies_from_baseline.py",
        "scripts/run_top3_policy_short_training.py",
        "scripts/run_counterfactual_diagnosis.py",
        "scripts/write_yolo_default_aug_50ep_reports.py",
        "outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.md",
        "outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.json",
        "outputs/audits/tiling_quality/tiling_quality_audit.md",
        "outputs/audits/tiling_quality/tiling_quality_audit.json",
        "outputs/audits/dataset_mapping/dataset_mapping_audit.md",
        "outputs/audits/dataset_mapping/dataset_mapping_audit.json",
        "outputs/audits/gpu_preflight/gpu_preflight_report.md",
        "outputs/audits/gpu_preflight/gpu_preflight_report.json",
        "outputs/audits/artifact_inventory/artifact_inventory.md",
        "outputs/experiments/20260517_tiled_baseline_20epoch/reports/summary.md",
        "outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_report.md",
        "outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_metrics.json",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_report.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_metrics.json",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/diagaug_50ep_report.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/diagaug_50ep_metrics.json",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/baseline_vs_diagaug.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/yolo_default_aug_50ep_report.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/yolo_default_aug_50ep_metrics.json",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/compare_baseline_yolo_default_diagaug.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/diagnosis/diagnosis.json",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/diagnosis/diagnosis_summary.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/policies/candidate_policies.json",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/policies/candidate_policies.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/proxy/proxy_ranking.json",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/proxy/proxy_ranking.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/proxy/proxy_safety_report.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/top3_policies/top3_policies.json",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/top3_policies/top3_policies.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/reports/policy_selection_trace.json",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/reports/policy_selection_trace.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain/reports/top3_policy_shorttrain_report.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain/reports/top3_policy_shorttrain_results.json",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/reports/counterfactual_diagnosis_report.md",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/counterfactual_summary.json",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/counterfactual_instances.csv",
        "outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/counterfactual_policy_ranking.json",
        "outputs/datasets/tiled/tiled_1024_ov20_smoke/dataset_summary.md",
        "outputs/datasets/tiled/tiled_1024_ov20_full/data.yaml",
        "outputs/datasets/tiled/tiled_1024_ov20_full/dataset_summary.md",
        "outputs/datasets/tiled/tiled_1024_ov20_full/tiled_dataset_report.md",
        "outputs/datasets/tiled/tiled_1024_ov20_full/tiled_dataset_report.json",
        "outputs/datasets/tiled/tiled_1024_ov20_full_safe/data.yaml",
        "outputs/datasets/tiled/tiled_1024_ov20_full_safe/dataset_summary.md",
        "outputs/datasets/tiled/tiled_1024_ov20_full_safe/tiled_dataset_report.md",
        "outputs/datasets/tiled/tiled_1024_ov20_full_safe/tiled_dataset_report.json",
        "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml",
        "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/class_filter_report.md",
        "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/class_filter_report.json",
        "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/dataset_summary.md",
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
