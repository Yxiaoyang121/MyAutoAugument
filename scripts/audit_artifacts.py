from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
RUNS_ROOT = PROJECT_ROOT / "runs"
REPORT_DIR = OUTPUTS_ROOT / "audits" / "artifact_inventory"
REPORT_JSON = REPORT_DIR / "artifact_inventory.json"
REPORT_MD = REPORT_DIR / "artifact_inventory.md"


@dataclass
class ArtifactRecord:
    path: str
    relative_path: str
    scope: str
    kind: str
    size_bytes: int
    size_mb: float
    last_modified: str | None
    inferred_use: str
    recommend_keep: bool
    recommend_archive: bool
    recommend_delete: bool
    recommended_target: str | None
    reason: str


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_inventory()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "outputs_root": str(OUTPUTS_ROOT),
        "runs_root": str(RUNS_ROOT),
        "records": [asdict(record) for record in records],
        "summary": summarize(records),
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(str(REPORT_JSON))
    print(str(REPORT_MD))


def build_inventory() -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    if OUTPUTS_ROOT.exists():
        for child in sorted(OUTPUTS_ROOT.iterdir(), key=lambda p: p.name.lower()):
            if child.name == "audits":
                records.append(make_record(child, scope="outputs", kind="top_level"))
                continue
            records.append(make_record(child, scope="outputs", kind="top_level"))
    if RUNS_ROOT.exists():
        for child in sorted(RUNS_ROOT.iterdir(), key=lambda p: p.name.lower()):
            records.append(make_record(child, scope="runs", kind="top_level"))
    detect = RUNS_ROOT / "detect"
    if detect.exists():
        for child in sorted(detect.iterdir(), key=lambda p: p.name.lower()):
            records.append(make_record(child, scope="runs/detect", kind="yolo_experiment"))
    return records


def make_record(path: Path, *, scope: str, kind: str) -> ArtifactRecord:
    size = directory_size(path) if path.is_dir() else path.stat().st_size
    mtime = latest_mtime(path)
    inferred_use = infer_use(path, scope)
    keep, archive, delete, target, reason = recommend(path, scope, inferred_use)
    return ArtifactRecord(
        path=str(path.resolve()),
        relative_path=relative(path),
        scope=scope,
        kind=kind,
        size_bytes=size,
        size_mb=round(size / (1024 * 1024), 3),
        last_modified=datetime.fromtimestamp(mtime).isoformat(timespec="seconds") if mtime else None,
        inferred_use=inferred_use,
        recommend_keep=keep,
        recommend_archive=archive,
        recommend_delete=delete,
        recommended_target=target,
        reason=reason,
    )


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def latest_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    latest = path.stat().st_mtime
    if path.is_dir():
        for item in path.rglob("*"):
            try:
                latest = max(latest, item.stat().st_mtime)
            except OSError:
                continue
    return latest


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def infer_use(path: Path, scope: str) -> str:
    rel = relative(path).replace("\\", "/").lower()
    name = path.name.lower()
    if scope == "runs/detect" or rel.startswith("runs/"):
        return "yolo_run"
    if "/archive/" in rel or rel.endswith("/archive") or name == "archive":
        return "obsolete"
    if "dataset" in name or "/datasets/" in rel or "tiled_dataset" in name:
        return "dataset"
    if "experiment" in rel or name.startswith(("baseline_", "closed_loop_", "my_policy_search", "advisor_", "tiled_baseline")):
        return "experiment"
    if "audit" in name or "/audits/" in rel or "diagnostics" in name or "metric_consistency" in rel:
        return "audit"
    if "debug" in name or "visualization" in name or "test_tmp" in name or name in {"tests", "test_copy_paste", "test_tmp"}:
        return "debug"
    if "smoke" in name:
        return "smoke"
    if "dryrun" in name or "dry_run" in name:
        return "dryrun"
    if name in {"project_snapshot_latest.md", "snapshots"}:
        return "audit"
    return "unknown"


def recommend(path: Path, scope: str, inferred_use: str) -> tuple[bool, bool, bool, str | None, str]:
    rel = relative(path).replace("\\", "/")
    archive_outputs = "outputs/archive/old_outputs_20260517"
    archive_runs = "outputs/archive/old_runs_20260517/runs_detect"
    if rel in {"outputs/project_snapshot_latest.md", "outputs/snapshots/project_snapshot_latest.md"}:
        return True, False, False, None, "Compatibility project snapshot path."
    if rel == "outputs/audits" or rel.startswith("outputs/audits/"):
        return True, False, False, None, "Normalized audit location."
    if rel == "outputs/snapshots" or rel.startswith("outputs/snapshots/"):
        return True, False, False, None, "Normalized audit or snapshot location."
    if rel == "outputs/datasets" or rel.startswith("outputs/datasets/"):
        return True, False, False, None, "Normalized generated dataset location."
    if rel == "outputs/experiments" or rel.startswith("outputs/experiments/"):
        return True, False, False, None, "Normalized experiment location."
    if rel == "outputs/archive" or rel.startswith("outputs/archive/"):
        return True, False, False, None, "Already archived."
    if rel.startswith("outputs/tiled_baseline_20epoch"):
        return True, True, False, "outputs/experiments/20260517_tiled_baseline_20epoch", "Important current baseline should be normalized into experiments."
    if rel.startswith("outputs/tiled_dataset_smoke"):
        return True, True, False, "outputs/datasets/tiled/tiled_1024_ov20_smoke", "Important smoke tiled dataset should be normalized into datasets."
    if rel.startswith("outputs/gpu_preflight_report"):
        return True, True, False, "outputs/audits/gpu_preflight", "GPU preflight report should live under audits."
    if scope == "runs/detect":
        return False, True, False, archive_runs, "Ultralytics auto output should not remain an official result path."
    if scope == "runs":
        if path.is_dir() and not any(path.iterdir()):
            return True, False, False, None, "Empty runs directory retained only as a placeholder; no official results remain here."
        return False, True, False, "outputs/archive/old_runs_20260517", "runs is not an official result store."
    if scope == "outputs" and inferred_use in {"smoke", "dryrun", "debug", "obsolete", "experiment", "dataset", "audit", "unknown"}:
        return False, True, False, archive_outputs, "Legacy root-level output should be archived or normalized."
    return False, False, False, None, "No recommendation rule matched."


def summarize(records: list[ArtifactRecord]) -> dict[str, object]:
    return {
        "count": len(records),
        "keep": sum(1 for record in records if record.recommend_keep),
        "archive": sum(1 for record in records if record.recommend_archive),
        "delete": sum(1 for record in records if record.recommend_delete),
        "unknown": [record.relative_path for record in records if record.inferred_use == "unknown"],
        "runs_detect_entries": [
            record.relative_path for record in records if record.scope == "runs/detect"
        ],
    }


def render_markdown(payload: dict[str, object]) -> str:
    records = payload["records"]
    lines = [
        "# Artifact Inventory",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Project root: `{payload['project_root']}`",
        f"- outputs root: `{payload['outputs_root']}`",
        f"- runs root: `{payload['runs_root']}`",
        "",
        "## Summary",
        "",
    ]
    summary = payload["summary"]
    for key in ["count", "keep", "archive", "delete"]:
        lines.append(f"- {key}: {summary[key]}")
    unknown = summary["unknown"]
    lines.append(f"- unknown entries: {len(unknown)}")
    lines.append("")
    lines.extend(
        [
            "## Inventory",
            "",
            "| path | scope | size MB | last modified | use | keep | archive | delete | target | reason |",
            "|---|---|---:|---|---|---|---|---|---|---|",
        ]
    )
    for record in records:
        lines.append(
            "| {path} | {scope} | {size_mb} | {last_modified} | {use} | {keep} | {archive} | {delete} | {target} | {reason} |".format(
                path=escape_md(record["relative_path"]),
                scope=record["scope"],
                size_mb=record["size_mb"],
                last_modified=record["last_modified"] or "",
                use=record["inferred_use"],
                keep=record["recommend_keep"],
                archive=record["recommend_archive"],
                delete=record["recommend_delete"],
                target=escape_md(record["recommended_target"] or ""),
                reason=escape_md(record["reason"]),
            )
        )
    runs_detect = summary["runs_detect_entries"]
    lines.extend(["", "## runs/detect Entries", ""])
    if runs_detect:
        for rel in runs_detect:
            lines.append(f"- {rel}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def escape_md(value: str) -> str:
    return value.replace("|", "\\|")


if __name__ == "__main__":
    main()
