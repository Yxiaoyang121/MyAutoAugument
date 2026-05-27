from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUN = (
    PROJECT_ROOT
    / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/runs/yolo_default_seed42"
)
CONTROL_RUN = PROJECT_ROOT / "outputs/experiments/yolo_default_inloop_no_feedback_control_50ep"
AUDIT_DIR = PROJECT_ROOT / "outputs/audits/yolo_default_inloop_parity"
SMOKE_NATIVE_RUN = AUDIT_DIR / "smoke_native_yolo_default_1ep"
SMOKE_INLOOP_RUN = AUDIT_DIR / "smoke_inloop_no_feedback_1ep"
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
AUG_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "copy_paste_mode",
    "auto_augment",
    "erasing",
    "close_mosaic",
)
CORE_ARGS = (
    "model",
    "data",
    "epochs",
    "batch",
    "imgsz",
    "device",
    "workers",
    "seed",
    "deterministic",
    "project",
    "name",
    "exist_ok",
    "plots",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "close_mosaic",
    "resume",
)
VAL_ARGS = ("split", "conf", "iou", "max_det", "save_json", "augment", "agnostic_nms", "classes", "half", "dnn")


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_audit(REFERENCE_RUN, CONTROL_RUN)
    write_json(AUDIT_DIR / "parity_audit.json", audit)
    (AUDIT_DIR / "parity_audit_report.md").write_text(build_report(audit), encoding="utf-8")
    print(json.dumps({"report": str((AUDIT_DIR / "parity_audit_report.md").resolve())}, indent=2))


def build_audit(reference_run: Path, control_run: Path) -> dict[str, Any]:
    ref_args = read_yaml(reference_run / "train/args.yaml")
    ctrl_args = read_yaml(control_run / "train/args.yaml")
    ref_results = read_results(reference_run / "train/results.csv")
    ctrl_results = read_results(control_run / "train/results.csv")
    ref_stats = read_json(reference_run / "reports/online_aug_stats.json")
    ctrl_stats = read_json(control_run / "reports/online_aug_stats.json")
    ref_final = read_json(reference_run / "reports/final_metrics.json")
    ctrl_final = read_json(control_run / "reports/final_metrics.json")
    ref_train_command = read_text(reference_run / "configs/train_command.txt") or read_text(reference_run / "logs/train.command.txt")
    ctrl_train_command = read_text(control_run / "configs/train_command.txt") or read_text(control_run / "logs/train.command.txt")
    ref_val_command = read_text(reference_run / "configs/val_command.txt") or read_text(reference_run / "logs/val.command.txt")
    ctrl_val_command = read_text(control_run / "configs/val_command.txt") or read_text(control_run / "logs/val.command.txt")
    ref_log = find_reference_log(reference_run)
    ctrl_log = control_run / "logs/control.stdout.log"
    ref_metrics = compact_metrics(ref_stats.get("val_metrics", {}))
    ctrl_metrics = compact_metrics(ctrl_stats.get("val_metrics", {}))
    result_curve = compare_metric_curves(ref_results, ctrl_results)
    args_diff = diff_keys(ref_args, ctrl_args, CORE_ARGS)
    aug_diff = diff_keys(ref_args, ctrl_args, AUG_KEYS)
    val_diff = diff_keys(ref_args, ctrl_args, VAL_ARGS)
    lr_diff = compare_lr_curves(ref_results, ctrl_results)
    command_findings = {
        "reference_train_command": ref_train_command,
        "control_train_command": ctrl_train_command,
        "reference_val_command": ref_val_command,
        "control_val_command": ctrl_val_command,
        "reference_uses_online_trainer": "OnlineAugDetectionTrainer" in ref_train_command,
        "control_uses_inloop_or_online_trainer": "InLoopFeedbackDetectionTrainer" in ctrl_train_command
        or "OnlineAugDetectionTrainer" in ctrl_train_command,
        "control_declares_default_trainer": "UltralyticsDefaultDetectionTrainer" in ctrl_train_command,
    }
    callback_evidence = read_json(control_run / "reports/epoch_records.json")
    root_cause = infer_root_cause(command_findings, callback_evidence, args_diff, aug_diff, val_diff, lr_diff)
    return {
        "reference_run": str(reference_run.resolve()),
        "control_run": str(control_run.resolve()),
        "reference_metrics": ref_metrics,
        "control_metrics": ctrl_metrics,
        "metric_delta_control_minus_reference": metric_delta(ctrl_metrics, ref_metrics),
        "args_diff": args_diff,
        "augmentation_args_diff": aug_diff,
        "validation_args_diff": val_diff,
        "lr_curve_diff": lr_diff,
        "results_curve_diff": result_curve,
        "commands": command_findings,
        "ultralytics": {
            "reference": extract_ultralytics_version(ref_log),
            "control": extract_ultralytics_version(ctrl_log),
        },
        "close_mosaic": {
            "reference_close_mosaic_arg": ref_args.get("close_mosaic"),
            "control_close_mosaic_arg": ctrl_args.get("close_mosaic"),
            "reference_log_triggered": contains_text(ref_log, "Closing dataloader mosaic"),
            "control_log_triggered": contains_text(ctrl_log, "Closing dataloader mosaic"),
        },
        "best_pt_selection": {
            "reference_best_pt": str((reference_run / "train/weights/best.pt").resolve()),
            "control_best_pt": str((control_run / "train/weights/best.pt").resolve()),
            "reference_val_uses_best_pt": uses_best_pt(ref_val_command, ref_final, reference_run),
            "control_val_uses_best_pt": uses_best_pt(ctrl_val_command, ctrl_final, control_run),
        },
        "industrial_aug_stats": {
            "reference_samples_augmented": ref_stats.get("samples_augmented"),
            "control_samples_augmented": ctrl_stats.get("samples_augmented"),
            "reference_ops": ref_stats.get("ops", {}),
            "control_ops": ctrl_stats.get("ops", {}),
        },
        "custom_entry_changed_behavior": root_cause["custom_entry_changed_behavior"],
        "root_cause": root_cause,
        "repair": {
            "implemented": True,
            "description": "feedback=false and industrial_aug=false now enters a native passthrough branch with no custom trainer, dataset, transform, or feedback callback.",
        },
        "parity_smoke": build_smoke_diff(SMOKE_NATIVE_RUN, SMOKE_INLOOP_RUN),
    }


def infer_root_cause(
    command_findings: dict[str, Any],
    callback_evidence: list[dict[str, Any]],
    args_diff: list[dict[str, Any]],
    aug_diff: list[dict[str, Any]],
    val_diff: list[dict[str, Any]],
    lr_diff: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    if command_findings["reference_uses_online_trainer"]:
        reasons.append("The selected reference run is not a pure native CLI run; it used OnlineAugDetectionTrainer with an empty passthrough policy.")
    if callback_evidence:
        reasons.append("The completed in-loop control recorded epoch callback evidence, so the old no-feedback branch still attached a no-op callback.")
    if any(item["key"] == "data" for item in args_diff):
        reasons.append("The reference used an absolute data YAML path while the control used the provided relative path.")
    if any(item["key"] == "project" for item in args_diff):
        reasons.append("Project/save_dir differ by design and should not affect training math.")
    if lr_diff.get("max_abs_lr_delta", 0.0) and lr_diff["max_abs_lr_delta"] > 1e-12:
        reasons.append("Learning-rate schedules are not identical in the captured results.")
    if aug_diff:
        reasons.append("Augmentation hyperparameters differ.")
    if val_diff:
        reasons.append("Validation arguments differ.")
    return {
        "custom_entry_changed_behavior": bool(callback_evidence),
        "primary": "parity baseline mismatch: reference used custom passthrough OnlineAugDetectionTrainer, while the control used native trainer plus a no-op callback in the old script",
        "reasons": reasons,
        "args_diff_count": len(args_diff),
        "augmentation_diff_count": len(aug_diff),
        "validation_diff_count": len(val_diff),
    }


def build_report(audit: dict[str, Any]) -> str:
    lines = [
        "# YOLO Default In-Loop Parity Audit",
        "",
        "## Runs",
        "",
        f"- Reference run: `{audit['reference_run']}`",
        f"- In-loop no-feedback control run: `{audit['control_run']}`",
        "",
        "## Metrics",
        "",
        metric_table(audit["reference_metrics"], audit["control_metrics"], audit["metric_delta_control_minus_reference"]),
        "",
        "## Command Findings",
        "",
        f"- Reference uses `OnlineAugDetectionTrainer`: `{str(audit['commands']['reference_uses_online_trainer']).lower()}`",
        f"- Control declares default trainer: `{str(audit['commands']['control_declares_default_trainer']).lower()}`",
        f"- Control uses custom trainer by command: `{str(audit['commands']['control_uses_inloop_or_online_trainer']).lower()}`",
        "",
        "Reference train command:",
        "",
        f"```text\n{audit['commands']['reference_train_command']}\n```",
        "",
        "Control train command:",
        "",
        f"```text\n{audit['commands']['control_train_command']}\n```",
        "",
        "## args.yaml Diff",
        "",
        diff_table(audit["args_diff"]),
        "",
        "## Augmentation Args Diff",
        "",
        diff_table(audit["augmentation_args_diff"]),
        "",
        "## Validation Args Diff",
        "",
        diff_table(audit["validation_args_diff"]),
        "",
        "## Learning-Rate Curve",
        "",
        f"- Rows compared: `{audit['lr_curve_diff']['rows_compared']}`",
        f"- Max abs LR delta: `{audit['lr_curve_diff']['max_abs_lr_delta']:.12f}`",
        f"- Last LR reference/control: `{audit['lr_curve_diff']['reference_last_lr']}` / `{audit['lr_curve_diff']['control_last_lr']}`",
        "",
        "## Results Curve",
        "",
        result_curve_table(audit["results_curve_diff"]),
        "",
        "## close_mosaic",
        "",
        f"- Reference close_mosaic arg: `{audit['close_mosaic']['reference_close_mosaic_arg']}`",
        f"- Control close_mosaic arg: `{audit['close_mosaic']['control_close_mosaic_arg']}`",
        f"- Reference log triggered close_mosaic: `{str(audit['close_mosaic']['reference_log_triggered']).lower()}`",
        f"- Control log triggered close_mosaic: `{str(audit['close_mosaic']['control_log_triggered']).lower()}`",
        "",
        "## Best.pt Selection",
        "",
        f"- Reference val uses best.pt: `{str(audit['best_pt_selection']['reference_val_uses_best_pt']).lower()}`",
        f"- Control val uses best.pt: `{str(audit['best_pt_selection']['control_val_uses_best_pt']).lower()}`",
        "",
        "## Industrial Augmentation",
        "",
        f"- Reference samples_augmented: `{audit['industrial_aug_stats']['reference_samples_augmented']}`",
        f"- Control samples_augmented: `{audit['industrial_aug_stats']['control_samples_augmented']}`",
        f"- Reference ops: `{audit['industrial_aug_stats']['reference_ops']}`",
        f"- Control ops: `{audit['industrial_aug_stats']['control_ops']}`",
        "",
        "## Root Cause",
        "",
        f"- Custom entry changed behavior in completed control: `{str(audit['custom_entry_changed_behavior']).lower()}`",
        f"- Primary: {audit['root_cause']['primary']}",
    ]
    for reason in audit["root_cause"]["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Repair Recommendation",
            "",
            "- Do not use the selected reference as proof of pure native YOLO default, because it used `OnlineAugDetectionTrainer` with a passthrough policy.",
            "- For no-feedback parity, compare a native Python API smoke run against the in-loop no-feedback branch.",
            "- The in-loop script has been repaired so `feedback=false` and `industrial_aug=false` directly calls native `YOLO.train(**same_args)` and registers no custom trainer, dataset, transform, or feedback callback.",
            "",
            "## 1 Epoch Parity Smoke",
            "",
            f"- Native smoke run: `{audit['parity_smoke']['native_run']}`",
            f"- In-loop no-feedback smoke run: `{audit['parity_smoke']['inloop_run']}`",
            f"- Smoke completed: `{str(audit['parity_smoke']['completed']).lower()}`",
            f"- args.yaml differences except project/save_dir: `{audit['parity_smoke']['args_diff_excluding_output_count']}`",
            f"- metric/loss/lr max abs delta: `{audit['parity_smoke']['max_abs_result_delta']:.12f}`",
            f"- in-loop epoch callback records: `{audit['parity_smoke']['inloop_epoch_record_count']}`",
            f"- smoke parity passed: `{str(audit['parity_smoke']['passed']).lower()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def diff_keys(left: dict[str, Any], right: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for key in keys:
        left_value = normalize_value(left.get(key))
        right_value = normalize_value(right.get(key))
        if left_value != right_value:
            rows.append({"key": key, "reference": left_value, "control": right_value})
    return rows


def build_smoke_diff(native_run: Path, inloop_run: Path) -> dict[str, Any]:
    native_args = read_yaml(native_run / "train/args.yaml")
    inloop_args = read_yaml(inloop_run / "train/args.yaml")
    native_results = read_results(native_run / "train/results.csv")
    inloop_results = read_results(inloop_run / "train/results.csv")
    ignored = {"project", "save_dir"}
    keys = tuple(sorted((set(native_args) | set(inloop_args)) - ignored))
    args_diff = diff_keys(native_args, inloop_args, keys)
    max_delta = 0.0
    compared_columns: list[str] = []
    if native_results and inloop_results:
        for key, native_value in native_results[0].items():
            if key == "time":
                continue
            try:
                delta = abs(float(inloop_results[0].get(key, 0.0) or 0.0) - float(native_value or 0.0))
            except ValueError:
                continue
            compared_columns.append(key)
            max_delta = max(max_delta, delta)
    inloop_epoch_records = read_json(inloop_run / "reports/epoch_records.json")
    completed = bool(native_results and inloop_results and native_args and inloop_args)
    passed = completed and len(args_diff) == 0 and max_delta <= 1e-12 and isinstance(inloop_epoch_records, list) and len(inloop_epoch_records) == 0
    return {
        "native_run": str(native_run.resolve()),
        "inloop_run": str(inloop_run.resolve()),
        "completed": completed,
        "args_diff_excluding_output": args_diff,
        "args_diff_excluding_output_count": len(args_diff),
        "compared_result_columns": compared_columns,
        "max_abs_result_delta": max_delta,
        "inloop_epoch_record_count": len(inloop_epoch_records) if isinstance(inloop_epoch_records, list) else None,
        "passed": passed,
    }


def compare_lr_curves(reference: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("lr/pg0", "lr/pg1", "lr/pg2")
    max_delta = 0.0
    rows = min(len(reference), len(control))
    for index in range(rows):
        for key in keys:
            delta = abs(float(control[index].get(key, 0.0) or 0.0) - float(reference[index].get(key, 0.0) or 0.0))
            max_delta = max(max_delta, delta)
    return {
        "rows_compared": rows,
        "max_abs_lr_delta": max_delta,
        "reference_last_lr": {key: reference[-1].get(key) for key in keys} if reference else {},
        "control_last_lr": {key: control[-1].get(key) for key in keys} if control else {},
    }


def compare_metric_curves(reference: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, Any]:
    columns = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    rows = min(len(reference), len(control))
    summary = {"rows_compared": rows, "last_epoch": {}, "max_abs_delta": {}}
    for metric, column in columns.items():
        max_delta = 0.0
        for index in range(rows):
            delta = abs(float(control[index].get(column, 0.0) or 0.0) - float(reference[index].get(column, 0.0) or 0.0))
            max_delta = max(max_delta, delta)
        summary["max_abs_delta"][metric] = max_delta
        if rows:
            summary["last_epoch"][metric] = {
                "reference": float(reference[rows - 1].get(column, 0.0) or 0.0),
                "control": float(control[rows - 1].get(column, 0.0) or 0.0),
                "delta": float(control[rows - 1].get(column, 0.0) or 0.0) - float(reference[rows - 1].get(column, 0.0) or 0.0),
            }
    return summary


def result_curve_table(curve: dict[str, Any]) -> str:
    lines = ["| metric | last reference | last control | last delta | max abs delta |", "|---|---:|---:|---:|---:|"]
    for key in METRIC_KEYS:
        last = curve["last_epoch"].get(key, {})
        lines.append(
            f"| {label(key)} | {fmt(last.get('reference'))} | {fmt(last.get('control'))} | {fmt(last.get('delta'))} | {fmt(curve['max_abs_delta'].get(key))} |"
        )
    return "\n".join(lines)


def metric_table(reference: dict[str, Any], control: dict[str, Any], delta: dict[str, Any]) -> str:
    lines = ["| metric | reference | control | control-reference |", "|---|---:|---:|---:|"]
    for key in METRIC_KEYS:
        lines.append(f"| {label(key)} | {fmt(reference.get(key))} | {fmt(control.get(key))} | {fmt(delta.get(key))} |")
    return "\n".join(lines)


def diff_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No differences."
    lines = ["| key | reference | control |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['key']}` | `{row['reference']}` | `{row['control']}` |")
    return "\n".join(lines)


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def read_json(path: Path) -> Any:
    if not path.exists():
        return {} if path.suffix == ".json" else []
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace").strip() if path.exists() else ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_reference_log(reference_run: Path) -> Path:
    candidates = [
        reference_run.parents[1] / "logs/yolo_default.train.stdout.log",
        reference_run / "logs/train.stdout.log",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def uses_best_pt(command: str, final_payload: dict[str, Any], run_dir: Path) -> bool:
    if "weights\\best.pt" in command or "weights/best.pt" in command:
        return True
    train_payload = final_payload.get("train", {}) if isinstance(final_payload, dict) else {}
    best_pt = str(train_payload.get("best_pt", ""))
    return bool(best_pt) and best_pt.replace("/", "\\").endswith("train\\weights\\best.pt") and (run_dir / "train/weights/best.pt").exists()


def contains_text(path: Path, text: str) -> bool:
    return path.exists() and text in path.read_text(encoding="utf-8-sig", errors="replace")


def extract_ultralytics_version(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[:50]:
        match = re.search(r"Ultralytics\s+([0-9.]+)", line)
        if match:
            return match.group(1)
    return None


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in ("images", "instances", *METRIC_KEYS)}


def metric_delta(control: dict[str, Any], reference: dict[str, Any]) -> dict[str, float | None]:
    delta = {}
    for key in METRIC_KEYS:
        left = control.get(key)
        right = reference.get(key)
        delta[key] = None if left is None or right is None else float(left) - float(right)
    return delta


def normalize_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return "null"
    return str(value).replace("/", "\\")


def label(key: str) -> str:
    return {"precision": "Precision", "recall": "Recall", "map50": "mAP50", "map50_95": "mAP50-95"}[key]


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
