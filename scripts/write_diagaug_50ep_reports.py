from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep"
BASELINE_ID = "20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep"
RUN_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_ID
BASELINE_DIR = PROJECT_ROOT / "outputs" / "experiments" / BASELINE_ID
REPORT_DIR = RUN_DIR / "reports"
BASELINE_METRICS_PATH = BASELINE_DIR / "reports" / "baseline_50ep_metrics.json"
DATA_YAML = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position" / "data.yaml"

DISABLED_YOLO_AUGS = {
    "mosaic": 0,
    "mixup": 0,
    "copy_paste": 0,
    "hsv_h": 0,
    "hsv_s": 0,
    "hsv_v": 0,
    "degrees": 0,
    "translate": 0,
    "scale": 0,
    "shear": 0,
    "perspective": 0,
    "fliplr": 0,
    "flipud": 0,
}


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = read_json(BASELINE_METRICS_PATH)
    diagnosis = read_json(RUN_DIR / "diagnosis" / "diagnosis.json")
    candidate_policies = read_json(RUN_DIR / "policies" / "candidate_policies.json")
    proxy_ranking = read_json(RUN_DIR / "proxy" / "proxy_ranking.json")
    selected_policy = read_json(RUN_DIR / "short_training" / "selected_policy.json")
    dataset_report = read_json(RUN_DIR / "dataset_builder" / "dataset_build_report.json")

    train_stdout = RUN_DIR / "logs" / "final_train_batch2.stdout.log"
    train_stderr = RUN_DIR / "logs" / "final_train_batch2.stderr.log"
    val_stdout = RUN_DIR / "logs" / "final_val_batch2.stdout.log"
    val_stderr = RUN_DIR / "logs" / "final_val_batch2.stderr.log"
    val_metrics = parse_yolo_val_log(val_stdout, baseline)
    final_best = RUN_DIR / "train" / "weights" / "best.pt"
    final_last = RUN_DIR / "train" / "weights" / "last.pt"
    train_command = clean_text_value(read_text(RUN_DIR / "logs" / "final_train_batch2.command.txt"))
    val_command = clean_text_value(read_text(RUN_DIR / "logs" / "final_val_batch2.command.txt"))
    final_exitcode = clean_text_value(read_text(RUN_DIR / "logs" / "final_train_batch2.exitcode")) or None
    val_exitcode = clean_text_value(read_text(RUN_DIR / "logs" / "final_val_batch2.exitcode")) or None
    train_results = read_last_training_row(RUN_DIR / "train" / "results.csv")
    train_bbox_count = count_label_rows(Path(dataset_report["labels_train"]))
    val_bbox_count = count_label_rows(Path(dataset_report["labels_val"]))
    oom = detect_oom([train_stdout, train_stderr, val_stdout, val_stderr])

    comparison = build_comparison(baseline, val_metrics)
    copy_paste_policy_count = sum(contains_copy_paste(row.get("policy", {})) for row in proxy_ranking)
    copy_paste_hard_rejected = any(
        contains_copy_paste(row.get("policy", {})) and not bool(row.get("hard_filter_pass"))
        for row in proxy_ranking
    )
    selected_contains_copy_paste = contains_copy_paste(selected_policy)

    metrics_payload = {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_run_id": BASELINE_ID,
        "dataset": str(DATA_YAML.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "baseline_best_pt": baseline["artifacts"]["best_pt"],
        "model": "yolo11n.pt",
        "train_settings": {
            "epochs": 50,
            "imgsz": 1024,
            "batch": 2,
            "workers": 0,
            "device": 0,
            "yolo_builtin_augmentations_disabled": True,
            "disabled_augmentation_params": DISABLED_YOLO_AUGS,
        },
        "diagnostic_flow": {
            "diagnosis_json": rel(RUN_DIR / "diagnosis" / "diagnosis.json"),
            "diagnosis_vector": diagnosis.get("diagnosis_vector", {}),
            "candidate_policies_json": rel(RUN_DIR / "policies" / "candidate_policies.json"),
            "proxy_ranking_json": rel(RUN_DIR / "proxy" / "proxy_ranking.json"),
            "selected_policy_json": rel(RUN_DIR / "short_training" / "selected_policy.json"),
            "copy_paste_policy_count": copy_paste_policy_count,
            "copy_paste_hard_rejected": copy_paste_hard_rejected,
        },
        "selected_policy": selected_policy,
        "selected_policy_contains_copy_paste": selected_contains_copy_paste,
        "augmented_dataset": {
            "path": dataset_report["dataset_dir"],
            "data_yaml": dataset_report["data_yaml"],
            "train_images": int(dataset_report.get("total_train_images", 0)),
            "train_bboxes": train_bbox_count,
            "val_images": int(dataset_report.get("val_count", 0)),
            "val_bboxes": val_bbox_count,
            "original_train_count": int(dataset_report.get("original_train_count", 0)),
            "augmented_train_count": int(dataset_report.get("augmented_train_count", 0)),
            "augment_repeat": int(dataset_report.get("augment_repeat", 0)),
        },
        "commands": {
            "final_train_command": train_command,
            "final_val_command": val_command,
            "final_train_exitcode": parse_exitcode(final_exitcode),
            "final_val_exitcode": parse_exitcode(val_exitcode),
        },
        "artifacts": {
            "best_pt": str(final_best.resolve()),
            "last_pt": str(final_last.resolve()),
            "train_log": rel(train_stdout),
            "val_log": rel(val_stdout),
            "diagnosis_json": rel(RUN_DIR / "diagnosis" / "diagnosis.json"),
            "candidate_policies_json": rel(RUN_DIR / "policies" / "candidate_policies.json"),
            "proxy_ranking_json": rel(RUN_DIR / "proxy" / "proxy_ranking.json"),
            "selected_policy_json": rel(RUN_DIR / "short_training" / "selected_policy.json"),
            "augmented_dataset": dataset_report["dataset_dir"],
        },
        "final_metrics": val_metrics["overall"],
        "final_per_class": val_metrics["per_class"],
        "last_epoch_metrics_from_training_loop": train_results,
        "baseline_metrics": baseline["validation"]["overall"],
        "baseline_per_class": baseline["validation"]["per_class"],
        "comparison": comparison,
        "oom": oom,
        "notes": [
            "Final validation uses the original safe tiled no-OK/no-position data.yaml for a direct baseline comparison.",
            "The augmented dataset retains original training images and adds one diagnosis-driven augmented copy per train image.",
        ],
    }
    write_json(REPORT_DIR / "diagaug_50ep_metrics.json", metrics_payload)
    write_text(REPORT_DIR / "diagaug_50ep_report.md", build_diagaug_report(metrics_payload))
    write_text(REPORT_DIR / "baseline_vs_diagaug.md", build_comparison_report(metrics_payload))
    write_final_training_summary(metrics_payload)
    update_state_docs(metrics_payload)
    print(REPORT_DIR / "diagaug_50ep_metrics.json")
    print(REPORT_DIR / "diagaug_50ep_report.md")
    print(REPORT_DIR / "baseline_vs_diagaug.md")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8-sig", errors="replace")


def clean_text_value(value: str) -> str:
    return value.replace("\ufeff", "").strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def parse_yolo_val_log(path: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    text = strip_ansi(read_text(path))
    rows: list[dict[str, Any]] = []
    overall: dict[str, Any] | None = None
    class_names = {int(item["class_id"]): item["name"] for item in baseline["validation"]["per_class"]}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        values = numeric_tail(parts, 6)
        if values is None:
            continue
        images, instances, precision, recall, ap50, ap50_95 = values
        if parts[0] == "all":
            overall = {
                "images": int(images),
                "instances": int(instances),
                "precision": precision,
                "recall": recall,
                "map50": ap50,
                "map50_95": ap50_95,
            }
            continue
        if overall is None:
            continue
        if len(rows) >= len(class_names):
            continue
        class_id = len(rows)
        rows.append(
            {
                "class_id": class_id,
                "name": class_names.get(class_id, parts[0]),
                "log_name": parts[0],
                "images": int(images),
                "instances": int(instances),
                "precision": precision,
                "recall": recall,
                "ap50": ap50,
                "ap50_95": ap50_95,
            }
        )
    if overall is None:
        raise RuntimeError(f"Could not parse YOLO overall metrics from {path}")
    if len(rows) != len(class_names):
        raise RuntimeError(f"Expected {len(class_names)} per-class rows from {path}, parsed {len(rows)}")
    return {"overall": overall, "per_class": rows}


def numeric_tail(parts: list[str], count: int) -> list[float] | None:
    values: list[float] = []
    for item in reversed(parts):
        try:
            values.append(float(item))
        except ValueError:
            continue
        if len(values) == count:
            return list(reversed(values))
    return None


def read_last_training_row(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    last = rows[-1]
    keys = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    return {name: float(last[column]) for name, column in keys.items() if last.get(column)}


def count_label_rows(labels_dir: Path) -> int:
    total = 0
    for path in labels_dir.rglob("*.txt"):
        total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def detect_oom(paths: list[Path]) -> bool:
    patterns = ["out of memory", "cuda oom", "cuda out of memory"]
    text = "\n".join(read_text(path).lower() for path in paths)
    return any(pattern in text for pattern in patterns)


def parse_exitcode(value: str | None) -> int | str | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def contains_copy_paste(policy: dict[str, Any]) -> bool:
    return any(str(op.get("name", "")).lower() == "copy_paste" for op in policy.get("operations", []) or [])


def build_comparison(baseline: dict[str, Any], diag: dict[str, Any]) -> dict[str, Any]:
    base_overall = baseline["validation"]["overall"]
    diag_overall = diag["overall"]
    overall_delta = {
        "precision": diag_overall["precision"] - base_overall["precision"],
        "recall": diag_overall["recall"] - base_overall["recall"],
        "map50": diag_overall["map50"] - base_overall["map50"],
        "map50_95": diag_overall["map50_95"] - base_overall["map50_95"],
    }
    baseline_classes = {int(item["class_id"]): item for item in baseline["validation"]["per_class"]}
    class_delta = []
    for item in diag["per_class"]:
        class_id = int(item["class_id"])
        base = baseline_classes[class_id]
        class_delta.append(
            {
                "class_id": class_id,
                "name": item["name"],
                "baseline_recall": base["recall"],
                "diagaug_recall": item["recall"],
                "delta_recall": item["recall"] - base["recall"],
                "baseline_ap50": base["ap50"],
                "diagaug_ap50": item["ap50"],
                "delta_ap50": item["ap50"] - base["ap50"],
                "baseline_precision": base["precision"],
                "diagaug_precision": item["precision"],
                "delta_precision": item["precision"] - base["precision"],
            }
        )
    return {
        "overall_delta": overall_delta,
        "per_class_delta": class_delta,
        "largest_recall_gain": max(class_delta, key=lambda item: item["delta_recall"]),
        "largest_recall_drop": min(class_delta, key=lambda item: item["delta_recall"]),
        "largest_ap50_gain": max(class_delta, key=lambda item: item["delta_ap50"]),
        "largest_ap50_drop": min(class_delta, key=lambda item: item["delta_ap50"]),
    }


def build_diagaug_report(payload: dict[str, Any]) -> str:
    final = payload["final_metrics"]
    comparison = payload["comparison"]
    dataset = payload["augmented_dataset"]
    selected_policy = payload["selected_policy"]
    operations = ", ".join(
        f"{op['name']}(p={op['prob']:.3f}, s={op['strength']:.3f})"
        for op in selected_policy.get("operations", [])
    )
    lines = [
        "# Diagnosis-Driven Augmentation 50 Epoch Report",
        "",
        "## Run",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- Baseline best.pt: `{payload['baseline_best_pt']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Model: `{payload['model']}`",
        "- Training settings: `epochs=50 imgsz=1024 batch=2 workers=0 device=0`",
        "- YOLO built-in augmentations disabled: `"
        + " ".join(f"{key}={value}" for key, value in DISABLED_YOLO_AUGS.items())
        + "`",
        f"- OOM: `{str(payload['oom']).lower()}`",
        "",
        "## Selected Policy",
        "",
        f"- Policy: `{selected_policy.get('policy_id')}`",
        f"- Source issue: `{selected_policy.get('source_issue')}`",
        f"- Contains copy_paste: `{str(payload['selected_policy_contains_copy_paste']).lower()}`",
        f"- Operations: {operations}",
        "",
        "## Augmented Dataset",
        "",
        f"- Data YAML: `{dataset['data_yaml']}`",
        f"- Train images: `{dataset['train_images']}`",
        f"- Train bboxes: `{dataset['train_bboxes']}`",
        f"- Val images: `{dataset['val_images']}`",
        f"- Val bboxes: `{dataset['val_bboxes']}`",
        "",
        "## Final Metrics",
        "",
        f"- Precision: `{final['precision']:.3f}`",
        f"- Recall: `{final['recall']:.3f}`",
        f"- mAP50: `{final['map50']:.3f}`",
        f"- mAP50-95: `{final['map50_95']:.3f}`",
        "",
        "## Baseline Delta",
        "",
        metric_delta_table(comparison["overall_delta"]),
        "",
        "## Commands",
        "",
        "Training:",
        "",
        "```powershell",
        payload["commands"]["final_train_command"],
        "```",
        "",
        "Validation:",
        "",
        "```powershell",
        payload["commands"]["final_val_command"],
        "```",
    ]
    return "\n".join(lines) + "\n"


def build_comparison_report(payload: dict[str, Any]) -> str:
    base = payload["baseline_metrics"]
    final = payload["final_metrics"]
    comp = payload["comparison"]
    lines = [
        "# Baseline vs Diagnosis-Driven Augmentation",
        "",
        "## Overall",
        "",
        "| Metric | Baseline | DiagAug | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("map50", "mAP50"),
        ("map50_95", "mAP50-95"),
    ]:
        lines.append(f"| {label} | {base[key]:.3f} | {final[key]:.3f} | {comp['overall_delta'][key]:+.3f} |")
    lines.extend(
        [
            "",
            "## Per-Class Recall/AP50 Delta",
            "",
            "| class id | class | baseline Recall | diagaug Recall | Delta Recall | baseline AP50 | diagaug AP50 | Delta AP50 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in comp["per_class_delta"]:
        lines.append(
            "| {class_id} | {name} | {baseline_recall:.3f} | {diagaug_recall:.3f} | {delta_recall:+.3f} | "
            "{baseline_ap50:.3f} | {diagaug_ap50:.3f} | {delta_ap50:+.3f} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Largest Changes",
            "",
            "- Largest Recall gain: "
            + format_class_delta(comp["largest_recall_gain"], "delta_recall"),
            "- Largest Recall drop: "
            + format_class_delta(comp["largest_recall_drop"], "delta_recall"),
            "- Largest AP50 gain: "
            + format_class_delta(comp["largest_ap50_gain"], "delta_ap50"),
            "- Largest AP50 drop: "
            + format_class_delta(comp["largest_ap50_drop"], "delta_ap50"),
        ]
    )
    return "\n".join(lines) + "\n"


def write_final_training_summary(payload: dict[str, Any]) -> None:
    final_dir = RUN_DIR / "final_training"
    final_dir.mkdir(parents=True, exist_ok=True)
    write_text(final_dir / "train_command.txt", payload["commands"]["final_train_command"])
    write_text(final_dir / "val_command.txt", payload["commands"]["final_val_command"])
    write_json(
        final_dir / "final_train_metrics.json",
        {
            "stage": "final_training",
            "status": "completed",
            "data_yaml": payload["augmented_dataset"]["data_yaml"],
            "model": payload["model"],
            "epochs": payload["train_settings"]["epochs"],
            "imgsz": payload["train_settings"]["imgsz"],
            "batch": payload["train_settings"]["batch"],
            "workers": payload["train_settings"]["workers"],
            "device": payload["train_settings"]["device"],
            "yolo_builtin_augmentations_disabled": payload["train_settings"][
                "yolo_builtin_augmentations_disabled"
            ],
            "disabled_augmentation_params": payload["train_settings"]["disabled_augmentation_params"],
            "train_command": payload["commands"]["final_train_command"],
            "exitcode": payload["commands"]["final_train_exitcode"],
            "best_pt": payload["artifacts"]["best_pt"],
            "last_pt": payload["artifacts"]["last_pt"],
            "last_epoch_metrics_from_training_loop": payload["last_epoch_metrics_from_training_loop"],
            "oom": payload["oom"],
        },
    )
    write_json(
        final_dir / "final_val_metrics.json",
        {
            "stage": "final_validation",
            "status": "completed",
            "data_yaml": str(DATA_YAML),
            "command": payload["commands"]["final_val_command"],
            "exitcode": payload["commands"]["final_val_exitcode"],
            "metrics": payload["final_metrics"],
            "per_class": payload["final_per_class"],
            "baseline_delta": payload["comparison"]["overall_delta"],
            "oom": payload["oom"],
        },
    )
    write_text(final_dir / "final_report.md", build_final_training_report(payload))


def build_final_training_report(payload: dict[str, Any]) -> str:
    final = payload["final_metrics"]
    delta = payload["comparison"]["overall_delta"]
    lines = [
        "# Final YOLO Training Report",
        "",
        "- Status: completed",
        f"- Data YAML: `{payload['augmented_dataset']['data_yaml']}`",
        "- Training settings: `epochs=50 imgsz=1024 batch=2 workers=0 device=0`",
        "- YOLO built-in augmentations disabled: `"
        + " ".join(f"{key}={value}" for key, value in DISABLED_YOLO_AUGS.items())
        + "`",
        f"- Best weights: `{payload['artifacts']['best_pt']}`",
        f"- Precision: `{final['precision']:.3f}` ({delta['precision']:+.3f} vs baseline)",
        f"- Recall: `{final['recall']:.3f}` ({delta['recall']:+.3f} vs baseline)",
        f"- mAP50: `{final['map50']:.3f}` ({delta['map50']:+.3f} vs baseline)",
        f"- mAP50-95: `{final['map50_95']:.3f}` ({delta['map50_95']:+.3f} vs baseline)",
        f"- OOM: `{str(payload['oom']).lower()}`",
        f"- Final train exitcode: `{payload['commands']['final_train_exitcode']}`",
        f"- Final val exitcode: `{payload['commands']['final_val_exitcode']}`",
    ]
    return "\n".join(lines) + "\n"


def metric_delta_table(delta: dict[str, float]) -> str:
    rows = [
        "| Metric | Delta |",
        "| --- | ---: |",
        f"| Precision | {delta['precision']:+.3f} |",
        f"| Recall | {delta['recall']:+.3f} |",
        f"| mAP50 | {delta['map50']:+.3f} |",
        f"| mAP50-95 | {delta['map50_95']:+.3f} |",
    ]
    return "\n".join(rows)


def format_class_delta(item: dict[str, Any], key: str) -> str:
    return f"`{item['name']}` (class {item['class_id']}, {key}={item[key]:+.3f})"


def update_state_docs(payload: dict[str, Any]) -> None:
    section = build_state_section(payload)
    upsert_section(PROJECT_ROOT / "PROJECT_STATE.md", "DIAGAUG_50EP", section)
    upsert_section(PROJECT_ROOT / "CODEX_HANDOFF.md", "DIAGAUG_50EP", section)
    upsert_section(PROJECT_ROOT / "EXPERIMENT_LOG.md", "DIAGAUG_50EP", section)


def build_state_section(payload: dict[str, Any]) -> str:
    final = payload["final_metrics"]
    comp = payload["comparison"]
    dataset = payload["augmented_dataset"]
    selected = payload["selected_policy"]
    return "\n".join(
        [
            "## Diagnosis-Driven Augmentation 50 Epoch Result",
            "",
            f"- Run ID: `{payload['run_id']}`",
            f"- Dataset: `{payload['dataset']}`",
            f"- Baseline best.pt: `{payload['baseline_best_pt']}`",
            f"- Selected policy: `{selected.get('policy_id')}` from `{selected.get('source_issue')}`",
            f"- Selected policy contains copy_paste: `{str(payload['selected_policy_contains_copy_paste']).lower()}`",
            f"- Copy-paste candidates retained in proxy ranking: `{payload['diagnostic_flow']['copy_paste_policy_count']}`",
            f"- Copy-paste hard rejected: `{str(payload['diagnostic_flow']['copy_paste_hard_rejected']).lower()}`",
            f"- Augmented train images / bboxes: `{dataset['train_images']}` / `{dataset['train_bboxes']}`",
            f"- Precision: `{final['precision']:.3f}` ({comp['overall_delta']['precision']:+.3f} vs baseline)",
            f"- Recall: `{final['recall']:.3f}` ({comp['overall_delta']['recall']:+.3f} vs baseline)",
            f"- mAP50: `{final['map50']:.3f}` ({comp['overall_delta']['map50']:+.3f} vs baseline)",
            f"- mAP50-95: `{final['map50_95']:.3f}` ({comp['overall_delta']['map50_95']:+.3f} vs baseline)",
            f"- OOM: `{str(payload['oom']).lower()}`",
            f"- best.pt: `{payload['artifacts']['best_pt']}`",
            f"- Report: `outputs/experiments/{RUN_ID}/reports/diagaug_50ep_report.md`",
            f"- Metrics JSON: `outputs/experiments/{RUN_ID}/reports/diagaug_50ep_metrics.json`",
            f"- Baseline comparison: `outputs/experiments/{RUN_ID}/reports/baseline_vs_diagaug.md`",
        ]
    )


def upsert_section(path: Path, key: str, section: str) -> None:
    start = f"<!-- {key}_START -->"
    end = f"<!-- {key}_END -->"
    text = read_text(path)
    block = f"{start}\n{section.rstrip()}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _match: block, text)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    write_text(path, text)


if __name__ == "__main__":
    main()
