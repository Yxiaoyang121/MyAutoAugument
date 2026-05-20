from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep"
BASELINE_ID = "20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep"
DIAGAUG_ID = "20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep"

RUN_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_ID
BASELINE_DIR = PROJECT_ROOT / "outputs" / "experiments" / BASELINE_ID
DIAGAUG_DIR = PROJECT_ROOT / "outputs" / "experiments" / DIAGAUG_ID
REPORT_DIR = RUN_DIR / "reports"

DATA_YAML = (
    PROJECT_ROOT
    / "outputs"
    / "datasets"
    / "tiled"
    / "tiled_1024_ov20_full_safe_no_ok_position"
    / "data.yaml"
)
BASELINE_METRICS_PATH = BASELINE_DIR / "reports" / "baseline_50ep_metrics.json"
DIAGAUG_METRICS_PATH = DIAGAUG_DIR / "reports" / "diagaug_50ep_metrics.json"

YOLO_AUG_KEYS = [
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
]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = read_json(BASELINE_METRICS_PATH)
    diagaug = read_json(DIAGAUG_METRICS_PATH)
    train_args = load_simple_yaml(RUN_DIR / "train" / "args.yaml")
    val_metrics = parse_yolo_val_log(RUN_DIR / "logs" / "val_best.stdout.log", baseline)
    train_summary = read_training_summary(RUN_DIR / "train" / "results.csv")

    baseline_overall = baseline["validation"]["overall"]
    diagaug_overall = diagaug["final_metrics"]
    default_overall = val_metrics["overall"]
    comparisons = {
        "vs_baseline": build_comparison(
            baseline_overall,
            baseline["validation"]["per_class"],
            default_overall,
            val_metrics["per_class"],
            "baseline",
            "yolo_default",
        ),
        "vs_diagaug": build_comparison(
            diagaug_overall,
            diagaug["final_per_class"],
            default_overall,
            val_metrics["per_class"],
            "diagaug",
            "yolo_default",
        ),
        "three_way_per_class": build_three_way_per_class(
            baseline["validation"]["per_class"],
            diagaug["final_per_class"],
            val_metrics["per_class"],
        ),
    }

    commands = {
        "train_command": clean_text_value(read_text(RUN_DIR / "logs" / "train_batch2.command.txt")),
        "val_command": clean_text_value(read_text(RUN_DIR / "logs" / "val_best.command.txt")),
        "train_exitcode": parse_exitcode(read_text(RUN_DIR / "logs" / "train_batch2.exitcode").strip()),
        "val_exitcode": parse_exitcode(read_text(RUN_DIR / "logs" / "val_best.exitcode").strip()),
    }
    train_time = {
        "train_start_utc": clean_text_value(read_text(RUN_DIR / "logs" / "train_batch2.start_utc.txt")),
        "train_end_utc": clean_text_value(read_text(RUN_DIR / "logs" / "train_batch2.end_utc.txt")),
        "val_start_utc": clean_text_value(read_text(RUN_DIR / "logs" / "val_best.start_utc.txt")),
        "val_end_utc": clean_text_value(read_text(RUN_DIR / "logs" / "val_best.end_utc.txt")),
        "training_wall_seconds": wall_seconds(
            read_text(RUN_DIR / "logs" / "train_batch2.start_utc.txt"),
            read_text(RUN_DIR / "logs" / "train_batch2.end_utc.txt"),
        ),
        "validation_wall_seconds": wall_seconds(
            read_text(RUN_DIR / "logs" / "val_best.start_utc.txt"),
            read_text(RUN_DIR / "logs" / "val_best.end_utc.txt"),
        ),
        "training_loop_time_seconds": train_summary.get("last_epoch", {}).get("time_seconds"),
    }
    actual_aug = {key: train_args.get(key) for key in YOLO_AUG_KEYS if key in train_args}
    oom = detect_oom(
        [
            RUN_DIR / "logs" / "train_batch2.stdout.log",
            RUN_DIR / "logs" / "train_batch2.stderr.log",
            RUN_DIR / "logs" / "val_best.stdout.log",
            RUN_DIR / "logs" / "val_best.stderr.log",
        ]
    )

    payload = {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(DATA_YAML.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "model": "yolo11n.pt",
        "train_settings": {
            "epochs": int(train_args.get("epochs", 50)),
            "imgsz": int(train_args.get("imgsz", 1024)),
            "batch": int(train_args.get("batch", 2)),
            "workers": int(train_args.get("workers", 0)),
            "device": train_args.get("device", "0"),
            "yolo_default_augmentations_enabled": True,
            "actual_augmentation_params_from_args_yaml": actual_aug,
        },
        "commands": commands,
        "time": train_time,
        "artifacts": {
            "best_pt": str((RUN_DIR / "train" / "weights" / "best.pt").resolve()),
            "last_pt": str((RUN_DIR / "train" / "weights" / "last.pt").resolve()),
            "train_args_yaml": rel(RUN_DIR / "train" / "args.yaml"),
            "train_results_csv": rel(RUN_DIR / "train" / "results.csv"),
            "train_log": rel(RUN_DIR / "logs" / "train_batch2.stdout.log"),
            "val_log": rel(RUN_DIR / "logs" / "val_best.stdout.log"),
            "reports_dir": rel(REPORT_DIR),
        },
        "final_metrics": default_overall,
        "final_per_class": val_metrics["per_class"],
        "baseline_metrics": baseline_overall,
        "baseline_per_class": baseline["validation"]["per_class"],
        "diagaug_metrics": diagaug_overall,
        "diagaug_per_class": diagaug["final_per_class"],
        "comparisons": comparisons,
        "training_results": train_summary,
        "oom": oom,
        "notes": [
            "This control group intentionally leaves Ultralytics training augmentations at their defaults.",
            "Final validation uses the same safe tiled no-OK/no-position data.yaml as the baseline and DiagAug runs.",
        ],
    }

    write_json(REPORT_DIR / "yolo_default_aug_50ep_metrics.json", payload)
    write_text(REPORT_DIR / "yolo_default_aug_50ep_report.md", build_run_report(payload))
    write_text(REPORT_DIR / "compare_baseline_yolo_default_diagaug.md", build_compare_report(payload))
    update_state_docs(payload)
    print(REPORT_DIR / "yolo_default_aug_50ep_metrics.json")
    print(REPORT_DIR / "yolo_default_aug_50ep_report.md")
    print(REPORT_DIR / "compare_baseline_yolo_default_diagaug.md")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or raw[:200].count(b"\x00") > 20:
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def clean_text_value(value: str) -> str:
    return value.replace("\ufeff", "").strip()


def load_simple_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    parsed: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" not in line or line.startswith((" ", "\t", "#")):
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = parse_scalar(value.strip())
    return parsed


def parse_scalar(value: str) -> Any:
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


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
        if overall is None or len(rows) >= len(class_names):
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
        raise RuntimeError(f"Expected {len(class_names)} class rows from {path}, parsed {len(rows)}")
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


def read_training_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def metrics_from_row(row: dict[str, str]) -> dict[str, Any]:
        return {
            "epoch": int(float(row["epoch"])),
            "time_seconds": float(row["time"]),
            "precision": float(row["metrics/precision(B)"]),
            "recall": float(row["metrics/recall(B)"]),
            "map50": float(row["metrics/mAP50(B)"]),
            "map50_95": float(row["metrics/mAP50-95(B)"]),
        }

    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "last_epoch": metrics_from_row(rows[-1]),
        "best_epoch_by_map50_95": metrics_from_row(best),
        "epoch_count": len(rows),
    }


def build_comparison(
    reference_overall: dict[str, Any],
    reference_per_class: list[dict[str, Any]],
    target_overall: dict[str, Any],
    target_per_class: list[dict[str, Any]],
    reference_key: str,
    target_key: str,
) -> dict[str, Any]:
    overall_delta = {
        "precision": target_overall["precision"] - reference_overall["precision"],
        "recall": target_overall["recall"] - reference_overall["recall"],
        "map50": target_overall["map50"] - reference_overall["map50"],
        "map50_95": target_overall["map50_95"] - reference_overall["map50_95"],
    }
    ref_classes = {int(item["class_id"]): item for item in reference_per_class}
    class_delta: list[dict[str, Any]] = []
    for item in target_per_class:
        class_id = int(item["class_id"])
        ref = ref_classes[class_id]
        class_delta.append(
            {
                "class_id": class_id,
                "name": item["name"],
                f"{reference_key}_recall": ref["recall"],
                f"{target_key}_recall": item["recall"],
                "delta_recall": item["recall"] - ref["recall"],
                f"{reference_key}_ap50": ref["ap50"],
                f"{target_key}_ap50": item["ap50"],
                "delta_ap50": item["ap50"] - ref["ap50"],
                f"{reference_key}_precision": ref["precision"],
                f"{target_key}_precision": item["precision"],
                "delta_precision": item["precision"] - ref["precision"],
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


def build_three_way_per_class(
    baseline_per_class: list[dict[str, Any]],
    diagaug_per_class: list[dict[str, Any]],
    default_per_class: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = {int(item["class_id"]): item for item in baseline_per_class}
    diagaug = {int(item["class_id"]): item for item in diagaug_per_class}
    rows = []
    for item in default_per_class:
        class_id = int(item["class_id"])
        base = baseline[class_id]
        diag = diagaug[class_id]
        rows.append(
            {
                "class_id": class_id,
                "name": item["name"],
                "baseline_recall": base["recall"],
                "diagaug_recall": diag["recall"],
                "yolo_default_recall": item["recall"],
                "delta_recall_vs_baseline": item["recall"] - base["recall"],
                "delta_recall_vs_diagaug": item["recall"] - diag["recall"],
                "baseline_ap50": base["ap50"],
                "diagaug_ap50": diag["ap50"],
                "yolo_default_ap50": item["ap50"],
                "delta_ap50_vs_baseline": item["ap50"] - base["ap50"],
                "delta_ap50_vs_diagaug": item["ap50"] - diag["ap50"],
            }
        )
    return rows


def wall_seconds(start_text: str, end_text: str) -> float | None:
    start = parse_datetime(clean_text_value(start_text))
    end = parse_datetime(clean_text_value(end_text))
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_exitcode(value: str | None) -> int | str | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def detect_oom(paths: list[Path]) -> bool:
    patterns = ["out of memory", "cuda oom", "cuda out of memory"]
    text = "\n".join(read_text(path).lower() for path in paths)
    return any(pattern in text for pattern in patterns)


def build_run_report(payload: dict[str, Any]) -> str:
    final = payload["final_metrics"]
    train = payload["train_settings"]
    delta_base = payload["comparisons"]["vs_baseline"]["overall_delta"]
    delta_diag = payload["comparisons"]["vs_diagaug"]["overall_delta"]
    lines = [
        "# YOLO Default Augmentation 50 Epoch Report",
        "",
        "## Run",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Model: `{payload['model']}`",
        "- Training settings: "
        f"`epochs={train['epochs']} imgsz={train['imgsz']} batch={train['batch']} "
        f"workers={train['workers']} device={train['device']}`",
        "- YOLO default augmentations: enabled by leaving Ultralytics defaults untouched.",
        f"- OOM: `{str(payload['oom']).lower()}`",
        f"- Training wall time: `{format_seconds(payload['time']['training_wall_seconds'])}`",
        f"- best.pt: `{payload['artifacts']['best_pt']}`",
        "",
        "## Actual Ultralytics Augmentation Args",
        "",
        "| Parameter | Value |",
        "| --- | ---: |",
    ]
    for key, value in payload["train_settings"]["actual_augmentation_params_from_args_yaml"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Final Metrics",
            "",
            "| Metric | YOLO default | Delta vs baseline | Delta vs DiagAug |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, label in metric_labels():
        lines.append(
            f"| {label} | {final[key]:.3f} | {delta_base[key]:+.3f} | {delta_diag[key]:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "Training:",
            "",
            "```powershell",
            payload["commands"]["train_command"],
            "```",
            "",
            "Validation:",
            "",
            "```powershell",
            payload["commands"]["val_command"],
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def build_compare_report(payload: dict[str, Any]) -> str:
    baseline = payload["baseline_metrics"]
    diagaug = payload["diagaug_metrics"]
    default = payload["final_metrics"]
    comp_base = payload["comparisons"]["vs_baseline"]
    comp_diag = payload["comparisons"]["vs_diagaug"]
    lines = [
        "# Baseline vs YOLO Default Augmentation vs DiagAug",
        "",
        "## Overall",
        "",
        "| Metric | No-YOLO-Aug baseline | YOLO default aug | DiagAug | YOLO default delta vs baseline | YOLO default delta vs DiagAug |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in metric_labels():
        lines.append(
            f"| {label} | {baseline[key]:.3f} | {default[key]:.3f} | {diagaug[key]:.3f} | "
            f"{comp_base['overall_delta'][key]:+.3f} | {comp_diag['overall_delta'][key]:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Augmentation Settings",
            "",
            "| Group | Augmentation settings |",
            "| --- | --- |",
            "| No-YOLO-Aug baseline | `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0` |",
            "| YOLO default aug | `"
            + " ".join(
                f"{key}={value}"
                for key, value in payload["train_settings"]["actual_augmentation_params_from_args_yaml"].items()
            )
            + "` |",
            "| DiagAug | Diagnosis-selected offline augmented dataset, YOLO built-in augmentations disabled during final train. See DiagAug report for selected policy details. |",
            "",
            "## Per-Class Recall/AP50",
            "",
            "| class id | class | baseline R | YOLO default R | DiagAug R | default Delta R vs base | default Delta R vs DiagAug | baseline AP50 | YOLO default AP50 | DiagAug AP50 | default Delta AP50 vs base | default Delta AP50 vs DiagAug |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in payload["comparisons"]["three_way_per_class"]:
        lines.append(
            "| {class_id} | {name} | {baseline_recall:.3f} | {yolo_default_recall:.3f} | {diagaug_recall:.3f} | "
            "{delta_recall_vs_baseline:+.3f} | {delta_recall_vs_diagaug:+.3f} | {baseline_ap50:.3f} | "
            "{yolo_default_ap50:.3f} | {diagaug_ap50:.3f} | {delta_ap50_vs_baseline:+.3f} | "
            "{delta_ap50_vs_diagaug:+.3f} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Largest YOLO Default Changes",
            "",
            "- Largest Recall gain vs baseline: "
            + format_class_delta(comp_base["largest_recall_gain"], "delta_recall"),
            "- Largest Recall drop vs baseline: "
            + format_class_delta(comp_base["largest_recall_drop"], "delta_recall"),
            "- Largest AP50 gain vs baseline: "
            + format_class_delta(comp_base["largest_ap50_gain"], "delta_ap50"),
            "- Largest AP50 drop vs baseline: "
            + format_class_delta(comp_base["largest_ap50_drop"], "delta_ap50"),
            "- Largest Recall gain vs DiagAug: "
            + format_class_delta(comp_diag["largest_recall_gain"], "delta_recall"),
            "- Largest Recall drop vs DiagAug: "
            + format_class_delta(comp_diag["largest_recall_drop"], "delta_recall"),
            "- Largest AP50 gain vs DiagAug: "
            + format_class_delta(comp_diag["largest_ap50_gain"], "delta_ap50"),
            "- Largest AP50 drop vs DiagAug: "
            + format_class_delta(comp_diag["largest_ap50_drop"], "delta_ap50"),
        ]
    )
    return "\n".join(lines) + "\n"


def metric_labels() -> list[tuple[str, str]]:
    return [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("map50", "mAP50"),
        ("map50_95", "mAP50-95"),
    ]


def format_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    hours = value / 3600.0
    return f"{value:.1f}s ({hours:.2f}h)"


def format_class_delta(item: dict[str, Any], key: str) -> str:
    return f"`{item['name']}` (class {item['class_id']}, {key}={item[key]:+.3f})"


def update_state_docs(payload: dict[str, Any]) -> None:
    section = build_state_section(payload)
    upsert_section(PROJECT_ROOT / "PROJECT_STATE.md", "YOLO_DEFAULT_AUG_50EP", section)
    upsert_section(PROJECT_ROOT / "CODEX_HANDOFF.md", "YOLO_DEFAULT_AUG_50EP", section)
    upsert_section(PROJECT_ROOT / "EXPERIMENT_LOG.md", "YOLO_DEFAULT_AUG_50EP", section)


def build_state_section(payload: dict[str, Any]) -> str:
    final = payload["final_metrics"]
    delta_base = payload["comparisons"]["vs_baseline"]["overall_delta"]
    delta_diag = payload["comparisons"]["vs_diagaug"]["overall_delta"]
    train = payload["train_settings"]
    return "\n".join(
        [
            "## YOLO Default Augmentation 50 Epoch Control",
            "",
            f"- Run ID: `{payload['run_id']}`",
            f"- Dataset: `{payload['dataset']}`",
            f"- Model/settings: `yolo11n.pt epochs={train['epochs']} imgsz={train['imgsz']} batch={train['batch']} workers={train['workers']} device={train['device']}`",
            "- YOLO default augmentations enabled; actual args recorded from `train/args.yaml`.",
            f"- Precision: `{final['precision']:.3f}` ({delta_base['precision']:+.3f} vs baseline, {delta_diag['precision']:+.3f} vs DiagAug)",
            f"- Recall: `{final['recall']:.3f}` ({delta_base['recall']:+.3f} vs baseline, {delta_diag['recall']:+.3f} vs DiagAug)",
            f"- mAP50: `{final['map50']:.3f}` ({delta_base['map50']:+.3f} vs baseline, {delta_diag['map50']:+.3f} vs DiagAug)",
            f"- mAP50-95: `{final['map50_95']:.3f}` ({delta_base['map50_95']:+.3f} vs baseline, {delta_diag['map50_95']:+.3f} vs DiagAug)",
            f"- OOM: `{str(payload['oom']).lower()}`",
            f"- Training wall time: `{format_seconds(payload['time']['training_wall_seconds'])}`",
            f"- best.pt: `{payload['artifacts']['best_pt']}`",
            f"- Report: `outputs/experiments/{RUN_ID}/reports/yolo_default_aug_50ep_report.md`",
            f"- Comparison: `outputs/experiments/{RUN_ID}/reports/compare_baseline_yolo_default_diagaug.md`",
            f"- Metrics JSON: `outputs/experiments/{RUN_ID}/reports/yolo_default_aug_50ep_metrics.json`",
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
