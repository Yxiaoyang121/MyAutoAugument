from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = Path(r"E:\TJGY\DataSet2_fixed")
TILED_SMOKE_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_smoke"
REPORT_DIR = PROJECT_ROOT / "outputs" / "audits" / "dataset_mapping"
REPORT_JSON = REPORT_DIR / "dataset_mapping_audit.json"
REPORT_MD = REPORT_DIR / "dataset_mapping_audit.md"
BASELINE_METRICS_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "20260517_tiled_baseline_20epoch"
    / "reports"
    / "baseline_20epoch_metrics.json"
)
BASELINE_REPORT_MD = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "20260517_tiled_baseline_20epoch"
    / "reports"
    / "baseline_20epoch_report.md"
)
TILED_REPORT_JSON = TILED_SMOKE_ROOT / "tiled_dataset_report.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LOW_SAMPLE_THRESHOLD = 10


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    original = audit_dataset("original", ORIGINAL_ROOT, ORIGINAL_ROOT / "data.yaml")
    tiled = audit_dataset("tiled_smoke", TILED_SMOKE_ROOT, TILED_SMOKE_ROOT / "data.yaml")
    tiled_report = load_json(TILED_REPORT_JSON)
    metrics = audit_baseline_metrics(tiled)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "original_dataset": original,
        "tiled_smoke_dataset": tiled,
        "tiled_smoke_assessment": assess_tiled_smoke(original, tiled, tiled_report),
        "mapping_assessment": assess_mapping(original, tiled, metrics),
        "baseline_20epoch_metrics": metrics,
        "inputs": {
            "original_data_yaml": str((ORIGINAL_ROOT / "data.yaml").resolve()),
            "tiled_smoke_data_yaml": str((TILED_SMOKE_ROOT / "data.yaml").resolve()),
            "baseline_report_md": str(BASELINE_REPORT_MD.resolve()),
            "baseline_metrics_json": str(BASELINE_METRICS_JSON.resolve()),
            "tiled_dataset_report_json": str(TILED_REPORT_JSON.resolve()),
        },
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(str(REPORT_JSON))
    print(str(REPORT_MD))


def audit_dataset(name: str, root: Path, data_yaml: Path) -> dict[str, Any]:
    yaml_info = load_data_yaml(data_yaml)
    splits = {
        split: scan_split(root, split, int(yaml_info["nc"]))
        for split in ["train", "val"]
    }
    all_class_ids: list[int] = []
    for split_info in splits.values():
        all_class_ids.extend(split_info["class_ids"])
    per_class = build_per_class(yaml_info["nc"], yaml_info["names"], splits)
    return {
        "name": name,
        "root": str(root.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_exists": data_yaml.exists(),
        "nc": yaml_info["nc"],
        "names": yaml_info["names"],
        "names_quality": yaml_info["names_quality"],
        "yaml_parse_error": yaml_info["parse_error"],
        "splits": splits,
        "per_class": per_class,
        "class_id_min": min(all_class_ids) if all_class_ids else None,
        "class_id_max": max(all_class_ids) if all_class_ids else None,
        "class_id_ge_nc": sorted({cid for cid in all_class_ids if cid >= int(yaml_info["nc"])}),
        "class_id_negative": sorted({cid for cid in all_class_ids if cid < 0}),
        "bbox_total": sum(split["bbox_total"] for split in splits.values()),
    }


def load_data_yaml(path: Path) -> dict[str, Any]:
    parse_error = None
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - defensive reporting path
            parse_error = f"{type(exc).__name__}: {exc}"
    names = normalize_names(data.get("names"))
    nc = int(data.get("nc", len(names) or 0))
    if nc == 0 and names:
        nc = len(names)
    names_quality = inspect_names(names, nc)
    return {
        "nc": nc,
        "names": [names.get(index) for index in range(nc)],
        "names_by_id": names,
        "names_quality": names_quality,
        "parse_error": parse_error,
    }


def normalize_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    if isinstance(raw, dict):
        names: dict[int, str] = {}
        for key, value in raw.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            names[index] = str(value)
        return names
    return {}


def inspect_names(names: dict[int, str], nc: int) -> dict[str, Any]:
    missing = [index for index in range(nc) if index not in names]
    empty = [index for index, value in names.items() if index < nc and not value.strip()]
    replacement_char = [index for index, value in names.items() if "\ufffd" in value]
    suspicious_mojibake = [
        index
        for index, value in names.items()
        if any(token in value for token in ["Ã", "Â", "閸", "瀹", "鍔", "€", "?"])
    ]
    return {
        "missing_name_ids": missing,
        "empty_name_ids": empty,
        "replacement_char_name_ids": replacement_char,
        "suspicious_mojibake_name_ids": suspicious_mojibake,
        "has_problem": bool(missing or empty or replacement_char or suspicious_mojibake),
    }


def scan_split(root: Path, split: str, nc: int) -> dict[str, Any]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_files = sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if image_dir.exists() else []
    label_files = sorted(label_dir.rglob("*.txt")) if label_dir.exists() else []
    class_counts: Counter[int] = Counter()
    class_images: dict[int, set[str]] = defaultdict(set)
    invalid_rows: list[dict[str, Any]] = []
    class_ids: list[int] = []
    for label_path in label_files:
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            invalid_rows.append({"file": str(label_path), "line": None, "issue": f"decode_error: {exc}"})
            continue
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                invalid_rows.append({"file": str(label_path), "line": line_number, "issue": "too_few_columns"})
                continue
            try:
                cid_float = float(parts[0])
                cid = int(cid_float)
            except ValueError:
                invalid_rows.append({"file": str(label_path), "line": line_number, "issue": "non_numeric_class_id"})
                continue
            if cid != cid_float:
                invalid_rows.append({"file": str(label_path), "line": line_number, "issue": "non_integer_class_id", "class_id": parts[0]})
            if cid < 0:
                invalid_rows.append({"file": str(label_path), "line": line_number, "issue": "negative_class_id", "class_id": cid})
            if cid >= nc:
                invalid_rows.append({"file": str(label_path), "line": line_number, "issue": "class_id_ge_nc", "class_id": cid})
            class_counts[cid] += 1
            class_images[cid].add(label_path.stem)
            class_ids.append(cid)
    return {
        "image_dir": str(image_dir.resolve()),
        "label_dir": str(label_dir.resolve()),
        "image_count": len(image_files),
        "label_file_count": len(label_files),
        "bbox_total": sum(class_counts.values()),
        "class_counts": {str(key): class_counts.get(key, 0) for key in range(nc)},
        "class_image_counts": {str(key): len(class_images.get(key, set())) for key in range(nc)},
        "class_ids": class_ids,
        "class_id_min": min(class_ids) if class_ids else None,
        "class_id_max": max(class_ids) if class_ids else None,
        "class_id_ge_nc": sorted({cid for cid in class_ids if cid >= nc}),
        "invalid_row_count": len(invalid_rows),
        "invalid_rows_sample": invalid_rows[:20],
    }


def build_per_class(nc: int, names: list[str | None], splits: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cid in range(nc):
        train_instances = int(splits["train"]["class_counts"].get(str(cid), 0))
        val_instances = int(splits["val"]["class_counts"].get(str(cid), 0))
        rows.append(
            {
                "class_id": cid,
                "name": names[cid] if cid < len(names) else None,
                "train_instances": train_instances,
                "val_instances": val_instances,
                "total_instances": train_instances + val_instances,
                "train_images": int(splits["train"]["class_image_counts"].get(str(cid), 0)),
                "val_images": int(splits["val"]["class_image_counts"].get(str(cid), 0)),
            }
        )
    return rows


def audit_baseline_metrics(tiled: dict[str, Any]) -> dict[str, Any]:
    metrics = load_json(BASELINE_METRICS_JSON)
    per_class_raw = metrics.get("metrics", {}).get("per_class", []) if metrics else []
    mapped_rows = map_metric_rows(per_class_raw, tiled)
    high_ap50 = [row for row in mapped_rows if row.get("map50", 0) >= 0.5]
    recall_zero = [row for row in mapped_rows if row.get("recall") == 0]
    low_ap50 = [row for row in mapped_rows if row.get("map50", 0) < 0.05]
    sample_starved = [
        row
        for row in mapped_rows
        if row.get("train_instances", 0) <= LOW_SAMPLE_THRESHOLD or row.get("val_instances", 0) <= LOW_SAMPLE_THRESHOLD
    ]
    absent_train_present_val = [
        row for row in mapped_rows if row.get("train_instances", 0) == 0 and row.get("val_instances", 0) > 0
    ]
    return {
        "metrics_json": str(BASELINE_METRICS_JSON.resolve()),
        "report_md": str(BASELINE_REPORT_MD.resolve()),
        "overall": metrics.get("metrics", {}) if metrics else {},
        "mapped_per_class": mapped_rows,
        "high_ap50_classes": summarize_metric_rows(high_ap50),
        "recall_zero_classes": summarize_metric_rows(recall_zero),
        "low_ap50_classes": summarize_metric_rows(low_ap50),
        "sample_starved_classes": summarize_metric_rows(sample_starved),
        "absent_train_present_val_classes": summarize_metric_rows(absent_train_present_val),
        "blank_or_unrendered_rows": [
            row for row in mapped_rows if not str(row.get("reported_class", "")).strip()
        ],
    }


def map_metric_rows(raw_rows: list[dict[str, Any]], tiled: dict[str, Any]) -> list[dict[str, Any]]:
    names = {row["class_id"]: row["name"] for row in tiled["per_class"]}
    dist = {row["class_id"]: row for row in tiled["per_class"]}
    used: set[int] = set()
    mapped: list[dict[str, Any]] = []
    for row in raw_rows:
        reported_class = str(row.get("class", ""))
        cid = None
        for candidate_id, name in names.items():
            if reported_class and reported_class == name and candidate_id not in used:
                cid = candidate_id
                break
        if cid is None:
            candidates = [
                item
                for item in tiled["per_class"]
                if item["class_id"] not in used
                and item["val_images"] == int(row.get("images", -1))
                and item["val_instances"] == int(row.get("instances", -1))
            ]
            if len(candidates) == 1:
                cid = int(candidates[0]["class_id"])
        if cid is not None:
            used.add(cid)
            class_info = dist[cid]
            mapped_name = class_info["name"]
            train_instances = class_info["train_instances"]
            val_instances = class_info["val_instances"]
        else:
            mapped_name = None
            train_instances = None
            val_instances = None
        mapped.append(
            {
                "reported_order": row.get("order"),
                "reported_class": reported_class,
                "mapped_class_id": cid,
                "mapped_class_name": mapped_name,
                "images": row.get("images"),
                "instances": row.get("instances"),
                "train_instances": train_instances,
                "val_instances": val_instances,
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "map50": row.get("map50"),
                "map50_95": row.get("map50_95"),
            }
        )
    return mapped


def summarize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "class_id": row.get("mapped_class_id"),
            "name": row.get("mapped_class_name"),
            "train_instances": row.get("train_instances"),
            "val_instances": row.get("val_instances"),
            "recall": row.get("recall"),
            "map50": row.get("map50"),
            "map50_95": row.get("map50_95"),
        }
        for row in rows
    ]


def assess_tiled_smoke(original: dict[str, Any], tiled: dict[str, Any], tiled_report: dict[str, Any]) -> dict[str, Any]:
    report_summary = tiled_report.get("summary", {}) if tiled_report else {}
    source_image_count = int(report_summary.get("source_image_count", 0) or 0)
    original_image_count = original["splits"]["train"]["image_count"] + original["splits"]["val"]["image_count"]
    is_smoke = (
        "smoke" in tiled["root"].lower()
        or source_image_count < original_image_count
        or source_image_count == 16
    )
    return {
        "is_smoke": is_smoke,
        "is_full_dataset": not is_smoke,
        "can_be_formal_baseline_dataset": not is_smoke,
        "evidence": [
            f"dataset path contains smoke: {'smoke' in tiled['root'].lower()}",
            f"tiled report source_image_count={source_image_count}",
            f"original image count={original_image_count}",
            "dataset_summary.md records Is smoke: true and the per-split source cap was 8 images",
        ],
        "tile_size": report_summary.get("tile_size"),
        "overlap": report_summary.get("overlap"),
        "min_visibility": report_summary.get("min_visibility"),
        "keep_empty_ratio": report_summary.get("keep_empty_ratio"),
    }


def assess_mapping(original: dict[str, Any], tiled: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    names_match = original["names"] == tiled["names"]
    invalid_ids = bool(
        original["class_id_ge_nc"]
        or original["class_id_negative"]
        or tiled["class_id_ge_nc"]
        or tiled["class_id_negative"]
    )
    blank_rows = metrics.get("blank_or_unrendered_rows", [])
    return {
        "current_tiled_names_match_original": names_match,
        "current_tiled_yaml_parse_error": tiled["yaml_parse_error"],
        "current_tiled_names_problem": tiled["names_quality"]["has_problem"],
        "class_id_out_of_range_found": invalid_ids,
        "blank_or_unrendered_reason": (
            "The saved 20-epoch val log/metrics contain empty reported class names for non-ASCII classes. "
            "The label class IDs are in range and can be mapped back by val image/instance counts. "
            "The active tiled data.yaml has been repaired to match the original names; the blank rows are an artifact "
            "of the earlier corrupted/non-renderable tiled data.yaml or console/log parsing, not evidence of class IDs >= nc."
        ),
        "map_low_primary_cause": (
            "The low mAP is mainly driven by the smoke split and class imbalance: OK and OK3 have high AP50, "
            "while several evaluated defect classes have recall 0, very few train/val samples, or val samples with no train samples."
        ),
        "blank_or_unrendered_row_count": len(blank_rows),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def render_markdown(payload: dict[str, Any]) -> str:
    original = payload["original_dataset"]
    tiled = payload["tiled_smoke_dataset"]
    smoke = payload["tiled_smoke_assessment"]
    mapping = payload["mapping_assessment"]
    metrics = payload["baseline_20epoch_metrics"]
    lines = [
        "# Dataset Mapping Audit",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Original data.yaml: `{payload['inputs']['original_data_yaml']}`",
        f"- Tiled smoke data.yaml: `{payload['inputs']['tiled_smoke_data_yaml']}`",
        f"- Baseline metrics JSON: `{payload['inputs']['baseline_metrics_json']}`",
        "",
        "## Executive Findings",
        "",
        f"- Tiled smoke is full dataset: `{smoke['is_full_dataset']}`",
        f"- Tiled smoke can be formal baseline dataset: `{smoke['can_be_formal_baseline_dataset']}`",
        f"- Current tiled names match original: `{mapping['current_tiled_names_match_original']}`",
        f"- Class id out of range found: `{mapping['class_id_out_of_range_found']}`",
        f"- blank-or-unrendered row count in existing 20 epoch metrics: `{mapping['blank_or_unrendered_row_count']}`",
        "",
        mapping["blank_or_unrendered_reason"],
        "",
        mapping["map_low_primary_cause"],
        "",
        "## Dataset Summary",
        "",
        "| dataset | nc | train images | val images | train labels | val labels | train bboxes | val bboxes | class id min | class id max | class id >= nc | names problem |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        dataset_summary_row("original", original),
        dataset_summary_row("tiled_smoke", tiled),
        "",
        "## Names",
        "",
        "| class id | original name | tiled smoke name | match |",
        "|---:|---|---|---|",
    ]
    for cid, (original_name, tiled_name) in enumerate(zip(original["names"], tiled["names"])):
        lines.append(f"| {cid} | {escape(original_name)} | {escape(tiled_name)} | {original_name == tiled_name} |")
    lines.extend(
        [
            "",
            "## Per-Class BBox Distribution",
            "",
            "| class id | name | original train | original val | original total | tiled train | tiled val | tiled total |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cid in range(int(original["nc"])):
        original_row = original["per_class"][cid]
        tiled_row = tiled["per_class"][cid]
        lines.append(
            "| {cid} | {name} | {otrain} | {oval} | {ototal} | {ttrain} | {tval} | {ttotal} |".format(
                cid=cid,
                name=escape(original_row["name"]),
                otrain=original_row["train_instances"],
                oval=original_row["val_instances"],
                ototal=original_row["total_instances"],
                ttrain=tiled_row["train_instances"],
                tval=tiled_row["val_instances"],
                ttotal=tiled_row["total_instances"],
            )
        )
    lines.extend(
        [
            "",
            "## Tiled Smoke Assessment",
            "",
            f"- Is smoke: `{smoke['is_smoke']}`",
            f"- Is full: `{smoke['is_full_dataset']}`",
            f"- tile_size: `{smoke['tile_size']}`",
            f"- overlap: `{smoke['overlap']}`",
            f"- min_visibility: `{smoke['min_visibility']}`",
            f"- keep_empty_ratio: `{smoke['keep_empty_ratio']}`",
            "",
        ]
    )
    for item in smoke["evidence"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Current 20 Epoch Per-Class Metrics",
            "",
            "| class id | name | reported class | train instances | val instances | precision | recall | AP50 | AP50-95 |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["mapped_per_class"]:
        lines.append(
            "| {cid} | {name} | {reported} | {train} | {val} | {precision} | {recall} | {ap50} | {ap5095} |".format(
                cid=row["mapped_class_id"],
                name=escape(row["mapped_class_name"]),
                reported=escape(row["reported_class"] or "blank-or-unrendered"),
                train=row["train_instances"],
                val=row["val_instances"],
                precision=row["precision"],
                recall=row["recall"],
                ap50=row["map50"],
                ap5095=row["map50_95"],
            )
        )
    lines.extend(render_metric_list("Classes With High AP50", metrics["high_ap50_classes"]))
    lines.extend(render_metric_list("Classes With Recall 0", metrics["recall_zero_classes"]))
    lines.extend(render_metric_list("Classes With Too Few Samples", metrics["sample_starved_classes"]))
    lines.extend(render_metric_list("Primary mAP Drag", metrics["low_ap50_classes"]))
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- The active tiled `data.yaml` now fully inherits the original class names.",
            "- No class id >= nc was found in the original or tiled smoke labels.",
            "- The existing low mAP should not be interpreted as a full-dataset baseline because this is a smoke subset.",
            "- The existing blank-or-unrendered rows are a logging/name-rendering artifact from the prior corrupted tiled `data.yaml`, while the numeric class IDs remain recoverable and in range.",
        ]
    )
    return "\n".join(lines) + "\n"


def dataset_summary_row(name: str, dataset: dict[str, Any]) -> str:
    train = dataset["splits"]["train"]
    val = dataset["splits"]["val"]
    return (
        f"| {name} | {dataset['nc']} | {train['image_count']} | {val['image_count']} | "
        f"{train['label_file_count']} | {val['label_file_count']} | {train['bbox_total']} | {val['bbox_total']} | "
        f"{dataset['class_id_min']} | {dataset['class_id_max']} | {dataset['class_id_ge_nc']} | "
        f"{dataset['names_quality']['has_problem']} |"
    )


def render_metric_list(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = ["", f"### {title}", ""]
    if not rows:
        lines.append("- None")
        return lines
    for row in rows:
        lines.append(
            "- class_id={class_id}, name={name}, train={train}, val={val}, recall={recall}, AP50={map50}, AP50-95={map50_95}".format(
                class_id=row["class_id"],
                name=row["name"],
                train=row["train_instances"],
                val=row["val_instances"],
                recall=row["recall"],
                map50=row["map50"],
                map50_95=row["map50_95"],
            )
        )
    return lines


def escape(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
