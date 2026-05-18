from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = Path(r"E:\TJGY\DataSet2_fixed")
TILED_FULL_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full"
REPORT_DIR = PROJECT_ROOT / "outputs" / "audits" / "dataset_mapping"
REPORT_JSON = REPORT_DIR / "full_tiled_dataset_mapping_audit.json"
REPORT_MD = REPORT_DIR / "full_tiled_dataset_mapping_audit.md"
TILED_REPORT_JSON = TILED_FULL_ROOT / "tiled_dataset_report.json"
TILING_QUALITY_JSON = PROJECT_ROOT / "outputs" / "audits" / "tiling_quality" / "tiling_quality_audit.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    original = audit_dataset("original", ORIGINAL_ROOT, ORIGINAL_ROOT / "data.yaml")
    tiled = audit_dataset("tiled_full", TILED_FULL_ROOT, TILED_FULL_ROOT / "data.yaml")
    tiled_report = load_json(TILED_REPORT_JSON)
    tiling_quality = load_json(TILING_QUALITY_JSON)
    full_assessment = assess_full_tiled(original, tiled, tiled_report, tiling_quality)
    mapping_assessment = assess_mapping(original, tiled, full_assessment)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "original_dataset": original,
        "tiled_full_dataset": tiled,
        "tiled_full_report": tiled_report,
        "tiling_quality_audit": tiling_quality,
        "tiled_full_assessment": full_assessment,
        "mapping_assessment": mapping_assessment,
        "inputs": {
            "original_data_yaml": str((ORIGINAL_ROOT / "data.yaml").resolve()),
            "tiled_full_data_yaml": str((TILED_FULL_ROOT / "data.yaml").resolve()),
            "tiled_dataset_report_json": str(TILED_REPORT_JSON.resolve()),
            "tiling_quality_audit_json": str(TILING_QUALITY_JSON.resolve()),
        },
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(str(REPORT_JSON))
    print(str(REPORT_MD))


def audit_dataset(name: str, root: Path, data_yaml: Path) -> dict[str, Any]:
    yaml_info = load_data_yaml(data_yaml)
    splits = {split: scan_split(root, split, int(yaml_info["nc"])) for split in ["train", "val"]}
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
            data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        except Exception as exc:  # pragma: no cover - defensive reporting path
            parse_error = f"{type(exc).__name__}: {exc}"
    names_by_id = normalize_names(data.get("names"))
    nc = int(data.get("nc", len(names_by_id) or 0))
    if nc == 0 and names_by_id:
        nc = len(names_by_id)
    names = [names_by_id.get(index) for index in range(nc)]
    return {
        "nc": nc,
        "names": names,
        "names_by_id": names_by_id,
        "names_quality": inspect_names(names, nc),
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


def inspect_names(names: list[str | None], nc: int) -> dict[str, Any]:
    missing = [index for index in range(nc) if index >= len(names) or names[index] is None]
    empty = [index for index, value in enumerate(names[:nc]) if value is not None and not str(value).strip()]
    replacement_char = [index for index, value in enumerate(names[:nc]) if value is not None and "\ufffd" in str(value)]
    mojibake_tokens = ["Ã", "Â", "å", "æ", "ç", "é", "闁", "鐎", "閸", "鈧", "脙", "脗", "锟斤拷"]
    suspicious_mojibake = [
        index
        for index, value in enumerate(names[:nc])
        if value is not None and any(token in str(value) for token in mojibake_tokens)
    ]
    question_mark_in_non_ascii = [
        index
        for index, value in enumerate(names[:nc])
        if value is not None and "?" in str(value) and any(ord(ch) > 127 for ch in str(value))
    ]
    contains_cjk = [
        index
        for index, value in enumerate(names[:nc])
        if value is not None and any("\u4e00" <= ch <= "\u9fff" for ch in str(value))
    ]
    chinese_name_damage_found = bool(replacement_char or suspicious_mojibake or question_mark_in_non_ascii)
    return {
        "missing_name_ids": missing,
        "empty_name_ids": empty,
        "replacement_char_name_ids": replacement_char,
        "suspicious_mojibake_name_ids": sorted(set(suspicious_mojibake + question_mark_in_non_ascii)),
        "contains_cjk_name_ids": contains_cjk,
        "chinese_name_damage_found": chinese_name_damage_found,
        "has_problem": bool(missing or empty or chinese_name_damage_found),
    }


def scan_split(root: Path, split: str, nc: int) -> dict[str, Any]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_files = (
        sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if image_dir.exists()
        else []
    )
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
        "bbox_total": int(sum(class_counts.values())),
        "class_counts": {str(key): int(class_counts.get(key, 0)) for key in range(nc)},
        "class_counts_all": {str(key): int(value) for key, value in sorted(class_counts.items())},
        "class_image_counts": {str(key): len(class_images.get(key, set())) for key in range(nc)},
        "class_ids": class_ids,
        "class_id_min": min(class_ids) if class_ids else None,
        "class_id_max": max(class_ids) if class_ids else None,
        "class_id_ge_nc": sorted({cid for cid in class_ids if cid >= nc}),
        "class_id_negative": sorted({cid for cid in class_ids if cid < 0}),
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


def assess_full_tiled(
    original: dict[str, Any],
    tiled: dict[str, Any],
    tiled_report: dict[str, Any],
    tiling_quality: dict[str, Any],
) -> dict[str, Any]:
    report_summary = tiled_report.get("summary", {}) if tiled_report else {}
    report_per_split = tiled_report.get("per_split", {}) if tiled_report else {}
    original_train_images = int(original["splits"]["train"]["image_count"])
    original_val_images = int(original["splits"]["val"]["image_count"])
    report_train_source = int(report_per_split.get("train", {}).get("source_image_count", -1))
    report_val_source = int(report_per_split.get("val", {}).get("source_image_count", -1))
    source_counts_match = report_train_source == original_train_images and report_val_source == original_val_images
    path_contains_smoke = "smoke" in tiled["root"].lower()
    is_full_dataset = bool(tiled["data_yaml_exists"] and source_counts_match and not path_contains_smoke)
    quality_summary = tiling_quality.get("summary", {}) if tiling_quality else {}
    truncated_ratio = float(quality_summary.get("border_truncated_bbox_ratio", 0.0) or 0.0)
    severe_truncated_ratio = float(quality_summary.get("severe_truncated_visibility_lt_0.7_ratio", 0.0) or 0.0)
    tiling_quality_unsafe = bool(truncated_ratio > 0.0 or severe_truncated_ratio > 0.0)
    return {
        "is_full_dataset": is_full_dataset,
        "can_be_formal_baseline_dataset": False,
        "tiling_quality_unsafe": tiling_quality_unsafe,
        "border_truncated_bbox_ratio": truncated_ratio,
        "severe_truncated_visibility_lt_0.7_ratio": severe_truncated_ratio,
        "unsafe_reason": (
            "Partial-object bbox risk was found by outputs/audits/tiling_quality/tiling_quality_audit.json; "
            "use outputs/datasets/tiled/tiled_1024_ov20_full_safe/ for formal baseline."
            if tiling_quality_unsafe
            else None
        ),
        "path_contains_smoke": path_contains_smoke,
        "source_counts_match_original": source_counts_match,
        "original_train_images": original_train_images,
        "original_val_images": original_val_images,
        "reported_source_train_images": report_train_source,
        "reported_source_val_images": report_val_source,
        "tiled_train_images": tiled["splits"]["train"]["image_count"],
        "tiled_val_images": tiled["splits"]["val"]["image_count"],
        "tile_size": report_summary.get("tile_size"),
        "overlap": report_summary.get("overlap"),
        "min_visibility": report_summary.get("min_visibility"),
        "keep_empty_ratio": report_summary.get("keep_empty_ratio"),
        "empty_tile_retained_count": report_summary.get("empty_tile_retained_count"),
        "dropped_bbox_count": report_summary.get("dropped_bbox_count"),
        "dropped_bbox_reasons": report_summary.get("dropped_bbox_reasons"),
    }


def assess_mapping(original: dict[str, Any], tiled: dict[str, Any], full_assessment: dict[str, Any]) -> dict[str, Any]:
    names_match = original["names"] == tiled["names"]
    invalid_ids = bool(
        original["class_id_ge_nc"]
        or original["class_id_negative"]
        or tiled["class_id_ge_nc"]
        or tiled["class_id_negative"]
    )
    tiled_names_problem = bool(tiled["names_quality"]["has_problem"])
    can_formal = bool(
        full_assessment["is_full_dataset"]
        and names_match
        and not invalid_ids
        and not tiled_names_problem
        and not full_assessment.get("tiling_quality_unsafe", False)
    )
    full_assessment["can_be_formal_baseline_dataset"] = can_formal
    return {
        "tiled_names_match_original": names_match,
        "tiled_yaml_parse_error": tiled["yaml_parse_error"],
        "tiled_names_problem": tiled_names_problem,
        "tiled_chinese_names_damaged": tiled["names_quality"]["chinese_name_damage_found"],
        "class_id_out_of_range_found": invalid_ids,
        "can_be_formal_baseline_dataset": can_formal,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def render_markdown(payload: dict[str, Any]) -> str:
    original = payload["original_dataset"]
    tiled = payload["tiled_full_dataset"]
    full = payload["tiled_full_assessment"]
    mapping = payload["mapping_assessment"]
    lines = [
        "# Full Tiled Dataset Mapping Audit",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Original data.yaml: `{payload['inputs']['original_data_yaml']}`",
        f"- Full tiled data.yaml: `{payload['inputs']['tiled_full_data_yaml']}`",
        f"- Tiled dataset report JSON: `{payload['inputs']['tiled_dataset_report_json']}`",
        f"- Tiling quality audit JSON: `{payload['inputs']['tiling_quality_audit_json']}`",
        "",
        "## Executive Findings",
        "",
        f"- Full tiled dataset: `{full['is_full_dataset']}`",
        f"- Can be formal baseline dataset: `{mapping['can_be_formal_baseline_dataset']}`",
        f"- Tiling quality unsafe: `{full['tiling_quality_unsafe']}`",
        f"- Border-truncated bbox ratio: `{full['border_truncated_bbox_ratio']:.4f}`",
        f"- Severe truncated visibility < 0.7 ratio: `{full['severe_truncated_visibility_lt_0.7_ratio']:.4f}`",
        f"- Tiled names match original: `{mapping['tiled_names_match_original']}`",
        f"- Class id out of range found: `{mapping['class_id_out_of_range_found']}`",
        f"- Chinese class names damaged: `{mapping['tiled_chinese_names_damaged']}`",
        f"- Source train/val counts match original: `{full['source_counts_match_original']}`",
        "",
        "## Dataset Summary",
        "",
        "| dataset | nc | train images | val images | train labels | val labels | train bboxes | val bboxes | class id min | class id max | class id >= nc | names problem |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        dataset_summary_row("original", original),
        dataset_summary_row("tiled_full", tiled),
        "",
        "## Tiling Summary",
        "",
        f"- tile_size: `{full['tile_size']}`",
        f"- overlap: `{full['overlap']}`",
        f"- min_visibility: `{full['min_visibility']}`",
        f"- keep_empty_ratio: `{full['keep_empty_ratio']}`",
        f"- original train/val images reported by builder: `{full['reported_source_train_images']} / {full['reported_source_val_images']}`",
        f"- tiled train/val images: `{full['tiled_train_images']} / {full['tiled_val_images']}`",
        f"- empty tiles retained: `{full['empty_tile_retained_count']}`",
        f"- dropped bboxes: `{full['dropped_bbox_count']}`",
        f"- dropped bbox reasons: `{full['dropped_bbox_reasons']}`",
        f"- unsafe reason: `{full['unsafe_reason']}`",
        "",
        "## Names",
        "",
        "| class id | original name | tiled full name | match |",
        "|---:|---|---|---|",
    ]
    for cid in range(max(int(original["nc"]), int(tiled["nc"]))):
        original_name = original["names"][cid] if cid < len(original["names"]) else None
        tiled_name = tiled["names"][cid] if cid < len(tiled["names"]) else None
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
            "## Conclusion",
            "",
            f"- Full source coverage confirmed: `{full['is_full_dataset']}`.",
            f"- Tiled `data.yaml` inherits original names: `{mapping['tiled_names_match_original']}`.",
            f"- No class id >= nc or negative class id found: `{not mapping['class_id_out_of_range_found']}`.",
            f"- Chinese class names are intact: `{not mapping['tiled_chinese_names_damaged']}`.",
            f"- Formal baseline dataset readiness: `{mapping['can_be_formal_baseline_dataset']}`.",
            "- The old full tiled dataset is superseded by `outputs/datasets/tiled/tiled_1024_ov20_full_safe/` for formal baseline work.",
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


def escape(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
