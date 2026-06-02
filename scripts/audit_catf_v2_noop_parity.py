from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = ROOT / "outputs/experiments/catf_v2_noop_parity_smoke"
CONTROL_ROOT = ROOT / "outputs/experiments/catf_v2_noop_control_50ep"
CATF_MULTI_ROOT = ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2"
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CATF-v2 no-op parity against clean native YOLO default.")
    parser.add_argument("--smoke-root", default=str(SMOKE_ROOT))
    parser.add_argument("--control-root", default=str(CONTROL_ROOT))
    parser.add_argument("--catf-multiseed-root", default=str(CATF_MULTI_ROOT))
    args = parser.parse_args()

    smoke_root = Path(args.smoke_root)
    control_root = Path(args.control_root)
    catf_root = Path(args.catf_multiseed_root)
    reports = smoke_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    smoke_payload = build_smoke_payload(smoke_root)
    write_json(reports / "noop_parity_report.json", smoke_payload)
    write_md(reports / "noop_parity_report.md", build_smoke_markdown(smoke_payload))

    random_payload = build_random_audit(smoke_payload)
    write_json(reports / "random_path_audit.json", random_payload)
    write_md(reports / "random_path_audit.md", build_random_markdown(random_payload))

    control_payload = build_seed1_control_payload(control_root, catf_root)
    if control_payload:
        control_reports = control_root / "reports"
        control_reports.mkdir(parents=True, exist_ok=True)
        write_json(control_reports / "noop_control_seed1_metrics.json", control_payload)
        write_md(control_reports / "noop_control_seed1_report.md", build_seed1_markdown(control_payload))

    print(json.dumps({"smoke": smoke_payload["summary"], "seed1_control": control_payload.get("summary") if control_payload else None}, indent=2))


def build_smoke_payload(root: Path) -> dict[str, Any]:
    clean_dir = root / "clean_native_1ep"
    noop_dir = root / "catf_v2_noop_1ep"
    clean_metrics = load_metrics(clean_dir)
    noop_metrics = load_metrics(noop_dir)
    clean_args = read_yaml(clean_dir / "train" / "args.yaml")
    noop_args = read_yaml(noop_dir / "train" / "args.yaml")
    clean_rows = read_results(clean_dir / "train" / "results.csv")
    noop_rows = read_results(noop_dir / "train" / "results.csv")
    result_diff = diff_results(clean_rows, noop_rows)
    args_diff = diff_mapping(clean_args, noop_args)
    metric_diff = metric_delta(noop_metrics, clean_metrics)
    online_stats = read_json(noop_dir / "reports" / "online_aug_stats.json")
    roi_stats = read_json(noop_dir / "reports" / "roi_aug_stats.json")
    noop_payload = read_json(noop_dir / "reports" / "final_metrics.json")
    parity_pass = (
        all(abs(float(value or 0.0)) <= 1e-9 for value in metric_diff.values())
        and result_diff["max_abs_numeric_diff_excluding_time"] <= 1e-9
        and int(online_stats.get("samples_augmented", -1)) == 0
        and int(roi_stats.get("roi_aug_applied", -1)) == 0
        and int(online_stats.get("router_random_draw_count", -1)) == 0
        and int(noop_payload.get("summary", {}).get("policy_update_applied_count", -1)) == 0
    )
    return {
        "clean_dir": str(clean_dir),
        "noop_dir": str(noop_dir),
        "clean_metrics": clean_metrics,
        "noop_metrics": noop_metrics,
        "metric_delta_noop_minus_clean": metric_diff,
        "args_diff": args_diff,
        "result_curve_diff": result_diff,
        "online_aug_stats": online_stats,
        "roi_aug_stats": roi_stats,
        "callback_records": {
            "feedback_epochs": noop_payload.get("summary", {}).get("feedback_epochs", []),
            "callback_invocations": len(noop_payload.get("epoch_records", []) or []),
            "policy_update_applied_count": noop_payload.get("summary", {}).get("policy_update_applied_count"),
        },
        "summary": {
            "catf_v2_noop_1ep_parity_pass": parity_pass,
            "metrics_identical": all(abs(float(value or 0.0)) <= 1e-9 for value in metric_diff.values()),
            "results_csv_identical_numeric_excluding_time": result_diff["max_abs_numeric_diff_excluding_time"] <= 1e-9,
            "wall_time_diff": result_diff["time_diff"],
            "args_diff_count": len(args_diff),
            "industrial_applied": int(online_stats.get("samples_augmented", 0)),
            "roi_applied": int(roi_stats.get("roi_aug_applied", 0)),
            "router_random_draw_count": int(online_stats.get("router_random_draw_count", 0)),
            "noop_transform_calls": int(online_stats.get("noop_transform_calls", 0)),
            "policy_update_applied_count": noop_payload.get("summary", {}).get("policy_update_applied_count"),
        },
    }


def build_random_audit(smoke_payload: dict[str, Any]) -> dict[str, Any]:
    stats = smoke_payload.get("online_aug_stats", {})
    roi_stats = smoke_payload.get("roi_aug_stats", {})
    return {
        "catf_noop": bool(stats.get("catf_noop")),
        "sample_router_built": bool(stats.get("sample_router_built")),
        "noop_transform_calls": int(stats.get("noop_transform_calls", 0)),
        "router_apply_calls": int(stats.get("samples_seen", 0)),
        "router_random_draw_count": int(stats.get("router_random_draw_count", 0)),
        "industrial_samples_augmented": int(stats.get("samples_augmented", 0)),
        "roi_aug_applied": int(roi_stats.get("roi_aug_applied", 0)),
        "policy_update_applied_count": smoke_payload.get("callback_records", {}).get("policy_update_applied_count"),
        "static_path_assessment": {
            "transform_short_circuits_before_label_rewrite": True,
            "sample_router_probability_draws_skipped": int(stats.get("samples_seen", 0)) == 0,
            "roi_random_selection_skipped": int(roi_stats.get("roi_aug_applied", 0)) == 0,
            "policy_matrix_update_skipped": smoke_payload.get("callback_records", {}).get("policy_update_applied_count") == 0,
            "sample_order_mutation_supported": False,
            "dataloader_workers": 0,
        },
        "conclusion": (
            "CATF-v2 noop did not consume router augmentation RNG in the smoke."
            if int(stats.get("router_random_draw_count", 0)) == 0
            else "CATF-v2 noop consumed router augmentation RNG; investigate sample_router path."
        ),
    }


def build_seed1_control_payload(control_root: Path, catf_root: Path) -> dict[str, Any]:
    noop_dir = control_root / "seed_1"
    if not (noop_dir / "reports" / "final_metrics.json").exists():
        return {}
    clean_dir = catf_root / "seed_1" / "clean_native_yolo_default"
    catf_dir = catf_root / "seed_1" / "catf_v2"
    clean_metrics = load_metrics(clean_dir)
    noop_metrics = load_metrics(noop_dir)
    catf_metrics = load_metrics(catf_dir)
    online_stats = read_json(noop_dir / "reports" / "online_aug_stats.json")
    roi_stats = read_json(noop_dir / "reports" / "roi_aug_stats.json")
    payload = read_json(noop_dir / "reports" / "final_metrics.json")
    d_clean = metric_delta(noop_metrics, clean_metrics)
    d_catf = metric_delta(noop_metrics, catf_metrics)
    constraint = constraint_failed(noop_metrics, clean_metrics)
    reproduces_clean = all(abs(float(value or 0.0)) <= 1e-9 for value in d_clean.values())
    return {
        "seed": 1,
        "clean_dir": str(clean_dir),
        "catf_v2_dir": str(catf_dir),
        "noop_dir": str(noop_dir),
        "clean_metrics": clean_metrics,
        "catf_v2_metrics": catf_metrics,
        "noop_metrics": noop_metrics,
        "delta_noop_minus_clean": d_clean,
        "delta_noop_minus_catf_v2": d_catf,
        "constraint_failed": constraint["constraint_failed"],
        "constraint_reasons": constraint["failure_reasons"],
        "online_aug_stats": online_stats,
        "roi_aug_stats": roi_stats,
        "feedback_epochs": payload.get("summary", {}).get("feedback_epochs", []),
        "epoch_continuous": payload.get("summary", {}).get("epoch_continuous"),
        "summary": {
            "seed1_noop_reproduces_clean_native": reproduces_clean,
            "seed1_noop_constraint_failed": constraint["constraint_failed"],
            "industrial_applied": int(online_stats.get("samples_augmented", 0)),
            "roi_applied": int(roi_stats.get("roi_aug_applied", 0)),
            "policy_update_applied_count": payload.get("summary", {}).get("policy_update_applied_count"),
            "router_random_draw_count": int(online_stats.get("router_random_draw_count", 0)),
            "catf_v2_seed1_gain_explained_by_noop_framework": not reproduces_clean,
        },
    }


def build_smoke_markdown(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# CATF-v2 No-op 1ep Parity Audit",
        "",
        f"- Clean run: `{payload['clean_dir']}`",
        f"- CATF-v2 noop run: `{payload['noop_dir']}`",
        f"- 1ep parity pass: `{str(s['catf_v2_noop_1ep_parity_pass']).lower()}`",
        f"- Metrics identical: `{str(s['metrics_identical']).lower()}`",
        f"- results.csv numeric max abs diff excluding time: `{payload['result_curve_diff']['max_abs_numeric_diff_excluding_time']:.12g}`",
        f"- results.csv wall-clock time diff: `{payload['result_curve_diff']['time_diff']:.6f}`",
        f"- args.yaml diff count: `{s['args_diff_count']}`",
        f"- Industrial samples augmented: `{s['industrial_applied']}`",
        f"- ROI applied: `{s['roi_applied']}`",
        f"- Router random draw count: `{s['router_random_draw_count']}`",
        f"- No-op transform calls: `{s['noop_transform_calls']}`",
        f"- Policy update applied count: `{s['policy_update_applied_count']}`",
        "",
        "## Metric Delta",
        "",
        "| metric | clean | noop | noop-clean |",
        "|---|---:|---:|---:|",
    ]
    for key in METRIC_KEYS:
        lines.append(
            f"| {key} | {payload['clean_metrics'].get(key, 0.0):.6f} | {payload['noop_metrics'].get(key, 0.0):.6f} | "
            f"{payload['metric_delta_noop_minus_clean'].get(key, 0.0):+.6f} |"
        )
    lines.extend(["", "## Args Diff", ""])
    if payload["args_diff"]:
        lines.extend(["| key | clean | noop |", "|---|---|---|"])
        for item in payload["args_diff"][:80]:
            lines.append(f"| `{item['key']}` | `{item['left']}` | `{item['right']}` |")
    else:
        lines.append("- No args diff.")
    return "\n".join(lines)


def build_random_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CATF-v2 No-op Random Path Audit",
        "",
        f"- CATF no-op: `{str(payload['catf_noop']).lower()}`",
        f"- Sample router built: `{str(payload['sample_router_built']).lower()}`",
        f"- No-op transform calls: `{payload['noop_transform_calls']}`",
        f"- Router apply calls: `{payload['router_apply_calls']}`",
        f"- Router random draw count: `{payload['router_random_draw_count']}`",
        f"- Industrial samples augmented: `{payload['industrial_samples_augmented']}`",
        f"- ROI applied: `{payload['roi_aug_applied']}`",
        f"- Policy update applied count: `{payload['policy_update_applied_count']}`",
        "",
        "## Static Path Assessment",
        "",
    ]
    for key, value in payload["static_path_assessment"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", f"Conclusion: {payload['conclusion']}"])
    return "\n".join(lines)


def build_seed1_markdown(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# CATF-v2 No-op Seed1 50ep Control",
        "",
        f"- Clean native: `{payload['clean_dir']}`",
        f"- CATF-v2: `{payload['catf_v2_dir']}`",
        f"- CATF-v2 noop: `{payload['noop_dir']}`",
        f"- Noop reproduces clean native: `{str(s['seed1_noop_reproduces_clean_native']).lower()}`",
        f"- Noop constraint failed: `{str(s['seed1_noop_constraint_failed']).lower()}`",
        f"- Industrial samples augmented: `{s['industrial_applied']}`",
        f"- ROI applied: `{s['roi_applied']}`",
        f"- Policy update applied count: `{s['policy_update_applied_count']}`",
        f"- Router random draw count: `{s['router_random_draw_count']}`",
        f"- CATF-v2 seed1 gain explained by no-op framework: `{str(s['catf_v2_seed1_gain_explained_by_noop_framework']).lower()}`",
        "",
        "## Metrics",
        "",
        "| metric | clean | noop | CATF-v2 | noop-clean | noop-CATF-v2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in METRIC_KEYS:
        lines.append(
            f"| {key} | {payload['clean_metrics'].get(key, 0.0):.6f} | {payload['noop_metrics'].get(key, 0.0):.6f} | "
            f"{payload['catf_v2_metrics'].get(key, 0.0):.6f} | {payload['delta_noop_minus_clean'].get(key, 0.0):+.6f} | "
            f"{payload['delta_noop_minus_catf_v2'].get(key, 0.0):+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- CATF-v2 noop reproduces clean native exactly, so the CATF-v2 framework path by itself did not explain seed1's CATF-v2 result."
                if s["seed1_noop_reproduces_clean_native"]
                else "- CATF-v2 noop does not reproduce clean native, so the framework path itself can perturb training."
            ),
            "- Diagnosis-only had already shown the callback alone is not a source of drift; this no-op control isolates the custom trainer/router path.",
        ]
    )
    return "\n".join(lines)


def load_metrics(run_dir: Path) -> dict[str, float]:
    candidates = [
        run_dir / "reports" / "final_metrics.json",
        run_dir / "reports" / "clean_native_yolo_default_metrics.json",
        run_dir / "reports" / "inloop_no_feedback_control_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            payload = read_json(path)
            metrics = extract_metrics(payload)
            if metrics:
                return metrics
    rows = read_results(run_dir / "train" / "results.csv")
    if rows:
        return extract_metrics_from_results_row(rows[-1])
    return {key: math.nan for key in METRIC_KEYS}


def extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    candidates = [
        payload.get("val", {}).get("metrics") if isinstance(payload.get("val"), dict) else None,
        payload.get("metrics") if isinstance(payload.get("metrics"), dict) else None,
        payload.get("final_metrics") if isinstance(payload.get("final_metrics"), dict) else None,
    ]
    if isinstance(payload.get("summary"), dict):
        candidates.append(payload["summary"].get("val_metrics"))
    if isinstance(payload.get("val_metrics"), dict):
        candidates.append(payload.get("val_metrics"))
    for item in candidates:
        if isinstance(item, dict) and all(key in item for key in METRIC_KEYS):
            return {key: float(item[key]) for key in METRIC_KEYS}
    return {}


def extract_metrics_from_results_row(row: dict[str, str]) -> dict[str, float]:
    mapping = {
        "precision": ["metrics/precision(B)", "precision"],
        "recall": ["metrics/recall(B)", "recall"],
        "map50": ["metrics/mAP50(B)", "map50"],
        "map50_95": ["metrics/mAP50-95(B)", "map50_95"],
    }
    out = {}
    for key, names in mapping.items():
        for name in names:
            if name in row:
                out[key] = float(row[name])
                break
    return out


def metric_delta(current: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {key: float(current.get(key, 0.0)) - float(reference.get(key, 0.0)) for key in METRIC_KEYS}


def constraint_failed(current: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    delta = metric_delta(current, reference)
    reasons = []
    if delta["precision"] < -0.01:
        reasons.append("precision_drop_gt_0.01")
    if delta["map50"] < -0.01:
        reasons.append("map50_drop_gt_0.01")
    if delta["map50_95"] < -0.01:
        reasons.append("map50_95_drop_gt_0.01")
    return {"constraint_failed": bool(reasons), "failure_reasons": reasons}


def diff_mapping(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key in sorted(set(left) | set(right)):
        lv = left.get(key)
        rv = right.get(key)
        if lv != rv:
            out.append({"key": key, "left": lv, "right": rv})
    return out


def diff_results(left_rows: list[dict[str, str]], right_rows: list[dict[str, str]]) -> dict[str, Any]:
    max_abs = 0.0
    max_abs_excluding_time = 0.0
    time_diff = 0.0
    diffs = []
    for idx, (left, right) in enumerate(zip(left_rows, right_rows), start=1):
        for key in sorted(set(left) & set(right)):
            try:
                lv = float(left[key])
                rv = float(right[key])
            except (TypeError, ValueError):
                continue
            delta = rv - lv
            max_abs = max(max_abs, abs(delta))
            if key == "time":
                time_diff = delta
                continue
            max_abs_excluding_time = max(max_abs_excluding_time, abs(delta))
            if abs(delta) > 1e-9:
                diffs.append({"epoch": idx, "column": key, "clean": lv, "noop": rv, "delta": delta})
    return {
        "row_count_clean": len(left_rows),
        "row_count_noop": len(right_rows),
        "same_row_count": len(left_rows) == len(right_rows),
        "max_abs_numeric_diff": max_abs,
        "max_abs_numeric_diff_excluding_time": max_abs_excluding_time,
        "time_diff": time_diff,
        "nonzero_numeric_diffs": diffs[:200],
    }


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
