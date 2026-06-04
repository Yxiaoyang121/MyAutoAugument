"""Re-optimize CATF-v2 thresholds on the official predict post-processing path.

This script is analysis-only. It reads Ultralytics YOLO.predict outputs and
searches per-class confidence thresholds. It never trains.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.catf_v2.per_class_thresholds import constraint_failures, metric_delta, threshold_changes  # noqa: E402
from scripts.evaluate_catf_v2_threshold_posthoc import CANONICAL_CLASS_NAMES, THRESHOLDS, evaluate, f4, fd  # noqa: E402
from scripts.validate_catf_v2_rc_official_path import (  # noqa: E402
    FIXED_ROOT,
    REPORTS,
    SEEDS,
    THRESHOLD_CONFIG,
    clean_official_metrics,
    clean_prediction_json,
    fixed_saved_official_metrics,
)


OFFICIAL_PATH_ROOT = REPORTS / "catf_v2_rc_official_path_validation"
OUT_JSON = REPORTS / "official_threshold_reoptimization.json"
OUT_MD = REPORTS / "official_threshold_reoptimization.md"
UNIFIED_JSON = REPORTS / "unified_precision_guard_thresholds.json"
PER_SEED_JSON = REPORTS / "per_seed_thresholds.json"
CONSERVATIVE_JSON = REPORTS / "conservative_default_thresholds.json"
DEFAULT_THRESHOLD = 0.25
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
HIGH_FP_PRIOR = {0, 1, 4, 8}
CONSERVATIVE_LOWERABLE = {5, 6, 7, 9, 12}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def class_names() -> dict[int, str]:
    return dict(CANONICAL_CLASS_NAMES)


def fixed_prediction_json(seed: int) -> Path:
    path = OFFICIAL_PATH_ROOT / f"seed_{seed}/fixed_predict/validation_predictions.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing official fixed predict JSON for seed {seed}: {path}")
    return path


def load_records() -> dict[int, dict[str, list[dict[str, Any]]]]:
    records: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for seed in SEEDS:
        clean_path = clean_prediction_json(seed)
        fixed_path = fixed_prediction_json(seed)
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean predict JSON for seed {seed}: {clean_path}")
        records[seed] = {
            "clean": read_json(clean_path)["records"],
            "fixed": read_json(fixed_path)["records"],
        }
    return records


def class_ids_from_records(records: dict[int, dict[str, list[dict[str, Any]]]]) -> list[int]:
    class_ids = set()
    for seed_records in records.values():
        for group_records in seed_records.values():
            for record in group_records:
                for item in record.get("ground_truth", []) + record.get("predictions", []):
                    class_ids.add(int(item["class_id"]))
    return sorted(class_ids)


def threshold_key(thresholds: dict[int, float], class_ids: list[int]) -> tuple[float, ...]:
    return tuple(round(float(thresholds.get(class_id, DEFAULT_THRESHOLD)), 2) for class_id in class_ids)


class EvaluatorCache:
    def __init__(self, records: dict[int, dict[str, list[dict[str, Any]]]], class_ids: list[int]) -> None:
        self.records = records
        self.class_ids = class_ids
        self.clean_default: dict[int, Any] = {}
        self.cache: dict[tuple[int, tuple[float, ...]], Any] = {}
        defaults = {class_id: DEFAULT_THRESHOLD for class_id in class_ids}
        for seed in SEEDS:
            self.clean_default[seed] = evaluate(records[seed]["clean"], defaults)

    def fixed_eval(self, seed: int, thresholds: dict[int, float]) -> Any:
        key = (seed, threshold_key(thresholds, self.class_ids))
        if key not in self.cache:
            self.cache[key] = evaluate(self.records[seed]["fixed"], thresholds)
        return self.cache[key]

    def seed_payload(self, seed: int, thresholds: dict[int, float]) -> dict[str, Any]:
        result = self.fixed_eval(seed, thresholds)
        reference = self.clean_default[seed].metrics
        failures = constraint_failures(result.metrics, reference)
        return {
            "metrics": result.metrics,
            "per_class": {str(cid): row for cid, row in result.per_class.items()},
            "delta_vs_clean_default": metric_delta(result.metrics, reference),
            "constraint_failed": bool(failures),
            "failure_reasons": failures,
        }

    def evaluate_thresholds(self, thresholds: dict[int, float]) -> dict[str, Any]:
        seeds: dict[str, Any] = {}
        for seed in SEEDS:
            seeds[str(seed)] = self.seed_payload(seed, thresholds)
        return summarize_scheme(thresholds, seeds)


def summarize_scheme(thresholds: dict[int, float], seeds: dict[str, Any]) -> dict[str, Any]:
    pass_count = sum(not row["constraint_failed"] for row in seeds.values())
    metrics = [row["metrics"] for row in seeds.values()]
    mean_metrics = {key: sum(row[key] for row in metrics) / max(1, len(metrics)) for key in METRIC_KEYS}
    failure_reasons = {seed: row["failure_reasons"] for seed, row in seeds.items() if row["constraint_failed"]}
    return {
        "thresholds": {str(cid): float(value) for cid, value in sorted(thresholds.items())},
        "seeds": seeds,
        "pass_count": pass_count,
        "mean_metrics": mean_metrics,
        "failure_reasons": failure_reasons,
    }


def scheme_score(summary: dict[str, Any]) -> float:
    pass_count = int(summary["pass_count"])
    penalty = 0.0
    for seed_row in summary["seeds"].values():
        for key in ("precision", "map50", "map50_95"):
            penalty += max(0.0, -0.01 - seed_row["delta_vs_clean_default"][key])
    mean = summary["mean_metrics"]
    return (
        pass_count * 10000.0
        - 1000.0 * penalty
        + 100.0 * mean["map50_95"]
        + 10.0 * mean["map50"]
        + mean["recall"]
        + 0.1 * mean["precision"]
    )


def greedy_unified_search(cache: EvaluatorCache, class_ids: list[int], starts: list[dict[int, float]], *, allowed: dict[int, list[float]] | None = None, max_passes: int = 6) -> dict[str, Any]:
    best_summary: dict[str, Any] | None = None
    for start in starts:
        thresholds = {class_id: float(start.get(class_id, DEFAULT_THRESHOLD)) for class_id in class_ids}
        current = cache.evaluate_thresholds(thresholds)
        current_score = scheme_score(current)
        steps = []
        for pass_idx in range(max_passes):
            improved = False
            for class_id in class_ids:
                candidates = allowed.get(class_id, THRESHOLDS) if allowed else THRESHOLDS
                best_local = (current_score, thresholds[class_id], current)
                for candidate in candidates:
                    trial = dict(thresholds)
                    trial[class_id] = candidate
                    summary = cache.evaluate_thresholds(trial)
                    score = scheme_score(summary)
                    if score > best_local[0] + 1e-10:
                        best_local = (score, candidate, summary)
                if best_local[1] != thresholds[class_id]:
                    before = thresholds[class_id]
                    thresholds[class_id] = float(best_local[1])
                    current_score = best_local[0]
                    current = best_local[2]
                    improved = True
                    steps.append({"pass": pass_idx + 1, "class_id": class_id, "before": before, "after": thresholds[class_id], "pass_count": current["pass_count"]})
            if not improved:
                break
        current["score"] = current_score
        current["steps"] = steps
        if best_summary is None or scheme_score(current) > scheme_score(best_summary):
            best_summary = current
    assert best_summary is not None
    return best_summary


def per_seed_search(cache: EvaluatorCache, class_ids: list[int]) -> dict[str, Any]:
    payload: dict[str, Any] = {"seeds": {}, "pass_count": 0}
    for seed in SEEDS:
        thresholds = {class_id: DEFAULT_THRESHOLD for class_id in class_ids}
        current = cache.seed_payload(seed, thresholds)
        current_score = single_seed_score(current)
        steps = []
        for pass_idx in range(5):
            improved = False
            for class_id in class_ids:
                best_local = (current_score, thresholds[class_id], current)
                for candidate in THRESHOLDS:
                    trial = dict(thresholds)
                    trial[class_id] = candidate
                    summary = cache.seed_payload(seed, trial)
                    score = single_seed_score(summary)
                    if score > best_local[0] + 1e-10:
                        best_local = (score, candidate, summary)
                if best_local[1] != thresholds[class_id]:
                    before = thresholds[class_id]
                    thresholds[class_id] = float(best_local[1])
                    current_score = best_local[0]
                    current = best_local[2]
                    improved = True
                    steps.append({"pass": pass_idx + 1, "class_id": class_id, "before": before, "after": thresholds[class_id], "constraint_failed": current["constraint_failed"]})
            if not improved:
                break
        payload["seeds"][str(seed)] = {
            **current,
            "thresholds": {str(cid): thresholds[cid] for cid in class_ids},
            "score": current_score,
            "steps": steps,
        }
        payload["pass_count"] += int(not current["constraint_failed"])
    return payload


def single_seed_score(seed_payload: dict[str, Any]) -> float:
    penalty = sum(max(0.0, -0.01 - seed_payload["delta_vs_clean_default"][key]) for key in ("precision", "map50", "map50_95"))
    feasible = not seed_payload["constraint_failed"]
    metrics = seed_payload["metrics"]
    return (
        (10000.0 if feasible else 0.0)
        - 1000.0 * penalty
        + 100.0 * metrics["map50_95"]
        + 10.0 * metrics["map50"]
        + metrics["recall"]
        + 0.1 * metrics["precision"]
    )


def conservative_search(cache: EvaluatorCache, class_ids: list[int]) -> dict[str, Any]:
    allowed = {}
    for class_id in class_ids:
        if class_id in HIGH_FP_PRIOR:
            allowed[class_id] = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
        elif class_id in CONSERVATIVE_LOWERABLE:
            allowed[class_id] = [0.15, 0.20, 0.25, 0.30, 0.35]
        else:
            allowed[class_id] = [0.25, 0.30, 0.35, 0.40]
    starts = [{class_id: DEFAULT_THRESHOLD for class_id in class_ids}]
    result = greedy_unified_search(cache, class_ids, starts, allowed=allowed, max_passes=3)
    # Enforce the intent: no more than five classes adjusted.
    changes = [cid for cid in class_ids if abs(float(result["thresholds"][str(cid)]) - DEFAULT_THRESHOLD) > 1e-9]
    if len(changes) > 5:
        ranked = sorted(changes, key=lambda cid: abs(float(result["thresholds"][str(cid)]) - DEFAULT_THRESHOLD), reverse=True)
        kept = set(ranked[:5])
        thresholds = {cid: (float(result["thresholds"][str(cid)]) if cid in kept else DEFAULT_THRESHOLD) for cid in class_ids}
        result = cache.evaluate_thresholds(thresholds)
        result["score"] = scheme_score(result)
        result["steps"] = [{"note": "trimmed_to_five_adjusted_classes", "kept": sorted(kept)}]
    return result


def old_rc_thresholds(class_ids: list[int]) -> dict[int, float]:
    payload = read_json(THRESHOLD_CONFIG)
    return {class_id: float(payload["classes"].get(str(class_id), {}).get("threshold", DEFAULT_THRESHOLD)) for class_id in class_ids}


def default_thresholds(class_ids: list[int]) -> dict[int, float]:
    return {class_id: DEFAULT_THRESHOLD for class_id in class_ids}


def high_precision_start(class_ids: list[int]) -> dict[int, float]:
    thresholds = default_thresholds(class_ids)
    for class_id in HIGH_FP_PRIOR:
        if class_id in thresholds:
            thresholds[class_id] = 0.70
    return thresholds


def defect_recall_start(class_ids: list[int]) -> dict[int, float]:
    thresholds = high_precision_start(class_ids)
    for class_id in {5, 6, 7, 9, 12}:
        if class_id in thresholds:
            thresholds[class_id] = 0.15
    return thresholds


def fixed_without_rc_summary() -> dict[str, Any]:
    failed = 0
    rows = {}
    for seed in SEEDS:
        clean = clean_official_metrics(seed)
        fixed = fixed_saved_official_metrics(seed)
        failures = constraint_failures(fixed, clean)
        rows[str(seed)] = {
            "clean_official_val_metrics": clean,
            "fixed_official_val_metrics": fixed,
            "delta_vs_clean": metric_delta(fixed, clean),
            "constraint_failed": bool(failures),
            "failure_reasons": failures,
        }
        failed += int(bool(failures))
    return {"pass_count": len(SEEDS) - failed, "seeds": rows}


def precision_drop_sources(cache: EvaluatorCache, seed: int, thresholds: dict[int, float], names: dict[int, str]) -> list[dict[str, Any]]:
    fixed = cache.seed_payload(seed, thresholds)
    clean = cache.clean_default[seed]
    rows = []
    for cid_s, row in fixed["per_class"].items():
        cid = int(cid_s)
        clean_row = clean.per_class.get(cid, {})
        delta_fp = float(row.get("fp", 0.0)) - float(clean_row.get("fp", 0.0))
        delta_precision = float(row.get("precision", 0.0)) - float(clean_row.get("precision", 0.0))
        rows.append(
            {
                "class_id": cid,
                "class_name": names.get(cid, str(cid)),
                "delta_fp": delta_fp,
                "delta_precision": delta_precision,
                "fixed_precision": row.get("precision"),
                "clean_precision": clean_row.get("precision"),
            }
        )
    return sorted(rows, key=lambda item: (item["delta_fp"], -item["delta_precision"]), reverse=True)


def write_threshold_config(path: Path, name: str, thresholds: dict[int, float], summary: dict[str, Any], names: dict[int, str]) -> None:
    payload = {
        "type": "official_path_threshold_config",
        "name": name,
        "threshold_range": THRESHOLDS,
        "default_threshold": DEFAULT_THRESHOLD,
        "pass_count": summary.get("pass_count"),
        "classes": {
            str(class_id): {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "threshold": float(thresholds[class_id]),
            }
            for class_id in sorted(thresholds)
        },
        "evaluation_summary": summary,
    }
    write_json(path, payload)


def table_row(seed: int, group: str, metrics: dict[str, float], delta: dict[str, float], failed: bool) -> str:
    return (
        f"| {seed} | {group} | {f4(metrics['precision'])} | {f4(metrics['recall'])} | {f4(metrics['map50'])} | {f4(metrics['map50_95'])} | "
        f"{fd(delta['precision'])} | {fd(delta['recall'])} | {fd(delta['map50'])} | {fd(delta['map50_95'])} | {str(failed).lower()} |"
    )


def main() -> None:
    records = load_records()
    class_ids = class_ids_from_records(records)
    names = class_names()
    cache = EvaluatorCache(records, class_ids)
    default = default_thresholds(class_ids)
    old_rc = old_rc_thresholds(class_ids)
    starts = [default, old_rc, high_precision_start(class_ids), defect_recall_start(class_ids)]

    fixed_no_rc = fixed_without_rc_summary()
    old_rc_summary = cache.evaluate_thresholds(old_rc)
    unified = greedy_unified_search(cache, class_ids, starts)
    per_seed = per_seed_search(cache, class_ids)
    conservative = conservative_search(cache, class_ids)

    unified_thresholds = {int(k): float(v) for k, v in unified["thresholds"].items()}
    conservative_thresholds = {int(k): float(v) for k, v in conservative["thresholds"].items()}
    write_threshold_config(UNIFIED_JSON, "unified_precision_guard_thresholds", unified_thresholds, unified, names)
    write_json(PER_SEED_JSON, {
        "type": "official_path_per_seed_threshold_config",
        "threshold_range": THRESHOLDS,
        "default_threshold": DEFAULT_THRESHOLD,
        "pass_count": per_seed["pass_count"],
        "seeds": per_seed["seeds"],
    })
    write_threshold_config(CONSERVATIVE_JSON, "conservative_default_thresholds", conservative_thresholds, conservative, names)

    old_precision_sources = {
        "seed_0": precision_drop_sources(cache, 0, old_rc, names)[:8],
        "seed_2": precision_drop_sources(cache, 2, old_rc, names)[:8],
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "official_predict_postprocess_threshold_reoptimization",
        "fixed_without_rc_official_val": fixed_no_rc,
        "old_rc_official_predict_summary": old_rc_summary,
        "unified_precision_guard": unified,
        "per_seed": per_seed,
        "conservative_default": conservative,
        "old_rc_precision_drop_sources": old_precision_sources,
        "answer_summary": {
            "fixed_without_rc_pass_count": fixed_no_rc["pass_count"],
            "old_rc_pass_count": old_rc_summary["pass_count"],
            "reoptimized_unified_pass_count": unified["pass_count"],
            "per_seed_pass_count": per_seed["pass_count"],
            "conservative_pass_count": conservative["pass_count"],
            "unified_3_of_3": unified["pass_count"] == len(SEEDS),
            "per_seed_3_of_3": per_seed["pass_count"] == len(SEEDS),
        },
    }
    write_json(OUT_JSON, payload)
    write_report(payload, names)
    print(json.dumps(payload["answer_summary"], ensure_ascii=False, indent=2))


def write_report(payload: dict[str, Any], names: dict[int, str]) -> None:
    lines = [
        "# Official-Path CATF-v2 Threshold Re-optimization",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "No training was run. This analysis uses existing Ultralytics `YOLO.predict(conf=0.10)` outputs and the official predict post-processing evaluator.",
        "",
        "## Pass Count Summary",
        "",
        "| scheme | pass_count | note |",
        "|---|---:|---|",
        f"| fixed CATF-v2 without RC | {payload['answer_summary']['fixed_without_rc_pass_count']}/3 | official Ultralytics val constraints |",
        f"| old unified RC | {payload['answer_summary']['old_rc_pass_count']}/3 | saved `catf_v2_rc_per_class_thresholds.json` |",
        f"| reoptimized unified RC | {payload['answer_summary']['reoptimized_unified_pass_count']}/3 | one threshold table for all seeds |",
        f"| per-seed RC | {payload['answer_summary']['per_seed_pass_count']}/3 | model/seed-specific deployment calibration |",
        f"| conservative default RC | {payload['answer_summary']['conservative_pass_count']}/3 | limited changes, high-FP prior protected |",
        "",
        "## Reoptimized Unified Thresholds",
        "",
        "| class | threshold |",
        "|---|---:|",
    ]
    for cid_s, threshold in payload["unified_precision_guard"]["thresholds"].items():
        cid = int(cid_s)
        lines.append(f"| {cid}:{names.get(cid, str(cid))} | {float(threshold):.2f} |")
    lines += [
        "",
        "## Unified Scheme Per-Seed Metrics",
        "",
        "| seed | P | R | mAP50 | mAP50-95 | dP | dR | dM50 | dM95 | constraint_failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in SEEDS:
        row = payload["unified_precision_guard"]["seeds"][str(seed)]
        lines.append(table_row(seed, "unified", row["metrics"], row["delta_vs_clean_default"], row["constraint_failed"]).replace("| unified | ", "| "))
    lines += [
        "",
        "## Per-Seed Scheme Metrics",
        "",
        "| seed | P | R | mAP50 | mAP50-95 | dP | dR | dM50 | dM95 | constraint_failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in SEEDS:
        row = payload["per_seed"]["seeds"][str(seed)]
        lines.append(table_row(seed, "per_seed", row["metrics"], row["delta_vs_clean_default"], row["constraint_failed"]).replace("| per_seed | ", "| "))
    lines += [
        "",
        "## Conservative Scheme Metrics",
        "",
        "| seed | P | R | mAP50 | mAP50-95 | dP | dR | dM50 | dM95 | constraint_failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in SEEDS:
        row = payload["conservative_default"]["seeds"][str(seed)]
        lines.append(table_row(seed, "conservative", row["metrics"], row["delta_vs_clean_default"], row["constraint_failed"]).replace("| conservative | ", "| "))
    lines += [
        "",
        "## Why Old RC Failed",
        "",
        "- The saved RC table lowered many classes to `0.10`, including domain high-FP-prior classes. That recovered Recall but introduced enough false positives to violate the Precision guard on seed0 and seed2.",
        "- Seed0 old RC failed Precision by `-0.0666`; seed2 old RC failed Precision by `-0.0148`.",
        "",
        "### Seed0 FP Sources Under Old RC",
        "",
        "| class | delta FP | delta Precision |",
        "|---|---:|---:|",
    ]
    for row in payload["old_rc_precision_drop_sources"]["seed_0"]:
        lines.append(f"| {row['class_id']}:{row['class_name']} | {row['delta_fp']:+.1f} | {fd(row['delta_precision'])} |")
    lines += [
        "",
        "### Seed2 FP Sources Under Old RC",
        "",
        "| class | delta FP | delta Precision |",
        "|---|---:|---:|",
    ]
    for row in payload["old_rc_precision_drop_sources"]["seed_2"]:
        lines.append(f"| {row['class_id']}:{row['class_name']} | {row['delta_fp']:+.1f} | {fd(row['delta_precision'])} |")
    lines += [
        "",
        "## Conclusions",
        "",
        f"- Unified threshold 3/3 possible: `{str(payload['answer_summary']['unified_3_of_3']).lower()}`.",
        f"- Per-seed threshold 3/3 possible: `{str(payload['answer_summary']['per_seed_3_of_3']).lower()}`.",
        "- If only per-seed calibration passes, RC should be framed as deployment calibration rather than the core training method.",
        "- Recommended paper line: fixed CATF-v2 is the training method; threshold calibration is a deployment-time calibration layer that must be validated on the official predict path.",
    ]
    write_md(OUT_MD, lines)


if __name__ == "__main__":
    main()

