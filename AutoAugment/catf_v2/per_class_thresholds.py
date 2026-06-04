from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


@dataclass(frozen=True)
class ThresholdCandidate:
    """A deployment-time per-class threshold candidate."""

    name: str
    thresholds: dict[int, float]
    metrics: dict[str, float]
    per_class: dict[int, dict[str, float]]
    score: float
    constraint_failed: bool
    failure_reasons: list[str]
    delta_vs_reference: dict[str, float]


def constraint_failures(metrics: dict[str, float], reference: dict[str, float], *, tolerance: float = 0.01) -> list[str]:
    failures: list[str] = []
    if metrics["precision"] < reference["precision"] - tolerance:
        failures.append("precision_drop_gt_tolerance")
    if metrics["map50"] < reference["map50"] - tolerance:
        failures.append("map50_drop_gt_tolerance")
    if metrics["map50_95"] < reference["map50_95"] - tolerance:
        failures.append("map50_95_drop_gt_tolerance")
    return failures


def metric_delta(metrics: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics[key]) - float(reference[key]) for key in METRIC_KEYS}


def rc_threshold_score(metrics: dict[str, float], reference: dict[str, float], *, tolerance: float = 0.01) -> float:
    """Score for CATF-v2-RC threshold selection.

    The score is recall-oriented, but it penalizes violation of the industrial
    precision/mAP constraints. It intentionally keeps mAP50-95 in the primary
    objective because seed0's remaining risk is localization/AP quality.
    """

    penalty = sum(max(0.0, reference[key] - tolerance - metrics[key]) for key in ("precision", "map50", "map50_95"))
    return (
        0.35 * metrics["recall"]
        + 0.25 * metrics["map50_95"]
        + 0.20 * metrics["map50"]
        + 0.20 * metrics["precision"]
        - 3.0 * penalty
    )


def select_safe_candidate(candidates: list[ThresholdCandidate], reference: dict[str, float]) -> ThresholdCandidate:
    """Select the best threshold candidate for constrained deployment.

    Feasible candidates are ranked by recall first, then mAP50-95, then mAP50.
    If none are feasible, return the highest-scoring candidate so reports can
    expose the least-bad fallback rather than silently failing.
    """

    feasible = [candidate for candidate in candidates if not candidate.constraint_failed]
    pool = feasible or candidates
    return max(
        pool,
        key=lambda candidate: (
            1 if not candidate.constraint_failed else 0,
            candidate.metrics["recall"],
            candidate.metrics["map50_95"],
            candidate.metrics["map50"],
            candidate.metrics["precision"],
            candidate.score,
        ),
    )


def threshold_changes(thresholds: dict[int, float], *, default_threshold: float = 0.25) -> dict[str, list[dict[str, Any]]]:
    raised: list[dict[str, Any]] = []
    lowered: list[dict[str, Any]] = []
    for class_id, threshold in sorted(thresholds.items()):
        row = {"class_id": int(class_id), "old_threshold": default_threshold, "new_threshold": float(threshold)}
        if threshold > default_threshold:
            raised.append(row)
        elif threshold < default_threshold:
            lowered.append(row)
    return {"raise": raised, "lower": lowered}


def robust_threshold_table(seed_thresholds: dict[int, dict[int, float]], *, default_threshold: float = 0.25) -> dict[int, float]:
    class_ids = sorted({class_id for thresholds in seed_thresholds.values() for class_id in thresholds})
    table: dict[int, float] = {}
    for class_id in class_ids:
        values = [thresholds.get(class_id, default_threshold) for thresholds in seed_thresholds.values()]
        table[class_id] = round(float(median(values)), 2)
    return table


def write_threshold_config(path: str | Path, thresholds: dict[int, float], *, class_names: dict[int, str] | None = None) -> None:
    class_names = class_names or {}
    payload = {
        "type": "per_class_threshold_config",
        "default_threshold": 0.25,
        "classes": {
            str(class_id): {
                "class_id": int(class_id),
                "class_name": class_names.get(class_id, str(class_id)),
                "threshold": float(threshold),
            }
            for class_id, threshold in sorted(thresholds.items())
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

