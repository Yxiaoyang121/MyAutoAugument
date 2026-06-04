from __future__ import annotations

from AutoAugment.catf_v2.per_class_thresholds import (
    ThresholdCandidate,
    constraint_failures,
    robust_threshold_table,
    select_safe_candidate,
    threshold_changes,
)


def _candidate(name: str, metrics: dict[str, float], failed: bool = False) -> ThresholdCandidate:
    reference = {"precision": 0.70, "recall": 0.70, "map50": 0.70, "map50_95": 0.50}
    return ThresholdCandidate(
        name=name,
        thresholds={0: 0.25},
        metrics=metrics,
        per_class={},
        score=metrics["recall"],
        constraint_failed=failed,
        failure_reasons=constraint_failures(metrics, reference),
        delta_vs_reference={key: metrics[key] - reference[key] for key in reference},
    )


def test_constraint_failures_respects_industrial_tolerance() -> None:
    reference = {"precision": 0.70, "recall": 0.70, "map50": 0.70, "map50_95": 0.50}
    metrics = {"precision": 0.689, "recall": 0.90, "map50": 0.691, "map50_95": 0.489}

    failures = constraint_failures(metrics, reference, tolerance=0.01)

    assert "precision_drop_gt_tolerance" in failures
    assert "map50_95_drop_gt_tolerance" in failures
    assert "map50_drop_gt_tolerance" not in failures


def test_select_safe_candidate_prefers_feasible_high_recall() -> None:
    failed_high_recall = _candidate(
        "failed",
        {"precision": 0.65, "recall": 0.95, "map50": 0.72, "map50_95": 0.52},
        failed=True,
    )
    safe_lower_recall = _candidate(
        "safe",
        {"precision": 0.70, "recall": 0.80, "map50": 0.71, "map50_95": 0.50},
        failed=False,
    )

    selected = select_safe_candidate([failed_high_recall, safe_lower_recall], {"precision": 0.70, "recall": 0.70, "map50": 0.70, "map50_95": 0.50})

    assert selected.name == "safe"


def test_robust_threshold_table_uses_median_across_seeds() -> None:
    table = robust_threshold_table(
        {
            0: {1: 0.10, 2: 0.35},
            1: {1: 0.20, 2: 0.65},
            2: {1: 0.10, 2: 0.25},
        }
    )

    assert table[1] == 0.10
    assert table[2] == 0.35


def test_threshold_changes_reports_raise_and_lower() -> None:
    changes = threshold_changes({0: 0.70, 1: 0.10, 2: 0.25})

    assert changes["raise"] == [{"class_id": 0, "old_threshold": 0.25, "new_threshold": 0.70}]
    assert changes["lower"] == [{"class_id": 1, "old_threshold": 0.25, "new_threshold": 0.10}]

