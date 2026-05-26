from scripts.run_yolo_default_diagnosis_constraints import evaluate_constraints, score_constraint_candidates


def row(key: str, *, precision: float, recall: float, map50: float, map50_95: float):
    delta = {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
    }
    return {
        "key": key,
        "label": key,
        "metrics": {"map50_95": 0.5 + map50_95},
        "delta_vs_yolo_default": delta,
        "constraints": evaluate_constraints(delta),
        "diagnosis": {"global": {"fp": 0, "localization_weak": 0}},
    }


def test_constraint_rejects_precision_and_map_drops() -> None:
    result = evaluate_constraints({"precision": -0.021, "recall": 0.1, "map50": -0.011, "map50_95": -0.02})
    assert result["passed"] is False
    assert "precision_drop_gt_0.02" in result["failures"]
    assert "map50_drop_gt_0.01" in result["failures"]
    assert "map50_95_drop_gt_0.01" in result["failures"]


def test_scoring_picks_recall_gain_under_constraints() -> None:
    rows = [
        row("yolo_default", precision=0.0, recall=0.0, map50=0.0, map50_95=0.0),
        row("safe_low", precision=-0.005, recall=0.02, map50=0.0, map50_95=-0.002),
        row("unsafe_high", precision=-0.05, recall=0.08, map50=0.0, map50_95=0.0),
    ]
    score = score_constraint_candidates(rows)
    assert score["best_key"] == "safe_low"
    assert score["recall_improved_under_constraints"] is True


def test_scoring_tiebreaks_by_map50_95() -> None:
    rows = [
        row("yolo_default", precision=0.0, recall=0.0, map50=0.0, map50_95=0.0),
        row("safe_a", precision=0.0, recall=0.01, map50=0.0, map50_95=-0.005),
        row("safe_b", precision=0.0, recall=0.01, map50=0.0, map50_95=0.004),
    ]
    score = score_constraint_candidates(rows)
    assert score["best_key"] == "safe_b"
