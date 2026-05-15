from __future__ import annotations

from pathlib import Path
from typing import Any

from AutoAugment.diagnostic_pipeline.common import write_json, write_markdown


def write_experiment_report(
    *,
    output_dir: str | Path,
    baseline_record: dict[str, Any] | None,
    diagnosis: dict[str, Any] | None,
    proxy_payload: dict[str, Any] | None,
    short_training_payload: dict[str, Any] | None,
    final_training_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Aggregate pipeline artifacts into paper-oriented experiment summaries."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline_metrics = (baseline_record or {}).get("metrics", {})
    final_metrics = (final_training_payload or {}).get("metrics", {})
    selected_policy = (short_training_payload or {}).get("selected_policy", {})
    comparison = {
        "Baseline": _row_from_metrics(baseline_metrics, status=(baseline_record or {}).get("status")),
        "Fixed Augment": _empty_row("not_run_in_pipeline"),
        "Random Augment": _empty_row("not_run_in_pipeline"),
        "Diagnosis-driven Augment": _row_from_metrics(final_metrics, status=(final_training_payload or {}).get("status")),
    }
    summary = {
        "stage": "experiment_report",
        "status": "completed",
        "method_positioning": "validation_error_diagnostic_driven_augmentation_framework",
        "not_network_structure_modification": True,
        "selected_policy": selected_policy,
        "global_diagnosis": (diagnosis or {}).get("global", {}),
        "diagnostic_issues": (diagnosis or {}).get("issues", []),
        "proxy_top": (proxy_payload or {}).get("ranking", [])[:5],
        "comparison": comparison,
        "artifacts": {
            "baseline_metrics": str((output.parent / "baseline" / "baseline_metrics.json").resolve()),
            "diagnosis_json": str((output.parent / "diagnosis" / "diagnosis.json").resolve()),
            "candidate_policies": str((output.parent / "policies" / "candidate_policies.json").resolve()),
            "proxy_metrics": str((output.parent / "proxy" / "proxy_metrics.json").resolve()),
            "selected_policy": str((output.parent / "short_training" / "selected_policy.json").resolve()),
        },
    }
    write_json(output / "experiment_summary.json", summary)
    write_summary_markdown(output / "experiment_summary.md", summary)
    write_tables_for_paper(output / "tables_for_paper.md", summary)
    return summary


def write_summary_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    """Write the human-readable experiment summary."""

    lines = [
        "# Experiment Summary",
        "",
        "This project is positioned as a validation-error diagnostic-driven data augmentation framework for YOLO training.",
        "It does not modify the YOLO Backbone, Neck, or Head.",
        "",
        "## Selected Policy",
        "",
        f"- Policy: {summary.get('selected_policy', {}).get('policy_id', summary.get('selected_policy', {}).get('name'))}",
        f"- Source issues: {', '.join(summary.get('selected_policy', {}).get('source_issues', []))}",
        "",
        "## Diagnosis",
    ]
    global_metrics = summary.get("global_diagnosis", {})
    lines.extend(
        [
            f"- TP: {global_metrics.get('tp')}",
            f"- FP: {global_metrics.get('fp')}",
            f"- FN: {global_metrics.get('fn')}",
            f"- Precision: {global_metrics.get('precision')}",
            f"- Recall: {global_metrics.get('recall')}",
            "",
            "## Method Comparison",
            "",
            _comparison_table(summary),
        ]
    )
    write_markdown(path, lines)


def write_tables_for_paper(path: str | Path, summary: dict[str, Any]) -> None:
    """Write Markdown tables that can be copied into a paper draft."""

    lines = [
        "# Tables for Paper",
        "",
        "## Main Comparison",
        "",
        _comparison_table(summary),
        "",
        "## Ablation Template",
        "",
        "| Method | mAP50 | mAP50-95 | Precision | Recall | small_object_recall | FP | FN | training_time | policy_search_cost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        "| Ours | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |",
        "| w/o Diagnosis | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |",
        "| w/o Proxy Filter | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |",
        "| w/o Short Training | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |",
        "| Fixed Strong Augment | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |",
        "| Fixed Weak Augment | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |",
    ]
    write_markdown(path, lines)


def _comparison_table(summary: dict[str, Any]) -> str:
    rows = [
        "| Method | mAP50 | mAP50-95 | Precision | Recall | small_object_recall | FP | FN | training_cost | policy_generation_cost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, item in summary.get("comparison", {}).items():
        rows.append(
            "| {name} | {map50} | {map5095} | {precision} | {recall} | {small_recall} | {fp} | {fn} | {training_cost} | {policy_cost} |".format(
                name=name,
                map50=_fmt(item.get("map50")),
                map5095=_fmt(item.get("map50_95")),
                precision=_fmt(item.get("precision")),
                recall=_fmt(item.get("recall")),
                small_recall=_fmt(item.get("small_object_recall")),
                fp=_fmt(item.get("fp")),
                fn=_fmt(item.get("fn")),
                training_cost=item.get("training_cost", item.get("status", "")),
                policy_cost=item.get("policy_generation_cost", ""),
            )
        )
    return "\n".join(rows)


def _row_from_metrics(metrics: dict[str, Any], *, status: str | None) -> dict[str, Any]:
    return {
        "status": status,
        "map50": metrics.get("map50", metrics.get("yolo_map50")),
        "map50_95": metrics.get("map50_95", metrics.get("yolo_map50_95")),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "small_object_recall": metrics.get("small_object_recall"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "training_cost": status,
        "policy_generation_cost": None,
    }


def _empty_row(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "map50": None,
        "map50_95": None,
        "precision": None,
        "recall": None,
        "small_object_recall": None,
        "fp": None,
        "fn": None,
        "training_cost": status,
        "policy_generation_cost": None,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
