from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from AutoAugment.diagnostic_pipeline.common import (
    find_yolo_weight,
    maybe_device_arg,
    metrics_from_logs,
    quote_path,
    run_logged_command,
    write_json,
)
from AutoAugment.diagnostic_pipeline.dataset_builder import write_standard_data_yaml
from AutoAugment.diagnostic_pipeline.diagnosis import run_error_diagnosis
from AutoAugment.diagnostic_pipeline.prediction import run_validation_prediction
from AutoAugment.policies import Policy, apply_policy_to_sample
from AutoAugment.utils import YoloImageRecord, copy_yolo_records, flatten_relative_stem, load_yolo_sample, save_yolo_sample


def run_short_training_selector(
    *,
    proxy_payload: dict[str, Any],
    train_records: list[YoloImageRecord],
    val_records: list[YoloImageRecord],
    output_dir: str | Path,
    data_yaml: str | Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int = 0,
    device: str | int | None = None,
    top_k: int = 3,
    class_names: dict[int, str] | None = None,
    seed: int = 42,
    dry_run: bool = False,
    skip_training: bool = False,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Run short YOLO training for top proxy-ranked policies and select a policy."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ranking = list(proxy_payload.get("ranking", []))
    top_rows = ranking[: max(1, int(top_k))]
    if not top_rows:
        raise ValueError("proxy ranking is empty; cannot select policy")

    trial_records: list[dict[str, Any]] = []
    if dry_run or skip_training:
        for index, row in enumerate(top_rows):
            trial_records.append(
                _planned_trial_record(
                    row=row,
                    trial_dir=output / "trials" / f"trial_{index:03d}_{row.get('policy_id', 'policy')}",
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    workers=workers,
                    model=model,
                    data_yaml=data_yaml,
                    dry_run=dry_run,
                    skip_training=skip_training,
                )
            )
        selected = _select_from_proxy_only(top_rows)
        payload = _short_training_payload(
            trial_records,
            selected_policy=selected["policy"],
            dry_run=dry_run,
            skipped=skip_training,
            selection_reason="proxy_only" if skip_training else "dry_run_plan",
        )
        _write_short_outputs(output, payload)
        return payload

    rng = np.random.default_rng(seed)
    for index, row in enumerate(top_rows):
        policy = Policy.from_dict(row["policy"])
        policy_id = str(row.get("policy_id", policy.name))
        trial_dir = output / "trials" / f"trial_{index:03d}_{policy_id}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_json(trial_dir / "policy.json", row["policy"])
        write_json(trial_dir / "proxy_metrics.json", row)
        trial_data_yaml = _build_short_trial_dataset(
            trial_dir=trial_dir,
            policy=policy,
            train_records=train_records,
            val_records=val_records,
            class_names=class_names,
            rng=np.random.default_rng(int(rng.integers(0, np.iinfo(np.uint32).max))),
        )
        train_command = (
            f"yolo detect train model={model} data={quote_path(trial_data_yaml)} epochs={epochs} "
            f"imgsz={imgsz} batch={batch} workers={workers}{maybe_device_arg(device)} "
            f"project={quote_path(trial_dir / 'train_runs')} name=train exist_ok=True"
        )
        train_result = run_logged_command(
            train_command,
            output_dir=trial_dir,
            prefix="train",
            timeout=timeout,
        )
        if train_result.returncode != 0:
            raise RuntimeError(f"Short-train failed for {policy_id}; see {train_result.stderr_log}")
        best_pt = find_yolo_weight(trial_dir / "train_runs" / "train", "best.pt")
        last_pt = find_yolo_weight(trial_dir / "train_runs" / "train", "last.pt")
        if best_pt is None:
            raise FileNotFoundError(f"Could not find short-train best.pt for {policy_id}")
        val_command = (
            f"yolo detect val model={quote_path(best_pt)} data={quote_path(trial_data_yaml)} "
            f"imgsz={imgsz} batch={batch} workers={workers}{maybe_device_arg(device)} "
            f"project={quote_path(trial_dir / 'val_runs')} name=val exist_ok=True"
        )
        val_result = run_logged_command(
            val_command,
            output_dir=trial_dir,
            prefix="val",
            timeout=timeout,
        )
        if val_result.returncode != 0:
            raise RuntimeError(f"Short validation failed for {policy_id}; see {val_result.stderr_log}")
        val_metrics = metrics_from_logs(val_result.stdout_log, val_result.stderr_log)
        diagnosis_metrics = _trial_prediction_diagnosis(
            trial_dir=trial_dir,
            best_pt=best_pt,
            val_images_dir=trial_dir / "dataset" / "images" / "val",
            val_labels_dir=trial_dir / "dataset" / "labels" / "val",
            imgsz=imgsz,
            workers=workers,
            device=device,
            class_names=class_names,
            timeout=timeout,
        )
        val_metrics.update(diagnosis_metrics)
        selection_score = _selection_score(val_metrics)
        val_metrics["selection_score"] = selection_score
        write_json(trial_dir / "val_metrics.json", val_metrics)
        trial_record = {
            "stage": "short_training_trial",
            "status": "completed",
            "policy_id": policy_id,
            "policy": row["policy"],
            "proxy_metrics": row,
            "train_command": train_result.command,
            "val_command": val_result.command,
            "train_stdout_log": str(train_result.stdout_log),
            "train_stderr_log": str(train_result.stderr_log),
            "val_stdout_log": str(val_result.stdout_log),
            "val_stderr_log": str(val_result.stderr_log),
            "best_pt": str(best_pt),
            "last_pt": str(last_pt) if last_pt else None,
            "val_metrics": val_metrics,
            "selection_score": selection_score,
        }
        write_json(trial_dir / "trial_record.json", trial_record)
        trial_records.append(trial_record)

    best_trial = max(trial_records, key=lambda item: float(item.get("selection_score", 0.0)))
    payload = _short_training_payload(
        trial_records,
        selected_policy=best_trial["policy"],
        dry_run=False,
        skipped=False,
        selection_reason="short_training_selection_score",
    )
    _write_short_outputs(output, payload)
    return payload


def _build_short_trial_dataset(
    *,
    trial_dir: Path,
    policy: Policy,
    train_records: list[YoloImageRecord],
    val_records: list[YoloImageRecord],
    class_names: dict[int, str] | None,
    rng: np.random.Generator,
) -> Path:
    dataset_dir = trial_dir / "dataset"
    train_images = dataset_dir / "images" / "train"
    train_labels = dataset_dir / "labels" / "train"
    val_images = dataset_dir / "images" / "val"
    val_labels = dataset_dir / "labels" / "val"
    copy_yolo_records(train_records, train_images, train_labels)
    copy_yolo_records(val_records, val_images, val_labels)
    for record in train_records:
        sample = load_yolo_sample(record)
        augmented = apply_policy_to_sample(sample, policy, rng=rng)
        stem = flatten_relative_stem(record.relative_path)
        suffix = record.image_path.suffix.lower()
        save_yolo_sample(augmented, train_images / f"{stem}_shortaug{suffix}", train_labels / f"{stem}_shortaug.txt")
    return write_standard_data_yaml(
        trial_dir / "data.yaml",
        dataset_dir=dataset_dir,
        class_names=class_names,
        train_labels_dir=train_labels,
        val_labels_dir=val_labels,
    )


def _trial_prediction_diagnosis(
    *,
    trial_dir: Path,
    best_pt: Path,
    val_images_dir: Path,
    val_labels_dir: Path,
    imgsz: int,
    workers: int,
    device: str | int | None,
    class_names: dict[int, str] | None,
    timeout: int | None,
) -> dict[str, Any]:
    pred_record = run_validation_prediction(
        weights=best_pt,
        val_images_dir=val_images_dir,
        val_labels_dir=val_labels_dir,
        output_dir=trial_dir / "diagnosis_prediction",
        imgsz=imgsz,
        workers=workers,
        device=device,
        dry_run=False,
        timeout=timeout,
    )
    diagnosis = run_error_diagnosis(
        val_images_dir=val_images_dir,
        val_labels_dir=val_labels_dir,
        predictions_dir=pred_record["predictions_dir"],
        output_dir=trial_dir / "diagnosis",
        class_names=class_names,
    )
    global_metrics = diagnosis.get("global", {})
    size_recall = diagnosis.get("size_recall", {})
    small = size_recall.get("small", {})
    tiny = size_recall.get("tiny", {})
    small_gt = int(small.get("gt_count", 0) or 0) + int(tiny.get("gt_count", 0) or 0)
    small_tp = int(small.get("tp_count", 0) or 0) + int(tiny.get("tp_count", 0) or 0)
    return {
        "small_object_recall": small_tp / max(1, small_gt),
        "fp": int(global_metrics.get("fp", 0)),
        "fn": int(global_metrics.get("fn", 0)),
        "tp": int(global_metrics.get("tp", 0)),
        "diagnosis_dir": str((trial_dir / "diagnosis").resolve()),
    }


def _selection_score(metrics: dict[str, Any]) -> float:
    map50 = float(metrics.get("map50", metrics.get("yolo_map50", 0.0)) or 0.0)
    map50_95 = float(metrics.get("map50_95", metrics.get("yolo_map50_95", 0.0)) or 0.0)
    recall = float(metrics.get("recall", 0.0) or 0.0)
    small_recall = float(metrics.get("small_object_recall", 0.0) or 0.0)
    fp = float(metrics.get("fp", 0.0) or 0.0)
    fn = float(metrics.get("fn", 0.0) or 0.0)
    penalty = min(0.15, 0.002 * (fp + fn))
    return float(0.30 * map50 + 0.35 * map50_95 + 0.20 * recall + 0.15 * small_recall - penalty)


def _planned_trial_record(
    *,
    row: dict[str, Any],
    trial_dir: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    model: str,
    data_yaml: str | Path,
    dry_run: bool,
    skip_training: bool,
) -> dict[str, Any]:
    trial_dir.mkdir(parents=True, exist_ok=True)
    policy = row.get("policy", {})
    write_json(trial_dir / "policy.json", policy)
    write_json(trial_dir / "proxy_metrics.json", row)
    command = (
        f"yolo detect train model={model} data={quote_path(data_yaml)} epochs={epochs} "
        f"imgsz={imgsz} batch={batch} workers={workers} project={quote_path(trial_dir / 'train_runs')} name=train exist_ok=True"
    )
    (trial_dir / "train_command.txt").write_text(command + "\n", encoding="utf-8")
    (trial_dir / "train_stdout.log").write_text("[planned] short training not executed\n", encoding="utf-8")
    (trial_dir / "train_stderr.log").write_text("", encoding="utf-8")
    val_metrics = {"status": "skipped" if skip_training else "planned", "dry_run": dry_run}
    write_json(trial_dir / "val_metrics.json", val_metrics)
    record = {
        "stage": "short_training_trial",
        "status": "skipped" if skip_training else "planned",
        "dry_run": dry_run,
        "skip_training": skip_training,
        "policy_id": row.get("policy_id"),
        "policy": policy,
        "proxy_metrics": row,
        "train_command": command,
        "val_metrics": val_metrics,
        "selection_score": float(row.get("proxy_score", 0.0) or 0.0),
    }
    write_json(trial_dir / "trial_record.json", record)
    return record


def _select_from_proxy_only(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda item: (
            bool(item.get("hard_filter_pass")),
            float(item.get("proxy_score", 0.0) or 0.0),
            -len(item.get("hard_filter_reasons", []) or []),
        ),
    )


def _short_training_payload(
    trial_records: list[dict[str, Any]],
    *,
    selected_policy: dict[str, Any],
    dry_run: bool,
    skipped: bool,
    selection_reason: str,
) -> dict[str, Any]:
    return {
        "stage": "low_cost_short_training_selector",
        "status": "skipped" if skipped else ("planned" if dry_run else "completed"),
        "dry_run": dry_run,
        "skipped": skipped,
        "selection_reason": selection_reason,
        "trial_count": len(trial_records),
        "trials": trial_records,
        "selected_policy": selected_policy,
    }


def _write_short_outputs(output: Path, payload: dict[str, Any]) -> None:
    write_json(output / "short_train_results.json", payload)
    write_json(output / "selected_policy.json", payload["selected_policy"])
