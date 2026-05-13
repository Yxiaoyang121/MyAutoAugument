from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from AutoAugment.diagnostics import generate_augmentation_advice
from AutoAugment.diagnostics.yolo_error_analysis import analyze_yolo_errors, write_analysis_outputs
from AutoAugment.policies import Policy, ensure_policy
from AutoAugment.search.proxy_metrics import (
    DEFAULT_EDGE_MARGIN,
    DEFAULT_RARE_CLASS_COUNT_THRESHOLD,
    DEFAULT_SMALL_AREA_THRESHOLD,
    DEFAULT_TINY_AREA_THRESHOLD,
    bbox_retention_raw,
    compute_proxy_score,
    compute_strength_penalty,
    yolo_bbox_edge_mask,
    yolo_bbox_safe_mask,
)
from AutoAugment.utils import IMAGE_EXTENSIONS


@dataclass
class EvaluationResult:
    score: float
    metrics: dict[str, Any]


class BaseEvaluator:
    def evaluate(
        self,
        trial_dir: str | Path,
        policy: Policy | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        raise NotImplementedError


class ProxyEvaluator(BaseEvaluator):
    """Fast evaluator for validating the search loop without training YOLO."""

    def __init__(self, *, score_version: str = "legacy") -> None:
        self.score_version = score_version

    def evaluate(
        self,
        trial_dir: str | Path,
        policy: Policy | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        trial_dir = Path(trial_dir)
        policy = ensure_policy(policy)
        context = context or {}
        images_root = Path(context.get("images_dir", trial_dir / "images"))
        labels_root = Path(context.get("labels_dir", trial_dir / "labels"))
        image_paths = sorted(
            path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            return EvaluationResult(0.0, {"reason": "no output images"})

        valid_images = 0
        means: list[float] = []
        stds: list[float] = []
        mean_deltas: list[float] = []
        saturation_rates: list[float] = []
        total_boxes = 0
        valid_boxes = 0
        safe_boxes = 0
        after_tiny_boxes = 0
        after_small_boxes = 0
        after_edge_boxes = 0
        after_class_counts: dict[int, int] = {}
        source_image_means = {
            str(key): float(value)
            for key, value in (context.get("source_image_means_by_output_name") or {}).items()
            if _is_number(value)
        }
        tiny_area_threshold = float(context.get("tiny_area_threshold", DEFAULT_TINY_AREA_THRESHOLD))
        small_area_threshold = float(context.get("small_area_threshold", DEFAULT_SMALL_AREA_THRESHOLD))
        edge_margin = float(context.get("edge_margin", DEFAULT_EDGE_MARGIN))
        rare_class_count_threshold = int(context.get("rare_class_count_threshold", DEFAULT_RARE_CLASS_COUNT_THRESHOLD))
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                continue
            valid_images += 1
            image_mean = float(image.mean())
            means.append(image_mean)
            stds.append(float(image.std()))
            saturation_rates.append(float(((image <= 3) | (image >= 252)).mean()))
            if image_path.name in source_image_means:
                mean_deltas.append(image_mean - source_image_means[image_path.name])
            try:
                relative = image_path.relative_to(images_root)
            except ValueError:
                relative = Path(image_path.name)
            label_path = labels_root / relative.with_suffix(".txt")
            labels, boxes = _read_yolo_labels_and_boxes(label_path)
            if boxes.size == 0:
                continue
            total_boxes += len(boxes)
            valid_boxes += int(_valid_yolo_mask(boxes).sum())
            safe_boxes += int(yolo_bbox_safe_mask(boxes).sum())
            areas = boxes[:, 2] * boxes[:, 3]
            after_tiny_boxes += int((areas <= tiny_area_threshold).sum())
            after_small_boxes += int((areas <= small_area_threshold).sum())
            after_edge_boxes += int(yolo_bbox_edge_mask(boxes, edge_margin=edge_margin).sum())
            for label in labels:
                class_id = int(label)
                after_class_counts[class_id] = after_class_counts.get(class_id, 0) + 1

        image_valid_rate = valid_images / max(1, len(image_paths))
        bbox_valid_rate = valid_boxes / total_boxes if total_boxes > 0 else 1.0
        bbox_safe_rate = safe_boxes / total_boxes if total_boxes > 0 else 1.0
        source_label_count = int(context.get("source_label_count", total_boxes))
        if source_label_count > 0:
            bbox_retention = min(1.2, total_boxes / source_label_count) / 1.2
        else:
            bbox_retention = 1.0
        true_bbox_retention = bbox_retention_raw(source_label_count, total_boxes)
        source_tiny_boxes = int(context.get("source_tiny_box_count", 0))
        source_small_boxes = int(context.get("source_small_box_count", 0))
        source_edge_boxes = int(context.get("source_edge_box_count", 0))
        tiny_target_retention = bbox_retention_raw(source_tiny_boxes, after_tiny_boxes)
        small_target_retention = bbox_retention_raw(source_small_boxes, after_small_boxes)
        edge_target_retention = bbox_retention_raw(source_edge_boxes, after_edge_boxes)
        source_class_counts = _normalize_class_counts(context.get("source_class_counts"))
        source_class_ids = {class_id for class_id, count in source_class_counts.items() if count > 0}
        after_class_ids = {class_id for class_id, count in after_class_counts.items() if count > 0}
        class_coverage_after = len(source_class_ids & after_class_ids) / len(source_class_ids) if source_class_ids else 1.0
        rare_class_ids = {
            class_id
            for class_id, count in source_class_counts.items()
            if 0 < count <= rare_class_count_threshold
        }
        rare_class_present = bool(rare_class_ids)
        rare_before_count = sum(source_class_counts.get(class_id, 0) for class_id in rare_class_ids)
        rare_after_count = sum(after_class_counts.get(class_id, 0) for class_id in rare_class_ids)
        rare_class_retention = bbox_retention_raw(rare_before_count, rare_after_count) if rare_class_present else 1.0
        mean_intensity = float(np.mean(means)) if means else 0.0
        mean_std = float(np.mean(stds)) if stds else 0.0
        exposure_score = float(np.clip(1.0 - max(0.0, abs(mean_intensity - 127.5) - 85.0) / 42.5, 0.0, 1.0))
        variance_score = float(np.clip(mean_std / 45.0, 0.0, 1.0))
        delta_std = float(np.std(mean_deltas)) if mean_deltas else None
        over_dark_or_over_bright_rate = float(np.mean([(value < 35.0 or value > 220.0) for value in means])) if means else 0.0
        saturation_rate = float(np.mean(saturation_rates)) if saturation_rates else 0.0
        exposure_diversity_score = (
            float(np.clip((delta_std or 0.0) / 18.0, 0.0, 1.0))
            * (1.0 - over_dark_or_over_bright_rate)
            * (1.0 - saturation_rate)
        )
        strength_penalty = compute_strength_penalty(policy)
        op_names = [operation.name for operation in policy.operations]
        diversity_score = len(set(op_names)) / max(1, len(op_names))
        legacy_score = (
            0.20 * image_valid_rate
            + 0.25 * bbox_valid_rate
            + 0.25 * bbox_retention
            + 0.12 * exposure_score
            + 0.10 * variance_score
            + 0.08 * diversity_score
        )
        metrics = {
            "image_count": len(image_paths),
            "valid_image_count": valid_images,
            "image_valid_rate": image_valid_rate,
            "total_boxes": total_boxes,
            "valid_boxes": valid_boxes,
            "bbox_valid_rate": bbox_valid_rate,
            "safe_boxes": safe_boxes,
            "bbox_safe_rate": bbox_safe_rate,
            "bbox_retention": bbox_retention,
            "bbox_retention_raw": true_bbox_retention,
            "source_tiny_box_count": source_tiny_boxes,
            "after_tiny_box_count": after_tiny_boxes,
            "tiny_target_retention": tiny_target_retention,
            "source_small_box_count": source_small_boxes,
            "after_small_box_count": after_small_boxes,
            "small_target_retention": small_target_retention,
            "source_edge_box_count": source_edge_boxes,
            "after_edge_box_count": after_edge_boxes,
            "edge_target_retention": edge_target_retention,
            "source_class_count": len(source_class_ids),
            "after_class_count": len(after_class_ids),
            "class_coverage_after": class_coverage_after,
            "rare_class_present": rare_class_present,
            "rare_class_ids": sorted(rare_class_ids),
            "rare_class_count_threshold": rare_class_count_threshold,
            "rare_class_retention": rare_class_retention,
            "mean_intensity": mean_intensity,
            "mean_std": mean_std,
            "exposure_score": exposure_score,
            "variance_score": variance_score,
            "image_mean_delta_std": delta_std,
            "over_dark_or_over_bright_rate": over_dark_or_over_bright_rate,
            "saturation_rate": saturation_rate,
            "exposure_diversity_score": exposure_diversity_score,
            "strength_penalty": strength_penalty,
            "diversity_score": diversity_score,
        }
        score_version = str(context.get("proxy_score_version", self.score_version))
        if score_version == "dataset2_v1":
            proxy_score, components, missing_components = compute_proxy_score(metrics, version=score_version)
            metrics["proxy_score"] = proxy_score
            metrics["proxy_score_version"] = score_version
            metrics["proxy_score_components"] = components
            metrics["missing_components"] = missing_components
            return EvaluationResult(float(proxy_score), metrics)
        metrics["legacy_proxy_score"] = float(legacy_score)
        return EvaluationResult(float(legacy_score), metrics)


class CommandEvaluator(BaseEvaluator):
    """Evaluator adapter for external training or validation commands."""

    def __init__(
        self,
        command_template: str,
        *,
        score_regex: str = r"score\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        timeout: int | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        if not command_template:
            raise ValueError("command_template is required")
        self.command_template = command_template
        self.score_regex = re.compile(score_regex)
        self.timeout = timeout
        self.cwd = Path(cwd) if cwd is not None else None

    def evaluate(
        self,
        trial_dir: str | Path,
        policy: Policy | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        trial_dir = Path(trial_dir).resolve()
        policy = ensure_policy(policy)
        context = context or {}
        format_values = {
            "dataset": str(trial_dir),
            "trial_dir": str(trial_dir),
            "dataset_yaml": str(trial_dir / "dataset.yaml"),
            "policy": policy.name,
        }
        format_values.update(context)
        command = self.command_template.format(**format_values)
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(self.cwd) if self.cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        output = f"{completed.stdout}\n{completed.stderr}"
        match = self.score_regex.search(output)
        if match is None:
            excerpt = output[-2000:]
            raise RuntimeError(
                "CommandEvaluator could not parse a score. "
                f"Regex: {self.score_regex.pattern}. Exit code: {completed.returncode}. Output tail:\n{excerpt}"
            )
        score = float(match.group(1))
        return EvaluationResult(
            score=score,
            metrics={
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )


class YoloCommandEvaluator(BaseEvaluator):
    """Run an external YOLO validation command and use parsed mAP as score."""

    def __init__(
        self,
        command_template: str,
        *,
        metric: str = "map50",
        dataset_yaml: str | Path | None = None,
        class_names: list[str] | None = None,
        nc: int | None = None,
        timeout: int | None = None,
        hybrid_proxy_weight: float = 0.0,
        workers: int = 0,
    ) -> None:
        if not command_template:
            raise ValueError("command_template is required for YoloCommandEvaluator")
        if metric not in {"map50", "map50_95"}:
            raise ValueError("metric must be one of: map50, map50_95")
        if not 0.0 <= hybrid_proxy_weight <= 1.0:
            raise ValueError("hybrid_proxy_weight must be in [0, 1]")
        if workers < 0:
            raise ValueError("workers must be non-negative")
        self.command_template = command_template
        self.metric = metric
        self.dataset_yaml = Path(dataset_yaml) if dataset_yaml is not None else None
        self.class_names = list(class_names) if class_names is not None else None
        self.nc = nc
        self.timeout = timeout
        self.hybrid_proxy_weight = float(hybrid_proxy_weight)
        self.workers = int(workers)
        self.proxy_evaluator = ProxyEvaluator()

    def evaluate(
        self,
        trial_dir: str | Path,
        policy: Policy | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        trial_dir = Path(trial_dir).resolve()
        policy = ensure_policy(policy)
        context = context or {}
        dataset_yaml = self._prepare_dataset_yaml(trial_dir)
        stdout_path = trial_dir / "yolo_stdout.log"
        stderr_path = trial_dir / "yolo_stderr.log"
        command = self._format_command(trial_dir, dataset_yaml, context)
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
        if completed.returncode != 0:
            raise RuntimeError(
                "YOLO validation command failed. "
                f"Exit code: {completed.returncode}. "
                f"See logs: {stdout_path} and {stderr_path}"
            )

        output = f"{completed.stdout}\n{completed.stderr}"
        parsed = parse_yolo_metrics(output)
        if self.metric not in parsed:
            raise RuntimeError(
                f"Could not parse required YOLO metric '{self.metric}'. "
                f"Parsed metrics: {parsed or '<none>'}. See logs: {stdout_path} and {stderr_path}"
            )
        yolo_score = float(parsed[self.metric])
        proxy_score: float | None = None
        proxy_metrics: dict[str, Any] | None = None
        final_score = yolo_score
        if self.hybrid_proxy_weight > 0.0:
            proxy_result = self.proxy_evaluator.evaluate(trial_dir, policy, context=context)
            proxy_score = float(proxy_result.score)
            proxy_metrics = proxy_result.metrics
            final_score = yolo_score * (1.0 - self.hybrid_proxy_weight) + proxy_score * self.hybrid_proxy_weight
        metrics = {
            "command": command,
            "returncode": completed.returncode,
            "dataset_yaml": str(dataset_yaml),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "yolo_map50": parsed.get("map50"),
            "yolo_map50_95": parsed.get("map50_95"),
            "yolo_metric_name": self.metric,
            "yolo_metric_score": yolo_score,
            "proxy_score": proxy_score,
            "proxy_metrics": proxy_metrics,
            "final_score": final_score,
            "hybrid_proxy_weight": self.hybrid_proxy_weight,
            "workers": self.workers,
        }
        return EvaluationResult(score=float(final_score), metrics=metrics)

    def _prepare_dataset_yaml(self, trial_dir: Path) -> Path:
        trial_yaml = trial_dir / "data.yaml"
        if self.dataset_yaml is not None:
            source = self.dataset_yaml.resolve()
            if not source.exists():
                raise FileNotFoundError(f"dataset yaml does not exist: {source}")
            if source != trial_yaml.resolve():
                shutil.copyfile(source, trial_yaml)
            return source
        write_yolo_data_yaml(trial_yaml, trial_dir, class_names=self.class_names, nc=self.nc)
        return trial_yaml.resolve()

    def _format_command(self, trial_dir: Path, dataset_yaml: Path, context: dict[str, Any]) -> str:
        images_dir = trial_dir / "images"
        labels_dir = trial_dir / "labels"
        raw_values = {
            "dataset": str(trial_dir),
            "trial_dir": str(trial_dir),
            "dataset_yaml": str(dataset_yaml),
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "workers": self.workers,
        }
        raw_values.update({key: str(value) for key, value in context.items()})
        values = {
            key: _quote_command_path(value) if key in {"dataset", "trial_dir", "dataset_yaml", "images_dir", "labels_dir"} else value
            for key, value in raw_values.items()
        }
        return self.command_template.format(**values)


class YoloTrainValEvaluator(BaseEvaluator):
    """Train a YOLO model per policy trial, then validate on a fixed val set."""

    DEFAULT_TRAIN_COMMAND = (
        "yolo detect train model={model} data={dataset_yaml} epochs={epochs} "
        "imgsz={imgsz} batch={batch} workers={workers} {freeze_arg} "
        "project={trial_dir}/train_runs name=train exist_ok=True"
    )
    DEFAULT_VAL_COMMAND = (
        "yolo detect val model={best_pt} data={dataset_yaml} imgsz={imgsz} "
        "batch={batch} workers={workers} project={trial_dir}/val_runs name=val exist_ok=True"
    )
    DEFAULT_PREDICT_COMMAND = (
        "yolo detect predict model={best_pt} source={val_images_dir} imgsz={imgsz} "
        "conf={diagnosis_conf} iou={diagnosis_iou} save_txt=True save_conf=True workers={workers} "
        "project={trial_dir}/diagnosis_runs name=pred exist_ok=True"
    )

    def __init__(
        self,
        train_command_template: str | None = None,
        val_command_template: str | None = None,
        *,
        metric: str = "map50",
        epochs: int = 10,
        imgsz: int = 640,
        batch: int = 8,
        model: str = "yolov8n.pt",
        timeout: int | None = None,
        hybrid_proxy_weight: float = 0.0,
        freeze_backbone: bool = False,
        diagnose_val_errors: bool = False,
        predict_command_template: str | None = None,
        diagnosis_conf: float = 0.25,
        diagnosis_iou: float = 0.5,
        diagnosis_match_iou: float | None = None,
        diagnosis_localization_weak_iou: float = 0.3,
        workers: int = 0,
    ) -> None:
        if metric not in {"map50", "map50_95"}:
            raise ValueError("metric must be one of: map50, map50_95")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if batch <= 0:
            raise ValueError("batch must be positive")
        if workers < 0:
            raise ValueError("workers must be non-negative")
        if not 0.0 <= hybrid_proxy_weight <= 1.0:
            raise ValueError("hybrid_proxy_weight must be in [0, 1]")
        self.train_command_template = train_command_template or self.DEFAULT_TRAIN_COMMAND
        self.val_command_template = val_command_template or self.DEFAULT_VAL_COMMAND
        self.metric = metric
        self.epochs = int(epochs)
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.workers = int(workers)
        self.model = model
        self.timeout = timeout
        self.hybrid_proxy_weight = float(hybrid_proxy_weight)
        self.freeze_backbone = bool(freeze_backbone)
        self.diagnose_val_errors = bool(diagnose_val_errors)
        self.predict_command_template = predict_command_template or self.DEFAULT_PREDICT_COMMAND
        self.diagnosis_conf = float(diagnosis_conf)
        self.diagnosis_iou = float(diagnosis_iou)
        self.diagnosis_match_iou = diagnosis_match_iou
        self.diagnosis_localization_weak_iou = float(diagnosis_localization_weak_iou)
        self.proxy_evaluator = ProxyEvaluator()

    def evaluate(
        self,
        trial_dir: str | Path,
        policy: Policy | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        trial_dir = Path(trial_dir).resolve()
        policy = ensure_policy(policy)
        context = context or {}
        dataset_dir = trial_dir / "dataset"
        train_images_dir = Path(context.get("images_dir", dataset_dir / "images" / "train")).resolve()
        train_labels_dir = Path(context.get("labels_dir", dataset_dir / "labels" / "train")).resolve()
        val_images_dir = Path(context.get("val_images_dir", "")).resolve()
        val_labels_dir = Path(context.get("val_labels_dir", "")).resolve()
        if not val_images_dir.exists():
            raise FileNotFoundError("YoloTrainValEvaluator requires a fixed validation images directory in context['val_images_dir']")
        if not val_labels_dir.exists():
            raise FileNotFoundError("YoloTrainValEvaluator requires a fixed validation labels directory in context['val_labels_dir']")

        val_reference_path = trial_dir / "val_reference.txt"
        val_reference_path.write_text(
            "\n".join(
                [
                    f"val_images_dir={val_images_dir}",
                    f"val_labels_dir={val_labels_dir}",
                    "validation_set_is_fixed=true",
                    "validation_set_augmented=false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        dataset_yaml = write_train_val_data_yaml(
            trial_dir / "data.yaml",
            dataset_dir,
            train_images_dir=train_images_dir,
            train_labels_dir=train_labels_dir,
            val_images_dir=val_images_dir,
            val_labels_dir=val_labels_dir,
            class_names=context.get("class_names"),
        )

        train_stdout_path = trial_dir / "train_stdout.log"
        train_stderr_path = trial_dir / "train_stderr.log"
        val_stdout_path = trial_dir / "val_stdout.log"
        val_stderr_path = trial_dir / "val_stderr.log"

        train_command = self._format_command(
            self.train_command_template,
            trial_dir=trial_dir,
            dataset_dir=dataset_dir,
            dataset_yaml=dataset_yaml,
            train_images_dir=train_images_dir,
            train_labels_dir=train_labels_dir,
            val_images_dir=val_images_dir,
            val_labels_dir=val_labels_dir,
        )
        train_completed = subprocess.run(
            train_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        train_stdout_path.write_text(train_completed.stdout, encoding="utf-8", errors="replace")
        train_stderr_path.write_text(train_completed.stderr, encoding="utf-8", errors="replace")
        if train_completed.returncode != 0:
            raise RuntimeError(
                "YOLO train command failed. "
                f"Exit code: {train_completed.returncode}. "
                f"See logs: {train_stdout_path} and {train_stderr_path}"
            )

        discovered_best = find_yolo_best_pt(trial_dir)
        weights_dir = trial_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        best_pt = weights_dir / "best.pt"
        if discovered_best.resolve() != best_pt.resolve():
            shutil.copyfile(discovered_best, best_pt)

        val_command = self._format_command(
            self.val_command_template,
            trial_dir=trial_dir,
            dataset_dir=dataset_dir,
            dataset_yaml=dataset_yaml,
            train_images_dir=train_images_dir,
            train_labels_dir=train_labels_dir,
            val_images_dir=val_images_dir,
            val_labels_dir=val_labels_dir,
            best_pt=best_pt,
        )
        val_completed = subprocess.run(
            val_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        val_stdout_path.write_text(val_completed.stdout, encoding="utf-8", errors="replace")
        val_stderr_path.write_text(val_completed.stderr, encoding="utf-8", errors="replace")
        if val_completed.returncode != 0:
            raise RuntimeError(
                "YOLO val command failed. "
                f"Exit code: {val_completed.returncode}. "
                f"See logs: {val_stdout_path} and {val_stderr_path}"
            )

        parsed = parse_yolo_metrics(f"{val_completed.stdout}\n{val_completed.stderr}")
        if self.metric not in parsed:
            raise RuntimeError(
                f"Could not parse required YOLO val metric '{self.metric}'. "
                f"Parsed metrics: {parsed or '<none>'}. See logs: {val_stdout_path} and {val_stderr_path}"
            )
        yolo_score = float(parsed[self.metric])
        proxy_score: float | None = None
        final_score = yolo_score
        if self.hybrid_proxy_weight > 0.0:
            proxy_result = self.proxy_evaluator.evaluate(
                trial_dir,
                policy,
                context={
                    **context,
                    "images_dir": str(train_images_dir),
                    "labels_dir": str(train_labels_dir),
                },
            )
            proxy_score = float(proxy_result.score)
            final_score = yolo_score * (1.0 - self.hybrid_proxy_weight) + proxy_score * self.hybrid_proxy_weight

        diagnosis_payload: dict[str, Any] | None = None
        if self.diagnose_val_errors:
            diagnosis_payload = self._run_trial_diagnostics(
                trial_dir=trial_dir,
                dataset_yaml=dataset_yaml,
                val_images_dir=val_images_dir,
                val_labels_dir=val_labels_dir,
                best_pt=best_pt,
                context=context,
            )

        metrics = {
            "yolo_map50": parsed.get("map50"),
            "yolo_map50_95": parsed.get("map50_95"),
            "yolo_metric_name": self.metric,
            "yolo_metric_score": yolo_score,
            "proxy_score": proxy_score,
            "hybrid_proxy_weight": self.hybrid_proxy_weight,
            "final_score": final_score,
            "best_pt": str(best_pt),
            "discovered_best_pt": str(discovered_best),
            "dataset_yaml": str(dataset_yaml),
            "train_command": train_command,
            "val_command": val_command,
            "train_stdout_log": str(train_stdout_path),
            "train_stderr_log": str(train_stderr_path),
            "val_stdout_log": str(val_stdout_path),
            "val_stderr_log": str(val_stderr_path),
            "search_epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "workers": self.workers,
            "freeze_backbone": self.freeze_backbone,
            "stage": "search_short_train",
        }
        if diagnosis_payload is not None:
            metrics.update(diagnosis_payload)
        return EvaluationResult(score=float(final_score), metrics=metrics)

    def _run_trial_diagnostics(
        self,
        *,
        trial_dir: Path,
        dataset_yaml: Path,
        val_images_dir: Path,
        val_labels_dir: Path,
        best_pt: Path,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        diagnosis_dir = trial_dir / "diagnosis"
        diagnosis_dir.mkdir(parents=True, exist_ok=True)
        predict_stdout_path = diagnosis_dir / "predict_stdout.log"
        predict_stderr_path = diagnosis_dir / "predict_stderr.log"
        predict_command = self._format_command(
            self.predict_command_template,
            trial_dir=trial_dir,
            dataset_dir=trial_dir / "dataset",
            dataset_yaml=dataset_yaml,
            val_images_dir=val_images_dir,
            val_labels_dir=val_labels_dir,
            best_pt=best_pt,
            diagnosis_conf=self.diagnosis_conf,
            diagnosis_iou=self.diagnosis_iou,
        )
        completed = subprocess.run(
            predict_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        predict_stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
        predict_stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
        if completed.returncode != 0:
            raise RuntimeError(
                "YOLO diagnostic predict command failed. "
                f"Exit code: {completed.returncode}. See logs: {predict_stdout_path} and {predict_stderr_path}"
            )
        predictions_dir = trial_dir / "diagnosis_runs" / "pred" / "labels"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        class_names = _class_names_for_diagnostics(context.get("class_names"))
        analysis = analyze_yolo_errors(
            images_dir=val_images_dir,
            labels_dir=val_labels_dir,
            predictions_dir=predictions_dir,
            class_names=class_names,
            match_iou=self.diagnosis_match_iou if self.diagnosis_match_iou is not None else self.diagnosis_iou,
            localization_weak_iou=self.diagnosis_localization_weak_iou,
        )
        advice = generate_augmentation_advice(analysis["summary"])
        write_analysis_outputs(diagnosis_dir, analysis=analysis, advice=advice)
        return {
            "diagnosis": advice,
            "val_analysis": analysis["summary"],
            "diagnosis_dir": str(diagnosis_dir),
            "diagnosis_predict_command": predict_command,
            "diagnosis_predict_stdout_log": str(predict_stdout_path),
            "diagnosis_predict_stderr_log": str(predict_stderr_path),
            "diagnosis_predictions_dir": str(predictions_dir),
        }

    def _format_command(self, template: str, **values: Any) -> str:
        command_values: dict[str, Any] = {
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "workers": self.workers,
            "model": self.model,
            "freeze": 10 if self.freeze_backbone else 0,
            "freeze_arg": "freeze=10" if self.freeze_backbone else "",
        }
        command_values.update(values)
        for key in [
            "dataset_yaml",
            "trial_dir",
            "dataset_dir",
            "train_images_dir",
            "train_labels_dir",
            "val_images_dir",
            "val_labels_dir",
            "best_pt",
        ]:
            if key in command_values:
                command_values[key] = _quote_command_path(_path_for_command(command_values[key]))
        return template.format(**command_values)


def parse_yolo_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    map50_95_patterns = [
        r"(?:metrics/)?mAP50[-_]?95(?:\([^)]+\))?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    ]
    map50_patterns = [
        r"(?:metrics/)?mAP50(?:\([^)]+\))?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    ]
    for pattern in map50_95_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            metrics["map50_95"] = float(match.group(1))
            break
    for pattern in map50_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            metrics["map50"] = float(match.group(1))
            break
    metrics.update(_parse_yolo_table_metrics(text))
    return metrics


def write_yolo_data_yaml(
    yaml_path: str | Path,
    trial_dir: str | Path,
    *,
    class_names: list[str] | None = None,
    nc: int | None = None,
) -> Path:
    yaml_path = Path(yaml_path)
    trial_dir = Path(trial_dir).resolve()
    detected_nc = _detect_nc_from_labels(trial_dir / "labels")
    if class_names is not None:
        resolved_nc = max(len(class_names), nc or 0, detected_nc)
        names = list(class_names) + [f"class{index}" for index in range(len(class_names), resolved_nc)]
    else:
        resolved_nc = max(nc or 0, detected_nc, 1)
        names = [f"class{index}" for index in range(resolved_nc)]
    lines = [
        f"path: {trial_dir.as_posix()}",
        "train: images",
        "val: images",
        f"nc: {resolved_nc}",
        "names:",
    ]
    for index, name in enumerate(names):
        lines.append(f"  {index}: {_yaml_quote(name)}")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def write_train_val_data_yaml(
    yaml_path: str | Path,
    dataset_dir: str | Path,
    *,
    train_images_dir: str | Path,
    train_labels_dir: str | Path,
    val_images_dir: str | Path,
    val_labels_dir: str | Path,
    class_names: list[str] | None = None,
) -> Path:
    yaml_path = Path(yaml_path)
    dataset_dir = Path(dataset_dir).resolve()
    train_labels_dir = Path(train_labels_dir).resolve()
    val_labels_dir = Path(val_labels_dir).resolve()
    val_images_dir = Path(val_images_dir).resolve()
    detected_nc = max(_detect_nc_from_labels(train_labels_dir), _detect_nc_from_labels(val_labels_dir), 1)
    if class_names:
        resolved_nc = max(len(class_names), detected_nc)
        names = list(class_names) + [f"class{index}" for index in range(len(class_names), resolved_nc)]
    else:
        resolved_nc = detected_nc
        names = [f"class{index}" for index in range(resolved_nc)]
    lines = [
        f"path: {dataset_dir.as_posix()}",
        "train: images/train",
        f"val: {val_images_dir.as_posix()}",
        f"nc: {resolved_nc}",
        "names:",
    ]
    for index, name in enumerate(names):
        lines.append(f"  {index}: {_yaml_quote(name)}")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path.resolve()


def find_yolo_best_pt(trial_dir: str | Path) -> Path:
    trial_dir = Path(trial_dir)
    direct = trial_dir / "train_runs" / "train" / "weights" / "best.pt"
    if direct.exists():
        return direct.resolve()
    train_runs = trial_dir / "train_runs"
    candidates = sorted(train_runs.rglob("best.pt"), key=lambda path: path.stat().st_mtime, reverse=True) if train_runs.exists() else []
    if candidates:
        return candidates[0].resolve()
    raise FileNotFoundError(
        "Could not find best.pt after YOLO training. "
        f"Expected {direct} or another best.pt under {train_runs}."
    )


def _read_yolo_boxes(label_path: Path) -> np.ndarray:
    return _read_yolo_labels_and_boxes(label_path)[1]


def _read_yolo_labels_and_boxes(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    labels: list[int] = []
    boxes: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        labels.append(int(float(parts[0])))
        boxes.append([float(value) for value in parts[1:5]])
    if not boxes:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    return np.asarray(labels, dtype=np.int64), np.asarray(boxes, dtype=np.float32)


def _normalize_class_counts(value: Any) -> dict[int, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[int, int] = {}
    for key, item in value.items():
        try:
            class_id = int(key)
            count = int(item)
        except (TypeError, ValueError):
            continue
        out[class_id] = count
    return out


def _valid_yolo_mask(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0) & (w > 0.0) & (h > 0.0) & (w <= 1.0) & (h <= 1.0)


def _parse_yolo_table_metrics(text: str) -> dict[str, float]:
    lines = [_strip_ansi(line).strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        tokens = line.split()
        if not tokens or tokens[0].lower() != "all":
            continue
        if not _near_yolo_metric_header(lines, index):
            continue
        values = [_float_or_none(token) for token in tokens[1:]]
        numeric_values = [value for value in values if value is not None]
        if len(numeric_values) < 6:
            continue
        # Ultralytics table rows end with: Box(P), R, mAP50, mAP50-95.
        # Header progress text and ANSI prefixes can shift token indexes, so
        # the row's numeric tail is safer than aligning with header tokens.
        return {
            "map50": numeric_values[-2],
            "map50_95": numeric_values[-1],
        }
    return {}


def _normalize_metric_token(token: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", token).lower()


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def _near_yolo_metric_header(lines: list[str], index: int) -> bool:
    start = max(0, index - 4)
    for candidate in lines[start:index]:
        normalized = _normalize_metric_token(candidate)
        if "map50" in normalized and "map5095" in normalized:
            return True
    return False


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _detect_nc_from_labels(labels_dir: Path) -> int:
    max_class_id = -1
    if not labels_dir.exists():
        return 1
    for label_path in labels_dir.rglob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                max_class_id = max(max_class_id, int(float(parts[0])))
            except ValueError:
                continue
    return max_class_id + 1 if max_class_id >= 0 else 1


def _yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _quote_command_path(value: str) -> str:
    if any(char.isspace() for char in value):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _path_for_command(value: Any) -> str:
    return Path(value).resolve().as_posix()


def _class_names_for_diagnostics(value: Any) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(key): str(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {index: str(item) for index, item in enumerate(value)}
    return {}
