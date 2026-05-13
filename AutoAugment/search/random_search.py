from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from AutoAugment.bbox.convert import xyxy_to_yolo
from AutoAugment.policies import Policy, SearchSpace, apply_policy_to_sample, default_detection_search_space
from AutoAugment.search.adaptive import DiagnosticPolicyUpdater, diagnose_trial_result, diff_policies_and_space
from AutoAugment.search.evaluator import BaseEvaluator, EvaluationResult, ProxyEvaluator
from AutoAugment.search.proxy_metrics import (
    DEFAULT_EDGE_MARGIN,
    DEFAULT_RARE_CLASS_COUNT_THRESHOLD,
    DEFAULT_SMALL_AREA_THRESHOLD,
    DEFAULT_TINY_AREA_THRESHOLD,
    apply_proxy_hard_filter,
    compute_proxy_score,
    select_proxy_candidate,
    yolo_bbox_edge_mask,
)
from AutoAugment.utils import (
    IMAGE_EXTENSIONS,
    find_yolo_records,
    flatten_relative_stem,
    load_yolo_sample,
    sample_records,
    save_yolo_sample,
)


@dataclass
class ProxyPrefilterConfig:
    enabled: bool = False
    candidate_policies: int = 24
    proxy_eval_samples: int = 96
    proxy_top_k: int = 1
    proxy_score_version: str = "dataset2_v1"
    hard_filter_profile: str = "dataset2_v1"
    keep_candidate_artifacts: bool = False
    artifact_mode: str = "metrics_only"
    max_proxy_artifact_gb: float = 5.0

    @classmethod
    def from_value(cls, value: "ProxyPrefilterConfig | dict[str, Any] | None") -> "ProxyPrefilterConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(
            enabled=bool(value.get("enabled", False)),
            candidate_policies=int(value.get("candidate_policies", 24)),
            proxy_eval_samples=int(value.get("proxy_eval_samples", 96)),
            proxy_top_k=int(value.get("proxy_top_k", 1)),
            proxy_score_version=str(value.get("proxy_score_version", "dataset2_v1")),
            hard_filter_profile=str(value.get("hard_filter_profile", "dataset2_v1")),
            keep_candidate_artifacts=bool(value.get("keep_candidate_artifacts", False)),
            artifact_mode=str(value.get("artifact_mode", "metrics_only")),
            max_proxy_artifact_gb=float(value.get("max_proxy_artifact_gb", 5.0)),
        )

    def __post_init__(self) -> None:
        if self.candidate_policies <= 0:
            raise ValueError("candidate_policies must be positive")
        if self.proxy_eval_samples <= 0:
            raise ValueError("proxy_eval_samples must be positive")
        if self.proxy_top_k <= 0:
            raise ValueError("proxy_top_k must be positive")
        if self.artifact_mode not in {"metrics_only", "selected_only", "all"}:
            raise ValueError("artifact_mode must be one of: metrics_only, selected_only, all")
        if self.max_proxy_artifact_gb <= 0:
            raise ValueError("max_proxy_artifact_gb must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "candidate_policies": self.candidate_policies,
            "proxy_eval_samples": self.proxy_eval_samples,
            "proxy_top_k": self.proxy_top_k,
            "proxy_score_version": self.proxy_score_version,
            "hard_filter_profile": self.hard_filter_profile,
            "keep_candidate_artifacts": self.keep_candidate_artifacts,
            "artifact_mode": self.artifact_mode,
            "max_proxy_artifact_gb": self.max_proxy_artifact_gb,
        }

    @property
    def effective_artifact_mode(self) -> str:
        if not self.keep_candidate_artifacts:
            return "metrics_only"
        return self.artifact_mode


@dataclass
class TrialResult:
    trial_index: int
    score: float
    metrics: dict[str, Any]
    policy: Policy
    trial_dir: Path
    before_policy: Policy | None = None
    diagnosis: dict[str, Any] | None = None
    adjust_reason: str | None = None
    after_policy: Policy | None = None
    policy_diff: dict[str, Any] | None = None
    score_before: float | None = None
    score_after: float | None = None
    accepted: bool | None = None
    rejected: bool | None = None
    yolo_log_paths: dict[str, str] | None = None
    proxy_prefilter: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "score": self.score,
            "metrics": self.metrics,
            "policy": self.policy.to_dict(),
            "trial_dir": str(self.trial_dir),
            "before_policy": self.before_policy.to_dict() if self.before_policy is not None else self.policy.to_dict(),
            "diagnosis": self.diagnosis or {},
            "adjust_reason": self.adjust_reason,
            "after_policy": self.after_policy.to_dict() if self.after_policy is not None else None,
            "policy_diff": self.policy_diff or {},
            "score_before": self.score_before,
            "score_after": self.score_after,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "workers": self.metrics.get("workers"),
            "yolo_log_paths": self.yolo_log_paths or _extract_log_paths(self.metrics),
            "proxy_prefilter": self.proxy_prefilter or {"enabled": False},
        }


class RandomSearch:
    def __init__(
        self,
        *,
        dataset_root: str | Path,
        output_dir: str | Path,
        num_trials: int = 10,
        num_samples: int | None = 20,
        seed: int = 42,
        search_space: SearchSpace | None = None,
        evaluator: BaseEvaluator | None = None,
        image_exts: Iterable[str] | None = None,
        keep_intermediate: bool = True,
        missing_label: str = "empty",
        records: list | None = None,
        trial_layout: str = "flat",
        context: dict[str, Any] | None = None,
        on_trial: Callable[[TrialResult, TrialResult | None], None] | None = None,
        adaptive_policy: bool = False,
        policy_updater: DiagnosticPolicyUpdater | None = None,
        stage_config: dict[str, Any] | None = None,
        proxy_prefilter: ProxyPrefilterConfig | dict[str, Any] | None = None,
    ) -> None:
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.num_trials = int(num_trials)
        self.num_samples = num_samples
        self.seed = int(seed)
        self.search_space = search_space or default_detection_search_space()
        self.evaluator = evaluator or ProxyEvaluator()
        self.image_exts = tuple(image_exts or IMAGE_EXTENSIONS)
        self.keep_intermediate = keep_intermediate
        self.missing_label = missing_label
        self.records = records
        self.trial_layout = trial_layout
        self.context = dict(context or {})
        self.on_trial = on_trial
        self.adaptive_policy = bool(adaptive_policy)
        self.policy_updater = policy_updater
        self.stage_config = dict(stage_config or {})
        self.proxy_prefilter = ProxyPrefilterConfig.from_value(proxy_prefilter)
        if self.trial_layout not in {"flat", "train_yolo"}:
            raise ValueError("trial_layout must be one of: flat, train_yolo")

    def run(self) -> list[TrialResult]:
        rng = np.random.default_rng(self.seed)
        records = self.records or find_yolo_records(self.dataset_root, image_exts=self.image_exts, missing_label=self.missing_label)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trials_root = self.output_dir / "trials"
        trials_root.mkdir(parents=True, exist_ok=True)
        results: list[TrialResult] = []
        best: TrialResult | None = None
        updater = self.policy_updater
        if self.adaptive_policy and updater is None:
            updater = DiagnosticPolicyUpdater(self.search_space)
        pending_policy: Policy | None = None
        previous_metrics: dict[str, Any] | None = None
        for trial_index in range(self.num_trials):
            trial_dir = trials_root / f"trial_{trial_index:03d}"
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
            if self.proxy_prefilter.enabled:
                selected = sample_records(records, self.num_samples, rng)
                policy, proxy_prefilter_record = self._select_policy_with_proxy_prefilter(
                    trial_dir=trial_dir,
                    trial_index=trial_index,
                    records=selected,
                    rng=rng,
                    updater=updater,
                )
            else:
                policy = pending_policy or (
                    updater.sample_policy(rng, name=f"policy_trial_{trial_index:03d}")
                    if updater is not None
                    else self.search_space.sample_policy(rng, name=f"policy_trial_{trial_index:03d}")
                )
                selected = sample_records(records, self.num_samples, rng)
                proxy_prefilter_record = {"enabled": False}
            policy.name = f"policy_trial_{trial_index:03d}"
            before_policy = policy
            score_before = best.score if best is not None else None
            image_dir, label_dir = self._trial_image_label_dirs(trial_dir)
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            policy.save(trial_dir / "policy.json")
            source_label_count = self._write_trial_dataset(trial_dir, selected, policy, rng, image_dir, label_dir)
            context = {
                **self.context,
                "source_label_count": source_label_count,
                "dataset_root": str(Path(self.dataset_root).resolve()),
                "output_dir": str(self.output_dir.resolve()),
                "trial_index": trial_index,
                "trial_layout": self.trial_layout,
                "train_dataset_dir": str((trial_dir / "dataset").resolve()) if self.trial_layout == "train_yolo" else None,
                "images_dir": str(image_dir.resolve()),
                "labels_dir": str(label_dir.resolve()),
                "stage_config": self.stage_config,
            }
            evaluation = self.evaluator.evaluate(
                trial_dir,
                policy,
                context=context,
            )
            accepted = best is None or float(evaluation.score) > float(best.score)
            score_after = float(evaluation.score) if accepted or best is None else float(best.score)
            diagnosis = diagnose_trial_result(evaluation.metrics, context=context, previous_metrics=previous_metrics)
            after_policy: Policy | None = None
            policy_diff: dict[str, Any] = {}
            adjust_reason = "adaptive_policy_disabled"
            pending_policy = None
            if updater is not None:
                adjustment = updater.update(
                    evaluated_policy=policy,
                    metrics=evaluation.metrics,
                    diagnosis=diagnosis,
                    accepted=accepted,
                    rng=rng,
                    next_policy_name=f"policy_trial_{trial_index + 1:03d}",
                )
                after_policy = adjustment.after_policy
                policy_diff = adjustment.policy_diff
                adjust_reason = adjustment.adjust_reason
                pending_policy = None if self.proxy_prefilter.enabled else after_policy
            else:
                policy_diff = diff_policies_and_space(before_policy=policy, after_policy=None)
            diagnosis = dict(diagnosis)
            diagnosis["policy_diff"] = policy_diff
            result = TrialResult(
                trial_index=trial_index,
                score=float(evaluation.score),
                metrics=evaluation.metrics,
                policy=policy,
                trial_dir=trial_dir,
                before_policy=before_policy,
                diagnosis=diagnosis,
                adjust_reason=adjust_reason,
                after_policy=after_policy,
                policy_diff=policy_diff,
                score_before=score_before,
                score_after=score_after,
                accepted=accepted,
                rejected=not accepted,
                yolo_log_paths=_extract_log_paths(evaluation.metrics),
                proxy_prefilter=proxy_prefilter_record,
            )
            (trial_dir / "metrics.json").write_text(
                json.dumps(result.metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (trial_dir / "diagnosis.json").write_text(
                json.dumps(result.diagnosis or {}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (trial_dir / "trial_record.json").write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            results.append(result)
            if accepted:
                best = result
                policy.save(self.output_dir / "best_policy.json")
            self._write_results(results)
            self._write_policy_history(results)
            if self.on_trial is not None:
                self.on_trial(result, best)
            if not self.keep_intermediate and best is not None:
                self._cleanup_intermediate(results, best)
            previous_metrics = evaluation.metrics
        if best is None:
            raise RuntimeError("search finished without a valid trial")
        self._write_results(results)
        self._write_policy_history(results)
        best.policy.save(self.output_dir / "best_policy.json")
        return results

    def _select_policy_with_proxy_prefilter(
        self,
        *,
        trial_dir: Path,
        trial_index: int,
        records: list,
        rng: np.random.Generator,
        updater: DiagnosticPolicyUpdater | None,
    ) -> tuple[Policy, dict[str, Any]]:
        config = self.proxy_prefilter
        trial_dir.mkdir(parents=True, exist_ok=True)
        candidates_root = trial_dir / "proxy_candidates"
        candidates_root.mkdir(parents=True, exist_ok=True)
        proxy_tmp_root = trial_dir / "proxy_tmp"
        cleanup_errors: list[str] = []
        proxy_tmp_cleaned = True
        stale_proxy_tmp_cleaned = False
        artifact_cleanup_triggered = False
        effective_artifact_mode = config.effective_artifact_mode
        max_artifact_bytes = int(config.max_proxy_artifact_gb * 1024 * 1024 * 1024)
        if proxy_tmp_root.exists():
            stale_proxy_tmp_cleaned = _cleanup_path(proxy_tmp_root, cleanup_errors)
            proxy_tmp_cleaned = stale_proxy_tmp_cleaned
        proxy_records = _select_proxy_eval_records(records, config.proxy_eval_samples, rng)
        source_context = _collect_source_proxy_context(proxy_records)
        candidate_rows: list[dict[str, Any]] = []
        candidate_policies: dict[int, Policy] = {}
        candidate_seeds: dict[int, int] = {}
        proxy_evaluator = ProxyEvaluator(score_version=config.proxy_score_version)
        search_space = updater.current_search_space if updater is not None else self.search_space

        for candidate_index in range(config.candidate_policies):
            candidate_name = f"policy_trial_{trial_index:03d}_candidate_{candidate_index:03d}"
            policy = search_space.sample_policy(rng, name=candidate_name)
            candidate_policies[candidate_index] = policy
            candidate_dir = candidates_root / f"candidate_{candidate_index:03d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            tmp_candidate_dir = proxy_tmp_root / f"candidate_{candidate_index:03d}"
            images_dir = tmp_candidate_dir / "images"
            labels_dir = tmp_candidate_dir / "labels"
            policy.save(candidate_dir / "policy.json")

            candidate_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            candidate_seeds[candidate_index] = candidate_seed
            candidate_rng = np.random.default_rng(candidate_seed)
            try:
                images_dir.mkdir(parents=True, exist_ok=True)
                labels_dir.mkdir(parents=True, exist_ok=True)
                self._write_trial_dataset(tmp_candidate_dir, proxy_records, policy, candidate_rng, images_dir, labels_dir)
                if _directory_size(tmp_candidate_dir) > max_artifact_bytes:
                    artifact_cleanup_triggered = True
                    effective_artifact_mode = "metrics_only"
                context = {
                    **self.context,
                    **source_context,
                    "dataset_root": str(Path(self.dataset_root).resolve()),
                    "output_dir": str(self.output_dir.resolve()),
                    "trial_index": trial_index,
                    "candidate_index": candidate_index,
                    "images_dir": str(images_dir.resolve()),
                    "labels_dir": str(labels_dir.resolve()),
                    "proxy_score_version": config.proxy_score_version,
                }
                evaluation = proxy_evaluator.evaluate(tmp_candidate_dir, policy, context=context)
                metrics = dict(evaluation.metrics)
                proxy_score, components, missing_components = compute_proxy_score(metrics, version=config.proxy_score_version)
                hard_filter_pass, hard_filter_reasons = apply_proxy_hard_filter(
                    metrics,
                    profile=config.hard_filter_profile,
                )
                candidate_dataset_retained = False
                if effective_artifact_mode == "all" and not artifact_cleanup_triggered:
                    _move_candidate_artifacts(tmp_candidate_dir, candidate_dir, cleanup_errors)
                    candidate_dataset_retained = _candidate_has_image_artifacts(candidate_dir)
                    if _proxy_artifact_bytes(candidates_root, proxy_tmp_root) > max_artifact_bytes:
                        artifact_cleanup_triggered = True
                        effective_artifact_mode = "metrics_only"
                        _remove_candidate_image_artifacts(candidates_root, cleanup_errors)
                        candidate_dataset_retained = False
                metrics.update(
                    {
                        "candidate_index": candidate_index,
                        "candidate_seed": candidate_seed,
                        "policy_name": policy.name,
                        "operation_names": [operation.name for operation in policy.operations],
                        "proxy_score": proxy_score,
                        "proxy_score_version": config.proxy_score_version,
                        "proxy_score_components": components,
                        "missing_components": missing_components,
                        "hard_filter_profile": config.hard_filter_profile,
                        "hard_filter_pass": hard_filter_pass,
                        "hard_filter_reasons": hard_filter_reasons,
                        "candidate_dataset_retained": candidate_dataset_retained,
                    }
                )
                (candidate_dir / "proxy_metrics.json").write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                candidate_rows.append(_candidate_summary_row(metrics))
            finally:
                if tmp_candidate_dir.exists():
                    proxy_tmp_cleaned = _cleanup_path(tmp_candidate_dir, cleanup_errors)

        selection = select_proxy_candidate(candidate_rows)
        selected_policy = candidate_policies[selection.selected_index]
        for row in candidate_rows:
            row["selected"] = int(row["candidate_index"]) == selection.selected_index
        ranked = _rank_proxy_candidates(candidate_rows)
        top_candidates_summary = [_compact_candidate_summary(row) for row in ranked[: config.proxy_top_k]]
        selected_row = next(row for row in candidate_rows if int(row["candidate_index"]) == selection.selected_index)
        if (
            effective_artifact_mode == "selected_only"
            and config.keep_candidate_artifacts
            and not artifact_cleanup_triggered
        ):
            selected_candidate_dir = candidates_root / f"candidate_{selection.selected_index:03d}"
            selected_images_dir = selected_candidate_dir / "images"
            selected_labels_dir = selected_candidate_dir / "labels"
            selected_rng = np.random.default_rng(candidate_seeds[selection.selected_index])
            self._write_trial_dataset(
                selected_candidate_dir,
                proxy_records,
                selected_policy,
                selected_rng,
                selected_images_dir,
                selected_labels_dir,
            )
            if _proxy_artifact_bytes(candidates_root, proxy_tmp_root) > max_artifact_bytes:
                artifact_cleanup_triggered = True
                effective_artifact_mode = "metrics_only"
                _remove_candidate_image_artifacts(candidates_root, cleanup_errors)
        if proxy_tmp_root.exists():
            proxy_tmp_cleaned = _cleanup_path(proxy_tmp_root, cleanup_errors)
        candidate_image_artifacts_kept = _candidate_image_artifacts_exist(candidates_root)
        candidate_artifact_bytes = _proxy_artifact_bytes(candidates_root, proxy_tmp_root)
        if not candidate_image_artifacts_kept and effective_artifact_mode != "metrics_only":
            effective_artifact_mode = "metrics_only"
        selected_payload = {
            "selected_candidate_index": selection.selected_index,
            "selected_candidate_proxy_score": selected_row["proxy_score"],
            "hard_filter_pass": selected_row["hard_filter_pass"],
            "fallback_used": selection.fallback_used,
            "fallback_reason": selection.fallback_reason,
            "selected_policy": selected_policy.to_dict(),
            "top_candidates_summary": top_candidates_summary,
            "hard_filter_profile": config.hard_filter_profile,
            "proxy_score_version": config.proxy_score_version,
            "artifact_mode": effective_artifact_mode,
            "requested_artifact_mode": config.artifact_mode,
            "keep_candidate_artifacts": config.keep_candidate_artifacts,
            "proxy_tmp_cleaned": proxy_tmp_cleaned,
            "stale_proxy_tmp_cleaned": stale_proxy_tmp_cleaned,
            "candidate_image_artifacts_kept": candidate_image_artifacts_kept,
            "candidate_artifact_bytes": candidate_artifact_bytes,
            "proxy_artifact_cleanup_triggered": artifact_cleanup_triggered,
            "cleanup_errors": cleanup_errors,
        }
        (trial_dir / "selected_policy_by_proxy.json").write_text(
            json.dumps(selected_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_proxy_candidates_summary(trial_dir / "proxy_candidates_summary.csv", candidate_rows)
        return selected_policy, {
            **config.to_dict(),
            "selected_candidate_index": selection.selected_index,
            "selected_candidate_proxy_score": selected_row["proxy_score"],
            "hard_filter_pass": selected_row["hard_filter_pass"],
            "fallback_used": selection.fallback_used,
            "fallback_reason": selection.fallback_reason,
            "top_candidates_summary": top_candidates_summary,
            "artifact_mode": effective_artifact_mode,
            "requested_artifact_mode": config.artifact_mode,
            "keep_candidate_artifacts": config.keep_candidate_artifacts,
            "proxy_tmp_cleaned": proxy_tmp_cleaned,
            "stale_proxy_tmp_cleaned": stale_proxy_tmp_cleaned,
            "candidate_image_artifacts_kept": candidate_image_artifacts_kept,
            "candidate_artifact_bytes": candidate_artifact_bytes,
            "proxy_artifact_cleanup_triggered": artifact_cleanup_triggered,
            "cleanup_errors": cleanup_errors,
        }

    def _write_trial_dataset(
        self,
        trial_dir: Path,
        records: list,
        policy: Policy,
        rng: np.random.Generator,
        image_dir: Path,
        label_dir: Path,
    ) -> int:
        source_label_count = 0
        for record in records:
            sample = load_yolo_sample(record)
            source_label_count += len(sample["labels"])
            augmented = apply_policy_to_sample(sample, policy, rng=rng)
            stem = flatten_relative_stem(record.relative_path)
            image_name = f"{stem}{record.image_path.suffix.lower()}"
            label_name = f"{stem}.txt"
            save_yolo_sample(augmented, image_dir / image_name, label_dir / label_name)
        return source_label_count

    def _trial_image_label_dirs(self, trial_dir: Path) -> tuple[Path, Path]:
        if self.trial_layout == "train_yolo":
            return trial_dir / "dataset" / "images" / "train", trial_dir / "dataset" / "labels" / "train"
        return trial_dir / "images", trial_dir / "labels"

    def _write_results(self, results: list[TrialResult]) -> None:
        json_path = self.output_dir / "trials.json"
        json_path.write_text(
            json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        csv_path = self.output_dir / "trials.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "trial_index",
                    "score",
                    "score_before",
                    "score_after",
                    "accepted",
                    "rejected",
                    "trial_dir",
                    "policy_name",
                    "adjust_reason",
                    "main_issues_json",
                    "yolo_log_paths_json",
                    "metrics_json",
                    "policy_json",
                    "before_policy_json",
                    "after_policy_json",
                    "policy_diff_json",
                    "diagnosis_json",
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "trial_index": result.trial_index,
                        "score": f"{result.score:.8f}",
                        "score_before": "" if result.score_before is None else f"{result.score_before:.8f}",
                        "score_after": "" if result.score_after is None else f"{result.score_after:.8f}",
                        "accepted": result.accepted,
                        "rejected": result.rejected,
                        "trial_dir": str(result.trial_dir),
                        "policy_name": result.policy.name,
                        "adjust_reason": result.adjust_reason,
                        "main_issues_json": json.dumps((result.diagnosis or {}).get("main_issues", []), ensure_ascii=False, sort_keys=True),
                        "yolo_log_paths_json": json.dumps(result.yolo_log_paths or {}, ensure_ascii=False, sort_keys=True),
                        "metrics_json": json.dumps(result.metrics, ensure_ascii=False, sort_keys=True),
                        "policy_json": json.dumps(result.policy.to_dict(), ensure_ascii=False, sort_keys=True),
                        "before_policy_json": json.dumps((result.before_policy or result.policy).to_dict(), ensure_ascii=False, sort_keys=True),
                        "after_policy_json": json.dumps(result.after_policy.to_dict() if result.after_policy is not None else None, ensure_ascii=False, sort_keys=True),
                        "policy_diff_json": json.dumps(result.policy_diff or {}, ensure_ascii=False, sort_keys=True),
                        "diagnosis_json": json.dumps(result.diagnosis or {}, ensure_ascii=False, sort_keys=True),
                    }
                )

    def _write_policy_history(self, results: list[TrialResult]) -> None:
        jsonl_path = self.output_dir / "policy_history.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) for result in results) + ("\n" if results else ""),
            encoding="utf-8",
        )
        csv_path = self.output_dir / "policy_history.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "trial_index",
                    "score_before",
                    "score",
                    "score_after",
                    "accepted",
                    "adjust_reason",
                    "before_policy",
                    "after_policy",
                    "main_issues",
                    "diff_summary",
                    "policy_diff",
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "trial_index": result.trial_index,
                        "score_before": result.score_before,
                        "score": result.score,
                        "score_after": result.score_after,
                        "accepted": result.accepted,
                        "adjust_reason": result.adjust_reason,
                        "before_policy": json.dumps((result.before_policy or result.policy).to_dict(), ensure_ascii=False, sort_keys=True),
                        "after_policy": json.dumps(result.after_policy.to_dict() if result.after_policy is not None else None, ensure_ascii=False, sort_keys=True),
                        "main_issues": json.dumps((result.diagnosis or {}).get("main_issues", []), ensure_ascii=False, sort_keys=True),
                        "diff_summary": _summarize_policy_diff(result.policy_diff or {}),
                        "policy_diff": json.dumps(result.policy_diff or {}, ensure_ascii=False, sort_keys=True),
                    }
                )
            md_lines = [
                "# Policy History",
                "",
                "| trial | score_before | score | score_after | accepted | adjust_reason | diff_summary |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for result in results:
                md_lines.append(
                    "| {trial} | {before} | {score:.6f} | {after} | {accepted} | {reason} | {diff} |".format(
                        trial=result.trial_index,
                        before="" if result.score_before is None else f"{result.score_before:.6f}",
                        score=result.score,
                        after="" if result.score_after is None else f"{result.score_after:.6f}",
                        accepted=result.accepted,
                        reason=str(result.adjust_reason or "").replace("|", "/"),
                        diff=_summarize_policy_diff(result.policy_diff or {}).replace("|", "/"),
                    )
                )
        (self.output_dir / "policy_history.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    def _cleanup_intermediate(self, results: list[TrialResult], best: TrialResult) -> None:
        for result in results:
            if result.trial_dir == best.trial_dir:
                continue
            if result.trial_dir.exists():
                shutil.rmtree(result.trial_dir)


def _cleanup_path(path: Path, cleanup_errors: list[str]) -> bool:
    def _retry_remove(func: Any, path_text: str, exc_info: Any) -> None:
        try:
            Path(path_text).chmod(0o700)
            func(path_text)
        except Exception as exc:  # pragma: no cover - platform-specific filesystem failures
            cleanup_errors.append(f"{path_text}: {exc}")

    try:
        shutil.rmtree(path, onerror=_retry_remove)
        if path.exists():
            cleanup_errors.append(f"{path}: cleanup incomplete")
        return not path.exists()
    except FileNotFoundError:
        return True
    except Exception as exc:  # pragma: no cover - platform-specific filesystem failures
        cleanup_errors.append(f"{path}: {exc}")
        return False


def _move_candidate_artifacts(tmp_candidate_dir: Path, candidate_dir: Path, cleanup_errors: list[str]) -> None:
    for name in ("images", "labels"):
        source = tmp_candidate_dir / name
        destination = candidate_dir / name
        if not source.exists():
            continue
        if destination.exists():
            _cleanup_path(destination, cleanup_errors)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(source), str(destination))
        except Exception as exc:  # pragma: no cover - platform-specific filesystem failures
            cleanup_errors.append(f"{source} -> {destination}: {exc}")


def _remove_candidate_image_artifacts(candidates_root: Path, cleanup_errors: list[str]) -> None:
    if not candidates_root.exists():
        return
    for candidate_dir in candidates_root.glob("candidate_*"):
        if not candidate_dir.is_dir():
            continue
        for name in ("images", "labels"):
            path = candidate_dir / name
            if path.exists():
                _cleanup_path(path, cleanup_errors)


def _candidate_has_image_artifacts(candidate_dir: Path) -> bool:
    return (candidate_dir / "images").exists() or (candidate_dir / "labels").exists()


def _candidate_image_artifacts_exist(candidates_root: Path) -> bool:
    if not candidates_root.exists():
        return False
    return any(_candidate_has_image_artifacts(path) for path in candidates_root.glob("candidate_*") if path.is_dir())


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total


def _proxy_artifact_bytes(candidates_root: Path, proxy_tmp_root: Path) -> int:
    return _directory_size(candidates_root) + _directory_size(proxy_tmp_root)


def _select_proxy_eval_records(records: list, num_samples: int, rng: np.random.Generator) -> list:
    if num_samples <= 0 or num_samples >= len(records):
        return list(records)
    metas = [_proxy_record_meta(record) for record in records]
    class_counts: dict[int, int] = {}
    for meta in metas:
        for class_id, count in meta["class_counts"].items():
            class_counts[class_id] = class_counts.get(class_id, 0) + count

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    for class_id in sorted(class_counts, key=lambda item: (class_counts[item], item)):
        candidates = [index for index, meta in enumerate(metas) if class_id in meta["class_ids"] and index not in selected_set]
        if not candidates:
            continue
        chosen = int(rng.choice(candidates))
        selected_indices.append(chosen)
        selected_set.add(chosen)
        if len(selected_indices) >= num_samples:
            return [records[index] for index in selected_indices]

    rare_cutoff = DEFAULT_RARE_CLASS_COUNT_THRESHOLD
    rare_classes = {class_id for class_id, count in class_counts.items() if count <= rare_cutoff}
    remaining = [index for index in range(len(records)) if index not in selected_set]
    random_tiebreakers = {index: float(rng.random()) for index in remaining}
    remaining.sort(
        key=lambda index: (
            sum(metas[index]["class_counts"].get(class_id, 0) for class_id in rare_classes),
            metas[index]["tiny_count"],
            metas[index]["small_count"],
            metas[index]["edge_count"],
            random_tiebreakers[index],
        ),
        reverse=True,
    )
    for index in remaining:
        selected_indices.append(index)
        if len(selected_indices) >= num_samples:
            break
    return [records[index] for index in selected_indices]


def _proxy_record_meta(record: Any) -> dict[str, Any]:
    labels, boxes = _read_yolo_label_file(record.label_path)
    class_counts: dict[int, int] = {}
    for label in labels:
        class_id = int(label)
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    areas = boxes[:, 2] * boxes[:, 3] if boxes.size else np.zeros((0,), dtype=np.float32)
    return {
        "class_ids": set(class_counts),
        "class_counts": class_counts,
        "tiny_count": int((areas <= DEFAULT_TINY_AREA_THRESHOLD).sum()) if boxes.size else 0,
        "small_count": int((areas <= DEFAULT_SMALL_AREA_THRESHOLD).sum()) if boxes.size else 0,
        "edge_count": int(yolo_bbox_edge_mask(boxes, edge_margin=DEFAULT_EDGE_MARGIN).sum()) if boxes.size else 0,
    }


def _collect_source_proxy_context(records: list) -> dict[str, Any]:
    source_label_count = 0
    tiny_count = 0
    small_count = 0
    edge_count = 0
    class_counts: dict[int, int] = {}
    image_means_by_output_name: dict[str, float] = {}
    for record in records:
        sample = load_yolo_sample(record)
        image = sample["image"]
        height, width = image.shape[:2]
        image_name = f"{flatten_relative_stem(record.relative_path)}{record.image_path.suffix.lower()}"
        image_means_by_output_name[image_name] = float(image.mean())
        labels = np.asarray(sample["labels"], dtype=np.int64)
        bboxes = np.asarray(sample["bboxes"], dtype=np.float32).reshape(-1, 4)
        yolo_boxes = xyxy_to_yolo(bboxes, width, height)
        source_label_count += len(labels)
        if yolo_boxes.size:
            areas = yolo_boxes[:, 2] * yolo_boxes[:, 3]
            tiny_count += int((areas <= DEFAULT_TINY_AREA_THRESHOLD).sum())
            small_count += int((areas <= DEFAULT_SMALL_AREA_THRESHOLD).sum())
            edge_count += int(yolo_bbox_edge_mask(yolo_boxes, edge_margin=DEFAULT_EDGE_MARGIN).sum())
        for label in labels:
            class_id = int(label)
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    return {
        "source_label_count": source_label_count,
        "source_tiny_box_count": tiny_count,
        "source_small_box_count": small_count,
        "source_edge_box_count": edge_count,
        "source_class_counts": class_counts,
        "source_image_means_by_output_name": image_means_by_output_name,
        "tiny_area_threshold": DEFAULT_TINY_AREA_THRESHOLD,
        "small_area_threshold": DEFAULT_SMALL_AREA_THRESHOLD,
        "edge_margin": DEFAULT_EDGE_MARGIN,
        "rare_class_count_threshold": DEFAULT_RARE_CLASS_COUNT_THRESHOLD,
    }


def _read_yolo_label_file(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def _candidate_summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_index": int(metrics["candidate_index"]),
        "policy_name": metrics["policy_name"],
        "operation_names": list(metrics.get("operation_names", [])),
        "proxy_score": float(metrics["proxy_score"]),
        "hard_filter_pass": bool(metrics["hard_filter_pass"]),
        "hard_filter_reasons": list(metrics.get("hard_filter_reasons", [])),
        "bbox_safe_rate": _float_or_none(metrics.get("bbox_safe_rate")),
        "bbox_valid_rate": _float_or_none(metrics.get("bbox_valid_rate")),
        "bbox_retention_raw": _float_or_none(metrics.get("bbox_retention_raw")),
        "small_target_retention": _float_or_none(metrics.get("small_target_retention")),
        "tiny_target_retention": _float_or_none(metrics.get("tiny_target_retention")),
        "edge_target_retention": _float_or_none(metrics.get("edge_target_retention")),
        "class_coverage_after": _float_or_none(metrics.get("class_coverage_after")),
        "rare_class_retention": _float_or_none(metrics.get("rare_class_retention")),
        "exposure_diversity_score": _float_or_none(metrics.get("exposure_diversity_score")),
        "strength_penalty": _float_or_none(metrics.get("strength_penalty")),
        "selected": False,
    }


def _rank_proxy_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            bool(row.get("hard_filter_pass")),
            float(row.get("proxy_score") or 0.0),
            -len(row.get("hard_filter_reasons") or []),
        ),
        reverse=True,
    )


def _compact_candidate_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_index": row["candidate_index"],
        "policy_name": row["policy_name"],
        "operation_names": row["operation_names"],
        "proxy_score": row["proxy_score"],
        "hard_filter_pass": row["hard_filter_pass"],
        "hard_filter_reasons": row["hard_filter_reasons"],
        "selected": row["selected"],
    }


def _write_proxy_candidates_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_index",
        "policy_name",
        "operation_names",
        "proxy_score",
        "hard_filter_pass",
        "hard_filter_reasons",
        "bbox_safe_rate",
        "bbox_valid_rate",
        "bbox_retention_raw",
        "small_target_retention",
        "tiny_target_retention",
        "edge_target_retention",
        "class_coverage_after",
        "rare_class_retention",
        "exposure_diversity_score",
        "strength_penalty",
        "selected",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["operation_names"] = json.dumps(out["operation_names"], ensure_ascii=False)
            out["hard_filter_reasons"] = json.dumps(out["hard_filter_reasons"], ensure_ascii=False)
            writer.writerow(out)


def _extract_log_paths(metrics: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key, value in metrics.items():
        if not isinstance(value, str):
            continue
        normalized = key.lower()
        if normalized.endswith("_log") or normalized.endswith("_logs") or "stdout_log" in normalized or "stderr_log" in normalized:
            paths[key] = value
    return paths


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_policy_diff(policy_diff: dict[str, Any]) -> str:
    if not policy_diff:
        return "not_recorded"
    parts: list[str] = []
    weight_changes = policy_diff.get("operation_weight_changes") or policy_diff.get("search_space_weight_delta") or {}
    if weight_changes:
        items = []
        for name, change in list(weight_changes.items())[:4]:
            items.append(f"{name}:{float(change['before']):.3g}->{float(change['after']):.3g}")
        parts.append("weights " + ", ".join(items))
    prob_changes = policy_diff.get("prob_range_changes") or {}
    if prob_changes:
        items = []
        for name, change in list(prob_changes.items())[:3]:
            before = change["before"]
            after = change["after"]
            items.append(f"{name}:[{before[0]:.2g},{before[1]:.2g}]->[{after[0]:.2g},{after[1]:.2g}]")
        parts.append("prob " + ", ".join(items))
    strength_changes = policy_diff.get("strength_range_changes") or {}
    if strength_changes:
        items = []
        for name, change in list(strength_changes.items())[:3]:
            before = change["before"]
            after = change["after"]
            items.append(f"{name}:[{before[0]:.2g},{before[1]:.2g}]->[{after[0]:.2g},{after[1]:.2g}]")
        parts.append("strength " + ", ".join(items))
    if not parts:
        return "unchanged"
    return "; ".join(parts)
