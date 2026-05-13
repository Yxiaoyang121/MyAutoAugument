from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def make_run_name(run_name: str | None = None, now: datetime | None = None) -> str:
    if run_name:
        return run_name
    current = now or datetime.now()
    return current.strftime("run_%Y%m%d_%H%M%S")


def default_run_output_dir(kind: str, run_name: str | None = None, root: str | Path = "outputs/runs") -> Path:
    return Path(root) / kind / make_run_name(run_name)


def pytest_output_dir() -> Path:
    return Path("outputs/tests/pytest_tmp")


def smoke_output_dir() -> Path:
    return Path("outputs/tests/smoke")


def resolve_output_dir(output: str | Path | None, kind: str, run_name: str | None = None) -> Path:
    if output is not None:
        return Path(output)
    return default_run_output_dir(kind, run_name=run_name)


def write_json_file(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataset_path_file(path: str | Path, *, supplied_path: str | Path, resolved_path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"supplied_path={supplied_path}",
            f"resolved_absolute_path={Path(resolved_path).resolve()}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_policy_search_readme(
    output_dir: str | Path,
    *,
    dataset_path: str | Path,
    evaluator: str,
    metric: str | None = None,
    search_space_json: str | Path | None = None,
) -> None:
    output = Path(output_dir)
    lines = [
        "Policy search result",
        "",
        f"Input dataset: {Path(dataset_path).resolve()}",
        f"Search output: {output.resolve()}",
        f"Trial directory: {(output / 'trials').resolve()}",
        f"Best policy: {(output / 'best_policy.json').resolve()}",
        f"Trials CSV: {(output / 'trials.csv').resolve()}",
        f"Trials JSON: {(output / 'trials.json').resolve()}",
        f"Policy history JSONL: {(output / 'policy_history.jsonl').resolve()}",
        f"Policy history CSV: {(output / 'policy_history.csv').resolve()}",
        f"Policy history Markdown: {(output / 'policy_history.md').resolve()}",
        f"Evaluator: {evaluator}",
    ]
    if metric:
        lines.append(f"YOLO metric: {metric}")
    if search_space_json:
        lines.append(f"Diagnostic advisor search space: {Path(search_space_json).resolve()}")
    else:
        lines.append("Diagnostic advisor search space: not used")
    lines.extend(
        [
            "",
            "Each trial is stored under trials/trial_xxx with images/, labels/, policy.json, metrics.json,",
            "and, for evaluator=yolo, data.yaml plus yolo_stdout.log and yolo_stderr.log.",
            "For evaluator=train_yolo, each trial stores dataset/images/train, dataset/labels/train,",
            "data.yaml, train_stdout.log, train_stderr.log, val_stdout.log, val_stderr.log, and weights/best.pt.",
            "",
        ]
    )
    (output / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def write_apply_policy_readme(
    output_dir: str | Path,
    *,
    dataset_path: str | Path,
    policy_path: str | Path,
) -> None:
    output = Path(output_dir)
    lines = [
        "Augmented YOLO dataset",
        "",
        f"Source dataset: {Path(dataset_path).resolve()}",
        f"Policy: {Path(policy_path).resolve()}",
        f"Output dataset: {output.resolve()}",
        f"Images: {(output / 'images').resolve()}",
        f"Labels: {(output / 'labels').resolve()}",
        f"Applied policy copy: {(output / 'applied_policy.json').resolve()}",
        "",
    ]
    (output / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
