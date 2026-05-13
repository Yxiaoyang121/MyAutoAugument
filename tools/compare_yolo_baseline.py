from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.policies import Policy, apply_policy_to_sample
from AutoAugment.search import find_yolo_best_pt, parse_yolo_metrics, write_train_val_data_yaml
from AutoAugment.utils import (
    copy_yolo_records,
    find_yolo_records_from_dirs,
    flatten_relative_stem,
    load_yolo_sample,
    save_yolo_sample,
)
from tools.visualize_yolo_dataset import visualize_split


IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO no-augmentation baseline with a saved best policy.")
    parser.add_argument("--dataset", required=True, help="Tiled YOLO dataset root with images/train,val and labels/train,val.")
    parser.add_argument("--policy", required=True, help="Best policy JSON to apply to the training split.")
    parser.add_argument("--output", required=True, help="Comparison output directory.")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=20, help="Maximum sample visualizations to write.")
    parser.add_argument("--reuse-baseline-dir", default=None, help="Existing comparison directory with baseline_no_aug metrics and artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    policy_path = Path(args.policy).resolve()
    output = Path(args.output).resolve()

    if not dataset.exists():
        raise FileNotFoundError(f"dataset does not exist: {dataset}")
    if not policy_path.exists():
        raise FileNotFoundError(f"policy does not exist: {policy_path}")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")

    output.mkdir(parents=True, exist_ok=True)
    train_records = find_yolo_records_from_dirs(
        dataset / "images" / "train",
        dataset / "labels" / "train",
        missing_label="error",
    )
    val_records = find_yolo_records_from_dirs(
        dataset / "images" / "val",
        dataset / "labels" / "val",
        missing_label="error",
    )
    val_images = output / "val_fixed" / "images" / "val"
    val_labels = output / "val_fixed" / "labels" / "val"
    copy_yolo_records(val_records, val_images, val_labels)

    policy = Policy.load(policy_path)
    config = {
        "dataset": str(dataset),
        "policy": str(policy_path),
        "output": str(output),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "fixed_val_images": str(val_images.resolve()),
        "fixed_val_labels": str(val_labels.resolve()),
        "reuse_baseline_dir": str(Path(args.reuse_baseline_dir).resolve()) if args.reuse_baseline_dir else None,
    }
    write_json(output / "run_config.json", config)

    results = []
    if args.reuse_baseline_dir:
        results.append(load_reused_baseline(Path(args.reuse_baseline_dir).resolve(), output))
        groups = [("advisor_best_policy_aug", policy)]
    else:
        groups = [
            ("baseline_no_aug", None),
            ("best_policy_aug", policy),
        ]
    for name, group_policy in groups:
        group_dir = output / name
        group_dir.mkdir(parents=True, exist_ok=True)
        if group_policy is None:
            prepare_no_aug_dataset(group_dir, train_records)
        else:
            prepare_augmented_dataset(group_dir, train_records, group_policy, seed=args.seed)
            group_policy.save(group_dir / "applied_policy.json")
            visualize_split(
                group_dir / "dataset" / "images" / "train",
                group_dir / "dataset" / "labels" / "train",
                group_dir / "best_policy_augmented_samples",
                max_images=args.max_images,
            )
        metrics = train_validate_group(
            group_name=name,
            group_dir=group_dir,
            val_images=val_images,
            val_labels=val_labels,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
        )
        metrics["group"] = name
        metrics["train_image_count"] = count_images(group_dir / "dataset" / "images" / "train")
        metrics["val_image_count"] = count_images(val_images)
        results.append(metrics)

    write_compare_outputs(output, results, policy)
    print("Baseline comparison completed:")
    for item in results:
        print(
            f"- {item['group']}: mAP50={item.get('map50')} "
            f"mAP50-95={item.get('map50_95')} precision={item.get('precision')} recall={item.get('recall')}"
        )
    best = max(results, key=lambda item: float(item.get("map50") or 0.0))
    print(f"- Better by mAP50: {best['group']}")
    print(f"- Output: {output}")


def prepare_no_aug_dataset(group_dir: Path, train_records: list[Any]) -> None:
    copy_yolo_records(
        train_records,
        group_dir / "dataset" / "images" / "train",
        group_dir / "dataset" / "labels" / "train",
    )


def load_reused_baseline(reuse_dir: Path, output_dir: Path) -> dict[str, Any]:
    metrics_path = reuse_dir / "compare_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"reused baseline compare_metrics.json does not exist: {metrics_path}")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    baseline = next((item for item in data.get("results", []) if item.get("group") == "baseline_no_aug"), None)
    if baseline is None:
        raise ValueError(f"could not find baseline_no_aug in {metrics_path}")
    target_dir = output_dir / "baseline_no_aug"
    target_dir.mkdir(parents=True, exist_ok=True)
    source_best = Path(baseline.get("best_pt", ""))
    if source_best.exists():
        weights_dir = target_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_best, weights_dir / "best.pt")
    for relative in ["data.yaml", "metrics.json", "val_stdout.log", "val_stderr.log", "predict_stdout.log", "predict_stderr.log"]:
        source = reuse_dir / "baseline_no_aug" / relative
        if source.exists():
            destination = target_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    source_predict = reuse_dir / "baseline_no_aug" / "predict_runs" / "val"
    if source_predict.exists():
        shutil.copytree(source_predict, target_dir / "predict_runs" / "val", dirs_exist_ok=True)
    reused = dict(baseline)
    reused["group"] = "baseline_no_aug"
    reused["reused_from"] = str(reuse_dir)
    if (target_dir / "weights" / "best.pt").exists():
        reused["best_pt"] = str((target_dir / "weights" / "best.pt").resolve())
    if (target_dir / "predict_runs" / "val").exists():
        reused["predict_output_dir"] = str((target_dir / "predict_runs" / "val").resolve())
    write_json(target_dir / "metrics.json", reused)
    return reused


def prepare_augmented_dataset(group_dir: Path, train_records: list[Any], policy: Policy, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    images_dir = group_dir / "dataset" / "images" / "train"
    labels_dir = group_dir / "dataset" / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for record in train_records:
        sample = load_yolo_sample(record)
        augmented = apply_policy_to_sample(sample, policy, rng=rng)
        stem = flatten_relative_stem(record.relative_path)
        image_path = images_dir / f"{stem}{record.image_path.suffix.lower()}"
        label_path = labels_dir / f"{stem}.txt"
        save_yolo_sample(augmented, image_path, label_path)


def train_validate_group(
    *,
    group_name: str,
    group_dir: Path,
    val_images: Path,
    val_labels: Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    dataset_dir = group_dir / "dataset"
    data_yaml = write_train_val_data_yaml(
        group_dir / "data.yaml",
        dataset_dir,
        train_images_dir=dataset_dir / "images" / "train",
        train_labels_dir=dataset_dir / "labels" / "train",
        val_images_dir=val_images,
        val_labels_dir=val_labels,
    )
    write_text(
        group_dir / "val_reference.txt",
        "\n".join(
            [
                f"val_images_dir={val_images.resolve()}",
                f"val_labels_dir={val_labels.resolve()}",
                "validation_set_is_fixed=true",
                "validation_set_augmented=false",
                "",
            ]
        ),
    )

    train_command = (
        f"yolo detect train model={quote_path(model)} data={quote_path(data_yaml)} "
        f"epochs={epochs} imgsz={imgsz} batch={batch} seed={seed} workers={workers} "
        f"project={quote_path(group_dir / 'train_runs')} name=train exist_ok=True"
    )
    train_completed = run_command(train_command)
    write_text(group_dir / "train_stdout.log", train_completed.stdout)
    write_text(group_dir / "train_stderr.log", train_completed.stderr)
    if train_completed.returncode != 0:
        raise RuntimeError(f"{group_name} training failed, see {group_dir / 'train_stderr.log'}")

    discovered_best = find_yolo_best_pt(group_dir)
    weights_dir = group_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_pt = weights_dir / "best.pt"
    if discovered_best.resolve() != best_pt.resolve():
        shutil.copyfile(discovered_best, best_pt)

    val_command = (
        f"yolo detect val model={quote_path(best_pt)} data={quote_path(data_yaml)} "
        f"imgsz={imgsz} batch={batch} seed={seed} workers={workers} "
        f"project={quote_path(group_dir / 'val_runs')} name=val exist_ok=True"
    )
    val_completed = run_command(val_command)
    write_text(group_dir / "val_stdout.log", val_completed.stdout)
    write_text(group_dir / "val_stderr.log", val_completed.stderr)
    if val_completed.returncode != 0:
        raise RuntimeError(f"{group_name} validation failed, see {group_dir / 'val_stderr.log'}")

    parsed = parse_yolo_metrics(f"{val_completed.stdout}\n{val_completed.stderr}")
    pr = parse_precision_recall(f"{val_completed.stdout}\n{val_completed.stderr}")
    metrics = {
        "map50": parsed.get("map50"),
        "map50_95": parsed.get("map50_95"),
        "precision": pr.get("precision"),
        "recall": pr.get("recall"),
        "best_pt": str(best_pt.resolve()),
        "discovered_best_pt": str(discovered_best.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "train_command": train_command,
        "val_command": val_command,
        "workers": workers,
    }
    predict_result = run_val_prediction(group_dir, best_pt, val_images, imgsz=imgsz, workers=workers, seed=seed)
    metrics.update(predict_result)
    write_json(group_dir / "metrics.json", metrics)
    return metrics


def run_val_prediction(group_dir: Path, best_pt: Path, val_images: Path, *, imgsz: int, workers: int, seed: int) -> dict[str, Any]:
    command = (
        f"yolo detect predict model={quote_path(best_pt)} source={quote_path(val_images)} "
        f"imgsz={imgsz} seed={seed} workers={workers} project={quote_path(group_dir / 'predict_runs')} name=val exist_ok=True save=True"
    )
    completed = run_command(command)
    write_text(group_dir / "predict_stdout.log", completed.stdout)
    write_text(group_dir / "predict_stderr.log", completed.stderr)
    result = {
        "predict_command": command,
        "predict_output_dir": str((group_dir / "predict_runs" / "val").resolve()),
        "predict_returncode": completed.returncode,
        "workers": workers,
    }
    if completed.returncode != 0:
        result["predict_error"] = f"prediction failed, see {group_dir / 'predict_stderr.log'}"
    return result


def parse_precision_recall(text: str) -> dict[str, float]:
    lines = [strip_ansi(line).strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        tokens = line.split()
        if not tokens or tokens[0].lower() != "all":
            continue
        if not near_metric_header(lines, index):
            continue
        values = [float_or_none(token) for token in tokens[1:]]
        numeric_values = [value for value in values if value is not None]
        if len(numeric_values) >= 6:
            return {"precision": numeric_values[-4], "recall": numeric_values[-3]}
    return {}


def near_metric_header(lines: list[str], index: int) -> bool:
    start = max(0, index - 4)
    for candidate in lines[start:index]:
        normalized = "".join(char for char in candidate.lower() if char.isalnum())
        if "map50" in normalized and "map5095" in normalized:
            return True
    return False


def strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def write_compare_outputs(output: Path, results: list[dict[str, Any]], policy: Policy) -> None:
    fieldnames = [
        "group",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "train_image_count",
        "val_image_count",
        "best_pt",
        "data_yaml",
        "predict_output_dir",
    ]
    with (output / "compare_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})
    write_json(output / "compare_metrics.json", {"results": results})
    write_compare_plot(output / "compare_map.png", results)
    write_readme(output / "README.txt", results, policy)


def write_compare_plot(path: Path, results: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [row["group"] for row in results]
    metrics = ["map50", "map50_95", "precision", "recall"]
    x = np.arange(len(labels))
    width = 0.18
    plt.figure(figsize=(9, 5))
    for offset, metric in enumerate(metrics):
        values = [0.0 if row.get(metric) is None else float(row[metric]) for row in results]
        plt.bar(x + (offset - 1.5) * width, values, width, label=metric)
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("metric")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_readme(path: Path, results: list[dict[str, Any]], policy: Policy) -> None:
    baseline = next(row for row in results if row["group"] == "baseline_no_aug")
    best_policy = next(row for row in results if row["group"] != "baseline_no_aug")
    map50_delta = float(best_policy["map50"]) - float(baseline["map50"])
    map50_95_delta = float(best_policy["map50_95"]) - float(baseline["map50_95"])
    lines = [
        "YOLO baseline comparison",
        "",
        "This run compares the tiled dataset without augmentation against the saved best policy.",
        "It does not run random search.",
        "",
        "Training config:",
        "- model=yolov8n.pt",
        "- epochs=30",
        "- imgsz=640",
        "- batch=4",
        "- seed=42",
        f"- workers={results[0].get('workers')}",
        "",
        "Best policy:",
        json.dumps(policy.to_dict(), ensure_ascii=False, indent=2),
        "",
        "Metrics:",
    ]
    for row in results:
        lines.append(
            f"- {row['group']}: mAP50={row.get('map50')} mAP50-95={row.get('map50_95')} "
            f"precision={row.get('precision')} recall={row.get('recall')}"
        )
    lines.extend(
        [
            "",
            f"mAP50 delta {best_policy['group']} - baseline_no_aug: {map50_delta:.6f}",
            f"mAP50-95 delta {best_policy['group']} - baseline_no_aug: {map50_95_delta:.6f}",
            f"Recommendation: {'use best_policy for formal training' if map50_delta > 0 else 'do not prefer best_policy based on this comparison'}",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def count_images(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def quote_path(value: str | Path) -> str:
    text = str(value)
    if isinstance(value, Path):
        text = value.resolve().as_posix()
    if any(char.isspace() for char in text):
        return f'"{text}"'
    return text


def run_command(command: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        command,
        shell=True,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
