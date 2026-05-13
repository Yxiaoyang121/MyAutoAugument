from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.visualize_yolo_dataset import visualize_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual reports from a completed policy search output.")
    parser.add_argument("--search-output", required=True, help="Policy search output directory.")
    parser.add_argument("--output", required=True, help="Visualization report output directory.")
    parser.add_argument("--max-images", type=int, default=20, help="Maximum best-trial sample images to visualize.")
    return parser.parse_args()


def visualize_policy_search_results(
    search_output: str | Path,
    output_root: str | Path,
    max_images: int = 20,
) -> dict[str, Any]:
    search_dir = Path(search_output)
    output_dir = Path(output_root)
    if not search_dir.exists():
        raise FileNotFoundError(f"search output directory does not exist: {search_dir}")
    if max_images <= 0:
        raise ValueError("--max-images must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_trials(search_dir / "trials.csv")
    if not rows:
        raise ValueError(f"no trial rows found in {search_dir / 'trials.csv'}")
    best = max(rows, key=lambda row: row["score"])

    write_curve(output_dir / "score_curve.png", rows, "score", "Score")
    write_curve(output_dir / "map50_curve.png", rows, "map50", "mAP50")
    if any(row.get("proxy_score") is not None for row in rows):
        write_curve(output_dir / "proxy_score_curve.png", rows, "proxy_score", "Proxy Score")

    write_top_policies(output_dir, rows)
    write_best_policy_summary(search_dir / "best_policy.json", output_dir / "best_policy_summary.txt")
    write_best_trial_val_summary(Path(best["trial_dir"]) / "val_stdout.log", output_dir / "best_trial_val_summary.txt")

    best_trial_dir = Path(best["trial_dir"])
    best_sample_output = output_dir / "best_trial_samples"
    visualize_split(
        images_dir=best_trial_dir / "dataset" / "images" / "train",
        labels_dir=best_trial_dir / "dataset" / "labels" / "train",
        output_dir=best_sample_output,
        max_images=max_images,
    )
    return {
        "best_trial_index": best["trial_index"],
        "best_score": best["score"],
        "best_map50": best["map50"],
        "best_trial_dir": str(best_trial_dir),
        "output_dir": str(output_dir),
    }


def load_trials(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"trials.csv does not exist: {csv_path}")
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metrics = json.loads(row.get("metrics_json") or "{}")
            policy = json.loads(row.get("policy_json") or "{}")
            rows.append(
                {
                    "trial_index": int(row["trial_index"]),
                    "score": float(row["score"]),
                    "trial_dir": row["trial_dir"],
                    "policy_name": row.get("policy_name", ""),
                    "metrics": metrics,
                    "policy": policy,
                    "map50": _float_or_none(metrics.get("yolo_map50")),
                    "map50_95": _float_or_none(metrics.get("yolo_map50_95")),
                    "proxy_score": _float_or_none(metrics.get("proxy_score")),
                    "best_pt": metrics.get("best_pt", ""),
                    "data_yaml": metrics.get("dataset_yaml", ""),
                }
            )
    return rows


def write_curve(path: Path, rows: list[dict[str, Any]], key: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_index", key])
            for row in rows:
                writer.writerow([row["trial_index"], row.get(key)])
        raise RuntimeError(f"matplotlib is required to write {path}. Wrote fallback {csv_path}") from exc

    xs = [row["trial_index"] for row in rows]
    ys = [0.0 if row.get(key) is None else float(row[key]) for row in rows]
    plt.figure(figsize=(9, 4.8))
    plt.plot(xs, ys, marker="o", linewidth=1.6, markersize=3)
    plt.xlabel("trial_index")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def write_top_policies(output_dir: Path, rows: list[dict[str, Any]], limit: int = 10) -> None:
    top = sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]
    csv_path = output_dir / "top_10_policies.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["trial_index", "score", "map50", "proxy_score", "policy_name", "best_pt", "operations_json"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in top:
            writer.writerow(
                {
                    "trial_index": row["trial_index"],
                    "score": f"{row['score']:.8f}",
                    "map50": row["map50"],
                    "proxy_score": row["proxy_score"],
                    "policy_name": row["policy_name"],
                    "best_pt": row["best_pt"],
                    "operations_json": json.dumps(row["policy"].get("operations", []), ensure_ascii=False),
                }
            )
    lines = ["Top 10 policies", ""]
    for rank, row in enumerate(top, start=1):
        lines.append(
            f"{rank}. trial={row['trial_index']} score={row['score']:.8f} "
            f"map50={row['map50']} proxy_score={row['proxy_score']} policy={row['policy_name']}"
        )
    (output_dir / "top_10_policies.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_policy_summary(policy_path: Path, output_path: Path) -> None:
    if not policy_path.exists():
        raise FileNotFoundError(f"best_policy.json does not exist: {policy_path}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    lines = [f"Best policy: {policy.get('name', '')}", "", "Operations:"]
    for index, operation in enumerate(policy.get("operations", []), start=1):
        lines.append(
            f"{index}. name={operation.get('name')} prob={operation.get('prob')} "
            f"strength={operation.get('strength')} params={json.dumps(operation.get('params', {}), ensure_ascii=False)}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_trial_val_summary(val_log_path: Path, output_path: Path) -> None:
    if not val_log_path.exists():
        raise FileNotFoundError(f"val_stdout.log does not exist: {val_log_path}")
    lines = []
    for line in val_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if "mAP50" in stripped or stripped.startswith("all") or "Class" in stripped:
            lines.append(stripped)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    result = visualize_policy_search_results(args.search_output, args.output, max_images=args.max_images)
    print("Policy search visualization completed:")
    print(f"- Search output: {Path(args.search_output).resolve()}")
    print(f"- Visualization output: {Path(args.output).resolve()}")
    print(f"- Best trial: {result['best_trial_index']}")
    print(f"- Best score: {result['best_score']:.8f}")
    print(f"- Best mAP50: {result['best_map50']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
