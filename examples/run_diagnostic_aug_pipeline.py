from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyze_yolo_val_errors import main as analyze_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the diagnostic augmentation advisor step and print the recommended train_yolo command."
    )
    parser.add_argument("--dataset", required=True, help="YOLO dataset root.")
    parser.add_argument("--weights", required=True, help="Baseline or existing best.pt used for val prediction.")
    parser.add_argument("--output", default="outputs/diagnostics/baseline_error_analysis")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--search-output", default="outputs/advisor_policy_search")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--search-epochs", type=int, default=5)
    parser.add_argument("--final-epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_argv = sys.argv[:]
    sys.argv = [
        "tools/analyze_yolo_val_errors.py",
        "--dataset",
        args.dataset,
        "--weights",
        args.weights,
        "--output",
        args.output,
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--workers",
        str(args.workers),
    ]
    try:
        analyze_main()
    finally:
        sys.argv = original_argv

    advisor_json = Path(args.output).resolve() / "advisor_search_space.json"
    print("")
    print("Recommended diagnostic-driven train_yolo command:")
    print("python examples\\run_policy_search.py ^")
    print(f"  --dataset {Path(args.dataset).resolve()} ^")
    print(f"  --output {args.search_output} ^")
    print(f"  --trials {args.trials} ^")
    print(f"  --search-samples {args.samples} ^")
    print("  --seed 42 ^")
    print("  --evaluator train_yolo ^")
    print("  --metric map50_95 ^")
    print("  --model yolov8n.pt ^")
    print(f"  --epochs {args.epochs} ^")
    print(f"  --search-epochs {args.search_epochs} ^")
    print(f"  --final-epochs {args.final_epochs} ^")
    print(f"  --imgsz {args.imgsz} ^")
    print(f"  --batch {args.batch} ^")
    print(f"  --workers {args.workers} ^")
    print("  --hybrid-proxy-weight 0 ^")
    print(f"  --search-space-json {advisor_json}")
    print("")
    print("This script does not start the diagnostic-driven policy search automatically.")
    print("Final full-dataset training is not implemented in run_policy_search.py; do not pass --final-full-dataset until it is implemented.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
