from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.formats.yolo import load_yolo_labels
from AutoAugment.utils import resolve_path
from AutoAugment.visualize import draw_bboxes


IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO dataset labels by drawing bboxes on images.")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root with images/train,val and labels/train,val.")
    parser.add_argument("--output", required=True, help="Visualization output directory.")
    parser.add_argument("--max-images", type=int, default=20, help="Maximum images per split.")
    return parser.parse_args()


def visualize_dataset(dataset_root: str | Path, output_root: str | Path, max_images: int = 20) -> dict[str, int]:
    dataset = resolve_path(dataset_root)
    output = resolve_path(output_root)
    if max_images <= 0:
        raise ValueError("--max-images must be positive")
    counts: dict[str, int] = {}
    for split in ["train", "val"]:
        counts[split] = visualize_split(
            images_dir=dataset / "images" / split,
            labels_dir=dataset / "labels" / split,
            output_dir=output / split,
            max_images=max_images,
        )
    return counts


def visualize_split(images_dir: Path, labels_dir: Path, output_dir: Path, max_images: int = 20) -> int:
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory does not exist: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels directory does not exist: {labels_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    written = 0
    for image_path in image_paths[:max_images]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        relative = image_path.relative_to(images_dir)
        label_path = labels_dir / relative.with_suffix(".txt")
        labels, bboxes = load_yolo_labels(label_path, width, height)
        visualized = draw_bboxes({"image": image, "labels": labels, "bboxes": bboxes})
        output_path = output_dir / f"{_flatten_stem(relative)}_vis.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), visualized):
            raise IOError(f"failed to write visualization: {output_path}")
        written += 1
    return written


def _flatten_stem(path: Path) -> str:
    return "__".join(path.with_suffix("").parts)


def main() -> None:
    args = parse_args()
    counts = visualize_dataset(args.dataset, args.output, max_images=args.max_images)
    print("YOLO visualization completed:")
    print(f"- Dataset: {resolve_path(args.dataset)}")
    print(f"- Output: {resolve_path(args.output)}")
    print(f"- Train visualizations: {counts.get('train', 0)}")
    print(f"- Val visualizations: {counts.get('val', 0)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
