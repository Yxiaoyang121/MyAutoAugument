from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject

from gui.adapters import AugmentationAdapter


class PreviewController(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = AugmentationAdapter()

    def run_preview(
        self,
        image_path: str | Path,
        label_path: str | Path | None,
        policy: dict[str, Any],
        *,
        seed: int = 42,
        deterministic_preview: bool = False,
    ) -> dict[str, Any]:
        sample = self.adapter.load_yolo_sample(image_path, label_path)
        augmented = self.adapter.apply_policy_to_sample(
            sample,
            policy,
            seed=seed,
            deterministic_preview=deterministic_preview,
        )
        return {
            "sample": sample,
            "augmented": augmented,
            "before_validity": self.adapter.bbox_validity(sample["image"], sample["bboxes"]),
            "after_validity": self.adapter.bbox_validity(augmented["image"], augmented["bboxes"]),
        }
