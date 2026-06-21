from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from AutoAugment.augmentations import apply_augmentation
from AutoAugment.formats.yolo import load_yolo_labels
from AutoAugment.policies import Policy, apply_policy, ensure_policy


class AugmentationAdapter:
    def load_yolo_sample(self, image_path: str | Path, label_path: str | Path | None = None) -> dict[str, Any]:
        image_file = Path(image_path)
        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"读取图像失败: {image_file}")
        height, width = image.shape[:2]
        if label_path:
            labels, bboxes = load_yolo_labels(label_path, width, height)
        else:
            labels = np.zeros((0,), dtype=np.int64)
            bboxes = np.zeros((0, 4), dtype=np.float32)
        return {"image": image, "labels": labels, "bboxes": bboxes, "image_path": str(image_file)}

    def apply_policy_to_sample(
        self,
        sample: dict[str, Any],
        policy: Policy | dict[str, Any],
        *,
        seed: int = 42,
        deterministic_preview: bool = False,
    ) -> dict[str, Any]:
        if deterministic_preview:
            image, labels, bboxes = self._apply_policy_for_preview(sample["image"], sample["labels"], sample["bboxes"], policy)
        else:
            rng = np.random.default_rng(seed)
            image, labels, bboxes = apply_policy(sample["image"], sample["labels"], sample["bboxes"], policy, rng=rng)
        result = dict(sample)
        result.update({"image": image, "labels": labels, "bboxes": bboxes})
        return result

    def _apply_policy_for_preview(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
        policy: Policy | dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply every configured operation once for a stable strategy preview.

        Training/search still uses operation probabilities and random sampling. The
        preview path is intentionally deterministic so the page shows the policy
        the user assembled instead of one random draw from that policy.
        """
        current_image = image.copy()
        current_labels = np.asarray(labels if labels is not None else [], dtype=np.int64).copy()
        current_bboxes = np.asarray(bboxes if bboxes is not None else np.zeros((0, 4)), dtype=np.float32).reshape(-1, 4).copy()
        preview_rng = np.random.default_rng(0)
        for operation in ensure_policy(policy).operations:
            if operation.prob <= 0 or operation.strength <= 0:
                continue
            params = self._deterministic_preview_params(operation.name, operation.params, operation.strength)
            direct = self._apply_direct_preview_operation(
                operation.name,
                current_image,
                current_labels,
                current_bboxes,
                params,
                operation.strength,
            )
            if direct is not None:
                current_image, current_labels, current_bboxes = direct
            else:
                current_image, current_labels, current_bboxes = apply_augmentation(
                    operation.name,
                    current_image,
                    current_labels,
                    current_bboxes,
                    params=params,
                    strength=operation.strength,
                    rng=preview_rng,
                )
        return current_image, current_labels, current_bboxes

    def _deterministic_preview_params(self, name: str, params: dict[str, Any], strength: float) -> dict[str, Any]:
        resolved = dict(params or {})
        s = float(np.clip(strength, 0.0, 1.0))
        op = name.strip().lower()
        if op == "rotate" and "angle" not in resolved:
            resolved["angle"] = float(resolved.get("max_angle", 15.0)) * s
        elif op == "translate":
            max_translate = float(resolved.get("max_translate", 0.1)) * s
            resolved.setdefault("dx", max_translate)
            resolved.setdefault("dy", 0.0)
        elif op == "scale" and "scale" not in resolved:
            resolved["scale"] = 1.0 + float(resolved.get("max_delta", 0.25)) * s
        elif op == "motion_blur":
            resolved.setdefault("angle", 0.0)
            resolved.setdefault("max_kernel", max(3, int(resolved.get("max_kernel", 11))))
        elif op == "gaussian_blur":
            resolved.setdefault("sigma", max(0.1, 2.0 * s))
        return resolved

    def _apply_direct_preview_operation(
        self,
        name: str,
        image: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
        params: dict[str, Any],
        strength: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        op = name.strip().lower()
        s = float(np.clip(strength, 0.0, 1.0))
        labels_out = labels.copy()
        bboxes_out = bboxes.astype(np.float32, copy=True)
        if op == "brightness":
            delta = float(params.get("max_delta", 0.25)) * s * 255.0
            return self._clip_image_like(image.astype(np.float32) + delta, image), labels_out, bboxes_out
        if op == "contrast":
            factor = 1.0 + float(params.get("max_delta", 0.5)) * s
            mean = image.astype(np.float32).mean(axis=(0, 1), keepdims=True)
            out = (image.astype(np.float32) - mean) * factor + mean
            return self._clip_image_like(out, image), labels_out, bboxes_out
        if op == "gamma":
            gamma_min = float(params.get("min_gamma", 0.7))
            gamma_value = max(0.1, 1.0 - (1.0 - gamma_min) * s)
            table = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma_value * 255.0).astype(np.uint8)
            if image.dtype == np.uint8:
                return cv2.LUT(image, table), labels_out, bboxes_out
        if op == "saturation" and image.ndim == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= 1.0 + float(params.get("max_delta", 0.5)) * s
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), labels_out, bboxes_out
        return None

    def _clip_image_like(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if np.issubdtype(reference.dtype, np.integer):
            info = np.iinfo(reference.dtype)
            return np.clip(image, 0, info.max).astype(reference.dtype)
        return np.clip(image, 0.0, 1.0).astype(reference.dtype, copy=False)

    def bbox_validity(self, image: np.ndarray, bboxes: np.ndarray) -> dict[str, Any]:
        height, width = image.shape[:2]
        boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        if boxes.size == 0:
            return {"valid": True, "total": 0, "valid_count": 0, "invalid_count": 0, "invalid_indices": []}
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        valid = (x1 >= 0) & (y1 >= 0) & (x2 <= width) & (y2 <= height) & (x2 > x1) & (y2 > y1)
        invalid_indices = [int(index) for index, ok in enumerate(valid) if not bool(ok)]
        return {
            "valid": not invalid_indices,
            "total": int(len(boxes)),
            "valid_count": int(valid.sum()),
            "invalid_count": int(len(invalid_indices)),
            "invalid_indices": invalid_indices,
        }

    def draw_bboxes(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
        class_names: list[str] | dict[int, str] | None = None,
    ) -> np.ndarray:
        out = image.copy()
        names: dict[int, str] = {}
        if isinstance(class_names, dict):
            names = {int(key): str(value) for key, value in class_names.items()}
        elif isinstance(class_names, list):
            names = {index: value for index, value in enumerate(class_names)}
        for label, box in zip(labels, np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)):
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), (40, 190, 90), 2)
            text = names.get(int(label), str(int(label)))
            cv2.putText(out, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 190, 90), 1, cv2.LINE_AA)
        return out
