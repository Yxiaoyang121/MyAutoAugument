from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from AutoAugment.bbox.affine import apply_affine_to_bboxes
from AutoAugment.bbox.clip import clip_filter_bboxes
from AutoAugment.bbox.iou import bbox_iou
from AutoAugment.transforms.base import build_sample
from AutoAugment.transforms.geometric import HorizontalFlip, RandomCrop, Rotate, Scale, Translate, VerticalFlip

from .registry import register_augmentation


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _strength(strength: float) -> float:
    return float(np.clip(strength, 0.0, 1.0))


def _inputs(
    image: np.ndarray,
    labels: np.ndarray | None,
    bboxes: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    labels_arr = np.asarray(labels if labels is not None else [], dtype=np.int64)
    if labels_arr.ndim == 0:
        labels_arr = labels_arr.reshape(1)
    if labels_arr.ndim != 1:
        raise ValueError("labels must be a 1-D array of class ids")
    if bboxes is None:
        if len(labels_arr) != 0:
            raise ValueError("bboxes are required when labels are not empty")
        bboxes_arr = np.zeros((0, 4), dtype=np.float32)
    else:
        bboxes_arr = np.asarray(bboxes, dtype=np.float32)
        if bboxes_arr.ndim == 1 and bboxes_arr.size == 0:
            bboxes_arr = bboxes_arr.reshape(0, 4)
        if bboxes_arr.ndim != 2 or bboxes_arr.shape[1] != 4:
            raise ValueError("bboxes must be an Nx4 xyxy array")
    if len(labels_arr) != len(bboxes_arr):
        raise ValueError("labels and bboxes must have the same length")
    return image, labels_arr, bboxes_arr


def _copy_labels_bboxes(labels: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return labels.copy(), bboxes.astype(np.float32, copy=True)


def _sample(image: np.ndarray, labels: np.ndarray, bboxes: np.ndarray) -> dict[str, Any]:
    return {"image": image, "labels": labels, "bboxes": bboxes}


def _from_sample(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return sample["image"], np.asarray(sample["labels"]), np.asarray(sample["bboxes"], dtype=np.float32)


def _clip_image(image: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    target_dtype = np.dtype(dtype or image.dtype)
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        return np.clip(image, 0, info.max).astype(target_dtype)
    return np.clip(image, 0.0, 1.0).astype(target_dtype, copy=False)


def _uint8_for_cv(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 0.0
        scaled = image * 255.0 if max_value <= 1.0 else image
        return np.clip(scaled, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def _odd_kernel(max_kernel: int, strength: float, rng: np.random.Generator, min_kernel: int = 3) -> int:
    max_kernel = max(1, int(max_kernel))
    if max_kernel % 2 == 0:
        max_kernel -= 1
    if strength <= 0.0 or max_kernel < min_kernel:
        return 1
    candidates = [k for k in range(min_kernel, max_kernel + 1, 2)]
    limit = max(1, int(np.ceil(len(candidates) * strength)))
    return int(rng.choice(candidates[:limit]))


def _warp_affine(
    image: np.ndarray,
    labels: np.ndarray,
    bboxes: np.ndarray,
    matrix: np.ndarray,
    border_value: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    warped = cv2.warpAffine(
        image,
        matrix.astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    transformed = apply_affine_to_bboxes(bboxes, matrix)
    clipped, filtered_labels = clip_filter_bboxes(transformed, labels, width, height)
    return warped, filtered_labels, clipped


def _copy_paste_weights(
    labels: np.ndarray,
    bboxes: np.ndarray,
    width: int,
    height: int,
    params: dict[str, Any],
) -> np.ndarray:
    """Return source-box sampling weights for same-image copy-paste."""

    weights = np.ones((len(labels),), dtype=np.float64)
    target_classes = params.get("target_classes", None)
    if target_classes is not None:
        allowed = {int(value) for value in target_classes}
        weights *= np.asarray([1.0 if int(label) in allowed else 0.0 for label in labels], dtype=np.float64)

    if bool(params.get("prefer_small", True)):
        areas = np.maximum(1.0, (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]))
        image_area = max(1.0, float(width * height))
        weights *= 1.0 / np.sqrt(np.maximum(areas / image_area, 1e-6))

    if bool(params.get("class_balanced", False)):
        class_counts_param = params.get("class_counts") or params.get("dataset_class_counts")
        if isinstance(class_counts_param, dict) and class_counts_param:
            counts = []
            for label in labels:
                key = int(label)
                counts.append(float(class_counts_param.get(key, class_counts_param.get(str(key), 1.0))))
            class_counts = np.asarray(counts, dtype=np.float64)
        else:
            unique, counts = np.unique(labels, return_counts=True)
            local_counts = {int(label): float(count) for label, count in zip(unique, counts)}
            class_counts = np.asarray([local_counts[int(label)] for label in labels], dtype=np.float64)
        weights *= 1.0 / np.maximum(class_counts, 1.0)

    return weights


def _sample_copy_paste_source(
    labels: np.ndarray,
    bboxes: np.ndarray,
    width: int,
    height: int,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> int | None:
    """Sample one source bbox index, prioritizing small and minority-class defects when requested."""

    if len(labels) == 0:
        return None
    weights = _copy_paste_weights(labels, bboxes, width, height, params)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return None
    probabilities = weights / total
    return int(rng.choice(np.arange(len(labels)), p=probabilities))


def _find_copy_paste_location(
    patch_width: int,
    patch_height: int,
    width: int,
    height: int,
    existing_bboxes: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Find a low-overlap destination box for a copied defect patch."""

    if patch_width <= 0 or patch_height <= 0 or patch_width > width or patch_height > height:
        return None
    max_overlap = float(params.get("max_overlap", params.get("max_iou", 0.20)))
    fallback_overlap = float(params.get("fallback_max_overlap", 0.50))
    max_attempts = max(1, int(params.get("max_attempts", 40)))
    best_box: np.ndarray | None = None
    best_overlap = float("inf")
    max_x = max(0, width - patch_width)
    max_y = max(0, height - patch_height)
    for _ in range(max_attempts):
        x1 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        y1 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        candidate = np.asarray([x1, y1, x1 + patch_width, y1 + patch_height], dtype=np.float32)
        overlap = 0.0
        if existing_bboxes.size:
            overlap = float(np.max(bbox_iou(candidate.reshape(1, 4), existing_bboxes)))
        if overlap <= max_overlap:
            return candidate
        if overlap < best_overlap:
            best_overlap = overlap
            best_box = candidate
    if best_box is not None and best_overlap <= fallback_overlap:
        return best_box
    return None


def _write_copy_paste_debug(
    image: np.ndarray,
    original_bboxes: np.ndarray,
    pasted_bboxes: list[np.ndarray],
    debug_dir: str | Path,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """Write a lightweight debug visualization for copy-paste results."""

    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    canvas = _uint8_for_cv(image).copy()
    for box in original_bboxes:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 1)
    for box in pasted_bboxes:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 128, 255), 2)
    prefix = str(params.get("debug_prefix", "copy_paste"))
    filename = f"{prefix}_{int(rng.integers(0, 1_000_000)):06d}.jpg"
    cv2.imwrite(str(path / filename), canvas)


@register_augmentation("brightness", changes_bboxes=False)
def brightness(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    s = _strength(strength)
    max_delta = float(params.get("max_delta", 0.25))
    delta = _rng(rng).uniform(-max_delta * s, max_delta * s) * 255.0
    out = image.astype(np.float32) + delta
    return _clip_image(out, image.dtype), labels_out, bboxes_out


@register_augmentation("contrast", changes_bboxes=False)
def contrast(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    s = _strength(strength)
    max_delta = float(params.get("max_delta", 0.5))
    factor = 1.0 + _rng(rng).uniform(-max_delta * s, max_delta * s)
    mean = image.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    out = (image.astype(np.float32) - mean) * factor + mean
    return _clip_image(out, image.dtype), labels_out, bboxes_out


@register_augmentation("gamma", changes_bboxes=False)
def gamma(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    s = _strength(strength)
    gamma_min = float(params.get("min_gamma", 0.7))
    gamma_max = float(params.get("max_gamma", 1.5))
    low = 1.0 - (1.0 - gamma_min) * s
    high = 1.0 + (gamma_max - 1.0) * s
    gamma_value = float(_rng(rng).uniform(low, high))
    inv_gamma = 1.0 / max(gamma_value, 1e-6)
    table = ((np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma * 255.0).astype(np.uint8)
    if image.dtype == np.uint8:
        return cv2.LUT(image, table), labels_out, bboxes_out
    out = (np.clip(image.astype(np.float32), 0.0, 1.0) ** inv_gamma).astype(image.dtype, copy=False)
    return out, labels_out, bboxes_out


@register_augmentation("gaussian_noise", changes_bboxes=False)
def gaussian_noise(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    s = _strength(strength)
    std = float(params.get("std", params.get("max_std", 0.08))) * s * 255.0
    noise = _rng(rng).normal(0.0, std, size=image.shape)
    out = image.astype(np.float32) + noise
    return _clip_image(out, image.dtype), labels_out, bboxes_out


@register_augmentation("salt_pepper_noise", changes_bboxes=False)
def salt_pepper_noise(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    s = _strength(strength)
    amount = float(params.get("amount", params.get("max_amount", 0.02))) * s
    if amount <= 0:
        return image.copy(), labels_out, bboxes_out
    generator = _rng(rng)
    out = image.copy()
    mask = generator.random(image.shape[:2])
    salt = mask < amount / 2.0
    pepper = (mask >= amount / 2.0) & (mask < amount)
    if out.ndim == 2:
        out[salt] = 255
        out[pepper] = 0
    else:
        out[salt, :] = 255
        out[pepper, :] = 0
    return out, labels_out, bboxes_out


@register_augmentation("gaussian_blur", changes_bboxes=False)
def gaussian_blur(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    generator = _rng(rng)
    kernel = _odd_kernel(int(params.get("max_kernel", 9)), _strength(strength), generator)
    if kernel <= 1:
        return image.copy(), labels_out, bboxes_out
    # Keep the sampled sigma range ordered even for very low strengths.
    sigma_high = max(0.1, 2.0 * _strength(strength))
    sigma = float(params.get("sigma", generator.uniform(0.1, sigma_high)))
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma), labels_out, bboxes_out


@register_augmentation("motion_blur", changes_bboxes=False)
def motion_blur(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    generator = _rng(rng)
    kernel_size = _odd_kernel(int(params.get("max_kernel", 11)), _strength(strength), generator)
    if kernel_size <= 1:
        return image.copy(), labels_out, bboxes_out
    angle = float(params.get("angle", generator.uniform(0.0, 180.0)))
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((kernel_size / 2.0 - 0.5, kernel_size / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        return image.copy(), labels_out, bboxes_out
    kernel /= kernel_sum
    return cv2.filter2D(image, -1, kernel), labels_out, bboxes_out


@register_augmentation("median_blur", changes_bboxes=False)
def median_blur(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    kernel = _odd_kernel(int(params.get("max_kernel", 7)), _strength(strength), _rng(rng))
    if kernel <= 1:
        return image.copy(), labels_out, bboxes_out
    return cv2.medianBlur(image, kernel), labels_out, bboxes_out


@register_augmentation("sharpen", changes_bboxes=False)
def sharpen(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    s = _strength(strength)
    amount = float(params.get("amount", 1.2)) * s
    if amount <= 0:
        return image.copy(), labels_out, bboxes_out
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=float(params.get("sigma", 1.0)))
    out = cv2.addWeighted(image.astype(np.float32), 1.0 + amount, blurred.astype(np.float32), -amount, 0.0)
    return _clip_image(out, image.dtype), labels_out, bboxes_out


@register_augmentation("clahe", changes_bboxes=False)
def clahe(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    clip_limit = 1.0 + (float(params.get("max_clip_limit", 4.0)) - 1.0) * _strength(strength)
    tile_grid_size = tuple(params.get("tile_grid_size", (8, 8)))
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cv_image = _uint8_for_cv(image)
    if cv_image.ndim == 2:
        return clahe_op.apply(cv_image), labels_out, bboxes_out
    lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe_op.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), labels_out, bboxes_out


@register_augmentation("local_contrast", changes_bboxes=False)
def local_contrast(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    s = _strength(strength)
    if s <= 0:
        return image.copy(), labels_out, bboxes_out
    cv_image = _uint8_for_cv(image)
    clip_limit = 1.0 + (float(params.get("max_clip_limit", 2.5)) - 1.0) * s
    tile_grid_size = tuple(params.get("tile_grid_size", (8, 8)))
    blend = float(np.clip(params.get("blend", 0.65) * s, 0.0, 1.0))
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if cv_image.ndim == 2:
        enhanced = clahe_op.apply(cv_image)
    else:
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe_op.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = cv2.addWeighted(cv_image.astype(np.float32), 1.0 - blend, enhanced.astype(np.float32), blend, 0.0)
    return _clip_image(out, image.dtype), labels_out, bboxes_out


@register_augmentation("cutout", changes_bboxes=False)
def cutout(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    labels_out, bboxes_out = _copy_labels_bboxes(labels_arr, bboxes_arr)
    params = params or {}
    s = _strength(strength)
    if s <= 0:
        return image.copy(), labels_out, bboxes_out
    generator = _rng(rng)
    height, width = image.shape[:2]
    max_holes = max(1, int(params.get("max_holes", 3)))
    holes = int(generator.integers(1, max_holes + 1))
    max_fraction = float(params.get("max_fraction", 0.25)) * s
    fill_value = params.get("fill_value", None)
    out = image.copy()
    for _ in range(holes):
        erase_w = max(1, int(width * generator.uniform(0.04, max(0.05, max_fraction))))
        erase_h = max(1, int(height * generator.uniform(0.04, max(0.05, max_fraction))))
        x1 = int(generator.integers(0, max(1, width - erase_w + 1)))
        y1 = int(generator.integers(0, max(1, height - erase_h + 1)))
        x2 = min(width, x1 + erase_w)
        y2 = min(height, y1 + erase_h)
        if fill_value is None:
            value = [int(v) for v in np.mean(image.reshape(-1, image.shape[-1]), axis=0)] if image.ndim == 3 else int(image.mean())
        else:
            value = fill_value
        out[y1:y2, x1:x2] = value
    return out, labels_out, bboxes_out


@register_augmentation("copy_paste", changes_bboxes=True)
def copy_paste(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy bbox-level defect patches within one YOLO detection sample.

    This is the minimal executable variant used by the diagnostic pipeline. It
    duplicates existing object crops from the same image, pastes them into a
    low-overlap location, appends the corresponding label and clips all boxes
    to the image boundary.
    """

    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    params = params or {}
    s = _strength(strength)
    if s <= 0.0 or len(labels_arr) == 0:
        return image.copy(), labels_arr.copy(), bboxes_arr.astype(np.float32, copy=True)

    generator = _rng(rng)
    height, width = image.shape[:2]
    max_paste_count = max(1, int(params.get("max_paste_count", 1)))
    paste_count = max(1, int(round(max_paste_count * max(s, 0.2))))
    paste_count = min(max_paste_count, paste_count)
    min_patch_size = max(1, int(params.get("min_patch_size", 2)))

    out = image.copy()
    labels_out = labels_arr.copy()
    bboxes_out = bboxes_arr.astype(np.float32, copy=True)
    pasted_boxes: list[np.ndarray] = []

    for _ in range(paste_count):
        source_index = _sample_copy_paste_source(labels_arr, bboxes_arr, width, height, params, generator)
        if source_index is None:
            break
        source_box = bboxes_arr[source_index]
        x1 = max(0, int(np.floor(source_box[0])))
        y1 = max(0, int(np.floor(source_box[1])))
        x2 = min(width, int(np.ceil(source_box[2])))
        y2 = min(height, int(np.ceil(source_box[3])))
        patch_width = x2 - x1
        patch_height = y2 - y1
        if patch_width < min_patch_size or patch_height < min_patch_size:
            continue

        destination = _find_copy_paste_location(
            patch_width,
            patch_height,
            width,
            height,
            bboxes_out,
            params,
            generator,
        )
        if destination is None:
            continue

        dx1, dy1, dx2, dy2 = [int(round(float(value))) for value in destination]
        patch = image[y1:y2, x1:x2].copy()
        if patch.shape[0] != dy2 - dy1 or patch.shape[1] != dx2 - dx1:
            patch = cv2.resize(patch, (dx2 - dx1, dy2 - dy1), interpolation=cv2.INTER_LINEAR)
        out[dy1:dy2, dx1:dx2] = patch
        labels_out = np.concatenate([labels_out, np.asarray([labels_arr[source_index]], dtype=np.int64)])
        bboxes_out = np.vstack([bboxes_out, destination.astype(np.float32)])
        pasted_boxes.append(destination.astype(np.float32))

    clipped, filtered_labels = clip_filter_bboxes(bboxes_out, labels_out, width, height)
    if params.get("debug_dir") and pasted_boxes:
        _write_copy_paste_debug(out, bboxes_arr, pasted_boxes, params["debug_dir"], params, generator)
    return out, filtered_labels.astype(np.int64, copy=False), clipped.astype(np.float32, copy=False)


@register_augmentation("random_erasing", changes_bboxes=False)
def random_erasing(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return cutout(image, labels, bboxes, params, strength, rng)


@register_augmentation("horizontal_flip", changes_bboxes=True)
def horizontal_flip(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    if _strength(strength) <= 0:
        return image.copy(), labels_arr.copy(), bboxes_arr.copy()
    result = HorizontalFlip(probability=1.0)(_sample(image, labels_arr, bboxes_arr), rng=_rng(rng))
    return _from_sample(result)


@register_augmentation("vertical_flip", changes_bboxes=True)
def vertical_flip(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    if _strength(strength) <= 0:
        return image.copy(), labels_arr.copy(), bboxes_arr.copy()
    result = VerticalFlip(probability=1.0)(_sample(image, labels_arr, bboxes_arr), rng=_rng(rng))
    return _from_sample(result)


@register_augmentation("rotate", changes_bboxes=True)
def rotate(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    params = params or {}
    s = _strength(strength)
    max_angle = float(params.get("max_angle", 15.0)) * s
    transform = Rotate(angle=params.get("angle"), angle_range=(-max_angle, max_angle), probability=1.0)
    return _from_sample(transform(_sample(image, labels_arr, bboxes_arr), rng=_rng(rng)))


@register_augmentation("translate", changes_bboxes=True)
def translate(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    params = params or {}
    s = _strength(strength)
    max_translate = float(params.get("max_translate", 0.1)) * s
    transform = Translate(
        dx=params.get("dx"),
        dy=params.get("dy"),
        dx_range=(-max_translate, max_translate),
        dy_range=(-max_translate, max_translate),
        probability=1.0,
    )
    return _from_sample(transform(_sample(image, labels_arr, bboxes_arr), rng=_rng(rng)))


@register_augmentation("scale", changes_bboxes=True)
def scale(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    params = params or {}
    s = _strength(strength)
    max_delta = float(params.get("max_delta", 0.25)) * s
    scale_min = max(0.05, 1.0 - max_delta)
    scale_max = 1.0 + max_delta
    transform = Scale(scale=params.get("scale"), scale_range=(scale_min, scale_max), probability=1.0)
    return _from_sample(transform(_sample(image, labels_arr, bboxes_arr), rng=_rng(rng)))


@register_augmentation("affine", changes_bboxes=True)
def affine(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    params = params or {}
    generator = _rng(rng)
    s = _strength(strength)
    height, width = image.shape[:2]
    angle = float(params.get("angle", generator.uniform(-float(params.get("max_angle", 10.0)) * s, float(params.get("max_angle", 10.0)) * s)))
    scale_value = float(params.get("scale", generator.uniform(1.0 - float(params.get("max_scale_delta", 0.15)) * s, 1.0 + float(params.get("max_scale_delta", 0.15)) * s)))
    shear_x = np.deg2rad(float(params.get("shear_x", generator.uniform(-float(params.get("max_shear", 5.0)) * s, float(params.get("max_shear", 5.0)) * s))))
    shear_y = np.deg2rad(float(params.get("shear_y", generator.uniform(-float(params.get("max_shear", 5.0)) * s, float(params.get("max_shear", 5.0)) * s))))
    max_translate = float(params.get("max_translate", 0.08)) * s
    tx = float(params.get("tx", generator.uniform(-max_translate, max_translate))) * width
    ty = float(params.get("ty", generator.uniform(-max_translate, max_translate))) * height

    cx, cy = width / 2.0, height / 2.0
    cos_a = np.cos(np.deg2rad(angle)) * scale_value
    sin_a = np.sin(np.deg2rad(angle)) * scale_value
    center_to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    origin_to_center = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float32)
    rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    shear = np.array([[1, np.tan(shear_x), 0], [np.tan(shear_y), 1, 0], [0, 0, 1]], dtype=np.float32)
    matrix = (origin_to_center @ shear @ rotation @ center_to_origin)[:2, :]
    return _warp_affine(image, labels_arr, bboxes_arr, matrix)


@register_augmentation("crop", changes_bboxes=True)
def crop(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    params = params or {}
    height, width = image.shape[:2]
    s = _strength(strength)
    max_crop_fraction = float(params.get("max_crop_fraction", 0.4)) * s
    min_ratio = max(0.2, 1.0 - max_crop_fraction)
    width_range = params.get("width_range", (max(1, int(width * min_ratio)), width))
    height_range = params.get("height_range", (max(1, int(height * min_ratio)), height))
    transform = RandomCrop(
        crop_box=params.get("crop_box"),
        crop_size=params.get("crop_size"),
        width_range=width_range,
        height_range=height_range,
        min_visibility=float(params.get("min_visibility", 0.2)),
        probability=1.0,
    )
    return _from_sample(transform(_sample(image, labels_arr, bboxes_arr), rng=_rng(rng)))


@register_augmentation("resize_letterbox", changes_bboxes=True)
def resize_letterbox(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels_arr, bboxes_arr = _inputs(image, labels, bboxes)
    params = params or {}
    height, width = image.shape[:2]
    target_size = params.get("target_size", None)
    if target_size is not None:
        target_width, target_height = int(target_size[0]), int(target_size[1])
    else:
        target_width = int(params.get("width", width))
        target_height = int(params.get("height", height))
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target letterbox size must be positive")
    scale_value = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale_value)))
    new_height = max(1, int(round(height * scale_value)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    color = params.get("color", (114, 114, 114))
    out = np.full((target_height, target_width, *image.shape[2:]), color, dtype=image.dtype) if image.ndim == 3 else np.full((target_height, target_width), int(color[0] if isinstance(color, (list, tuple)) else color), dtype=image.dtype)
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    out[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    bboxes_out = bboxes_arr.copy()
    if len(bboxes_out) > 0:
        bboxes_out[:, [0, 2]] = bboxes_out[:, [0, 2]] * scale_value + pad_x
        bboxes_out[:, [1, 3]] = bboxes_out[:, [1, 3]] * scale_value + pad_y
    clipped, filtered_labels = clip_filter_bboxes(bboxes_out, labels_arr, target_width, target_height)
    return out, filtered_labels, clipped
