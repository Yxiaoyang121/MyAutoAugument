from __future__ import annotations

from typing import Any

from AutoAugment.augmentations import get_augmentation_spec, list_augmentations

from gui.models.augmentation_schema import AugmentationParamSchema as P
from gui.models.augmentation_schema import AugmentationSchema


CATEGORY_GEOMETRY = "几何增强"
CATEGORY_COLOR = "光照/颜色增强"
CATEGORY_NOISE = "噪声增强"
CATEGORY_INDUSTRIAL = "工业偏差类增强"


AUGMENTATION_SCHEMAS: dict[str, AugmentationSchema] = {
    "brightness": AugmentationSchema(
        "brightness",
        CATEGORY_COLOR,
        "亮度扰动，按 strength 缩放 max_delta 后在像素域加减亮度。",
        (P("max_delta", 0.25, "float", 0.0, 1.0),),
    ),
    "contrast": AugmentationSchema(
        "contrast",
        CATEGORY_COLOR,
        "对比度扰动，以图像均值为中心缩放像素差异。",
        (P("max_delta", 0.5, "float", 0.0, 2.0),),
    ),
    "gamma": AugmentationSchema(
        "gamma",
        CATEGORY_COLOR,
        "Gamma 非线性亮度调整，在 min_gamma 和 max_gamma 之间采样。",
        (P("min_gamma", 0.7, "float", 0.05, 5.0), P("max_gamma", 1.5, "float", 0.05, 5.0)),
    ),
    "gaussian_noise": AugmentationSchema(
        "gaussian_noise",
        CATEGORY_NOISE,
        "高斯噪声，std 或 max_std 会乘以 strength 并映射到 0..255 像素域。",
        (
            P("max_std", 0.08, "float", 0.0, 1.0),
            P("std", None, "float", 0.0, 1.0, include_in_policy=False, note="可选固定覆盖值"),
        ),
    ),
    "salt_pepper_noise": AugmentationSchema(
        "salt_pepper_noise",
        CATEGORY_NOISE,
        "椒盐噪声，amount 或 max_amount 控制被置黑/白的像素比例。",
        (
            P("max_amount", 0.02, "float", 0.0, 1.0),
            P("amount", None, "float", 0.0, 1.0, include_in_policy=False, note="可选固定覆盖值"),
        ),
    ),
    "gaussian_blur": AugmentationSchema(
        "gaussian_blur",
        CATEGORY_INDUSTRIAL,
        "高斯模糊，模拟离焦或成像模糊；bbox 不变。",
        (
            P("max_kernel", 9, "int", 1, 31),
            P("sigma", None, "float", 0.0, 10.0, include_in_policy=False, note="为空时后端随机采样"),
        ),
    ),
    "motion_blur": AugmentationSchema(
        "motion_blur",
        CATEGORY_INDUSTRIAL,
        "运动模糊，模拟传送带、相机抖动或曝光期间位移。",
        (
            P("max_kernel", 11, "int", 1, 31),
            P("angle", None, "float", 0.0, 180.0, include_in_policy=False, note="为空时后端随机采样"),
        ),
    ),
    "median_blur": AugmentationSchema(
        "median_blur",
        CATEGORY_INDUSTRIAL,
        "中值滤波模糊，常用于模拟去噪或细节损失。",
        (P("max_kernel", 7, "int", 1, 31),),
    ),
    "sharpen": AugmentationSchema(
        "sharpen",
        CATEGORY_INDUSTRIAL,
        "锐化增强，通过反遮罩提升局部边缘。",
        (P("amount", 1.2, "float", 0.0, 5.0), P("sigma", 1.0, "float", 0.0, 10.0)),
    ),
    "clahe": AugmentationSchema(
        "clahe",
        CATEGORY_INDUSTRIAL,
        "CLAHE 局部直方图均衡，改善低对比工业目标可见性。",
        (P("max_clip_limit", 4.0, "float", 1.0, 20.0), P("tile_grid_size", (8, 8), "list[int]", 1, None)),
    ),
    "local_contrast": AugmentationSchema(
        "local_contrast",
        CATEGORY_INDUSTRIAL,
        "Local contrast enhancement with bounded blend strength; bbox unchanged.",
        (
            P("max_clip_limit", 2.5, "float", 1.0, 20.0),
            P("tile_grid_size", (8, 8), "list[int]", 1, None),
            P("blend", 0.65, "float", 0.0, 1.0),
        ),
    ),
    "cutout": AugmentationSchema(
        "cutout",
        CATEGORY_INDUSTRIAL,
        "随机遮挡矩形区域，模拟遮挡、污渍或缺失观测；bbox 不变。",
        (
            P("max_holes", 3, "int", 1, 32),
            P("max_fraction", 0.25, "float", 0.0, 1.0),
            P("fill_value", None, "int|list[int]", 0, 255, include_in_policy=False, note="为空时使用图像均值"),
        ),
    ),
    "copy_paste": AugmentationSchema(
        "copy_paste",
        CATEGORY_INDUSTRIAL,
        "Copy and paste low-overlap object patches inside one sample; bbox changes.",
        (
            P("max_paste_count", 1, "int", 1, 16),
            P("max_overlap", 0.2, "float", 0.0, 1.0),
            P("fallback_max_overlap", 0.5, "float", 0.0, 1.0),
            P("max_attempts", 40, "int", 1, 500),
            P("min_patch_size", 2, "int", 1, None),
            P("prefer_small", True, "bool", None, None),
            P("class_balanced", False, "bool", None, None),
        ),
        changes_bboxes=True,
    ),
    "random_erasing": AugmentationSchema(
        "random_erasing",
        CATEGORY_INDUSTRIAL,
        "cutout 的后端别名，调用相同遮挡逻辑。",
        (
            P("max_holes", 3, "int", 1, 32),
            P("max_fraction", 0.25, "float", 0.0, 1.0),
            P("fill_value", None, "int|list[int]", 0, 255, include_in_policy=False),
        ),
    ),
    "horizontal_flip": AugmentationSchema(
        "horizontal_flip",
        CATEGORY_GEOMETRY,
        "水平翻转，同时按图像宽度更新并裁剪 bbox。",
        changes_bboxes=True,
    ),
    "vertical_flip": AugmentationSchema(
        "vertical_flip",
        CATEGORY_GEOMETRY,
        "垂直翻转，同时按图像高度更新并裁剪 bbox。",
        changes_bboxes=True,
    ),
    "rotate": AugmentationSchema(
        "rotate",
        CATEGORY_GEOMETRY,
        "绕图像中心仿射旋转，bbox 四角变换后重新取外接矩形。",
        (
            P("max_angle", 15.0, "float", 0.0, 180.0),
            P("angle", None, "float", -180.0, 180.0, include_in_policy=False, note="可选固定角度"),
        ),
        changes_bboxes=True,
    ),
    "translate": AugmentationSchema(
        "translate",
        CATEGORY_GEOMETRY,
        "水平/垂直平移，dx/dy 在绝对值不超过 1 时按图像宽高比例解释。",
        (
            P("max_translate", 0.1, "float", 0.0, 1.0),
            P("dx", None, "float", -1.0, 1.0, include_in_policy=False, note="可选固定水平位移"),
            P("dy", None, "float", -1.0, 1.0, include_in_policy=False, note="可选固定垂直位移"),
        ),
        changes_bboxes=True,
    ),
    "scale": AugmentationSchema(
        "scale",
        CATEGORY_GEOMETRY,
        "以图像中心缩放，bbox 经同一仿射矩阵变换并裁剪。",
        (
            P("max_delta", 0.25, "float", 0.0, 1.0),
            P("scale", None, "float", 0.05, 5.0, include_in_policy=False, note="可选固定缩放值"),
        ),
        changes_bboxes=True,
    ),
    "affine": AugmentationSchema(
        "affine",
        CATEGORY_GEOMETRY,
        "组合旋转、缩放、shear 和平移的仿射增强。",
        (
            P("max_angle", 10.0, "float", 0.0, 180.0),
            P("max_scale_delta", 0.15, "float", 0.0, 1.0),
            P("max_shear", 5.0, "float", 0.0, 45.0),
            P("max_translate", 0.08, "float", 0.0, 1.0),
            P("angle", None, "float", -180.0, 180.0, include_in_policy=False),
            P("scale", None, "float", 0.05, 5.0, include_in_policy=False),
            P("shear_x", None, "float", -45.0, 45.0, include_in_policy=False),
            P("shear_y", None, "float", -45.0, 45.0, include_in_policy=False),
            P("tx", None, "float", -1.0, 1.0, include_in_policy=False),
            P("ty", None, "float", -1.0, 1.0, include_in_policy=False),
        ),
        changes_bboxes=True,
    ),
    "crop": AugmentationSchema(
        "crop",
        CATEGORY_GEOMETRY,
        "随机裁剪，裁剪后平移 bbox 并按 min_visibility 过滤被截断过多的框。",
        (
            P("max_crop_fraction", 0.4, "float", 0.0, 0.95),
            P("min_visibility", 0.2, "float", 0.0, 1.0),
            P("crop_box", None, "list[int]", None, None, include_in_policy=False),
            P("crop_size", None, "list[int]", 1, None, include_in_policy=False),
            P("width_range", None, "list[int]", 1, None, include_in_policy=False),
            P("height_range", None, "list[int]", 1, None, include_in_policy=False),
        ),
        changes_bboxes=True,
    ),
    "resize_letterbox": AugmentationSchema(
        "resize_letterbox",
        CATEGORY_GEOMETRY,
        "等比例缩放并 letterbox 填充到目标尺寸，同步缩放和偏移 bbox。",
        (
            P("target_size", None, "list[int]", 1, None, include_in_policy=False, note="例如 [640, 640]"),
            P("width", None, "int", 1, None, include_in_policy=False),
            P("height", None, "int", 1, None, include_in_policy=False),
            P("color", (114, 114, 114), "list[int]", 0, 255),
        ),
        changes_bboxes=True,
    ),
}


class BackendCapabilityAdapter:
    """Expose real backend capabilities through a stable GUI-facing API."""

    def list_augmentations(self) -> list[AugmentationSchema]:
        registered = list_augmentations()
        schemas: list[AugmentationSchema] = []
        for name in registered:
            spec = get_augmentation_spec(name)
            schema = AUGMENTATION_SCHEMAS[name]
            schemas.append(
                AugmentationSchema(
                    name=schema.name,
                    category=schema.category,
                    description_zh=schema.description_zh,
                    params=schema.params,
                    changes_bboxes=bool(spec.changes_bboxes),
                    detection_suitable=schema.detection_suitable,
                    backend_reference=f"AutoAugment.augmentations.ops.{schema.name}",
                    notes=schema.notes,
                )
            )
        return schemas

    def schema_by_name(self) -> dict[str, AugmentationSchema]:
        return {item.name: item for item in self.list_augmentations()}

    def validate_schema(self) -> None:
        registered = set(list_augmentations())
        mapped = set(AUGMENTATION_SCHEMAS)
        missing = sorted(registered - mapped)
        stale = sorted(mapped - registered)
        if missing or stale:
            raise RuntimeError(f"augmentation schema mismatch; missing={missing}, stale={stale}")

    def dataset_formats(self) -> list[dict[str, Any]]:
        return [
            {
                "format": "YOLO",
                "status": "full workflow",
                "evidence": "AutoAugment.formats.yolo, AutoAugment.utils.yolo_dataset, RandomSearch train/val split",
            },
            {
                "format": "COCO",
                "status": "annotation conversion",
                "evidence": "AutoAugment.formats.coco and AutoAugment.datasets.coco_dataset",
            },
            {
                "format": "VOC",
                "status": "XML read / bbox conversion",
                "evidence": "AutoAugment.formats.voc and bbox.convert voc helpers",
            },
        ]

    def bbox_capabilities(self) -> list[str]:
        return [
            "xyxy internal representation",
            "YOLO xywh normalized <-> xyxy conversion",
            "COCO xywh <-> xyxy conversion",
            "VOC xyxy passthrough conversion",
            "affine corner transform and enclosing bbox recomputation",
            "clip to image bounds",
            "filter invalid bbox by min width, height, and area",
            "crop visibility filtering",
        ]

    def metrics(self) -> list[str]:
        return [
            "image_valid_rate",
            "bbox_valid_rate",
            "bbox_safe_rate",
            "bbox_retention",
            "bbox_retention_raw",
            "tiny_target_retention",
            "small_target_retention",
            "edge_target_retention",
            "class_coverage_after",
            "rare_class_retention",
            "exposure_score",
            "variance_score",
            "exposure_diversity_score",
            "strength_penalty",
            "diversity_score",
            "proxy_score",
            "yolo_map50",
            "yolo_map50_95",
            "final_score",
        ]

    def output_files(self) -> list[str]:
        return [
            "run_config.json",
            "stage_config.json",
            "input_dataset.txt",
            "README.txt",
            "trials.json",
            "trials.csv",
            "policy_history.jsonl",
            "policy_history.csv",
            "policy_history.md",
            "best_policy.json",
            "trials/trial_xxx/policy.json",
            "trials/trial_xxx/metrics.json",
            "trials/trial_xxx/diagnosis.json",
            "trials/trial_xxx/trial_record.json",
            "trials/trial_xxx/train_stdout.log",
            "trials/trial_xxx/train_stderr.log",
            "trials/trial_xxx/val_stdout.log",
            "trials/trial_xxx/val_stderr.log",
            "diagnosis/error_summary.json",
            "diagnosis/augmentation_advice.json",
            "diagnosis/advisor_search_space.json",
        ]
