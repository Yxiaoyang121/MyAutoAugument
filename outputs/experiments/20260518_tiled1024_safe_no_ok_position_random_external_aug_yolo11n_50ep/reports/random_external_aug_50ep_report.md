# Random External Augmentation 50 Epoch Report

- Scope: random external augmentation control group, 50 epoch training.
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Augmented dataset: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep\dataset\final_dataset`
- Train images / bboxes: `4602` / `6364`
- Original train images: `2301`
- Random augmented train images: `2301`
- Val images / bboxes: `677` / `905`
- Random policy: `random_external_policy_seed42`
- Operations: `sharpen(p=0.238,s=0.343), brightness(p=0.681,s=0.414), cutout(p=0.151,s=0.140), horizontal_flip(p=0.448,s=1.000)`
- bbox_valid_rate: `1.000000`
- original_bbox_retention: `1.000000`
- safety_score: `0.940000`
- Precision: `0.750`
- Recall: `0.668`
- mAP50: `0.734`
- mAP50-95: `0.501`
- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep\train\weights\best.pt`
- OOM: `false`
- Training wall seconds: `21467.4`

## Random Policy

| operation | prob | strength | params |
|---|---:|---:|---|
| sharpen | 0.2377 | 0.3427 | `{"amount": 0.8}` |
| brightness | 0.6806 | 0.4144 | `{"max_delta": 0.2}` |
| cutout | 0.1512 | 0.1401 | `{"max_holes": 2, "max_fraction": 0.18}` |
| horizontal_flip | 0.4483 | 1.0000 | `{}` |

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
|---:|---|---:|---:|
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.995 | 0.967 |
| 2 | 加强筋打伤 | 0.875 | 0.982 |
| 3 | 开裂 | 1.000 | 0.995 |
| 4 | 油污 | 0.389 | 0.290 |
| 5 | 浅划伤 | 0.506 | 0.622 |
| 6 | 漏背锡 | 0.587 | 0.608 |
| 7 | 碰伤 | 0.513 | 0.638 |
| 8 | 脏污 | 0.381 | 0.446 |
| 9 | 轮廓划伤 | 0.607 | 0.709 |
| 10 | 锡丝残留 | 0.366 | 0.660 |
| 11 | 锡尖 | 0.896 | 0.929 |
| 12 | 锡膏 | 0.567 | 0.698 |

## Class-Aware Shorttrain Context

- `class_aware_policy_001` was evaluated only as a 5 epoch short-training policy selection check.
- It did not beat `diag_policy_001`; therefore this random external 50 epoch control is not compared as a formal 50 epoch class-aware result.
