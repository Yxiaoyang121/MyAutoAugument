# Class-Aware Policy Short-Training Report

- Scope: 5 epoch top1 short-training only; no formal 50 epoch training was run.
- Policy ID: `class_aware_policy_001`
- Dataset: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain\dataset\final_dataset\data.yaml`
- Train images / bboxes: `4602` / `6412`
- Val images / bboxes: `677` / `905`
- Modified augmented images: `1390`
- Unmodified augmented duplicates: `911`
- copy_paste new boxes: `49`
- Precision: `0.626`
- Recall: `0.680`
- mAP50: `0.685`
- mAP50-95: `0.456`
- short_train_score: `0.603477`
- score formula: `0.30*mAP50 + 0.35*mAP50-95 + 0.20*Recall + 0.15*small_object_recall`
- OOM: `false`

## Branches

- `photometric_branch`: target=`加强筋打伤, 开裂, 油污, 浅划伤, 漏背锡, 碰伤, 脏污, 轮廓划伤, 锡尖`, ops=`clahe, contrast, gamma, brightness, sharpen_mild`
- `copy_paste_branch`: target=`加强筋打伤, 开裂, 浅划伤, 锡丝残留`, ops=`class_balanced_copy_paste`
- `texture_branch`: target=`油污, 浅划伤, 碰伤, 脏污, 轮廓划伤, 锡膏`, ops=`sharpen, local_contrast, mild_noise`
- `localization_branch`: target=`漏背锡, 碰伤, 脏污, 轮廓划伤, 锡膏`, ops=`mild_translate, mild_scale`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
|---:|---|---:|---:|
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.971 | 0.990 |
| 2 | 加强筋打伤 | 0.750 | 0.836 |
| 3 | 开裂 | 0.815 | 0.663 |
| 4 | 油污 | 0.222 | 0.236 |
| 5 | 浅划伤 | 0.692 | 0.658 |
| 6 | 漏背锡 | 0.696 | 0.702 |
| 7 | 碰伤 | 0.569 | 0.613 |
| 8 | 脏污 | 0.337 | 0.353 |
| 9 | 轮廓划伤 | 0.655 | 0.749 |
| 10 | 锡丝残留 | 0.667 | 0.450 |
| 11 | 锡尖 | 0.815 | 0.932 |
| 12 | 锡膏 | 0.652 | 0.728 |
