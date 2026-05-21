# Per-Class Diagnosis Report

- Scope: class-aware error attribution and policy generation only.
- No YOLO training, no 50 epoch run, and no top3 short-training were executed.
- Baseline: `20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep`
- Existing diagnosis: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/diagnosis/diagnosis.json`
- Counterfactual diagnosis: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/counterfactual_summary.json`

## Required Answers

1. Current largest issue: Low contrast is still real, but counterfactual photometric recovery is partial (0.1050); class imbalance, texture confusion, and localization/size errors also matter.
2. Photometric-suitable classes: `加强筋打伤, 开裂, 油污, 浅划伤, 漏背锡, 碰伤, 脏污, 轮廓划伤, 锡尖`
3. copy-paste / class-balance classes: `加强筋打伤, 开裂, 浅划伤, 锡丝残留`
4. Texture-confusion classes: `油污, 浅划伤, 碰伤, 脏污, 轮廓划伤, 锡膏`
5. Low-support classes: `加强筋打伤, 开裂`
6. diag_policy_001 limitation: It applies one global photometric branch and cannot target low-support, texture, localization, or class-balance failures.
7. class-aware mixed policy expected improvement: It covers multiple error causes with class-specific branches and keeps copy_paste targeted to sparse/imbalanced classes.
8. Recommend top1 short-training next: `True`

## Per-Class Attribution

| class | train | val | P | R | AP50 | AP50-95 | TP | FP | FN | primary_error | attributions | severity | confidence | priority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| OK2 | 302 | 64 | 0.971 | 1.000 | 0.995 | 0.841 | 64 | 6 | 0 | `stable_or_low_priority` | stable_or_low_priority | 0.009 | 1.000 | 0.009 |
| OK3 | 946 | 274 | 0.932 | 0.978 | 0.970 | 0.777 | 269 | 8 | 5 | `stable_or_low_priority` | low_contrast_error, dark_object_error, small_object_error, tile_edge_error | 0.247 | 1.000 | 0.247 |
| 加强筋打伤 | 36 | 8 | 0.996 | 0.875 | 0.955 | 0.660 | 7 | 0 | 1 | `low_support_class` | low_support_class, class_sample_imbalance, low_contrast_error, dark_object_error, small_object_error | 0.283 | 0.350 | 0.099 |
| 开裂 | 17 | 2 | 0.000 | 0.000 | 0.111 | 0.092 | 0 | 2 | 2 | `low_support_class` | low_support_class, class_sample_imbalance, dark_object_error, class_confusion | 0.822 | 0.200 | 0.164 |
| 油污 | 137 | 18 | 0.254 | 0.333 | 0.242 | 0.111 | 5 | 19 | 13 | `high_fp_class` | low_recall_class, high_fp_class, low_contrast_error, dark_object_error, texture_confusion, tile_edge_error, class_confusion | 0.632 | 0.600 | 0.379 |
| 浅划伤 | 81 | 13 | 1.000 | 0.550 | 0.712 | 0.338 | 7 | 3 | 6 | `class_sample_imbalance` | class_sample_imbalance, low_recall_class, low_contrast_error, texture_confusion, small_object_error, class_confusion | 0.418 | 0.600 | 0.251 |
| 漏背锡 | 187 | 46 | 0.577 | 0.712 | 0.694 | 0.325 | 32 | 23 | 11 | `low_contrast_error` | low_contrast_error, weak_localization, class_confusion | 0.208 | 0.800 | 0.166 |
| 碰伤 | 860 | 269 | 0.681 | 0.617 | 0.617 | 0.368 | 165 | 83 | 92 | `low_recall_class` | low_recall_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, tile_edge_error, class_confusion | 0.439 | 1.000 | 0.439 |
| 脏污 | 157 | 42 | 0.490 | 0.381 | 0.440 | 0.235 | 18 | 9 | 20 | `high_fp_class` | low_recall_class, high_fp_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, class_confusion | 0.469 | 0.800 | 0.375 |
| 轮廓划伤 | 189 | 84 | 0.782 | 0.511 | 0.717 | 0.386 | 47 | 9 | 35 | `low_recall_class` | low_recall_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, tile_edge_error, class_confusion | 0.441 | 1.000 | 0.441 |
| 锡丝残留 | 32 | 12 | 0.634 | 0.583 | 0.619 | 0.375 | 5 | 1 | 7 | `class_sample_imbalance` | class_sample_imbalance, low_recall_class, small_object_error, tile_edge_error, class_confusion | 0.463 | 0.600 | 0.278 |
| 锡尖 | 108 | 27 | 0.825 | 0.852 | 0.906 | 0.685 | 22 | 4 | 5 | `low_contrast_error` | low_contrast_error, small_object_error, class_confusion | 0.188 | 0.800 | 0.150 |
| 锡膏 | 130 | 46 | 0.826 | 0.609 | 0.725 | 0.450 | 27 | 5 | 18 | `low_recall_class` | low_recall_class, texture_confusion, weak_localization, small_object_error, class_confusion | 0.277 | 0.800 | 0.221 |

## Evidence Details

- `OK2`: low_contrast=0.000, dark=0.000, small=0.000, edge=0.000, border_touch=0.000, weak_loc=0, cf_recovered=0, cf_rate=0.000, support=high
- `OK3`: low_contrast=1.000, dark=0.400, small=1.000, edge=0.800, border_touch=0.000, weak_loc=0, cf_recovered=3, cf_rate=0.600, support=high
- `加强筋打伤`: low_contrast=1.000, dark=1.000, small=1.000, edge=0.000, border_touch=0.000, weak_loc=0, cf_recovered=1, cf_rate=1.000, support=low
- `开裂`: low_contrast=0.000, dark=1.000, small=0.000, edge=0.000, border_touch=0.000, weak_loc=0, cf_recovered=0, cf_rate=0.000, support=very_low
- `油污`: low_contrast=0.692, dark=1.000, small=0.538, edge=0.462, border_touch=0.000, weak_loc=0, cf_recovered=1, cf_rate=0.091, support=medium
- `浅划伤`: low_contrast=0.667, dark=0.167, small=1.000, edge=0.167, border_touch=0.000, weak_loc=0, cf_recovered=0, cf_rate=0.000, support=medium
- `漏背锡`: low_contrast=0.000, dark=0.000, small=0.091, edge=0.091, border_touch=0.000, weak_loc=3, cf_recovered=1, cf_rate=0.100, support=medium_high
- `碰伤`: low_contrast=0.870, dark=0.576, small=0.957, edge=0.674, border_touch=0.043, weak_loc=12, cf_recovered=12, cf_rate=0.141, support=high
- `脏污`: low_contrast=0.100, dark=0.400, small=0.700, edge=0.250, border_touch=0.000, weak_loc=4, cf_recovered=5, cf_rate=0.250, support=medium_high
- `轮廓划伤`: low_contrast=0.771, dark=0.571, small=1.000, edge=0.629, border_touch=0.000, weak_loc=2, cf_recovered=2, cf_rate=0.061, support=high
- `锡丝残留`: low_contrast=0.000, dark=0.286, small=0.714, edge=1.000, border_touch=0.000, weak_loc=0, cf_recovered=0, cf_rate=0.000, support=medium
- `锡尖`: low_contrast=0.000, dark=0.000, small=1.000, edge=0.200, border_touch=0.000, weak_loc=0, cf_recovered=1, cf_rate=0.200, support=medium_high
- `锡膏`: low_contrast=0.000, dark=0.000, small=0.667, edge=0.167, border_touch=0.000, weak_loc=1, cf_recovered=0, cf_rate=0.000, support=medium_high
