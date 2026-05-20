# Baseline vs YOLO Default Augmentation vs DiagAug

## Overall

| Metric | No-YOLO-Aug baseline | YOLO default aug | DiagAug | YOLO default delta vs baseline | YOLO default delta vs DiagAug |
| --- | ---: | ---: | ---: | ---: | ---: |
| Precision | 0.690 | 0.785 | 0.686 | +0.095 | +0.099 |
| Recall | 0.615 | 0.676 | 0.688 | +0.061 | -0.012 |
| mAP50 | 0.669 | 0.735 | 0.717 | +0.066 | +0.018 |
| mAP50-95 | 0.434 | 0.476 | 0.496 | +0.042 | -0.020 |

## Augmentation Settings

| Group | Augmentation settings |
| --- | --- |
| No-YOLO-Aug baseline | `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0` |
| YOLO default aug | `hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=0.0 translate=0.1 scale=0.5 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.5 bgr=0.0 mosaic=1.0 mixup=0.0 cutmix=0.0 copy_paste=0.0 copy_paste_mode=flip auto_augment=randaugment erasing=0.4 close_mosaic=10` |
| DiagAug | Diagnosis-selected offline augmented dataset, YOLO built-in augmentations disabled during final train. See DiagAug report for selected policy details. |

## Per-Class Recall/AP50

| class id | class | baseline R | YOLO default R | DiagAug R | default Delta R vs base | default Delta R vs DiagAug | baseline AP50 | YOLO default AP50 | DiagAug AP50 | default Delta AP50 vs base | default Delta AP50 vs DiagAug |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | OK2 | 1.000 | 0.984 | 1.000 | -0.016 | -0.016 | 0.995 | 0.993 | 0.993 | -0.002 | +0.000 |
| 1 | OK3 | 0.978 | 0.967 | 0.989 | -0.011 | -0.022 | 0.970 | 0.994 | 0.973 | +0.024 | +0.021 |
| 2 | 加强筋打伤 | 0.875 | 0.875 | 0.625 | +0.000 | +0.250 | 0.955 | 0.949 | 0.747 | -0.006 | +0.202 |
| 3 | 开裂 | 0.000 | 1.000 | 1.000 | +1.000 | +0.000 | 0.111 | 0.663 | 0.995 | +0.552 | -0.332 |
| 4 | 油污 | 0.333 | 0.278 | 0.389 | -0.055 | -0.111 | 0.242 | 0.292 | 0.264 | +0.050 | +0.028 |
| 5 | 浅划伤 | 0.550 | 0.462 | 0.615 | -0.088 | -0.153 | 0.712 | 0.551 | 0.581 | -0.161 | -0.030 |
| 6 | 漏背锡 | 0.712 | 0.500 | 0.561 | -0.212 | -0.061 | 0.694 | 0.686 | 0.508 | -0.008 | +0.178 |
| 7 | 碰伤 | 0.617 | 0.565 | 0.546 | -0.052 | +0.019 | 0.617 | 0.723 | 0.640 | +0.106 | +0.083 |
| 8 | 脏污 | 0.381 | 0.500 | 0.476 | +0.119 | +0.024 | 0.440 | 0.517 | 0.443 | +0.077 | +0.074 |
| 9 | 轮廓划伤 | 0.511 | 0.571 | 0.563 | +0.060 | +0.008 | 0.717 | 0.747 | 0.666 | +0.030 | +0.081 |
| 10 | 锡丝残留 | 0.583 | 0.650 | 0.574 | +0.067 | +0.076 | 0.619 | 0.776 | 0.774 | +0.157 | +0.002 |
| 11 | 锡尖 | 0.852 | 0.773 | 0.888 | -0.079 | -0.115 | 0.906 | 0.951 | 0.974 | +0.045 | -0.023 |
| 12 | 锡膏 | 0.609 | 0.668 | 0.717 | +0.059 | -0.049 | 0.725 | 0.709 | 0.760 | -0.016 | -0.051 |

## Largest YOLO Default Changes

- Largest Recall gain vs baseline: `开裂` (class 3, delta_recall=+1.000)
- Largest Recall drop vs baseline: `漏背锡` (class 6, delta_recall=-0.212)
- Largest AP50 gain vs baseline: `开裂` (class 3, delta_ap50=+0.552)
- Largest AP50 drop vs baseline: `浅划伤` (class 5, delta_ap50=-0.161)
- Largest Recall gain vs DiagAug: `加强筋打伤` (class 2, delta_recall=+0.250)
- Largest Recall drop vs DiagAug: `浅划伤` (class 5, delta_recall=-0.153)
- Largest AP50 gain vs DiagAug: `加强筋打伤` (class 2, delta_ap50=+0.202)
- Largest AP50 drop vs DiagAug: `开裂` (class 3, delta_ap50=-0.332)
