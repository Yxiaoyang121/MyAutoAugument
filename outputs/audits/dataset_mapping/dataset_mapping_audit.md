# Dataset Mapping Audit

- Generated: 2026-05-17T23:24:43
- Original data.yaml: `E:\TJGY\DataSet2_fixed\data.yaml`
- Tiled smoke data.yaml: `E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_smoke\data.yaml`
- Baseline metrics JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260517_tiled_baseline_20epoch\reports\baseline_20epoch_metrics.json`

## Executive Findings

- Tiled smoke is full dataset: `False`
- Tiled smoke can be formal baseline dataset: `False`
- Current tiled names match original: `True`
- Class id out of range found: `False`
- blank-or-unrendered row count in existing 20 epoch metrics: `7`

The saved 20-epoch val log/metrics contain empty reported class names for non-ASCII classes. The label class IDs are in range and can be mapped back by val image/instance counts. The active tiled data.yaml has been repaired to match the original names; the blank rows are an artifact of the earlier corrupted/non-renderable tiled data.yaml or console/log parsing, not evidence of class IDs >= nc.

The low mAP is mainly driven by the smoke split and class imbalance: OK and OK3 have high AP50, while several evaluated defect classes have recall 0, very few train/val samples, or val samples with no train samples.

## Dataset Summary

| dataset | nc | train images | val images | train labels | val labels | train bboxes | val bboxes | class id min | class id max | class id >= nc | names problem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 15 | 461 | 116 | 461 | 116 | 2433 | 651 | 0 | 14 | [] | False |
| tiled_smoke | 15 | 107 | 86 | 107 | 86 | 232 | 143 | 0 | 14 | [] | False |

## Names

| class id | original name | tiled smoke name | match |
|---:|---|---|---|
| 0 | OK | OK | True |
| 1 | OK2 | OK2 | True |
| 2 | OK3 | OK3 | True |
| 3 | 加强筋打伤 | 加强筋打伤 | True |
| 4 | 定位 | 定位 | True |
| 5 | 开裂 | 开裂 | True |
| 6 | 油污 | 油污 | True |
| 7 | 浅划伤 | 浅划伤 | True |
| 8 | 漏背锡 | 漏背锡 | True |
| 9 | 碰伤 | 碰伤 | True |
| 10 | 脏污 | 脏污 | True |
| 11 | 轮廓划伤 | 轮廓划伤 | True |
| 12 | 锡丝残留 | 锡丝残留 | True |
| 13 | 锡尖 | 锡尖 | True |
| 14 | 锡膏 | 锡膏 | True |

## Per-Class BBox Distribution

| class id | name | original train | original val | original total | tiled train | tiled val | tiled total |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | OK | 446 | 114 | 560 | 6 | 21 | 27 |
| 1 | OK2 | 149 | 33 | 182 | 0 | 0 | 0 |
| 2 | OK3 | 418 | 115 | 533 | 48 | 40 | 88 |
| 3 | 加强筋打伤 | 16 | 4 | 20 | 0 | 0 | 0 |
| 4 | 定位 | 460 | 116 | 576 | 0 | 0 | 0 |
| 5 | 开裂 | 9 | 1 | 10 | 0 | 2 | 2 |
| 6 | 油污 | 97 | 13 | 110 | 96 | 21 | 117 |
| 7 | 浅划伤 | 33 | 6 | 39 | 0 | 0 | 0 |
| 8 | 漏背锡 | 88 | 23 | 111 | 5 | 6 | 11 |
| 9 | 碰伤 | 428 | 126 | 554 | 7 | 29 | 36 |
| 10 | 脏污 | 90 | 27 | 117 | 63 | 0 | 63 |
| 11 | 轮廓划伤 | 69 | 31 | 100 | 0 | 9 | 9 |
| 12 | 锡丝残留 | 20 | 7 | 27 | 3 | 3 | 6 |
| 13 | 锡尖 | 50 | 12 | 62 | 0 | 0 | 0 |
| 14 | 锡膏 | 60 | 23 | 83 | 4 | 12 | 16 |

## Tiled Smoke Assessment

- Is smoke: `True`
- Is full: `False`
- tile_size: `1024`
- overlap: `0.2`
- min_visibility: `0.3`
- keep_empty_ratio: `0.1`

- dataset path contains smoke: True
- tiled report source_image_count=16
- original image count=577
- dataset_summary.md records Is smoke: true and the per-split source cap was 8 images

## Current 20 Epoch Per-Class Metrics

| class id | name | reported class | train instances | val instances | precision | recall | AP50 | AP50-95 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | OK | OK | 6 | 21 | 0.951 | 0.857 | 0.964 | 0.841 |
| 2 | OK3 | OK3 | 48 | 40 | 0.939 | 0.774 | 0.919 | 0.626 |
| 5 | 开裂 | blank-or-unrendered | 0 | 2 | 1 | 0 | 0.00147 | 0.00105 |
| 6 | 油污 | blank-or-unrendered | 96 | 21 | 0.563 | 0.286 | 0.289 | 0.145 |
| 8 | 漏背锡 | blank-or-unrendered | 5 | 6 | 0 | 0 | 0.0255 | 0.0106 |
| 9 | 碰伤 | blank-or-unrendered | 7 | 29 | 1 | 0 | 0.00235 | 0.0013 |
| 11 | 轮廓划伤 | blank-or-unrendered | 0 | 9 | 1 | 0 | 0 | 0 |
| 12 | 锡丝残留 | blank-or-unrendered | 3 | 3 | 1 | 0 | 0.0101 | 0.00404 |
| 14 | 锡膏 | blank-or-unrendered | 4 | 12 | 1 | 0 | 0.00873 | 0.00367 |

### Classes With High AP50

- class_id=0, name=OK, train=6, val=21, recall=0.857, AP50=0.964, AP50-95=0.841
- class_id=2, name=OK3, train=48, val=40, recall=0.774, AP50=0.919, AP50-95=0.626

### Classes With Recall 0

- class_id=5, name=开裂, train=0, val=2, recall=0, AP50=0.00147, AP50-95=0.00105
- class_id=8, name=漏背锡, train=5, val=6, recall=0, AP50=0.0255, AP50-95=0.0106
- class_id=9, name=碰伤, train=7, val=29, recall=0, AP50=0.00235, AP50-95=0.0013
- class_id=11, name=轮廓划伤, train=0, val=9, recall=0, AP50=0, AP50-95=0
- class_id=12, name=锡丝残留, train=3, val=3, recall=0, AP50=0.0101, AP50-95=0.00404
- class_id=14, name=锡膏, train=4, val=12, recall=0, AP50=0.00873, AP50-95=0.00367

### Classes With Too Few Samples

- class_id=0, name=OK, train=6, val=21, recall=0.857, AP50=0.964, AP50-95=0.841
- class_id=5, name=开裂, train=0, val=2, recall=0, AP50=0.00147, AP50-95=0.00105
- class_id=8, name=漏背锡, train=5, val=6, recall=0, AP50=0.0255, AP50-95=0.0106
- class_id=9, name=碰伤, train=7, val=29, recall=0, AP50=0.00235, AP50-95=0.0013
- class_id=11, name=轮廓划伤, train=0, val=9, recall=0, AP50=0, AP50-95=0
- class_id=12, name=锡丝残留, train=3, val=3, recall=0, AP50=0.0101, AP50-95=0.00404
- class_id=14, name=锡膏, train=4, val=12, recall=0, AP50=0.00873, AP50-95=0.00367

### Primary mAP Drag

- class_id=5, name=开裂, train=0, val=2, recall=0, AP50=0.00147, AP50-95=0.00105
- class_id=8, name=漏背锡, train=5, val=6, recall=0, AP50=0.0255, AP50-95=0.0106
- class_id=9, name=碰伤, train=7, val=29, recall=0, AP50=0.00235, AP50-95=0.0013
- class_id=11, name=轮廓划伤, train=0, val=9, recall=0, AP50=0, AP50-95=0
- class_id=12, name=锡丝残留, train=3, val=3, recall=0, AP50=0.0101, AP50-95=0.00404
- class_id=14, name=锡膏, train=4, val=12, recall=0, AP50=0.00873, AP50-95=0.00367

## Conclusion

- The active tiled `data.yaml` now fully inherits the original class names.
- No class id >= nc was found in the original or tiled smoke labels.
- The existing low mAP should not be interpreted as a full-dataset baseline because this is a smoke subset.
- The existing blank-or-unrendered rows are a logging/name-rendering artifact from the prior corrupted tiled `data.yaml`, while the numeric class IDs remain recoverable and in range.
