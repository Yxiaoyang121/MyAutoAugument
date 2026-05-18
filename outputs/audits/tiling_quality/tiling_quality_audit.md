# Tiling Quality Audit

- Generated: 2026-05-18T10:26:35
- Dataset: `E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full`
- Source dataset: `E:\TJGY\DataSet2_fixed`
- Tiled report JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full\tiled_dataset_report.json`
- Border margin: `2.0`

## Summary

- Total bboxes: 7465
- Visibility < 0.5: 1091
- Visibility < 0.7: 1956
- Visibility < 0.8: 2405
- Visibility < 0.9: 2874
- Bboxes touching tile boundary: 3249 (0.4352)
- Border-truncated bboxes: 3197 (0.4283)
- Severe truncated bboxes with visibility < 0.7: 1956 (0.2620)
- Debug visualizations: `E:\TJGY\MinPaper\MyAutoAugument\outputs\audits\tiling_quality\debug_truncated_bboxes` (50 images)
- Unmatched bboxes: 0

## Per Class

| class id | name | bbox count | truncated | truncated ratio | touches boundary | avg visibility | vis <0.5 | vis <0.7 | vis <0.8 | vis <0.9 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | OK | 1879 | 1697 | 0.9031 | 1699 | 0.6655 | 534 | 1049 | 1286 | 1531 |
| 1 | OK2 | 485 | 119 | 0.2454 | 120 | 0.8947 | 57 | 87 | 97 | 110 |
| 2 | OK3 | 1942 | 722 | 0.3718 | 734 | 0.8726 | 192 | 384 | 516 | 640 |
| 3 | 加强筋打伤 | 48 | 4 | 0.0833 | 5 | 0.9693 | 2 | 2 | 3 | 3 |
| 4 | 定位 | 151 | 151 | 1.0000 | 151 | 0.3101 | 151 | 151 | 151 | 151 |
| 5 | 开裂 | 21 | 2 | 0.0952 | 2 | 0.9957 | 0 | 0 | 0 | 0 |
| 6 | 油污 | 270 | 115 | 0.4259 | 120 | 0.8395 | 43 | 69 | 87 | 106 |
| 7 | 浅划伤 | 94 | 0 | 0.0000 | 2 | 1.0000 | 0 | 0 | 0 | 0 |
| 8 | 漏背锡 | 275 | 42 | 0.1527 | 42 | 0.9450 | 16 | 22 | 28 | 34 |
| 9 | 碰伤 | 1243 | 115 | 0.0925 | 132 | 0.9666 | 30 | 70 | 82 | 103 |
| 10 | 脏污 | 292 | 93 | 0.3185 | 96 | 0.8792 | 34 | 61 | 68 | 76 |
| 11 | 轮廓划伤 | 345 | 72 | 0.2087 | 74 | 0.9351 | 15 | 31 | 45 | 65 |
| 12 | 锡丝残留 | 76 | 32 | 0.4211 | 32 | 0.8853 | 6 | 13 | 18 | 25 |
| 13 | 锡尖 | 145 | 10 | 0.0690 | 13 | 0.9750 | 2 | 6 | 8 | 10 |
| 14 | 锡膏 | 199 | 23 | 0.1156 | 27 | 0.9558 | 9 | 11 | 16 | 20 |
