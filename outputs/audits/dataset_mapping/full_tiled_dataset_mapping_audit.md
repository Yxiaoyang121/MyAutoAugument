# Full Tiled Dataset Mapping Audit

- Generated: 2026-05-18T09:35:45
- Original data.yaml: `E:\TJGY\DataSet2_fixed\data.yaml`
- Full tiled data.yaml: `E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full\data.yaml`
- Tiled dataset report JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full\tiled_dataset_report.json`

## Executive Findings

- Full tiled dataset: `True`
- Can be formal baseline dataset: `True`
- Tiled names match original: `True`
- Class id out of range found: `False`
- Chinese class names damaged: `False`
- Source train/val counts match original: `True`

## Dataset Summary

| dataset | nc | train images | val images | train labels | val labels | train bboxes | val bboxes | class id min | class id max | class id >= nc | names problem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 15 | 461 | 116 | 461 | 116 | 2433 | 651 | 0 | 14 | [] | False |
| tiled_full | 15 | 4155 | 1098 | 4155 | 1098 | 5865 | 1600 | 0 | 14 | [] | False |

## Tiling Summary

- tile_size: `1024`
- overlap: `0.2`
- min_visibility: `0.3`
- keep_empty_ratio: `0.1`
- original train/val images reported by builder: `461 / 116`
- tiled train/val images: `4155 / 1098`
- empty tiles retained: `478`
- dropped bboxes: `22202`
- dropped bbox reasons: `{'below_min_visibility': 6212, 'outside_tile': 15990}`

## Names

| class id | original name | tiled full name | match |
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
| 0 | OK | 446 | 114 | 560 | 1497 | 382 | 1879 |
| 1 | OK2 | 149 | 33 | 182 | 393 | 92 | 485 |
| 2 | OK3 | 418 | 115 | 533 | 1516 | 426 | 1942 |
| 3 | 加强筋打伤 | 16 | 4 | 20 | 40 | 8 | 48 |
| 4 | 定位 | 460 | 116 | 576 | 125 | 26 | 151 |
| 5 | 开裂 | 9 | 1 | 10 | 19 | 2 | 21 |
| 6 | 油污 | 97 | 13 | 110 | 234 | 36 | 270 |
| 7 | 浅划伤 | 33 | 6 | 39 | 81 | 13 | 94 |
| 8 | 漏背锡 | 88 | 23 | 111 | 222 | 53 | 275 |
| 9 | 碰伤 | 428 | 126 | 554 | 945 | 298 | 1243 |
| 10 | 脏污 | 90 | 27 | 117 | 232 | 60 | 292 |
| 11 | 轮廓划伤 | 69 | 31 | 100 | 244 | 101 | 345 |
| 12 | 锡丝残留 | 20 | 7 | 27 | 53 | 23 | 76 |
| 13 | 锡尖 | 50 | 12 | 62 | 116 | 29 | 145 |
| 14 | 锡膏 | 60 | 23 | 83 | 148 | 51 | 199 |

## Conclusion

- Full source coverage confirmed: `True`.
- Tiled `data.yaml` inherits original names: `True`.
- No class id >= nc or negative class id found: `True`.
- Chinese class names are intact: `True`.
- Formal baseline dataset readiness: `True`.
