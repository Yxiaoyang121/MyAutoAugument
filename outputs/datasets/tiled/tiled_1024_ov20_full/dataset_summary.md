# Dataset Summary

- Source dataset: E:\TJGY\DataSet2_fixed
- Output dataset: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full
- Original train/val images: 461 / 116
- Tiled train/val images: 4155 / 1098
- Original bboxes: 3084
- Tiled bboxes: 7465
- Empty tiles retained: 478
- Dropped bboxes in retained tiles: 22202
- Dropped bbox reasons: `{'below_min_visibility': 6212, 'outside_tile': 15990}`
- data.yaml nc: 15
- data.yaml names: `['OK', 'OK2', 'OK3', '加强筋打伤', '定位', '开裂', '油污', '浅划伤', '漏背锡', '碰伤', '脏污', '轮廓划伤', '锡丝残留', '锡尖', '锡膏']`
- Tiled class id min/max: 0 / 14
- Class id >= nc: []
- Class id out of range found: False
- Chinese class names damaged: False
- Can be formal baseline dataset: True

## Per-Class Tiled BBoxes

| class id | name | tiled train | tiled val | tiled total |
|---:|---|---:|---:|---:|
| 0 | OK | 1497 | 382 | 1879 |
| 1 | OK2 | 393 | 92 | 485 |
| 2 | OK3 | 1516 | 426 | 1942 |
| 3 | 加强筋打伤 | 40 | 8 | 48 |
| 4 | 定位 | 125 | 26 | 151 |
| 5 | 开裂 | 19 | 2 | 21 |
| 6 | 油污 | 234 | 36 | 270 |
| 7 | 浅划伤 | 81 | 13 | 94 |
| 8 | 漏背锡 | 222 | 53 | 275 |
| 9 | 碰伤 | 945 | 298 | 1243 |
| 10 | 脏污 | 232 | 60 | 292 |
| 11 | 轮廓划伤 | 244 | 101 | 345 |
| 12 | 锡丝残留 | 53 | 23 | 76 |
| 13 | 锡尖 | 116 | 29 | 145 |
| 14 | 锡膏 | 148 | 51 | 199 |
