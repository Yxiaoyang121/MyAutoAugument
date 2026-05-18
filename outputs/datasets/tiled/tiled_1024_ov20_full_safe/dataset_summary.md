# Dataset Summary

- Source dataset: E:\TJGY\DataSet2_fixed
- Output dataset: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe
- Original train/val images: 461 / 116
- Tiled train/val images: 2452 / 677
- Original bboxes: 3084
- Tiled bboxes: 4269
- Empty tiles retained: 285
- Dropped bboxes after intersection candidates: 13826
- Dropped bbox reasons: `{'below_min_area': 10, 'border_truncated': 3961, 'center_outside_tile': 9855}`
- Dropped for insufficient visibility: 13346
- Dropped for border truncation: 3961
- Obvious half-target bbox remains: False
- data.yaml nc: 15
- data.yaml names: `['OK', 'OK2', 'OK3', '加强筋打伤', '定位', '开裂', '油污', '浅划伤', '漏背锡', '碰伤', '脏污', '轮廓划伤', '锡丝残留', '锡尖', '锡膏']`
- Tiled class id min/max: 0 / 14
- Class id >= nc: []
- Class id out of range found: False
- Chinese class names damaged: False
- Can be formal baseline dataset: True

## Per-Class Tiled BBoxes

| class id | name | tiled train | tiled val | tiled total | dropped | retention rate |
|---:|---|---:|---:|---:|---:|---:|
| 0 | OK | 155 | 27 | 182 | 3159 | 0.0545 |
| 1 | OK2 | 302 | 64 | 366 | 273 | 0.5728 |
| 2 | OK3 | 946 | 274 | 1220 | 940 | 0.5648 |
| 3 | 加强筋打伤 | 36 | 8 | 44 | 9 | 0.8302 |
| 4 | 定位 | 0 | 0 | 0 | 8640 | 0.0000 |
| 5 | 开裂 | 17 | 2 | 19 | 8 | 0.7037 |
| 6 | 油污 | 137 | 18 | 155 | 204 | 0.4318 |
| 7 | 浅划伤 | 81 | 13 | 94 | 0 | 1.0000 |
| 8 | 漏背锡 | 187 | 46 | 233 | 104 | 0.6914 |
| 9 | 碰伤 | 860 | 269 | 1129 | 150 | 0.8827 |
| 10 | 脏污 | 157 | 42 | 199 | 133 | 0.5994 |
| 11 | 轮廓划伤 | 189 | 84 | 273 | 105 | 0.7222 |
| 12 | 锡丝残留 | 32 | 12 | 44 | 52 | 0.4583 |
| 13 | 锡尖 | 108 | 27 | 135 | 18 | 0.8824 |
| 14 | 锡膏 | 130 | 46 | 176 | 31 | 0.8502 |
