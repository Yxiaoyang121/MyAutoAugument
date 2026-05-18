# Tiled YOLO Dataset Report

- Source dataset: E:\TJGY\DataSet2_fixed
- Output dataset: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full
- Data YAML: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full\data.yaml
- Debug tiling: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full\debug_tiling
- tile_size: 1024
- overlap: 0.2
- min_visibility: 0.3
- keep_empty_ratio: 0.1
- seed: 42
- Original train images: 461
- Original val images: 116
- Tiled train images: 4155
- Tiled val images: 1098
- Original bboxes: 3084
- Tiled bboxes: 7465
- Dropped bboxes in retained tiles: 22202
- Dropped bbox reasons in retained tiles: `{'below_min_visibility': 6212, 'outside_tile': 15990}`
- Empty tiles retained: 478
- Debug tile visualizations: 30
- data.yaml nc: 15
- Tiled class id min: 0
- Tiled class id max: 14
- Any class id >= nc: False
- Any class id out of range: False
- Chinese class names damaged: False
- Can be formal baseline dataset: True

## Per Split

| split | original images | tiled images | original bboxes | tiled bboxes | empty retained | dropped bboxes | dropped reasons | class id min | class id max | class id >= nc |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| train | 461 | 4155 | 2433 | 5865 | 378 | 17364 | `{'below_min_visibility': 4882, 'outside_tile': 12482}` | 0 | 14 | [] |
| val | 116 | 1098 | 651 | 1600 | 100 | 4838 | `{'below_min_visibility': 1330, 'outside_tile': 3508}` | 0 | 14 | [] |

## Data YAML

- nc: 15
- names quality: `{'missing_name_ids': [], 'empty_name_ids': [], 'replacement_char_name_ids': [], 'suspicious_mojibake_name_ids': [], 'contains_cjk_name_ids': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 'chinese_name_damage_found': False, 'has_problem': False}`

| class id | name |
|---:|---|
| 0 | OK |
| 1 | OK2 |
| 2 | OK3 |
| 3 | 加强筋打伤 |
| 4 | 定位 |
| 5 | 开裂 |
| 6 | 油污 |
| 7 | 浅划伤 |
| 8 | 漏背锡 |
| 9 | 碰伤 |
| 10 | 脏污 |
| 11 | 轮廓划伤 |
| 12 | 锡丝残留 |
| 13 | 锡尖 |
| 14 | 锡膏 |

## Per-Class Instances

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

## Dropped BBoxes

| scope | total | reasons |
|---|---:|---|
| retained tiles | 22202 | `{'below_min_visibility': 6212, 'outside_tile': 15990}` |
| all candidate tiles before empty sampling | 38795 | `{'below_min_visibility': 10630, 'outside_tile': 28165}` |
