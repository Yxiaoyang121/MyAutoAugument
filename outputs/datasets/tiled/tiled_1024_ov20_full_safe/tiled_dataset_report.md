# Tiled YOLO Dataset Report

- Source dataset: E:\TJGY\DataSet2_fixed
- Output dataset: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe
- Data YAML: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe\data.yaml
- Debug tiling: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe\debug_tiling
- tile_size: 1024
- overlap: 0.2
- min_visibility: 0.7
- large_object_min_visibility: 0.9
- drop_border_truncated: True
- border_margin: 2.0
- require_box_center_inside: True
- keep_empty_ratio: 0.1
- seed: 42
- Original train images: 461
- Original val images: 116
- Tiled train images: 2452
- Tiled val images: 677
- Original bboxes: 3084
- Tiled bboxes: 4269
- Dropped bboxes after intersection candidates: 13826
- Dropped bbox reasons: `{'below_min_area': 10, 'border_truncated': 3961, 'center_outside_tile': 9855}`
- Dropped for insufficient visibility: 13346
- Dropped for border truncation: 3961
- Dropped for center outside tile: 9855
- Empty tiles retained: 285
- Debug tile visualizations: 100
- Obvious half-target bbox remains: False
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
| train | 461 | 2452 | 2433 | 3337 | 223 | 10962 | `{'below_min_area': 10, 'border_truncated': 3144, 'center_outside_tile': 7808, 'outside_tile': 22196}` | 0 | 14 | [] |
| val | 116 | 677 | 651 | 932 | 62 | 2864 | `{'border_truncated': 817, 'center_outside_tile': 2047, 'outside_tile': 5969}` | 0 | 14 | [] |

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

| class id | name | original train | original val | original total | tiled train | tiled val | tiled total | dropped | retention rate | required visibility |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | OK | 446 | 114 | 560 | 155 | 27 | 182 | 3159 | 0.0545 | 0.9 |
| 1 | OK2 | 149 | 33 | 182 | 302 | 64 | 366 | 273 | 0.5728 | 0.9 |
| 2 | OK3 | 418 | 115 | 533 | 946 | 274 | 1220 | 940 | 0.5648 | 0.9 |
| 3 | 加强筋打伤 | 16 | 4 | 20 | 36 | 8 | 44 | 9 | 0.8302 | 0.7 |
| 4 | 定位 | 460 | 116 | 576 | 0 | 0 | 0 | 8640 | 0.0000 | 0.9 |
| 5 | 开裂 | 9 | 1 | 10 | 17 | 2 | 19 | 8 | 0.7037 | 0.7 |
| 6 | 油污 | 97 | 13 | 110 | 137 | 18 | 155 | 204 | 0.4318 | 0.7 |
| 7 | 浅划伤 | 33 | 6 | 39 | 81 | 13 | 94 | 0 | 1.0000 | 0.7 |
| 8 | 漏背锡 | 88 | 23 | 111 | 187 | 46 | 233 | 104 | 0.6914 | 0.7 |
| 9 | 碰伤 | 428 | 126 | 554 | 860 | 269 | 1129 | 150 | 0.8827 | 0.7 |
| 10 | 脏污 | 90 | 27 | 117 | 157 | 42 | 199 | 133 | 0.5994 | 0.7 |
| 11 | 轮廓划伤 | 69 | 31 | 100 | 189 | 84 | 273 | 105 | 0.7222 | 0.7 |
| 12 | 锡丝残留 | 20 | 7 | 27 | 32 | 12 | 44 | 52 | 0.4583 | 0.7 |
| 13 | 锡尖 | 50 | 12 | 62 | 108 | 27 | 135 | 18 | 0.8824 | 0.7 |
| 14 | 锡膏 | 60 | 23 | 83 | 130 | 46 | 176 | 31 | 0.8502 | 0.7 |

## Dropped BBoxes

| scope | total | reasons |
|---|---:|---|
| selected retained tiles | 5047 | `{'below_min_area': 4, 'border_truncated': 1929, 'center_outside_tile': 3114}` |
| all candidate tiles before empty sampling | 13826 | `{'below_min_area': 10, 'border_truncated': 3961, 'center_outside_tile': 9855}` |

## Retained BBox Quality

- Retained visibility below required: 0
- Retained border truncated: 0
- Retained border touching: 52
- Retained visibility thresholds: `{'0.5': 0, '0.7': 0, '0.8': 0, '0.9': 0}`
