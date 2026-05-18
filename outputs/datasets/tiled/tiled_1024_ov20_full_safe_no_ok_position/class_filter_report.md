# Class Filter Report

- Source dataset: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe
- Output dataset: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position
- Data YAML: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position\data.yaml
- Debug samples: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position\debug_samples
- Seed: 42
- Train keep_empty_ratio: 0.1
- Val keep_empty_ratio: 1.0
- Val empty policy: keep_all

## Filter Summary

- Deleted classes: `['OK', '定位']`
- Kept classes: `['OK2', 'OK3', '加强筋打伤', '开裂', '油污', '浅划伤', '漏背锡', '碰伤', '脏污', '轮廓划伤', '锡丝残留', '锡尖', '锡膏']`
- Source train/val images: 2452 / 677
- Filtered train/val images: 2301 / 677
- Source bbox count: 4269
- Filtered bbox count: 4087
- Train bbox count before/after: 3337 / 3182
- Val bbox count before/after: 932 / 905
- Train empty tiles retained: 210
- Val empty tiles retained: 83
- Class id out of range: False
- Chinese class names damaged: False
- Formal baseline ready: True

## Class Mapping

| old class id | old name | new class id | new name |
|---:|---|---:|---|
| 1 | OK2 | 0 | OK2 |
| 2 | OK3 | 1 | OK3 |
| 3 | 加强筋打伤 | 2 | 加强筋打伤 |
| 5 | 开裂 | 3 | 开裂 |
| 6 | 油污 | 4 | 油污 |
| 7 | 浅划伤 | 5 | 浅划伤 |
| 8 | 漏背锡 | 6 | 漏背锡 |
| 9 | 碰伤 | 7 | 碰伤 |
| 10 | 脏污 | 8 | 脏污 |
| 11 | 轮廓划伤 | 9 | 轮廓划伤 |
| 12 | 锡丝残留 | 10 | 锡丝残留 |
| 13 | 锡尖 | 11 | 锡尖 |
| 14 | 锡膏 | 12 | 锡膏 |

## Class Counts

| new class id | old class id | name | source bboxes | filtered bboxes | retention |
|---:|---:|---|---:|---:|---:|
| 0 | 1 | OK2 | 366 | 366 | 1.0000 |
| 1 | 2 | OK3 | 1220 | 1220 | 1.0000 |
| 2 | 3 | 加强筋打伤 | 44 | 44 | 1.0000 |
| 3 | 5 | 开裂 | 19 | 19 | 1.0000 |
| 4 | 6 | 油污 | 155 | 155 | 1.0000 |
| 5 | 7 | 浅划伤 | 94 | 94 | 1.0000 |
| 6 | 8 | 漏背锡 | 233 | 233 | 1.0000 |
| 7 | 9 | 碰伤 | 1129 | 1129 | 1.0000 |
| 8 | 10 | 脏污 | 199 | 199 | 1.0000 |
| 9 | 11 | 轮廓划伤 | 273 | 273 | 1.0000 |
| 10 | 12 | 锡丝残留 | 44 | 44 | 1.0000 |
| 11 | 13 | 锡尖 | 135 | 135 | 1.0000 |
| 12 | 14 | 锡膏 | 176 | 176 | 1.0000 |

## Class Map

{
  "old_to_new": {
    "1": 0,
    "2": 1,
    "3": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "10": 8,
    "11": 9,
    "12": 10,
    "13": 11,
    "14": 12
  },
  "new_to_old": {
    "0": 1,
    "1": 2,
    "2": 3,
    "3": 5,
    "4": 6,
    "5": 7,
    "6": 8,
    "7": 9,
    "8": 10,
    "9": 11,
    "10": 12,
    "11": 13,
    "12": 14
  }
}
