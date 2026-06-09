# Paper-Mode CP-CATF Probe Split Report

## Summary

- Source data: `E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position\data.yaml`
- Data yaml: `E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position_paper_probe\data.yaml`
- Probe yaml: `E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position_paper_probe\probe.yaml`
- Split seed: `2026`
- Original train images: `2301`
- Train core images: `2071`
- Probe images: `230`
- Final val images: `677`
- Classes missing in probe: `[]`
- Classes too small in train_core: `[]`
- Train/probe/val no overlap: `true`

## Per-Class BBox Counts

| class_id | name | train_core | probe | val | original_train |
|---:|---|---:|---:|---:|---:|
| 0 | OK2 | 264 | 38 | 64 | 302 |
| 1 | OK3 | 844 | 102 | 274 | 946 |
| 2 | 加强筋打伤 | 30 | 6 | 8 | 36 |
| 3 | 开裂 | 14 | 3 | 2 | 17 |
| 4 | 油污 | 120 | 17 | 18 | 137 |
| 5 | 浅划伤 | 67 | 14 | 13 | 81 |
| 6 | 漏背锡 | 155 | 32 | 46 | 187 |
| 7 | 碰伤 | 727 | 133 | 269 | 860 |
| 8 | 脏污 | 129 | 28 | 42 | 157 |
| 9 | 轮廓划伤 | 159 | 30 | 84 | 189 |
| 10 | 锡丝残留 | 26 | 6 | 12 | 32 |
| 11 | 锡尖 | 96 | 12 | 27 | 108 |
| 12 | 锡膏 | 109 | 21 | 46 | 130 |
