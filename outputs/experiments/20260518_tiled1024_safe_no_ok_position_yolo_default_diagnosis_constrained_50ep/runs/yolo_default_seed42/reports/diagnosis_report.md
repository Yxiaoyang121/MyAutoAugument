# Diagnosis Report: YOLO default

- Run dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\yolo_default_seed42`
- Precision/Recall/mAP50/mAP50-95: `0.7132/0.7600/0.7759/0.5241`
- TP/FP/FN: `719/287/162`
- Localization weak: `24`

## FN Types

| class_id | class_name | gt | tp | fp | fn | precision | recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 碰伤 | 269 | 178 | 78 | 79 | 0.6953 | 0.6617 |
| 9 | 轮廓划伤 | 84 | 50 | 27 | 30 | 0.6494 | 0.5952 |
| 8 | 脏污 | 42 | 26 | 16 | 15 | 0.6190 | 0.6190 |
| 12 | 锡膏 | 46 | 33 | 13 | 13 | 0.7174 | 0.7174 |
| 4 | 油污 | 18 | 9 | 31 | 9 | 0.2250 | 0.5000 |
| 6 | 漏背锡 | 46 | 32 | 24 | 8 | 0.5714 | 0.6957 |
| 5 | 浅划伤 | 13 | 7 | 4 | 5 | 0.6364 | 0.5385 |
| 1 | OK3 | 274 | 273 | 66 | 1 | 0.8053 | 0.9964 |

## FP Types

| class_id | class_name | gt | tp | fp | fn | precision | recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 碰伤 | 269 | 178 | 78 | 79 | 0.6953 | 0.6617 |
| 1 | OK3 | 274 | 273 | 66 | 1 | 0.8053 | 0.9964 |
| 4 | 油污 | 18 | 9 | 31 | 9 | 0.2250 | 0.5000 |
| 9 | 轮廓划伤 | 84 | 50 | 27 | 30 | 0.6494 | 0.5952 |
| 6 | 漏背锡 | 46 | 32 | 24 | 8 | 0.5714 | 0.6957 |
| 0 | OK2 | 64 | 64 | 18 | 0 | 0.7805 | 1.0000 |
| 8 | 脏污 | 42 | 26 | 16 | 15 | 0.6190 | 0.6190 |
| 12 | 锡膏 | 46 | 33 | 13 | 13 | 0.7174 | 0.7174 |

## Quality And Position

- FN quality: `{"count": 162, "brightness_mean": 65.583122664246, "contrast_std_mean": 18.344309717065947, "low_contrast_count": 111, "dark_count": 88, "bright_count": 2, "low_contrast_rate": 0.6851851851851852, "dark_rate": 0.5432098765432098, "bright_rate": 0.012345679012345678}`
- FP quality: `{"count": 287, "brightness_mean": 65.74850661062855, "contrast_std_mean": 20.759474270737062, "low_contrast_count": 192, "dark_count": 151, "bright_count": 0, "low_contrast_rate": 0.6689895470383276, "dark_rate": 0.5261324041811847, "bright_rate": 0.0}`
- FN size: `{"tiny": {"gt_count": 177, "tp_count": 113, "fn_count": 58, "recall": 0.6384180790960452, "fn_rate": 0.327683615819209}, "small": {"gt_count": 412, "tp_count": 317, "fn_count": 83, "recall": 0.7694174757281553, "fn_rate": 0.20145631067961164}, "medium": {"gt_count": 307, "tp_count": 281, "fn_count": 20, "recall": 0.9153094462540716, "fn_rate": 0.06514657980456026}, "large": {"gt_count": 9, "tp_count": 8, "fn_count": 1, "recall": 0.8888888888888888, "fn_rate": 0.1111111111111111}}`
- FN position: `{"left": {"gt_count": 283, "tp_count": 226, "fn_count": 55, "recall": 0.7985865724381626, "fn_rate": 0.19434628975265017}, "center": {"gt_count": 338, "tp_count": 257, "fn_count": 69, "recall": 0.7603550295857988, "fn_rate": 0.20414201183431951}, "right": {"gt_count": 284, "tp_count": 236, "fn_count": 38, "recall": 0.8309859154929577, "fn_rate": 0.13380281690140844}, "top": {"gt_count": 190, "tp_count": 152, "fn_count": 26, "recall": 0.8, "fn_rate": 0.1368421052631579}, "middle": {"gt_count": 475, "tp_count": 382, "fn_count": 87, "recall": 0.8042105263157895, "fn_rate": 0.1831578947368421}, "bottom": {"gt_count": 240, "tp_count": 185, "fn_count": 49, "recall": 0.7708333333333334, "fn_rate": 0.20416666666666666}, "edge": {"gt_count": 178, "tp_count": 148, "fn_count": 24, "recall": 0.8314606741573034, "fn_rate": 0.1348314606741573}}`
