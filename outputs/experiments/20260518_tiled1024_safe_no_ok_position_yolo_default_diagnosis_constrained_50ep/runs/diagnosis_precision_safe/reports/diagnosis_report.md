# Diagnosis Report: YOLO default + diagnosis_precision_safe

- Run dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\diagnosis_precision_safe`
- Precision/Recall/mAP50/mAP50-95: `0.6647/0.7493/0.7557/0.5055`
- TP/FP/FN: `725/305/156`
- Localization weak: `24`

## FN Types

| class_id | class_name | gt | tp | fp | fn | precision | recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 碰伤 | 269 | 184 | 76 | 75 | 0.7077 | 0.6840 |
| 9 | 轮廓划伤 | 84 | 56 | 38 | 23 | 0.5957 | 0.6667 |
| 8 | 脏污 | 42 | 22 | 12 | 16 | 0.6471 | 0.5238 |
| 6 | 漏背锡 | 46 | 33 | 26 | 10 | 0.5593 | 0.7174 |
| 12 | 锡膏 | 46 | 34 | 12 | 10 | 0.7391 | 0.7391 |
| 4 | 油污 | 18 | 9 | 24 | 9 | 0.2727 | 0.5000 |
| 5 | 浅划伤 | 13 | 6 | 12 | 7 | 0.3333 | 0.4615 |
| 10 | 锡丝残留 | 12 | 9 | 5 | 3 | 0.6429 | 0.7500 |

## FP Types

| class_id | class_name | gt | tp | fp | fn | precision | recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 碰伤 | 269 | 184 | 76 | 75 | 0.7077 | 0.6840 |
| 1 | OK3 | 274 | 273 | 65 | 1 | 0.8077 | 0.9964 |
| 9 | 轮廓划伤 | 84 | 56 | 38 | 23 | 0.5957 | 0.6667 |
| 6 | 漏背锡 | 46 | 33 | 26 | 10 | 0.5593 | 0.7174 |
| 4 | 油污 | 18 | 9 | 24 | 9 | 0.2727 | 0.5000 |
| 0 | OK2 | 64 | 64 | 21 | 0 | 0.7529 | 1.0000 |
| 5 | 浅划伤 | 13 | 6 | 12 | 7 | 0.3333 | 0.4615 |
| 8 | 脏污 | 42 | 22 | 12 | 16 | 0.6471 | 0.5238 |

## Quality And Position

- FN quality: `{"count": 156, "brightness_mean": 65.0102008130286, "contrast_std_mean": 19.94031999114369, "low_contrast_count": 97, "dark_count": 81, "bright_count": 2, "low_contrast_rate": 0.6217948717948718, "dark_rate": 0.5192307692307693, "bright_rate": 0.01282051282051282}`
- FP quality: `{"count": 305, "brightness_mean": 68.61197828111945, "contrast_std_mean": 20.74982631873731, "low_contrast_count": 202, "dark_count": 155, "bright_count": 0, "low_contrast_rate": 0.6622950819672131, "dark_rate": 0.5081967213114754, "bright_rate": 0.0}`
- FN size: `{"tiny": {"gt_count": 177, "tp_count": 119, "fn_count": 54, "recall": 0.672316384180791, "fn_rate": 0.3050847457627119}, "small": {"gt_count": 412, "tp_count": 320, "fn_count": 78, "recall": 0.7766990291262136, "fn_rate": 0.18932038834951456}, "medium": {"gt_count": 307, "tp_count": 278, "fn_count": 23, "recall": 0.9055374592833876, "fn_rate": 0.0749185667752443}, "large": {"gt_count": 9, "tp_count": 8, "fn_count": 1, "recall": 0.8888888888888888, "fn_rate": 0.1111111111111111}}`
- FN position: `{"left": {"gt_count": 283, "tp_count": 234, "fn_count": 45, "recall": 0.8268551236749117, "fn_rate": 0.15901060070671377}, "center": {"gt_count": 338, "tp_count": 249, "fn_count": 79, "recall": 0.7366863905325444, "fn_rate": 0.23372781065088757}, "right": {"gt_count": 284, "tp_count": 242, "fn_count": 32, "recall": 0.852112676056338, "fn_rate": 0.11267605633802817}, "top": {"gt_count": 190, "tp_count": 146, "fn_count": 32, "recall": 0.7684210526315789, "fn_rate": 0.16842105263157894}, "middle": {"gt_count": 475, "tp_count": 387, "fn_count": 83, "recall": 0.8147368421052632, "fn_rate": 0.17473684210526316}, "bottom": {"gt_count": 240, "tp_count": 192, "fn_count": 41, "recall": 0.8, "fn_rate": 0.17083333333333334}, "edge": {"gt_count": 178, "tp_count": 149, "fn_count": 23, "recall": 0.8370786516853933, "fn_rate": 0.12921348314606743}}`
