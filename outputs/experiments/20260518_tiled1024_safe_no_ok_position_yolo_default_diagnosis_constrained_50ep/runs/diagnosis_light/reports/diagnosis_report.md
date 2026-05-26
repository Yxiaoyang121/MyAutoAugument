# Diagnosis Report: YOLO default + diagnosis_light

- Run dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\diagnosis_light`
- Precision/Recall/mAP50/mAP50-95: `0.7343/0.6862/0.7555/0.4993`
- TP/FP/FN: `720/278/172`
- Localization weak: `13`

## FN Types

| class_id | class_name | gt | tp | fp | fn | precision | recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 碰伤 | 269 | 188 | 75 | 75 | 0.7148 | 0.6989 |
| 9 | 轮廓划伤 | 84 | 56 | 45 | 28 | 0.5545 | 0.6667 |
| 8 | 脏污 | 42 | 18 | 7 | 24 | 0.7200 | 0.4286 |
| 4 | 油污 | 18 | 4 | 14 | 14 | 0.2222 | 0.2222 |
| 12 | 锡膏 | 46 | 32 | 5 | 14 | 0.8649 | 0.6957 |
| 5 | 浅划伤 | 13 | 5 | 8 | 8 | 0.3846 | 0.3846 |
| 6 | 漏背锡 | 46 | 34 | 24 | 5 | 0.5862 | 0.7391 |
| 1 | OK3 | 274 | 272 | 66 | 2 | 0.8047 | 0.9927 |

## FP Types

| class_id | class_name | gt | tp | fp | fn | precision | recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 碰伤 | 269 | 188 | 75 | 75 | 0.7148 | 0.6989 |
| 1 | OK3 | 274 | 272 | 66 | 2 | 0.8047 | 0.9927 |
| 9 | 轮廓划伤 | 84 | 56 | 45 | 28 | 0.5545 | 0.6667 |
| 6 | 漏背锡 | 46 | 34 | 24 | 5 | 0.5862 | 0.7391 |
| 0 | OK2 | 64 | 64 | 18 | 0 | 0.7805 | 1.0000 |
| 4 | 油污 | 18 | 4 | 14 | 14 | 0.2222 | 0.2222 |
| 5 | 浅划伤 | 13 | 5 | 8 | 8 | 0.3846 | 0.3846 |
| 8 | 脏污 | 42 | 18 | 7 | 24 | 0.7200 | 0.4286 |

## Quality And Position

- FN quality: `{"count": 172, "brightness_mean": 64.64445433347463, "contrast_std_mean": 19.174223484573535, "low_contrast_count": 110, "dark_count": 92, "bright_count": 2, "low_contrast_rate": 0.6395348837209303, "dark_rate": 0.5348837209302325, "bright_rate": 0.011627906976744186}`
- FP quality: `{"count": 278, "brightness_mean": 66.76817198880474, "contrast_std_mean": 19.756498300309953, "low_contrast_count": 195, "dark_count": 146, "bright_count": 0, "low_contrast_rate": 0.7014388489208633, "dark_rate": 0.5251798561151079, "bright_rate": 0.0}`
- FN size: `{"tiny": {"gt_count": 177, "tp_count": 117, "fn_count": 55, "recall": 0.6610169491525424, "fn_rate": 0.3107344632768362}, "small": {"gt_count": 412, "tp_count": 316, "fn_count": 95, "recall": 0.7669902912621359, "fn_rate": 0.23058252427184467}, "medium": {"gt_count": 307, "tp_count": 281, "fn_count": 19, "recall": 0.9153094462540716, "fn_rate": 0.06188925081433225}, "large": {"gt_count": 9, "tp_count": 6, "fn_count": 3, "recall": 0.6666666666666666, "fn_rate": 0.3333333333333333}}`
- FN position: `{"left": {"gt_count": 283, "tp_count": 230, "fn_count": 52, "recall": 0.8127208480565371, "fn_rate": 0.18374558303886926}, "center": {"gt_count": 338, "tp_count": 249, "fn_count": 85, "recall": 0.7366863905325444, "fn_rate": 0.2514792899408284}, "right": {"gt_count": 284, "tp_count": 241, "fn_count": 35, "recall": 0.8485915492957746, "fn_rate": 0.12323943661971831}, "top": {"gt_count": 190, "tp_count": 158, "fn_count": 26, "recall": 0.8315789473684211, "fn_rate": 0.1368421052631579}, "middle": {"gt_count": 475, "tp_count": 378, "fn_count": 92, "recall": 0.7957894736842105, "fn_rate": 0.1936842105263158}, "bottom": {"gt_count": 240, "tp_count": 184, "fn_count": 54, "recall": 0.7666666666666667, "fn_rate": 0.225}, "edge": {"gt_count": 178, "tp_count": 147, "fn_count": 29, "recall": 0.8258426966292135, "fn_rate": 0.16292134831460675}}`
