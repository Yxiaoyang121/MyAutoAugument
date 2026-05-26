# Diagnosis Report: YOLO default + diagnosis_recall_safe

- Run dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\diagnosis_recall_safe`
- Precision/Recall/mAP50/mAP50-95: `0.6712/0.7659/0.7411/0.5044`
- TP/FP/FN: `735/269/149`
- Localization weak: `21`

## FN Types

| class_id | class_name | gt | tp | fp | fn | precision | recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 碰伤 | 269 | 191 | 72 | 72 | 0.7262 | 0.7100 |
| 9 | 轮廓划伤 | 84 | 52 | 24 | 28 | 0.6842 | 0.6190 |
| 12 | 锡膏 | 46 | 31 | 14 | 12 | 0.6889 | 0.6739 |
| 4 | 油污 | 18 | 8 | 22 | 10 | 0.2667 | 0.4444 |
| 8 | 脏污 | 42 | 28 | 12 | 10 | 0.7000 | 0.6667 |
| 6 | 漏背锡 | 46 | 35 | 18 | 8 | 0.6604 | 0.7609 |
| 5 | 浅划伤 | 13 | 6 | 4 | 7 | 0.6000 | 0.4615 |
| 2 | 加强筋打伤 | 8 | 7 | 0 | 1 | 1.0000 | 0.8750 |

## FP Types

| class_id | class_name | gt | tp | fp | fn | precision | recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 碰伤 | 269 | 191 | 72 | 72 | 0.7262 | 0.7100 |
| 1 | OK3 | 274 | 274 | 67 | 0 | 0.8035 | 1.0000 |
| 9 | 轮廓划伤 | 84 | 52 | 24 | 28 | 0.6842 | 0.6190 |
| 4 | 油污 | 18 | 8 | 22 | 10 | 0.2667 | 0.4444 |
| 0 | OK2 | 64 | 64 | 20 | 0 | 0.7619 | 1.0000 |
| 6 | 漏背锡 | 46 | 35 | 18 | 8 | 0.6604 | 0.7609 |
| 12 | 锡膏 | 46 | 31 | 14 | 12 | 0.6889 | 0.6739 |
| 8 | 脏污 | 42 | 28 | 12 | 10 | 0.7000 | 0.6667 |

## Quality And Position

- FN quality: `{"count": 149, "brightness_mean": 67.36324116931087, "contrast_std_mean": 18.647077794485533, "low_contrast_count": 100, "dark_count": 73, "bright_count": 2, "low_contrast_rate": 0.6711409395973155, "dark_rate": 0.4899328859060403, "bright_rate": 0.013422818791946308}`
- FP quality: `{"count": 269, "brightness_mean": 68.06257591764303, "contrast_std_mean": 20.1576669242175, "low_contrast_count": 175, "dark_count": 140, "bright_count": 0, "low_contrast_rate": 0.6505576208178439, "dark_rate": 0.5204460966542751, "bright_rate": 0.0}`
- FN size: `{"tiny": {"gt_count": 177, "tp_count": 121, "fn_count": 51, "recall": 0.6836158192090396, "fn_rate": 0.288135593220339}, "small": {"gt_count": 412, "tp_count": 320, "fn_count": 81, "recall": 0.7766990291262136, "fn_rate": 0.1966019417475728}, "medium": {"gt_count": 307, "tp_count": 286, "fn_count": 16, "recall": 0.9315960912052117, "fn_rate": 0.05211726384364821}, "large": {"gt_count": 9, "tp_count": 8, "fn_count": 1, "recall": 0.8888888888888888, "fn_rate": 0.1111111111111111}}`
- FN position: `{"left": {"gt_count": 283, "tp_count": 232, "fn_count": 47, "recall": 0.8197879858657244, "fn_rate": 0.16607773851590105}, "center": {"gt_count": 338, "tp_count": 258, "fn_count": 71, "recall": 0.7633136094674556, "fn_rate": 0.21005917159763313}, "right": {"gt_count": 284, "tp_count": 245, "fn_count": 31, "recall": 0.8626760563380281, "fn_rate": 0.10915492957746478}, "top": {"gt_count": 190, "tp_count": 150, "fn_count": 29, "recall": 0.7894736842105263, "fn_rate": 0.15263157894736842}, "middle": {"gt_count": 475, "tp_count": 390, "fn_count": 80, "recall": 0.8210526315789474, "fn_rate": 0.16842105263157894}, "bottom": {"gt_count": 240, "tp_count": 195, "fn_count": 40, "recall": 0.8125, "fn_rate": 0.16666666666666666}, "edge": {"gt_count": 178, "tp_count": 150, "fn_count": 24, "recall": 0.8426966292134831, "fn_rate": 0.1348314606741573}}`
