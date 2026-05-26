# Diagnosis Report: custom_yolo_like_base old control

- Run dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep`
- Precision/Recall/mAP50/mAP50-95: `0.6661/0.7490/0.7129/0.4415`
- TP/FP/FN: `722/319/170`
- Localization weak: `13`

## FN Types

| class_id | class_name | gt | tp | fp | fn | precision | recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 碰伤 | 269 | 191 | 107 | 71 | 0.6409 | 0.7100 |
| 9 | 轮廓划伤 | 84 | 45 | 23 | 36 | 0.6618 | 0.5357 |
| 8 | 脏污 | 42 | 24 | 22 | 18 | 0.5217 | 0.5714 |
| 12 | 锡膏 | 46 | 33 | 25 | 13 | 0.5690 | 0.7174 |
| 4 | 油污 | 18 | 8 | 30 | 10 | 0.2105 | 0.4444 |
| 6 | 漏背锡 | 46 | 34 | 12 | 9 | 0.7391 | 0.7391 |
| 5 | 浅划伤 | 13 | 8 | 4 | 5 | 0.6667 | 0.6154 |
| 2 | 加强筋打伤 | 8 | 4 | 1 | 4 | 0.8000 | 0.5000 |

## FP Types

| class_id | class_name | gt | tp | fp | fn | precision | recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 碰伤 | 269 | 191 | 107 | 71 | 0.6409 | 0.7100 |
| 1 | OK3 | 274 | 273 | 59 | 1 | 0.8223 | 0.9964 |
| 4 | 油污 | 18 | 8 | 30 | 10 | 0.2105 | 0.4444 |
| 12 | 锡膏 | 46 | 33 | 25 | 13 | 0.5690 | 0.7174 |
| 9 | 轮廓划伤 | 84 | 45 | 23 | 36 | 0.6618 | 0.5357 |
| 8 | 脏污 | 42 | 24 | 22 | 18 | 0.5217 | 0.5714 |
| 0 | OK2 | 64 | 64 | 15 | 0 | 0.8101 | 1.0000 |
| 6 | 漏背锡 | 46 | 34 | 12 | 9 | 0.7391 | 0.7391 |

## Quality And Position

- FN quality: `{"count": 170, "brightness_mean": 67.37568306277372, "contrast_std_mean": 19.67728647774212, "low_contrast_count": 115, "dark_count": 81, "bright_count": 2, "low_contrast_rate": 0.6764705882352942, "dark_rate": 0.4764705882352941, "bright_rate": 0.011764705882352941}`
- FP quality: `{"count": 319, "brightness_mean": 65.38897861778871, "contrast_std_mean": 19.195005382803142, "low_contrast_count": 220, "dark_count": 168, "bright_count": 0, "low_contrast_rate": 0.6896551724137931, "dark_rate": 0.5266457680250783, "bright_rate": 0.0}`
- FN size: `{"tiny": {"gt_count": 177, "tp_count": 127, "fn_count": 45, "recall": 0.7175141242937854, "fn_rate": 0.2542372881355932}, "small": {"gt_count": 412, "tp_count": 303, "fn_count": 104, "recall": 0.7354368932038835, "fn_rate": 0.2524271844660194}, "medium": {"gt_count": 307, "tp_count": 284, "fn_count": 20, "recall": 0.9250814332247557, "fn_rate": 0.06514657980456026}, "large": {"gt_count": 9, "tp_count": 8, "fn_count": 1, "recall": 0.8888888888888888, "fn_rate": 0.1111111111111111}}`
- FN position: `{"left": {"gt_count": 283, "tp_count": 226, "fn_count": 56, "recall": 0.7985865724381626, "fn_rate": 0.1978798586572438}, "center": {"gt_count": 338, "tp_count": 255, "fn_count": 79, "recall": 0.7544378698224852, "fn_rate": 0.23372781065088757}, "right": {"gt_count": 284, "tp_count": 241, "fn_count": 35, "recall": 0.8485915492957746, "fn_rate": 0.12323943661971831}, "top": {"gt_count": 190, "tp_count": 155, "fn_count": 31, "recall": 0.8157894736842105, "fn_rate": 0.1631578947368421}, "middle": {"gt_count": 475, "tp_count": 378, "fn_count": 90, "recall": 0.7957894736842105, "fn_rate": 0.18947368421052632}, "bottom": {"gt_count": 240, "tp_count": 189, "fn_count": 49, "recall": 0.7875, "fn_rate": 0.20416666666666666}, "edge": {"gt_count": 178, "tp_count": 149, "fn_count": 27, "recall": 0.8370786516853933, "fn_rate": 0.15168539325842698}}`
