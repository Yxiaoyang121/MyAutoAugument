# In-Loop YOLO Default Feedback Smoke

## Answers

- Single-run continuous training: `true`
- No stage restart: `true`
- Epoch sequence continuous: `true`
- Optimizer/scheduler/EMA managed by one trainer: `true`
- close_mosaic managed by official YOLO: `true`
- Feedback updated after epoch 5: `true`
- Mutable policy affects later epochs: `true`
- YOLO default augmentation remains enabled: `true`
- BBox/class legal: `true`

## Augmentation Stats

- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- Samples seen: `115050`
- Samples augmented: `33143`
- Invalid bbox count: `0`
- Class id OOB count: `0`

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| brightness | 115050 | 3580 | 111470 |
| clahe | 115050 | 5753 | 109297 |
| contrast | 115050 | 3500 | 111550 |
| cutout_safe | 115050 | 3357 | 111693 |
| gamma | 115050 | 5671 | 109379 |
| local_contrast | 115050 | 6847 | 108203 |
| sharpen_mild | 115050 | 9141 | 105909 |

## Continuity Evidence

- Train dirs: `1`
- Stage dirs: `0`
- args.yaml epochs: `50`
- args.yaml resume: `False`
- args.yaml close_mosaic: `10`
- results.csv epoch sequence: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_feedback\seed_2\catf_feedback\train\weights\best.pt`

## Final Metrics

- Precision: `0.6846`
- Recall: `0.7166`
- mAP50: `0.7608`
- mAP50-95: `0.4800`
