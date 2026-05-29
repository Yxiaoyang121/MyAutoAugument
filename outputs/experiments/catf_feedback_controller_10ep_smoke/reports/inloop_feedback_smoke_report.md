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
- Samples seen: `23010`
- Samples augmented: `5744`
- Invalid bbox count: `0`
- Class id OOB count: `0`

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| brightness | 23010 | 486 | 22524 |
| clahe | 23010 | 861 | 22149 |
| contrast | 23010 | 460 | 22550 |
| cutout_safe | 23010 | 461 | 22549 |
| gamma | 23010 | 950 | 22060 |
| local_contrast | 23010 | 1384 | 21626 |
| sharpen_mild | 23010 | 1884 | 21126 |

## Continuity Evidence

- Train dirs: `1`
- Stage dirs: `0`
- args.yaml epochs: `10`
- args.yaml resume: `False`
- args.yaml close_mosaic: `10`
- results.csv epoch sequence: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_feedback_controller_10ep_smoke\train\weights\best.pt`

## Final Metrics

- Precision: `0.7572`
- Recall: `0.4420`
- mAP50: `0.6073`
- mAP50-95: `0.4204`
