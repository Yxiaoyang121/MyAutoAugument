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
- Samples augmented: `2974`
- Invalid bbox count: `0`
- Class id OOB count: `0`

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| clahe | 11505 | 570 | 10935 |
| cutout_safe | 11505 | 350 | 11155 |
| gamma | 11505 | 562 | 10943 |
| local_contrast | 11505 | 950 | 10555 |
| sharpen_mild | 11505 | 902 | 10603 |

## Continuity Evidence

- Train dirs: `1`
- Stage dirs: `0`
- args.yaml epochs: `10`
- args.yaml resume: `False`
- args.yaml close_mosaic: `10`
- results.csv epoch sequence: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_inloop_feedback_10ep_smoke\train\weights\best.pt`

## Final Metrics

- Precision: `0.5798`
- Recall: `0.5304`
- mAP50: `0.5480`
- mAP50-95: `0.3675`
