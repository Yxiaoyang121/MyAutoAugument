# In-Loop YOLO Default Feedback Smoke

## Answers

- Single-run continuous training: `true`
- No stage restart: `true`
- Epoch sequence continuous: `true`
- Optimizer/scheduler/EMA managed by one trainer: `true`
- close_mosaic managed by official YOLO: `true`
- Feedback updated after epoch 5: `true`
- Mutable policy affects later epochs: `false`
- YOLO default augmentation remains enabled: `true`
- BBox/class legal: `true`

## Augmentation Stats

- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- Samples seen: `0`
- Samples augmented: `0`
- Invalid bbox count: `0`
- Class id OOB count: `0`

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|

## Continuity Evidence

- Train dirs: `1`
- Stage dirs: `0`
- args.yaml epochs: `10`
- args.yaml resume: `False`
- args.yaml close_mosaic: `10`
- results.csv epoch sequence: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\diagnosis_only_inloop_control_10ep_smoke\train\weights\best.pt`

## Final Metrics

- Precision: `0.6952`
- Recall: `0.5176`
- mAP50: `0.5749`
- mAP50-95: `0.3701`
