# Metric Consistency Audit

- YOLO precision: None
- YOLO recall: None
- Diagnosis precision: 0.7952
- Diagnosis recall: 0.7381
- TP/FP/FN: 668/172/215
- Precision delta: None
- Recall delta: None

## Conclusion
Differences are expected unless YOLO val and diagnosis use identical prediction files, thresholds, and matching rules.

## Likely Causes
- YOLO val reports aggregate metrics from Ultralytics validation, while diagnosis replays saved predict labels.
- Diagnosis uses a fixed predict confidence threshold conf=0.25 and NMS IoU=0.5.
- Diagnosis TP/FP/FN uses a single greedy class-aware match_iou=0.5.
- mAP50/mAP50-95 are area-under-curve metrics across confidence thresholds, so they are not expected to equal single-threshold TP/FP/FN.
- localization_weak detections are counted outside TP/FN in diagnosis global metrics and can change recall denominators.
