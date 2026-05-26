# YOLO Default With Diagnosis Feedback Smoke

- Base augmentation: Ultralytics YOLO default augmentation remains enabled.
- Custom YOLO-like `mosaic4` / `randaugment_like`: `false`.
- Fixed augmented dataset generated: `false`.
- Stage count: `2`
- Policy history updates: `1`

## Final Metrics

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.5259 | 0.3311 | 0.3109 | 0.1961 |

## Constraint Scoring

- Accepted final strategy: `false`
- Failures: `precision_drop_gt_0.01, map50_drop_gt_0.01, map50_95_drop_gt_0.01`

## Stage Summary

| stage | epochs | P | R | mAP50 | mAP50-95 | adjustments |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.5863 | 0.1715 | 0.1771 | 0.1157 | 25 |
| 1 | 1 | 0.5259 | 0.3311 | 0.3109 | 0.1961 | 0 |

## Reference Comparison

| reference | P | R | mAP50 | mAP50-95 | dP | dR | d_mAP50 | d_mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO default seed=42 | 0.7132 | 0.7600 | 0.7759 | 0.5241 | -0.1873 | -0.4289 | -0.4650 | -0.3280 |
| baseline no aug | 0.6900 | 0.6150 | 0.6690 | 0.4340 | -0.1641 | -0.2839 | -0.3581 | -0.2379 |
| offline DiagAug | 0.6860 | 0.6880 | 0.7170 | 0.4960 | -0.1601 | -0.3569 | -0.4061 | -0.2999 |
| offline random external | 0.7500 | 0.6680 | 0.7340 | 0.5010 | -0.2241 | -0.3369 | -0.4231 | -0.3049 |
