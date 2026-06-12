# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7588 | 0.7132 | 0.0456 |
| Recall | 0.6878 | 0.7600 | -0.0723 |
| mAP50 | 0.7779 | 0.7759 | 0.0021 |
| mAP50-95 | 0.5204 | 0.5241 | -0.0037 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7588 | 0.7262 | 0.0326 |
| Recall | 0.6878 | 0.6844 | 0.0034 |
| mAP50 | 0.7779 | 0.7616 | 0.0163 |
| mAP50-95 | 0.5204 | 0.5250 | -0.0047 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `false`
