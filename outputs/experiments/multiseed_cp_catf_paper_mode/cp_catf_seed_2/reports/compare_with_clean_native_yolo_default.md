# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6290 | 0.7132 | -0.0842 |
| Recall | 0.6385 | 0.7600 | -0.1215 |
| mAP50 | 0.6590 | 0.7759 | -0.1169 |
| mAP50-95 | 0.4381 | 0.5241 | -0.0860 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6290 | 0.7262 | -0.0973 |
| Recall | 0.6385 | 0.6844 | -0.0458 |
| mAP50 | 0.6590 | 0.7616 | -0.1026 |
| mAP50-95 | 0.4381 | 0.5250 | -0.0869 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
