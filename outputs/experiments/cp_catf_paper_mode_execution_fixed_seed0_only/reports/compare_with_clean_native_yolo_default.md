# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7399 | 0.7132 | 0.0267 |
| Recall | 0.6763 | 0.7600 | -0.0837 |
| mAP50 | 0.7593 | 0.7759 | -0.0165 |
| mAP50-95 | 0.5028 | 0.5241 | -0.0213 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7399 | 0.7262 | 0.0137 |
| Recall | 0.6763 | 0.6844 | -0.0081 |
| mAP50 | 0.7593 | 0.7616 | -0.0023 |
| mAP50-95 | 0.5028 | 0.5250 | -0.0223 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
