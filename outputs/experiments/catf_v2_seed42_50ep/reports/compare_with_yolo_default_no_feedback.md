# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7498 | 0.7132 | 0.0366 |
| Recall | 0.7257 | 0.7600 | -0.0343 |
| mAP50 | 0.7679 | 0.7759 | -0.0080 |
| mAP50-95 | 0.5212 | 0.5241 | -0.0029 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7498 | 0.7262 | 0.0236 |
| Recall | 0.7257 | 0.6844 | 0.0413 |
| mAP50 | 0.7679 | 0.7616 | 0.0062 |
| mAP50-95 | 0.5212 | 0.5250 | -0.0039 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `false`
