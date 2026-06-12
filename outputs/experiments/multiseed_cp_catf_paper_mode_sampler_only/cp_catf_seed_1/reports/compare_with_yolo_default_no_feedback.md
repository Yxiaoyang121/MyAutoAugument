# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7085 | 0.7132 | -0.0047 |
| Recall | 0.7258 | 0.7600 | -0.0342 |
| mAP50 | 0.7412 | 0.7759 | -0.0347 |
| mAP50-95 | 0.5112 | 0.5241 | -0.0129 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7085 | 0.7262 | -0.0177 |
| Recall | 0.7258 | 0.6844 | 0.0414 |
| mAP50 | 0.7412 | 0.7616 | -0.0204 |
| mAP50-95 | 0.5112 | 0.5250 | -0.0138 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
