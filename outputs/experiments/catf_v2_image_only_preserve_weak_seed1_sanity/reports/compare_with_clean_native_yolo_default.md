# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7997 | 0.7132 | 0.0865 |
| Recall | 0.6974 | 0.7600 | -0.0626 |
| mAP50 | 0.7787 | 0.7759 | 0.0029 |
| mAP50-95 | 0.5169 | 0.5241 | -0.0071 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7997 | 0.7262 | 0.0735 |
| Recall | 0.6974 | 0.6844 | 0.0130 |
| mAP50 | 0.7787 | 0.7616 | 0.0171 |
| mAP50-95 | 0.5169 | 0.5250 | -0.0081 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `false`
