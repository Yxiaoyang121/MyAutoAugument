# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7785 | 0.7132 | 0.0653 |
| Recall | 0.6697 | 0.7600 | -0.0904 |
| mAP50 | 0.7437 | 0.7759 | -0.0322 |
| mAP50-95 | 0.4895 | 0.5241 | -0.0345 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7785 | 0.7262 | 0.0523 |
| Recall | 0.6697 | 0.6844 | -0.0147 |
| mAP50 | 0.7437 | 0.7616 | -0.0180 |
| mAP50-95 | 0.4895 | 0.5250 | -0.0355 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
