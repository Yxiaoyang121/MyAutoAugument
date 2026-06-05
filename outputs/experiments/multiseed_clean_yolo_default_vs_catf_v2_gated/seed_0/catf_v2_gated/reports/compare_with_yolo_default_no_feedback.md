# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7785 | 0.7846 | -0.0060 |
| Recall | 0.6697 | 0.6765 | -0.0068 |
| mAP50 | 0.7437 | 0.7347 | 0.0090 |
| mAP50-95 | 0.4895 | 0.4759 | 0.0136 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7785 | 0.7846 | -0.0060 |
| Recall | 0.6697 | 0.6765 | -0.0068 |
| mAP50 | 0.7437 | 0.7347 | 0.0090 |
| mAP50-95 | 0.4895 | 0.4759 | 0.0136 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `false`
