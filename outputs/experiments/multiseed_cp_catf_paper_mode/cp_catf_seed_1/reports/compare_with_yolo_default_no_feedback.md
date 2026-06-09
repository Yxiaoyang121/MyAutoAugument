# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7220 | 0.7132 | 0.0088 |
| Recall | 0.7582 | 0.7600 | -0.0018 |
| mAP50 | 0.7777 | 0.7759 | 0.0018 |
| mAP50-95 | 0.5251 | 0.5241 | 0.0010 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7220 | 0.7262 | -0.0042 |
| Recall | 0.7582 | 0.6844 | 0.0739 |
| mAP50 | 0.7777 | 0.7616 | 0.0160 |
| mAP50-95 | 0.5251 | 0.5250 | 0.0000 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `false`
