# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6324 | 0.7132 | -0.0808 |
| Recall | 0.4960 | 0.7600 | -0.2640 |
| mAP50 | 0.5659 | 0.7759 | -0.2100 |
| mAP50-95 | 0.3750 | 0.5241 | -0.1491 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6324 | 0.7262 | -0.0938 |
| Recall | 0.4960 | 0.6844 | -0.1884 |
| mAP50 | 0.5659 | 0.7616 | -0.1957 |
| mAP50-95 | 0.3750 | 0.5250 | -0.1501 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
