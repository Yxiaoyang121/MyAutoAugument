# In-Loop No-Feedback YOLO Default Control

## Integrity

- Training success: `true`
- Single train run: `true`
- Stage restart count: `0`
- Epoch sequence continuous: `true`
- args.yaml epochs: `1`
- close_mosaic official/global: `true`
- YOLO default augmentation enabled: `true`
- Industrial augmentation enabled: `false`
- Feedback enabled: `false`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_noop_parity_smoke\clean_native_1ep\train\weights\best.pt`

## Metrics

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.8277 | 0.7725 | 0.0552 |
| Recall | 0.1977 | 0.6477 | -0.4499 |
| mAP50 | 0.2127 | 0.7542 | -0.5415 |
| mAP50-95 | 0.1464 | 0.4799 | -0.3335 |

## Answer

- Close to YOLO default reference: `false`
- If this control is not close, feedback 50ep should not be trusted because the entrypoint changed training behavior.
