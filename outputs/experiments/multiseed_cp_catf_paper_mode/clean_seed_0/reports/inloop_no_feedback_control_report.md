# In-Loop No-Feedback YOLO Default Control

## Integrity

- Training success: `true`
- Single train run: `true`
- Stage restart count: `0`
- Epoch sequence continuous: `true`
- args.yaml epochs: `50`
- close_mosaic official/global: `true`
- YOLO default augmentation enabled: `true`
- Industrial augmentation enabled: `false`
- Feedback enabled: `false`
- Train image count: `2071`
- Fixed augmented dataset generated: `false`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode\clean_seed_0\train\weights\best.pt`

## Metrics

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7513 | 0.7132 | 0.0381 |
| Recall | 0.6763 | 0.7600 | -0.0837 |
| mAP50 | 0.7566 | 0.7759 | -0.0192 |
| mAP50-95 | 0.5114 | 0.5241 | -0.0126 |

## Answer

- Close to YOLO default reference: `false`
- If this control is not close, feedback 50ep should not be trusted because the entrypoint changed training behavior.
