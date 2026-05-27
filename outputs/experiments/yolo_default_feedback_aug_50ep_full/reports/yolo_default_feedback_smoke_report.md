# YOLO Default With Diagnosis Feedback 50 Epoch

- Base augmentation: Ultralytics YOLO default augmentation remains enabled.
- Custom YOLO-like `mosaic4` / `randaugment_like`: `false`.
- Fixed augmented dataset generated: `false`.
- Stage count: `10`
- Policy history updates: `9`
- Feedback 50 epoch completed: `true`

## Final Metrics

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.7712 | 0.6689 | 0.7439 | 0.4993 |

## Constraint Scoring

- Accepted final strategy: `false`
- Failures: `map50_drop_gt_0.01, map50_95_drop_gt_0.01`
- Delta vs YOLO default seed=42: P `+0.0580`, R `-0.0911`, mAP50 `-0.0320`, mAP50-95 `-0.0248`

## Required Answers

- 1. Feedback 50 epoch success: `true`
- 2. Final P/R/mAP50/mAP50-95: `0.7712/0.6689/0.7439/0.4993`
- 3. Exceeds YOLO default: `false`
- 4. Recall improved without breaking P/mAP: `false`
- 5. Increased parameters: `official_yolo_aug.hsv_v.value, official_yolo_aug.translate.value, official_yolo_aug.scale.value, custom_industrial_policy.clahe.prob, custom_industrial_policy.clahe.strength, custom_industrial_policy.gamma.prob, custom_industrial_policy.gamma.strength, custom_industrial_policy.sharpen_mild.prob, custom_industrial_policy.sharpen_mild.strength, official_yolo_aug.erasing.value, custom_industrial_policy.cutout_safe.prob, custom_industrial_policy.cutout_safe.strength, custom_industrial_policy.local_contrast.prob, custom_industrial_policy.local_contrast.strength`
- 6. Decreased parameters: `official_yolo_aug.hsv_v.value, custom_industrial_policy.clahe.prob, custom_industrial_policy.clahe.strength, custom_industrial_policy.gamma.prob, custom_industrial_policy.gamma.strength, custom_industrial_policy.local_contrast.prob, custom_industrial_policy.local_contrast.strength, official_yolo_aug.translate.value, official_yolo_aug.scale.value, official_yolo_aug.mosaic.value`
- 7. Policy updates stable: `false`
- 8. Stage degradation observed: `true`
- 9. Constraint failed: `true`
- 10. Worthy as paper main method now: `false`

## Stage Summary

| stage | epochs | P | R | mAP50 | mAP50-95 | adjustments |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 0.7506 | 0.3847 | 0.4371 | 0.2760 | 25 |
| 1 | 5 | 0.4914 | 0.5393 | 0.5142 | 0.3251 | 34 |
| 2 | 5 | 0.5366 | 0.6782 | 0.6582 | 0.4287 | 34 |
| 3 | 5 | 0.5293 | 0.6694 | 0.6469 | 0.4153 | 30 |
| 4 | 5 | 0.6623 | 0.6739 | 0.6758 | 0.4476 | 30 |
| 5 | 5 | 0.6966 | 0.6789 | 0.7434 | 0.5035 | 27 |
| 6 | 5 | 0.7153 | 0.6768 | 0.7415 | 0.4764 | 26 |
| 7 | 5 | 0.7474 | 0.6808 | 0.7537 | 0.5184 | 25 |
| 8 | 5 | 0.6764 | 0.6996 | 0.7255 | 0.4825 | 24 |
| 9 | 5 | 0.7712 | 0.6689 | 0.7439 | 0.4993 | 0 |

## Reference Comparison

| reference | P | R | mAP50 | mAP50-95 | dP | dR | d_mAP50 | d_mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO default seed=42 | 0.7132 | 0.7600 | 0.7759 | 0.5241 | +0.0580 | -0.0911 | -0.0320 | -0.0248 |
| baseline no aug | 0.6900 | 0.6150 | 0.6690 | 0.4340 | +0.0812 | +0.0539 | +0.0749 | +0.0653 |
| offline DiagAug | 0.6860 | 0.6880 | 0.7170 | 0.4960 | +0.0852 | -0.0191 | +0.0269 | +0.0033 |
| offline random external | 0.7500 | 0.6680 | 0.7340 | 0.5010 | +0.0212 | +0.0009 | +0.0099 | -0.0017 |
