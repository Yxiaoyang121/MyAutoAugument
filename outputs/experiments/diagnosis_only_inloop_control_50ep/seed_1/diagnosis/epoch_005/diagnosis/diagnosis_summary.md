# Validation Error Diagnosis

- Status: completed
- TP: 529
- FP: 273
- FN: 370
- Precision: 0.6596
- Recall: 0.5845

## Diagnosis Vector
- small_object_score: 0.5144
- low_contrast_score: 0.5286
- class_imbalance_score: 0.9953
- localization_score: 0.0066
- false_positive_score: 0.3404

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- background_interference severity=medium suggestions=background-diversity, light-noise, illumination-jitter
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=16 fn=0 precision=0.8000 recall=1.0000
- OK3: gt=274 tp=274 fp=71 fn=0 precision=0.7942 recall=1.0000
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=0 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=6 fp=5 fn=37 precision=0.5455 recall=0.1304
- 碰伤: gt=269 tp=96 fp=17 fn=172 precision=0.8496 recall=0.3569
- 脏污: gt=42 tp=5 fp=31 fn=36 precision=0.1389 recall=0.1190
- 轮廓划伤: gt=84 tp=30 fp=15 fn=53 precision=0.6667 recall=0.3571
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=14 fp=1 fn=13 precision=0.9333 recall=0.5185
- 锡膏: gt=46 tp=40 fp=117 fn=6 precision=0.2548 recall=0.8696
