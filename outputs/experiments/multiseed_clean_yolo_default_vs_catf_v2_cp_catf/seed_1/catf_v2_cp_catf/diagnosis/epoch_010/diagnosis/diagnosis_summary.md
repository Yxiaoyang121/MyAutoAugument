# Validation Error Diagnosis

- Status: completed
- TP: 641
- FP: 372
- FN: 240
- Precision: 0.6328
- Recall: 0.7083

## Diagnosis Vector
- small_object_score: 0.3497
- low_contrast_score: 0.4902
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3672

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=271 fp=69 fn=1 precision=0.7971 recall=0.9891
- 加强筋打伤: gt=8 tp=2 fp=0 fn=6 precision=1.0000 recall=0.2500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=3 fp=33 fn=15 precision=0.0833 recall=0.1667
- 浅划伤: gt=13 tp=4 fp=0 fn=9 precision=1.0000 recall=0.3077
- 漏背锡: gt=46 tp=27 fp=48 fn=14 precision=0.3600 recall=0.5870
- 碰伤: gt=269 tp=166 fp=126 fn=93 precision=0.5685 recall=0.6171
- 脏污: gt=42 tp=6 fp=4 fn=36 precision=0.6000 recall=0.1429
- 轮廓划伤: gt=84 tp=42 fp=37 fn=38 precision=0.5316 recall=0.5000
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=27 fp=8 fn=0 precision=0.7714 recall=1.0000
- 锡膏: gt=46 tp=29 fp=24 fn=14 precision=0.5472 recall=0.6304
