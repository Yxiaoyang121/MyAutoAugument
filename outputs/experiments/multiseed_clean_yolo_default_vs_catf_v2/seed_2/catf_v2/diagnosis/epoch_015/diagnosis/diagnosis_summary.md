# Validation Error Diagnosis

- Status: completed
- TP: 643
- FP: 311
- FN: 238
- Precision: 0.6740
- Recall: 0.7105

## Diagnosis Vector
- small_object_score: 0.3616
- low_contrast_score: 0.5050
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3260

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=269 fp=57 fn=5 precision=0.8252 recall=0.9818
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=40 fn=14 precision=0.0476 recall=0.1111
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=38 fp=46 fn=8 precision=0.4524 recall=0.8261
- 碰伤: gt=269 tp=175 fp=78 fn=85 precision=0.6917 recall=0.6506
- 脏污: gt=42 tp=16 fp=19 fn=21 precision=0.4571 recall=0.3810
- 轮廓划伤: gt=84 tp=30 fp=24 fn=46 precision=0.5556 recall=0.3571
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=20 fp=3 fn=7 precision=0.8696 recall=0.7407
- 锡膏: gt=46 tp=23 fp=19 fn=23 precision=0.5476 recall=0.5000
