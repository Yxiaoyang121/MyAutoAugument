# Validation Error Diagnosis

- Status: completed
- TP: 377
- FP: 116
- FN: 64
- Precision: 0.7647
- Recall: 0.8529

## Diagnosis Vector
- small_object_score: 0.1773
- low_contrast_score: 0.6180
- class_imbalance_score: 0.9353
- localization_score: 0.0023
- false_positive_score: 0.2353

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=3 fn=0 precision=0.9268 recall=1.0000
- OK3: gt=102 tp=102 fp=23 fn=0 precision=0.8160 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=5 fn=0 precision=0.3750 recall=1.0000
- 油污: gt=17 tp=4 fp=5 fn=13 precision=0.4444 recall=0.2353
- 浅划伤: gt=14 tp=2 fp=7 fn=12 precision=0.2222 recall=0.1429
- 漏背锡: gt=32 tp=29 fp=12 fn=3 precision=0.7073 recall=0.9062
- 碰伤: gt=133 tp=107 fp=21 fn=26 precision=0.8359 recall=0.8045
- 脏污: gt=28 tp=20 fp=17 fn=7 precision=0.5405 recall=0.7143
- 轮廓划伤: gt=30 tp=30 fp=10 fn=0 precision=0.7500 recall=1.0000
- 锡丝残留: gt=6 tp=3 fp=2 fn=3 precision=0.6000 recall=0.5000
- 锡尖: gt=12 tp=12 fp=9 fn=0 precision=0.5714 recall=1.0000
- 锡膏: gt=21 tp=21 fp=2 fn=0 precision=0.9130 recall=1.0000
