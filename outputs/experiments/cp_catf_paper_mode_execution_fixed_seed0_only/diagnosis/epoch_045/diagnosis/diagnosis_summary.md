# Validation Error Diagnosis

- Status: completed
- TP: 379
- FP: 119
- FN: 59
- Precision: 0.7610
- Recall: 0.8575

## Diagnosis Vector
- small_object_score: 0.1706
- low_contrast_score: 0.6263
- class_imbalance_score: 0.9442
- localization_score: 0.0090
- false_positive_score: 0.2390

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=102 fp=22 fn=0 precision=0.8226 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=15 fn=0 precision=0.1667 recall=1.0000
- 油污: gt=17 tp=2 fp=6 fn=13 precision=0.2500 recall=0.1176
- 浅划伤: gt=14 tp=7 fp=8 fn=6 precision=0.4667 recall=0.5000
- 漏背锡: gt=32 tp=28 fp=14 fn=4 precision=0.6667 recall=0.8750
- 碰伤: gt=133 tp=108 fp=23 fn=25 precision=0.8244 recall=0.8120
- 脏污: gt=28 tp=20 fp=17 fn=7 precision=0.5405 recall=0.7143
- 轮廓划伤: gt=30 tp=29 fp=2 fn=1 precision=0.9355 recall=0.9667
- 锡丝残留: gt=6 tp=4 fp=2 fn=2 precision=0.6667 recall=0.6667
- 锡尖: gt=12 tp=12 fp=8 fn=0 precision=0.6000 recall=1.0000
- 锡膏: gt=21 tp=20 fp=1 fn=1 precision=0.9524 recall=0.9524
