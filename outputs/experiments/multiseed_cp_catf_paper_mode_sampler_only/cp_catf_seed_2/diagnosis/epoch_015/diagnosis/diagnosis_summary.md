# Validation Error Diagnosis

- Status: completed
- TP: 342
- FP: 190
- FN: 87
- Precision: 0.6429
- Recall: 0.7738

## Diagnosis Vector
- small_object_score: 0.2575
- low_contrast_score: 0.5500
- class_imbalance_score: 0.9853
- localization_score: 0.0294
- false_positive_score: 0.3571

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=3 fn=0 precision=0.9268 recall=1.0000
- OK3: gt=102 tp=100 fp=22 fn=1 precision=0.8197 recall=0.9804
- 加强筋打伤: gt=6 tp=4 fp=0 fn=2 precision=1.0000 recall=0.6667
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=4 fp=11 fn=13 precision=0.2667 recall=0.2353
- 浅划伤: gt=14 tp=7 fp=29 fn=5 precision=0.1944 recall=0.5000
- 漏背锡: gt=32 tp=24 fp=24 fn=5 precision=0.5000 recall=0.7500
- 碰伤: gt=133 tp=97 fp=23 fn=34 precision=0.8083 recall=0.7293
- 脏污: gt=28 tp=14 fp=28 fn=13 precision=0.3333 recall=0.5000
- 轮廓划伤: gt=30 tp=27 fp=22 fn=1 precision=0.5510 recall=0.9000
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=12 fp=10 fn=0 precision=0.5455 recall=1.0000
- 锡膏: gt=21 tp=15 fp=18 fn=4 precision=0.4545 recall=0.7143
