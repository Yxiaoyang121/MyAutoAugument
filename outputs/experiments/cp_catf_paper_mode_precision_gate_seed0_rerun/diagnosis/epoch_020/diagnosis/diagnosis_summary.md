# Validation Error Diagnosis

- Status: completed
- TP: 316
- FP: 134
- FN: 114
- Precision: 0.7022
- Recall: 0.7149

## Diagnosis Vector
- small_object_score: 0.3445
- low_contrast_score: 0.6066
- class_imbalance_score: 0.9353
- localization_score: 0.0271
- false_positive_score: 0.2978

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=101 fp=24 fn=1 precision=0.8080 recall=0.9902
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=1 fp=0 fn=2 precision=1.0000 recall=0.3333
- 油污: gt=17 tp=3 fp=2 fn=14 precision=0.6000 recall=0.1765
- 浅划伤: gt=14 tp=2 fp=7 fn=8 precision=0.2222 recall=0.1429
- 漏背锡: gt=32 tp=23 fp=12 fn=7 precision=0.6571 recall=0.7188
- 碰伤: gt=133 tp=75 fp=11 fn=56 precision=0.8721 recall=0.5639
- 脏污: gt=28 tp=14 fp=55 fn=11 precision=0.2029 recall=0.5000
- 轮廓划伤: gt=30 tp=25 fp=5 fn=4 precision=0.8333 recall=0.8333
- 锡丝残留: gt=6 tp=3 fp=2 fn=3 precision=0.6000 recall=0.5000
- 锡尖: gt=12 tp=11 fp=13 fn=1 precision=0.4583 recall=0.9167
- 锡膏: gt=21 tp=14 fp=1 fn=7 precision=0.9333 recall=0.6667
