# Validation Error Diagnosis

- Status: completed
- TP: 353
- FP: 118
- FN: 82
- Precision: 0.7495
- Recall: 0.7986

## Diagnosis Vector
- small_object_score: 0.2341
- low_contrast_score: 0.5610
- class_imbalance_score: 0.9853
- localization_score: 0.0158
- false_positive_score: 0.2505

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=101 fp=23 fn=1 precision=0.8145 recall=0.9902
- 加强筋打伤: gt=6 tp=6 fp=2 fn=0 precision=0.7500 recall=1.0000
- 开裂: gt=3 tp=3 fp=6 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=17 tp=4 fp=6 fn=13 precision=0.4000 recall=0.2353
- 浅划伤: gt=14 tp=0 fp=0 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=18 fp=3 fn=13 precision=0.8571 recall=0.5625
- 碰伤: gt=133 tp=105 fp=26 fn=27 precision=0.8015 recall=0.7895
- 脏污: gt=28 tp=17 fp=24 fn=6 precision=0.4146 recall=0.6071
- 轮廓划伤: gt=30 tp=26 fp=9 fn=4 precision=0.7429 recall=0.8667
- 锡丝残留: gt=6 tp=6 fp=3 fn=0 precision=0.6667 recall=1.0000
- 锡尖: gt=12 tp=12 fp=12 fn=0 precision=0.5000 recall=1.0000
- 锡膏: gt=21 tp=17 fp=3 fn=4 precision=0.8500 recall=0.8095
