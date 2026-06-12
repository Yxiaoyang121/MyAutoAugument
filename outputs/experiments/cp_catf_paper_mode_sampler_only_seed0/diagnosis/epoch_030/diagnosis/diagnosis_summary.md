# Validation Error Diagnosis

- Status: completed
- TP: 350
- FP: 118
- FN: 86
- Precision: 0.7479
- Recall: 0.7919

## Diagnosis Vector
- small_object_score: 0.2241
- low_contrast_score: 0.5023
- class_imbalance_score: 0.9236
- localization_score: 0.0136
- false_positive_score: 0.2521

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=101 fp=23 fn=1 precision=0.8145 recall=0.9902
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=8 fn=0 precision=0.2727 recall=1.0000
- 油污: gt=17 tp=3 fp=1 fn=14 precision=0.7500 recall=0.1765
- 浅划伤: gt=14 tp=7 fp=8 fn=6 precision=0.4667 recall=0.5000
- 漏背锡: gt=32 tp=18 fp=7 fn=13 precision=0.7200 recall=0.5625
- 碰伤: gt=133 tp=102 fp=28 fn=31 precision=0.7846 recall=0.7669
- 脏污: gt=28 tp=12 fp=29 fn=12 precision=0.2927 recall=0.4286
- 轮廓划伤: gt=30 tp=28 fp=3 fn=2 precision=0.9032 recall=0.9333
- 锡丝残留: gt=6 tp=3 fp=2 fn=3 precision=0.6000 recall=0.5000
- 锡尖: gt=12 tp=11 fp=7 fn=1 precision=0.6111 recall=0.9167
- 锡膏: gt=21 tp=18 fp=0 fn=3 precision=1.0000 recall=0.8571
