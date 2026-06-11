# Validation Error Diagnosis

- Status: completed
- TP: 319
- FP: 165
- FN: 117
- Precision: 0.6591
- Recall: 0.7217

## Diagnosis Vector
- small_object_score: 0.3278
- low_contrast_score: 0.5013
- class_imbalance_score: 0.9853
- localization_score: 0.0136
- false_positive_score: 0.3409

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=101 fp=26 fn=1 precision=0.7953 recall=0.9902
- 加强筋打伤: gt=6 tp=4 fp=0 fn=2 precision=1.0000 recall=0.6667
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=2 fp=16 fn=14 precision=0.1111 recall=0.1176
- 浅划伤: gt=14 tp=0 fp=2 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=25 fp=20 fn=7 precision=0.5556 recall=0.7812
- 碰伤: gt=133 tp=91 fp=27 fn=42 precision=0.7712 recall=0.6842
- 脏污: gt=28 tp=14 fp=52 fn=9 precision=0.2121 recall=0.5000
- 轮廓划伤: gt=30 tp=29 fp=11 fn=1 precision=0.7250 recall=0.9667
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=7 fp=9 fn=5 precision=0.4375 recall=0.5833
- 锡膏: gt=21 tp=8 fp=0 fn=13 precision=1.0000 recall=0.3810
