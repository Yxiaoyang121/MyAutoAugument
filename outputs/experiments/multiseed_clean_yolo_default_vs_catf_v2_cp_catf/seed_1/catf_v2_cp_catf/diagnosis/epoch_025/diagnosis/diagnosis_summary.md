# Validation Error Diagnosis

- Status: completed
- TP: 654
- FP: 268
- FN: 239
- Precision: 0.7093
- Recall: 0.7227

## Diagnosis Vector
- small_object_score: 0.3463
- low_contrast_score: 0.5546
- class_imbalance_score: 0.9953
- localization_score: 0.0133
- false_positive_score: 0.2907

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=271 fp=64 fn=3 precision=0.8090 recall=0.9891
- 加强筋打伤: gt=8 tp=4 fp=0 fn=4 precision=1.0000 recall=0.5000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=18 fn=14 precision=0.1818 recall=0.2222
- 浅划伤: gt=13 tp=5 fp=0 fn=8 precision=1.0000 recall=0.3846
- 漏背锡: gt=46 tp=31 fp=40 fn=10 precision=0.4366 recall=0.6739
- 碰伤: gt=269 tp=158 fp=44 fn=109 precision=0.7822 recall=0.5874
- 脏污: gt=42 tp=18 fp=26 fn=24 precision=0.4091 recall=0.4286
- 轮廓划伤: gt=84 tp=40 fp=31 fn=41 precision=0.5634 recall=0.4762
- 锡丝残留: gt=12 tp=1 fp=1 fn=11 precision=0.5000 recall=0.0833
- 锡尖: gt=27 tp=27 fp=9 fn=0 precision=0.7500 recall=1.0000
- 锡膏: gt=46 tp=31 fp=13 fn=13 precision=0.7045 recall=0.6739
