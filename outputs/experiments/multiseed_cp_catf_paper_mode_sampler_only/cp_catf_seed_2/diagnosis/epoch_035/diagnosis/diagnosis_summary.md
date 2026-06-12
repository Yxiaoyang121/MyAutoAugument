# Validation Error Diagnosis

- Status: completed
- TP: 364
- FP: 136
- FN: 72
- Precision: 0.7280
- Recall: 0.8235

## Diagnosis Vector
- small_object_score: 0.2241
- low_contrast_score: 0.6438
- class_imbalance_score: 0.8618
- localization_score: 0.0136
- false_positive_score: 0.2720

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=100 fp=22 fn=2 precision=0.8197 recall=0.9804
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=7 fn=0 precision=0.3000 recall=1.0000
- 油污: gt=17 tp=6 fp=11 fn=10 precision=0.3529 recall=0.3529
- 浅划伤: gt=14 tp=6 fp=2 fn=8 precision=0.7500 recall=0.4286
- 漏背锡: gt=32 tp=28 fp=12 fn=2 precision=0.7000 recall=0.8750
- 碰伤: gt=133 tp=99 fp=16 fn=34 precision=0.8609 recall=0.7444
- 脏污: gt=28 tp=13 fp=36 fn=12 precision=0.2653 recall=0.4643
- 轮廓划伤: gt=30 tp=28 fp=8 fn=2 precision=0.7778 recall=0.9333
- 锡丝残留: gt=6 tp=6 fp=2 fn=0 precision=0.7500 recall=1.0000
- 锡尖: gt=12 tp=12 fp=8 fn=0 precision=0.6000 recall=1.0000
- 锡膏: gt=21 tp=19 fp=9 fn=2 precision=0.6786 recall=0.9048
