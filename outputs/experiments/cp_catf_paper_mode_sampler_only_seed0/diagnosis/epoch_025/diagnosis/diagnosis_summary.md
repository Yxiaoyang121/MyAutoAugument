# Validation Error Diagnosis

- Status: completed
- TP: 360
- FP: 133
- FN: 79
- Precision: 0.7302
- Recall: 0.8145

## Diagnosis Vector
- small_object_score: 0.2107
- low_contrast_score: 0.5247
- class_imbalance_score: 0.9236
- localization_score: 0.0068
- false_positive_score: 0.2698

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=100 fp=22 fn=2 precision=0.8197 recall=0.9804
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=11 fn=0 precision=0.2143 recall=1.0000
- 油污: gt=17 tp=3 fp=4 fn=14 precision=0.4286 recall=0.1765
- 浅划伤: gt=14 tp=8 fp=10 fn=6 precision=0.4444 recall=0.5714
- 漏背锡: gt=32 tp=21 fp=9 fn=10 precision=0.7000 recall=0.6562
- 碰伤: gt=133 tp=102 fp=23 fn=30 precision=0.8160 recall=0.7669
- 脏污: gt=28 tp=13 fp=19 fn=14 precision=0.4062 recall=0.4643
- 轮廓划伤: gt=30 tp=30 fp=16 fn=0 precision=0.6522 recall=1.0000
- 锡丝残留: gt=6 tp=5 fp=1 fn=1 precision=0.8333 recall=0.8333
- 锡尖: gt=12 tp=12 fp=10 fn=0 precision=0.5455 recall=1.0000
- 锡膏: gt=21 tp=19 fp=6 fn=2 precision=0.7600 recall=0.9048
