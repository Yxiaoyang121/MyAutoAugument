# Validation Error Diagnosis

- Status: completed
- TP: 370
- FP: 141
- FN: 60
- Precision: 0.7241
- Recall: 0.8371

## Diagnosis Vector
- small_object_score: 0.1873
- low_contrast_score: 0.6242
- class_imbalance_score: 0.9236
- localization_score: 0.0271
- false_positive_score: 0.2759

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=100 fp=24 fn=2 precision=0.8065 recall=0.9804
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=17 tp=3 fp=8 fn=14 precision=0.2727 recall=0.1765
- 浅划伤: gt=14 tp=6 fp=10 fn=4 precision=0.3750 recall=0.4286
- 漏背锡: gt=32 tp=30 fp=11 fn=1 precision=0.7317 recall=0.9375
- 碰伤: gt=133 tp=108 fp=26 fn=24 precision=0.8060 recall=0.8120
- 脏污: gt=28 tp=15 fp=32 fn=8 precision=0.3191 recall=0.5357
- 轮廓划伤: gt=30 tp=26 fp=14 fn=3 precision=0.6500 recall=0.8667
- 锡丝残留: gt=6 tp=3 fp=2 fn=3 precision=0.6000 recall=0.5000
- 锡尖: gt=12 tp=12 fp=9 fn=0 precision=0.5714 recall=1.0000
- 锡膏: gt=21 tp=20 fp=3 fn=1 precision=0.8696 recall=0.9524
