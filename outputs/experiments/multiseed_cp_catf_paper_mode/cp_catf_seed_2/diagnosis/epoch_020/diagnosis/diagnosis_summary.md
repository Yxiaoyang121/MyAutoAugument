# Validation Error Diagnosis

- Status: completed
- TP: 314
- FP: 156
- FN: 117
- Precision: 0.6681
- Recall: 0.7104

## Diagnosis Vector
- small_object_score: 0.3445
- low_contrast_score: 0.5846
- class_imbalance_score: 0.9853
- localization_score: 0.0249
- false_positive_score: 0.3319

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=4 fn=0 precision=0.9048 recall=1.0000
- OK3: gt=102 tp=100 fp=24 fn=1 precision=0.8065 recall=0.9804
- 加强筋打伤: gt=6 tp=6 fp=4 fn=0 precision=0.6000 recall=1.0000
- 开裂: gt=3 tp=0 fp=2 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=3 fp=16 fn=14 precision=0.1579 recall=0.1765
- 浅划伤: gt=14 tp=3 fp=12 fn=8 precision=0.2000 recall=0.2143
- 漏背锡: gt=32 tp=22 fp=22 fn=6 precision=0.5000 recall=0.6875
- 碰伤: gt=133 tp=68 fp=7 fn=65 precision=0.9067 recall=0.5113
- 脏污: gt=28 tp=14 fp=23 fn=12 precision=0.3784 recall=0.5000
- 轮廓划伤: gt=30 tp=30 fp=16 fn=0 precision=0.6522 recall=1.0000
- 锡丝残留: gt=6 tp=1 fp=1 fn=5 precision=0.5000 recall=0.1667
- 锡尖: gt=12 tp=12 fp=23 fn=0 precision=0.3429 recall=1.0000
- 锡膏: gt=21 tp=17 fp=2 fn=3 precision=0.8947 recall=0.8095
