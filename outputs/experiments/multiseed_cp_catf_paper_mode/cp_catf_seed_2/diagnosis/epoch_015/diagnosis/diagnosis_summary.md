# Validation Error Diagnosis

- Status: completed
- TP: 328
- FP: 122
- FN: 107
- Precision: 0.7289
- Recall: 0.7421

## Diagnosis Vector
- small_object_score: 0.2977
- low_contrast_score: 0.4995
- class_imbalance_score: 0.9853
- localization_score: 0.0158
- false_positive_score: 0.2711

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=100 fp=22 fn=2 precision=0.8197 recall=0.9804
- 加强筋打伤: gt=6 tp=4 fp=0 fn=2 precision=1.0000 recall=0.6667
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=3 fp=13 fn=14 precision=0.1875 recall=0.1765
- 浅划伤: gt=14 tp=0 fp=2 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=22 fp=18 fn=8 precision=0.5500 recall=0.6875
- 碰伤: gt=133 tp=100 fp=33 fn=32 precision=0.7519 recall=0.7519
- 脏污: gt=28 tp=6 fp=6 fn=22 precision=0.5000 recall=0.2143
- 轮廓划伤: gt=30 tp=28 fp=10 fn=0 precision=0.7368 recall=0.9333
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=10 fp=9 fn=1 precision=0.5263 recall=0.8333
- 锡膏: gt=21 tp=17 fp=8 fn=3 precision=0.6800 recall=0.8095
