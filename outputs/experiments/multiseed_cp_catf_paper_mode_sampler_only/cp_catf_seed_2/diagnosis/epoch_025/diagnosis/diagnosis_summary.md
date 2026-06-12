# Validation Error Diagnosis

- Status: completed
- TP: 347
- FP: 119
- FN: 88
- Precision: 0.7446
- Recall: 0.7851

## Diagnosis Vector
- small_object_score: 0.2441
- low_contrast_score: 0.4869
- class_imbalance_score: 0.9853
- localization_score: 0.0158
- false_positive_score: 0.2554

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=32 fp=2 fn=6 precision=0.9412 recall=0.8421
- OK3: gt=102 tp=100 fp=22 fn=2 precision=0.8197 recall=0.9804
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=5 fp=4 fn=12 precision=0.5556 recall=0.2941
- 浅划伤: gt=14 tp=3 fp=5 fn=10 precision=0.3750 recall=0.2143
- 漏背锡: gt=32 tp=22 fp=14 fn=8 precision=0.6111 recall=0.6875
- 碰伤: gt=133 tp=110 fp=25 fn=23 precision=0.8148 recall=0.8271
- 脏污: gt=28 tp=10 fp=14 fn=14 precision=0.4167 recall=0.3571
- 轮廓划伤: gt=30 tp=28 fp=15 fn=2 precision=0.6512 recall=0.9333
- 锡丝残留: gt=6 tp=5 fp=4 fn=1 precision=0.5556 recall=0.8333
- 锡尖: gt=12 tp=9 fp=7 fn=3 precision=0.5625 recall=0.7500
- 锡膏: gt=21 tp=17 fp=7 fn=4 precision=0.7083 recall=0.8095
