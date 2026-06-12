# Validation Error Diagnosis

- Status: completed
- TP: 374
- FP: 118
- FN: 66
- Precision: 0.7602
- Recall: 0.8462

## Diagnosis Vector
- small_object_score: 0.1940
- low_contrast_score: 0.5598
- class_imbalance_score: 0.9353
- localization_score: 0.0045
- false_positive_score: 0.2398

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=100 fp=20 fn=2 precision=0.8333 recall=0.9804
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=17 tp=7 fp=10 fn=10 precision=0.4118 recall=0.4118
- 浅划伤: gt=14 tp=2 fp=1 fn=12 precision=0.6667 recall=0.1429
- 漏背锡: gt=32 tp=28 fp=9 fn=4 precision=0.7568 recall=0.8750
- 碰伤: gt=133 tp=112 fp=20 fn=21 precision=0.8485 recall=0.8421
- 脏污: gt=28 tp=16 fp=31 fn=10 precision=0.3404 recall=0.5714
- 轮廓划伤: gt=30 tp=30 fp=9 fn=0 precision=0.7692 recall=1.0000
- 锡丝残留: gt=6 tp=5 fp=4 fn=1 precision=0.5556 recall=0.8333
- 锡尖: gt=12 tp=12 fp=10 fn=0 precision=0.5455 recall=1.0000
- 锡膏: gt=21 tp=15 fp=3 fn=6 precision=0.8333 recall=0.7143
