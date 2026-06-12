# Validation Error Diagnosis

- Status: completed
- TP: 300
- FP: 166
- FN: 131
- Precision: 0.6438
- Recall: 0.6787

## Diagnosis Vector
- small_object_score: 0.3679
- low_contrast_score: 0.4897
- class_imbalance_score: 0.9853
- localization_score: 0.0249
- false_positive_score: 0.3562

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=99 fp=25 fn=2 precision=0.7984 recall=0.9706
- 加强筋打伤: gt=6 tp=1 fp=0 fn=5 precision=1.0000 recall=0.1667
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=0 fp=1 fn=17 precision=0.0000 recall=0.0000
- 浅划伤: gt=14 tp=0 fp=0 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=17 fp=11 fn=10 precision=0.6071 recall=0.5312
- 碰伤: gt=133 tp=87 fp=35 fn=44 precision=0.7131 recall=0.6541
- 脏污: gt=28 tp=4 fp=5 fn=22 precision=0.4444 recall=0.1429
- 轮廓划伤: gt=30 tp=28 fp=16 fn=1 precision=0.6364 recall=0.9333
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=8 fp=3 fn=4 precision=0.7273 recall=0.6667
- 锡膏: gt=21 tp=18 fp=69 fn=3 precision=0.2069 recall=0.8571
