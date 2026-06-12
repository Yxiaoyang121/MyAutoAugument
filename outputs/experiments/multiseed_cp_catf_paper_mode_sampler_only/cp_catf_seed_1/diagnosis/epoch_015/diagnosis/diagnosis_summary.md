# Validation Error Diagnosis

- Status: completed
- TP: 345
- FP: 217
- FN: 89
- Precision: 0.6139
- Recall: 0.7805

## Diagnosis Vector
- small_object_score: 0.2575
- low_contrast_score: 0.5382
- class_imbalance_score: 0.9853
- localization_score: 0.0181
- false_positive_score: 0.3861

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=100 fp=23 fn=2 precision=0.8130 recall=0.9804
- 加强筋打伤: gt=6 tp=3 fp=0 fn=3 precision=1.0000 recall=0.5000
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=1 fp=14 fn=16 precision=0.0667 recall=0.0588
- 浅划伤: gt=14 tp=1 fp=1 fn=13 precision=0.5000 recall=0.0714
- 漏背锡: gt=32 tp=22 fp=43 fn=8 precision=0.3385 recall=0.6875
- 碰伤: gt=133 tp=106 fp=25 fn=24 precision=0.8092 recall=0.7970
- 脏污: gt=28 tp=17 fp=65 fn=8 precision=0.2073 recall=0.6071
- 轮廓划伤: gt=30 tp=27 fp=17 fn=3 precision=0.6136 recall=0.9000
- 锡丝残留: gt=6 tp=2 fp=1 fn=4 precision=0.6667 recall=0.3333
- 锡尖: gt=12 tp=12 fp=10 fn=0 precision=0.5455 recall=1.0000
- 锡膏: gt=21 tp=16 fp=16 fn=5 precision=0.5000 recall=0.7619
