# Validation Error Diagnosis

- Status: completed
- TP: 349
- FP: 179
- FN: 91
- Precision: 0.6610
- Recall: 0.7896

## Diagnosis Vector
- small_object_score: 0.2609
- low_contrast_score: 0.6407
- class_imbalance_score: 0.9853
- localization_score: 0.0045
- false_positive_score: 0.3390

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=101 fp=23 fn=1 precision=0.8145 recall=0.9902
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=2 fp=16 fn=15 precision=0.1111 recall=0.1176
- 浅划伤: gt=14 tp=2 fp=3 fn=12 precision=0.4000 recall=0.1429
- 漏背锡: gt=32 tp=31 fp=14 fn=1 precision=0.6889 recall=0.9688
- 碰伤: gt=133 tp=92 fp=22 fn=40 precision=0.8070 recall=0.6917
- 脏污: gt=28 tp=18 fp=58 fn=9 precision=0.2368 recall=0.6429
- 轮廓划伤: gt=30 tp=27 fp=8 fn=3 precision=0.7714 recall=0.9000
- 锡丝残留: gt=6 tp=0 fp=1 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=12 fp=18 fn=0 precision=0.4000 recall=1.0000
- 锡膏: gt=21 tp=20 fp=14 fn=1 precision=0.5882 recall=0.9524
