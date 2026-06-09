# Validation Error Diagnosis

- Status: completed
- TP: 375
- FP: 147
- FN: 63
- Precision: 0.7184
- Recall: 0.8484

## Diagnosis Vector
- small_object_score: 0.1973
- low_contrast_score: 0.6294
- class_imbalance_score: 0.9353
- localization_score: 0.0090
- false_positive_score: 0.2816

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=102 fp=23 fn=0 precision=0.8160 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=1 fn=0 precision=0.7500 recall=1.0000
- 油污: gt=17 tp=4 fp=7 fn=13 precision=0.3636 recall=0.2353
- 浅划伤: gt=14 tp=2 fp=8 fn=10 precision=0.2000 recall=0.1429
- 漏背锡: gt=32 tp=29 fp=14 fn=3 precision=0.6744 recall=0.9062
- 碰伤: gt=133 tp=106 fp=23 fn=26 precision=0.8217 recall=0.7970
- 脏污: gt=28 tp=19 fp=43 fn=8 precision=0.3065 recall=0.6786
- 轮廓划伤: gt=30 tp=29 fp=8 fn=1 precision=0.7838 recall=0.9667
- 锡丝残留: gt=6 tp=6 fp=3 fn=0 precision=0.6667 recall=1.0000
- 锡尖: gt=12 tp=12 fp=10 fn=0 precision=0.5455 recall=1.0000
- 锡膏: gt=21 tp=19 fp=5 fn=2 precision=0.7917 recall=0.9048
