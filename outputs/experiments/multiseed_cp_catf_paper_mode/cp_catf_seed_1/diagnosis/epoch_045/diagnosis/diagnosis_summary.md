# Validation Error Diagnosis

- Status: completed
- TP: 382
- FP: 124
- FN: 55
- Precision: 0.7549
- Recall: 0.8643

## Diagnosis Vector
- small_object_score: 0.1672
- low_contrast_score: 0.6300
- class_imbalance_score: 0.9030
- localization_score: 0.0113
- false_positive_score: 0.2451

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=102 fp=23 fn=0 precision=0.8160 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=17 tp=4 fp=5 fn=12 precision=0.4444 recall=0.2353
- 浅划伤: gt=14 tp=5 fp=11 fn=9 precision=0.3125 recall=0.3571
- 漏背锡: gt=32 tp=29 fp=8 fn=3 precision=0.7838 recall=0.9062
- 碰伤: gt=133 tp=111 fp=20 fn=21 precision=0.8473 recall=0.8346
- 脏污: gt=28 tp=18 fp=30 fn=7 precision=0.3750 recall=0.6429
- 轮廓划伤: gt=30 tp=29 fp=12 fn=1 precision=0.7073 recall=0.9667
- 锡丝残留: gt=6 tp=6 fp=4 fn=0 precision=0.6000 recall=1.0000
- 锡尖: gt=12 tp=11 fp=4 fn=1 precision=0.7333 recall=0.9167
- 锡膏: gt=21 tp=20 fp=5 fn=1 precision=0.8000 recall=0.9524
