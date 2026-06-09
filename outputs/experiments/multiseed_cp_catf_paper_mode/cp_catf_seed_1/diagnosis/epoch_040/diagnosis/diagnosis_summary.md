# Validation Error Diagnosis

- Status: completed
- TP: 374
- FP: 98
- FN: 62
- Precision: 0.7924
- Recall: 0.8462

## Diagnosis Vector
- small_object_score: 0.1906
- low_contrast_score: 0.6605
- class_imbalance_score: 0.9103
- localization_score: 0.0136
- false_positive_score: 0.2076

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=102 fp=23 fn=0 precision=0.8160 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=17 tp=4 fp=3 fn=12 precision=0.5714 recall=0.2353
- 浅划伤: gt=14 tp=3 fp=9 fn=10 precision=0.2500 recall=0.2143
- 漏背锡: gt=32 tp=27 fp=10 fn=4 precision=0.7297 recall=0.8438
- 碰伤: gt=133 tp=106 fp=12 fn=27 precision=0.8983 recall=0.7970
- 脏污: gt=28 tp=19 fp=18 fn=8 precision=0.5135 recall=0.6786
- 轮廓划伤: gt=30 tp=28 fp=9 fn=1 precision=0.7568 recall=0.9333
- 锡丝残留: gt=6 tp=5 fp=3 fn=0 precision=0.6250 recall=0.8333
- 锡尖: gt=12 tp=12 fp=4 fn=0 precision=0.7500 recall=1.0000
- 锡膏: gt=21 tp=21 fp=4 fn=0 precision=0.8400 recall=1.0000
