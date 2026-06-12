# Validation Error Diagnosis

- Status: completed
- TP: 382
- FP: 143
- FN: 51
- Precision: 0.7276
- Recall: 0.8643

## Diagnosis Vector
- small_object_score: 0.1538
- low_contrast_score: 0.6206
- class_imbalance_score: 0.9442
- localization_score: 0.0204
- false_positive_score: 0.2724

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=102 fp=24 fn=0 precision=0.8095 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=2 fn=0 precision=0.7500 recall=1.0000
- 开裂: gt=3 tp=3 fp=8 fn=0 precision=0.2727 recall=1.0000
- 油污: gt=17 tp=2 fp=11 fn=14 precision=0.1538 recall=0.1176
- 浅划伤: gt=14 tp=7 fp=11 fn=4 precision=0.3889 recall=0.5000
- 漏背锡: gt=32 tp=30 fp=14 fn=2 precision=0.6818 recall=0.9375
- 碰伤: gt=133 tp=112 fp=27 fn=21 precision=0.8058 recall=0.8421
- 脏污: gt=28 tp=15 fp=16 fn=8 precision=0.4839 recall=0.5357
- 轮廓划伤: gt=30 tp=30 fp=8 fn=0 precision=0.7895 recall=1.0000
- 锡丝残留: gt=6 tp=4 fp=2 fn=2 precision=0.6667 recall=0.6667
- 锡尖: gt=12 tp=12 fp=11 fn=0 precision=0.5217 recall=1.0000
- 锡膏: gt=21 tp=21 fp=7 fn=0 precision=0.7500 recall=1.0000
