# Validation Error Diagnosis

- Status: completed
- TP: 365
- FP: 112
- FN: 71
- Precision: 0.7652
- Recall: 0.8258

## Diagnosis Vector
- small_object_score: 0.2074
- low_contrast_score: 0.5458
- class_imbalance_score: 0.9103
- localization_score: 0.0136
- false_positive_score: 0.2348

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=102 fp=24 fn=0 precision=0.8095 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=2 fn=0 precision=0.7500 recall=1.0000
- 开裂: gt=3 tp=3 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=17 tp=5 fp=5 fn=12 precision=0.5000 recall=0.2941
- 浅划伤: gt=14 tp=3 fp=6 fn=10 precision=0.3333 recall=0.2143
- 漏背锡: gt=32 tp=27 fp=15 fn=3 precision=0.6429 recall=0.8438
- 碰伤: gt=133 tp=105 fp=18 fn=27 precision=0.8537 recall=0.7895
- 脏污: gt=28 tp=15 fp=17 fn=12 precision=0.4688 recall=0.5357
- 轮廓划伤: gt=30 tp=29 fp=11 fn=1 precision=0.7250 recall=0.9667
- 锡丝残留: gt=6 tp=5 fp=2 fn=1 precision=0.7143 recall=0.8333
- 锡尖: gt=12 tp=10 fp=7 fn=2 precision=0.5882 recall=0.8333
- 锡膏: gt=21 tp=17 fp=3 fn=3 precision=0.8500 recall=0.8095
