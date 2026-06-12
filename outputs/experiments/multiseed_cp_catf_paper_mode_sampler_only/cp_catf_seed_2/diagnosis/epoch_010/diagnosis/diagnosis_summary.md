# Validation Error Diagnosis

- Status: completed
- TP: 284
- FP: 102
- FN: 152
- Precision: 0.7358
- Recall: 0.6425

## Diagnosis Vector
- small_object_score: 0.3746
- low_contrast_score: 0.4681
- class_imbalance_score: 0.9853
- localization_score: 0.0136
- false_positive_score: 0.2642

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=99 fp=23 fn=3 precision=0.8115 recall=0.9706
- 加强筋打伤: gt=6 tp=5 fp=0 fn=1 precision=1.0000 recall=0.8333
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=1 fp=9 fn=16 precision=0.1000 recall=0.0588
- 浅划伤: gt=14 tp=0 fp=1 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=10 fp=6 fn=21 precision=0.6250 recall=0.3125
- 碰伤: gt=133 tp=97 fp=33 fn=33 precision=0.7462 recall=0.7293
- 脏污: gt=28 tp=2 fp=0 fn=26 precision=1.0000 recall=0.0714
- 轮廓划伤: gt=30 tp=14 fp=7 fn=16 precision=0.6667 recall=0.4667
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=4 fp=3 fn=8 precision=0.5714 recall=0.3333
- 锡膏: gt=21 tp=14 fp=18 fn=5 precision=0.4375 recall=0.6667
