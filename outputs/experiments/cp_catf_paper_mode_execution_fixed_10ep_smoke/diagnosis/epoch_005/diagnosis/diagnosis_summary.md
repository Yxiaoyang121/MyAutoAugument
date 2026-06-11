# Validation Error Diagnosis

- Status: completed
- TP: 299
- FP: 141
- FN: 131
- Precision: 0.6795
- Recall: 0.6765

## Diagnosis Vector
- small_object_score: 0.3512
- low_contrast_score: 0.4397
- class_imbalance_score: 0.9853
- localization_score: 0.0271
- false_positive_score: 0.3205

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=3 fn=0 precision=0.9268 recall=1.0000
- OK3: gt=102 tp=100 fp=19 fn=0 precision=0.8403 recall=0.9804
- 加强筋打伤: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=0 fp=0 fn=17 precision=0.0000 recall=0.0000
- 浅划伤: gt=14 tp=0 fp=0 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=18 fp=35 fn=9 precision=0.3396 recall=0.5625
- 碰伤: gt=133 tp=99 fp=41 fn=30 precision=0.7071 recall=0.7444
- 脏污: gt=28 tp=0 fp=0 fn=28 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=30 tp=20 fp=14 fn=9 precision=0.5882 recall=0.6667
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=12 fp=21 fn=0 precision=0.3636 recall=1.0000
- 锡膏: gt=21 tp=12 fp=8 fn=9 precision=0.6000 recall=0.5714
