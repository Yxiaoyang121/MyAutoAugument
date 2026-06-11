# Validation Error Diagnosis

- Status: completed
- TP: 369
- FP: 113
- FN: 66
- Precision: 0.7656
- Recall: 0.8348

## Diagnosis Vector
- small_object_score: 0.1973
- low_contrast_score: 0.5750
- class_imbalance_score: 0.8824
- localization_score: 0.0158
- false_positive_score: 0.2344

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=3 fn=0 precision=0.9268 recall=1.0000
- OK3: gt=102 tp=102 fp=23 fn=0 precision=0.8160 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=2 fp=0 fn=1 precision=1.0000 recall=0.6667
- 油污: gt=17 tp=5 fp=6 fn=12 precision=0.4545 recall=0.2941
- 浅划伤: gt=14 tp=6 fp=8 fn=5 precision=0.4286 recall=0.4286
- 漏背锡: gt=32 tp=27 fp=13 fn=4 precision=0.6750 recall=0.8438
- 碰伤: gt=133 tp=107 fp=25 fn=25 precision=0.8106 recall=0.8045
- 脏污: gt=28 tp=17 fp=18 fn=9 precision=0.4857 recall=0.6071
- 轮廓划伤: gt=30 tp=28 fp=2 fn=2 precision=0.9333 recall=0.9333
- 锡丝残留: gt=6 tp=3 fp=2 fn=3 precision=0.6000 recall=0.5000
- 锡尖: gt=12 tp=12 fp=12 fn=0 precision=0.5000 recall=1.0000
- 锡膏: gt=21 tp=16 fp=0 fn=5 precision=1.0000 recall=0.7619
