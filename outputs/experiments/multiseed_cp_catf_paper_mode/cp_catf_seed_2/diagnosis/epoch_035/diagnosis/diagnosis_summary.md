# Validation Error Diagnosis

- Status: completed
- TP: 370
- FP: 163
- FN: 63
- Precision: 0.6942
- Recall: 0.8371

## Diagnosis Vector
- small_object_score: 0.2007
- low_contrast_score: 0.5722
- class_imbalance_score: 0.8603
- localization_score: 0.0204
- false_positive_score: 0.3058

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=102 fp=22 fn=0 precision=0.8226 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=17 tp=8 fp=12 fn=9 precision=0.4000 recall=0.4706
- 浅划伤: gt=14 tp=5 fp=4 fn=8 precision=0.5556 recall=0.3571
- 漏背锡: gt=32 tp=29 fp=20 fn=1 precision=0.5918 recall=0.9062
- 碰伤: gt=133 tp=105 fp=28 fn=27 precision=0.7895 recall=0.7895
- 脏污: gt=28 tp=13 fp=49 fn=11 precision=0.2097 recall=0.4643
- 轮廓划伤: gt=30 tp=29 fp=9 fn=1 precision=0.7632 recall=0.9667
- 锡丝残留: gt=6 tp=6 fp=2 fn=0 precision=0.7500 recall=1.0000
- 锡尖: gt=12 tp=10 fp=8 fn=2 precision=0.5556 recall=0.8333
- 锡膏: gt=21 tp=16 fp=6 fn=4 precision=0.7273 recall=0.7619
