# Validation Error Diagnosis

- Status: completed
- TP: 387
- FP: 130
- FN: 48
- Precision: 0.7485
- Recall: 0.8756

## Diagnosis Vector
- small_object_score: 0.1472
- low_contrast_score: 0.6708
- class_imbalance_score: 0.8824
- localization_score: 0.0158
- false_positive_score: 0.2515

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=102 fp=22 fn=0 precision=0.8226 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=4 fn=0 precision=0.4286 recall=1.0000
- 油污: gt=17 tp=5 fp=9 fn=10 precision=0.3571 recall=0.2941
- 浅划伤: gt=14 tp=5 fp=7 fn=7 precision=0.4167 recall=0.3571
- 漏背锡: gt=32 tp=27 fp=15 fn=3 precision=0.6429 recall=0.8438
- 碰伤: gt=133 tp=110 fp=26 fn=23 precision=0.8088 recall=0.8271
- 脏污: gt=28 tp=24 fp=18 fn=3 precision=0.5714 recall=0.8571
- 轮廓划伤: gt=30 tp=30 fp=13 fn=0 precision=0.6977 recall=1.0000
- 锡丝残留: gt=6 tp=4 fp=2 fn=2 precision=0.6667 recall=0.6667
- 锡尖: gt=12 tp=12 fp=5 fn=0 precision=0.7059 recall=1.0000
- 锡膏: gt=21 tp=21 fp=8 fn=0 precision=0.7241 recall=1.0000
