# Validation Error Diagnosis

- Status: completed
- TP: 338
- FP: 139
- FN: 97
- Precision: 0.7086
- Recall: 0.7647

## Diagnosis Vector
- small_object_score: 0.2508
- low_contrast_score: 0.5000
- class_imbalance_score: 0.9853
- localization_score: 0.0158
- false_positive_score: 0.2914

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=100 fp=24 fn=2 precision=0.8065 recall=0.9804
- 加强筋打伤: gt=6 tp=6 fp=3 fn=0 precision=0.6667 recall=1.0000
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=2 fp=5 fn=15 precision=0.2857 recall=0.1176
- 浅划伤: gt=14 tp=3 fp=1 fn=11 precision=0.7500 recall=0.2143
- 漏背锡: gt=32 tp=15 fp=3 fn=17 precision=0.8333 recall=0.4688
- 碰伤: gt=133 tp=106 fp=28 fn=25 precision=0.7910 recall=0.7970
- 脏污: gt=28 tp=6 fp=9 fn=17 precision=0.4000 recall=0.2143
- 轮廓划伤: gt=30 tp=27 fp=30 fn=3 precision=0.4737 recall=0.9000
- 锡丝残留: gt=6 tp=5 fp=3 fn=1 precision=0.6250 recall=0.8333
- 锡尖: gt=12 tp=12 fp=7 fn=0 precision=0.6316 recall=1.0000
- 锡膏: gt=21 tp=18 fp=24 fn=3 precision=0.4286 recall=0.8571
