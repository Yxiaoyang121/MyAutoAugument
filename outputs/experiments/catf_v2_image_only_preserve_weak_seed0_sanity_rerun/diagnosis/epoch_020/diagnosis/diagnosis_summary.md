# Validation Error Diagnosis

- Status: completed
- TP: 677
- FP: 305
- FN: 216
- Precision: 0.6894
- Recall: 0.7481

## Diagnosis Vector
- small_object_score: 0.3260
- low_contrast_score: 0.5176
- class_imbalance_score: 0.9953
- localization_score: 0.0133
- false_positive_score: 0.3106

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=26 fn=0 precision=0.7111 recall=1.0000
- OK3: gt=274 tp=272 fp=69 fn=2 precision=0.7977 recall=0.9927
- 加强筋打伤: gt=8 tp=3 fp=0 fn=5 precision=1.0000 recall=0.3750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=34 fn=12 precision=0.1500 recall=0.3333
- 浅划伤: gt=13 tp=3 fp=0 fn=9 precision=1.0000 recall=0.2308
- 漏背锡: gt=46 tp=31 fp=37 fn=11 precision=0.4559 recall=0.6739
- 碰伤: gt=269 tp=171 fp=63 fn=93 precision=0.7308 recall=0.6357
- 脏污: gt=42 tp=11 fp=7 fn=31 precision=0.6111 recall=0.2619
- 轮廓划伤: gt=84 tp=52 fp=33 fn=32 precision=0.6118 recall=0.6190
- 锡丝残留: gt=12 tp=9 fp=7 fn=3 precision=0.5625 recall=0.7500
- 锡尖: gt=27 tp=18 fp=2 fn=8 precision=0.9000 recall=0.6667
- 锡膏: gt=46 tp=37 fp=27 fn=8 precision=0.5781 recall=0.8043
