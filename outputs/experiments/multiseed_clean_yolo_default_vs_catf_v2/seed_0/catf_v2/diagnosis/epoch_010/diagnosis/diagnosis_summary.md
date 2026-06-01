# Validation Error Diagnosis

- Status: completed
- TP: 632
- FP: 390
- FN: 253
- Precision: 0.6184
- Recall: 0.6983

## Diagnosis Vector
- small_object_score: 0.3667
- low_contrast_score: 0.4966
- class_imbalance_score: 0.9953
- localization_score: 0.0221
- false_positive_score: 0.3816

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=273 fp=59 fn=1 precision=0.8223 recall=0.9964
- 加强筋打伤: gt=8 tp=4 fp=0 fn=4 precision=1.0000 recall=0.5000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=3 fp=24 fn=15 precision=0.1111 recall=0.1667
- 浅划伤: gt=13 tp=6 fp=1 fn=7 precision=0.8571 recall=0.4615
- 漏背锡: gt=46 tp=15 fp=15 fn=23 precision=0.5000 recall=0.3261
- 碰伤: gt=269 tp=165 fp=78 fn=98 precision=0.6790 recall=0.6134
- 脏污: gt=42 tp=24 fp=92 fn=15 precision=0.2069 recall=0.5714
- 轮廓划伤: gt=84 tp=35 fp=47 fn=48 precision=0.4268 recall=0.4167
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=8 fp=4 fn=19 precision=0.6667 recall=0.2963
- 锡膏: gt=46 tp=35 fp=48 fn=9 precision=0.4217 recall=0.7609
