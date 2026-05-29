# Validation Error Diagnosis

- Status: completed
- TP: 543
- FP: 304
- FN: 348
- Precision: 0.6411
- Recall: 0.6000

## Diagnosis Vector
- small_object_score: 0.4924
- low_contrast_score: 0.5213
- class_imbalance_score: 0.9953
- localization_score: 0.0155
- false_positive_score: 0.3589

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=270 fp=71 fn=2 precision=0.7918 recall=0.9854
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=0 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=5 fp=12 fn=40 precision=0.2941 recall=0.1087
- 碰伤: gt=269 tp=111 fp=44 fn=151 precision=0.7161 recall=0.4126
- 脏污: gt=42 tp=8 fp=23 fn=34 precision=0.2581 recall=0.1905
- 轮廓划伤: gt=84 tp=33 fp=30 fn=49 precision=0.5238 recall=0.3929
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=13 fp=8 fn=12 precision=0.6190 recall=0.4815
- 锡膏: gt=46 tp=39 fp=94 fn=7 precision=0.2932 recall=0.8478
