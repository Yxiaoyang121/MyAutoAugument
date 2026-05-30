# Validation Error Diagnosis

- Status: completed
- TP: 575
- FP: 368
- FN: 306
- Precision: 0.6098
- Recall: 0.6354

## Diagnosis Vector
- small_object_score: 0.4329
- low_contrast_score: 0.4897
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3902

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=14 fn=0 precision=0.8205 recall=1.0000
- OK3: gt=274 tp=269 fp=65 fn=5 precision=0.8054 recall=0.9818
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=12 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=16 fp=21 fn=28 precision=0.4324 recall=0.3478
- 碰伤: gt=269 tp=132 fp=100 fn=126 precision=0.5690 recall=0.4907
- 脏污: gt=42 tp=6 fp=53 fn=36 precision=0.1017 recall=0.1429
- 轮廓划伤: gt=84 tp=46 fp=57 fn=31 precision=0.4466 recall=0.5476
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=19 fp=6 fn=8 precision=0.7600 recall=0.7037
- 锡膏: gt=46 tp=23 fp=40 fn=19 precision=0.3651 recall=0.5000
