# Validation Error Diagnosis

- Status: completed
- TP: 538
- FP: 244
- FN: 346
- Precision: 0.6880
- Recall: 0.5945

## Diagnosis Vector
- small_object_score: 0.4788
- low_contrast_score: 0.4880
- class_imbalance_score: 0.9953
- localization_score: 0.0232
- false_positive_score: 0.3120

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=29 fn=0 precision=0.6882 recall=1.0000
- OK3: gt=274 tp=255 fp=41 fn=19 precision=0.8615 recall=0.9307
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=0 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=14 fp=8 fn=28 precision=0.6364 recall=0.3043
- 碰伤: gt=269 tp=155 fp=95 fn=104 precision=0.6200 recall=0.5762
- 脏污: gt=42 tp=0 fp=3 fn=42 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=84 tp=15 fp=20 fn=65 precision=0.4286 recall=0.1786
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=22 fp=28 fn=4 precision=0.4400 recall=0.8148
- 锡膏: gt=46 tp=13 fp=20 fn=31 precision=0.3939 recall=0.2826
