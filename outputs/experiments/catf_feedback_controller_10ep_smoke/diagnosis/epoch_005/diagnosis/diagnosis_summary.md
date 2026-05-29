# Validation Error Diagnosis

- Status: completed
- TP: 555
- FP: 277
- FN: 342
- Precision: 0.6671
- Recall: 0.6133

## Diagnosis Vector
- small_object_score: 0.4601
- low_contrast_score: 0.5053
- class_imbalance_score: 0.9953
- localization_score: 0.0088
- false_positive_score: 0.3329

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=29 fn=0 precision=0.6882 recall=1.0000
- OK3: gt=274 tp=269 fp=46 fn=5 precision=0.8540 recall=0.9818
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=2 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=6 fp=5 fn=7 precision=0.5455 recall=0.4615
- 漏背锡: gt=46 tp=17 fp=37 fn=24 precision=0.3148 recall=0.3696
- 碰伤: gt=269 tp=116 fp=71 fn=150 precision=0.6203 recall=0.4312
- 脏污: gt=42 tp=0 fp=0 fn=42 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=84 tp=43 fp=62 fn=41 precision=0.4095 recall=0.5119
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=25 fp=22 fn=2 precision=0.5319 recall=0.9259
- 锡膏: gt=46 tp=15 fp=3 fn=31 precision=0.8333 recall=0.3261
